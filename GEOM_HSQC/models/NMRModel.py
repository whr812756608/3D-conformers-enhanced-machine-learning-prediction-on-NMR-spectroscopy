import torch
import torch.nn as nn

num_solvent_class = 9

class NodeEncodeInterface(nn.Module):
    def __init__(self,
                 node_encoder,
                 hidden_channels=256, #Node Embedding Dimmension
                 c_out_channels=1,
                 h_out_channels=1,
                 c_out_hidden = [256, 512],
                 h_out_hidden = [256, 512],
                 c_solvent_emb_dim = 32,
                 h_solvent_emb_dim = 32,
                 use_solvent=True):
        super(NodeEncodeInterface, self).__init__()
        self.hidden_channels = hidden_channels
        self.c_out_channels = c_out_channels
        self.h_out_channels = h_out_channels
        self.node_encoder = node_encoder
        self.use_solvent = use_solvent
        if self.use_solvent:
            self.lin_out_c = Projection(hidden_channels + c_solvent_emb_dim, c_out_channels, c_out_hidden) # DO NOT USE SOLVENT ON C, + solvent_emb_dim
            self.lin_out_h = Projection(hidden_channels + h_solvent_emb_dim, h_out_channels, h_out_hidden)
            self.c_solvent_embedding = torch.nn.Embedding(num_solvent_class, c_solvent_emb_dim)
            self.h_solvent_embedding = torch.nn.Embedding(num_solvent_class, h_solvent_emb_dim)
        else:
            self.lin_out_c = Projection(hidden_channels, c_out_channels, c_out_hidden)
            self.lin_out_h = Projection(hidden_channels, h_out_channels, h_out_hidden)

    def predict(self, x, data):
        z = data.x
        batch = data.batch

        # Identify carbon and hydrogen nodes
        carbon_nodes = (z[:, 0] == 5).nonzero(as_tuple=True)[0]  # Assuming the first feature indicates the atom type
        hydrogen_nodes = (z[:, 0] == 0).nonzero(as_tuple=True)[0]

        c_features = x[carbon_nodes]
        # h_features = x[hydrogen_nodes]
        
        # solvent impact
        if self.use_solvent:
            solvent_class = data.solvent_class 
            ##### Embedding solvent class
            c_solvent_class = self.c_solvent_embedding(solvent_class)
            h_solvent_class = self.h_solvent_embedding(solvent_class)
            # batch=npz_data.batch
            c_solvent_class_per_node = c_solvent_class[batch[carbon_nodes]]
            # h_solvent_class_per_node = h_solvent_class[batch[hydrogen_nodes]]
            c_features = torch.concat([c_solvent_class_per_node, c_features], dim=1)
            # h_features = torch.concat([h_solvent_class_per_node, h_features], dim=1)

        # for each c, gather its h features and predict h shifts
        h_features_average = []
        c_idx_connected_h = [] # this is the graph index that represents C node connected to H
        c_pred_idx = []
        # Loop through each carbon node
        for c_idx, c_node in enumerate(carbon_nodes):
            # Find edges where the carbon node is a source
            connected_edges = (data.edge_index[0] == c_node).nonzero(as_tuple=True)[0]

            # Find corresponding target nodes
            connected_hydrogens = [data.edge_index[1, e].item() for e in connected_edges if
                                   data.edge_index[1, e] in hydrogen_nodes]

            if len(connected_hydrogens) > 0:
                c_idx_connected_h.append(c_node)
                c_pred_idx.append(c_idx)
                # Extract features for these hydrogen nodes
                h_node_features = x[connected_hydrogens]
                h_solvent_class_per_node = h_solvent_class[batch[connected_hydrogens]]
                h_node_features = torch.concat([h_solvent_class_per_node, h_node_features], dim=1)
                
                # Calculate the average of these features
                avg_h_features = torch.mean(h_node_features, dim=0)
                h_features_average.append(avg_h_features)
            else:
                continue

        h_features = torch.stack(h_features_average)

        out_c = self.lin_out_c(c_features)
        out_c = out_c[c_pred_idx]
        out_h = self.lin_out_h(h_features)
        # print(out_h.shape)
        out = [out_c, out_h]
        return out, c_idx_connected_h

    def forward(self, batch_data):
        x =  self.node_encoder(batch_data)
        out = self.predict(x, batch_data)

        return out

class Projection(nn.Module):
    def __init__(self, input_size=2048, output_size=128, hidden_sizes=[512, 512], dropout=0.2):
        super(Projection, self).__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # Combine all layers
        self.model = nn.Sequential(*layers)

    def reset_parameters(self):
        for layer in self.model:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


    def forward(self, x):
        out = self.model(x)
        return out