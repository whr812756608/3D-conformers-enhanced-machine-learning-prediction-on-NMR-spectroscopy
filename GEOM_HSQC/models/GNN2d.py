import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros
import torch.nn as nn
from torch_geometric.utils import scatter
from torch_geometric.nn import NNConv
from torch_geometric.nn.inits import zeros

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3
num_hybridization_type = 7

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 4
num_solvent_class = 9

# class Projection(nn.Module):
#     def __init__(self, input_size=2048, output_size=128, hidden_sizes=[512, 512], dropout=0.2):
#         super(Projection, self).__init__()

#         layers = []
#         layers.append(nn.Linear(input_size, hidden_sizes[0]))
#         for i in range(len(hidden_sizes) - 1):
#             layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
#             layers.append(nn.Dropout(dropout))
#             layers.append(nn.ReLU())

#         # Output layer
#         layers.append(nn.Linear(hidden_sizes[-1], output_size))

#         # Combine all layers
#         self.model = nn.Sequential(*layers)

#     def reset_parameters(self):
#         for layer in self.model:
#             if hasattr(layer, 'reset_parameters'):
#                 layer.reset_parameters()


#     def forward(self, x):
#         out = self.model(x)
#         return out
class GNNNodeEncoder(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0.2, gnn_type="gin", aggr='add'):
        super(GNNNodeEncoder, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        assert gnn_type in ["gin", "gcn", "gat", "graphsage", "nnconv"], "GNN type not implemented."

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)
        self.x_embedding3 = torch.nn.Embedding(num_hybridization_type, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr=aggr))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim, aggr=aggr))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim, aggr=aggr))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim, aggr=aggr))
            elif gnn_type == "nnconv":
                self.gnns.append(NNConv(emb_dim, aggr=aggr, bias=True))
            else:
                raise ValueError("Invalid graph convolution type.")

        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batch):
        x, edge_index, edge_attr, b = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        x = self.x_embedding1(x[:, 0].long()) + self.x_embedding2(x[:, 1].long()) + self.x_embedding3(x[:, 2].long())

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

            ### Different implementations of JK
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)
        else:
            raise ValueError("Invalid Jump knowledge.")

        return node_representation


# class NodeEncodeInterface(nn.Module):
#     def __init__(self,
#                  node_encoder,
#                  hidden_channels=256, #Node Embedding Dimmension
#                  c_out_channels=1,
#                  h_out_channels=1,
#                  c_out_hidden = [256, 512],
#                  h_out_hidden = [256, 512],
#                  c_solvent_emb_dim = 32,
#                  h_solvent_emb_dim = 32,
#                  use_solvent=True):
#         super(NodeEncodeInterface, self).__init__()
#         self.hidden_channels = hidden_channels
#         self.c_out_channels = c_out_channels
#         self.h_out_channels = h_out_channels
#         self.node_encoder = node_encoder
#         self.use_solvent = use_solvent
#         if self.use_solvent:
#             self.lin_out_c = Projection(hidden_channels + c_solvent_emb_dim, c_out_channels, c_out_hidden) # DO NOT USE SOLVENT ON C, + solvent_emb_dim
#             self.lin_out_h = Projection(hidden_channels + h_solvent_emb_dim, h_out_channels, h_out_hidden)
#             self.c_solvent_embedding = torch.nn.Embedding(num_solvent_class, c_solvent_emb_dim)
#             self.h_solvent_embedding = torch.nn.Embedding(num_solvent_class, h_solvent_emb_dim)
#         else:
#             self.lin_out_c = Projection(hidden_channels, c_out_channels, c_out_hidden)
#             self.lin_out_h = Projection(hidden_channels, h_out_channels, h_out_hidden)

#     def predict(self, x, npz_data):
#         z = npz_data.x
#         batch = npz_data.batch

#         # Identify carbon and hydrogen nodes
#         carbon_nodes = (z[:, 0] == 5).nonzero(as_tuple=True)[0]  # Assuming the first feature indicates the atom type
#         hydrogen_nodes = (z[:, 0] == 0).nonzero(as_tuple=True)[0]

#         c_features = x[carbon_nodes]
#         h_features = x[hydrogen_nodes]
        
#         # solvent impact
#         if self.use_solvent:
#             solvent_class = npz_data.solvent_class
#             ##### Embedding solvent class
#             c_solvent_class = self.c_solvent_embedding(solvent_class)
#             h_solvent_class = self.h_solvent_embedding(solvent_class)
#             # batch=npz_data.batch
#             c_solvent_class_per_node = c_solvent_class[batch[carbon_nodes]]
#             h_solvent_class_per_node = h_solvent_class[batch[hydrogen_nodes]]
#             c_features = torch.concat([c_solvent_class_per_node, c_features], dim=1)
#             h_features = torch.concat([h_solvent_class_per_node, h_features], dim=1)

#         # for each c, gather its h features and predict h shifts
#         h_features_average = []
#         c_idx_connected_h = [] # this is the graph index that represents C node connected to H
#         c_pred_idx = []
#         # Loop through each carbon node
#         for c_idx, c_node in enumerate(carbon_nodes):
#             # Find edges where the carbon node is a source
#             connected_edges = (npz_data.edge_index[0] == c_node).nonzero(as_tuple=True)[0]

#             # Find corresponding target nodes
#             connected_hydrogens = [npz_data.edge_index[1, e].item() for e in connected_edges if
#                                    npz_data.edge_index[1, e] in hydrogen_nodes]

#             if len(connected_hydrogens) > 0:
#                 c_idx_connected_h.append(c_node)
#                 c_pred_idx.append(c_idx)
#                 # Extract features for these hydrogen nodes
#                 h_node_features = x[connected_hydrogens]
#                 # Calculate the average of these features
#                 avg_h_features = torch.mean(h_node_features, dim=0)
#                 h_features_average.append(avg_h_features)
#             else:
#                 continue

#         h_features = torch.stack(h_features_average)

#         out_c = self.lin_out_c(c_features)
#         out_c = out_c[c_pred_idx]
#         out_h = self.lin_out_h(h_features)
#         # print(out_h.shape)
#         out = [out_c, out_h]
#         return out, c_idx_connected_h

#     def forward(self, batch_data):
#         x =  self.node_encoder(batch_data)
#         out = self.predict(x, batch_data)

#         return out



class GNNGraphEncoder(torch.nn.Module):
    def __init__(self, node_encoder, emb_dim, graph_pooling="add"):
        super(GNNGraphEncoder, self).__init__()
        self.node_encoder = node_encoder
        self.emb_dim = emb_dim

        if graph_pooling in ["sum", "add"]:
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")

        self.readout = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(emb_dim, emb_dim))

    def forward(self, batch):

        node_representation = self.node_encoder(batch)
        graph_embedding = self.pool(node_representation, batch.batch)
        graph_embedding = self.readout(graph_embedding)
        return graph_embedding


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.

    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        # edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) # only use bond type as edge attr

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_index = edge_index[0]
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
        row, col = edge_index
        deg = scatter(edge_weight, row, dim=0, dim_size=num_nodes, reduce='sum')
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        norm = self.norm(edge_index, x.size(0), x.dtype)
        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_index = edge_index[0]
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
        row, col = edge_index
        deg = scatter(edge_weight, row, dim=0, dim_size=num_nodes, reduce='sum')
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        norm = self.norm(edge_index, x.size(0), x.dtype)
        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class NNConv(MessagePassing):
    """
    Reference: `"Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ and `"Neural Message Passing for NMR Chemical Shift Prediction"
    <https://pubs.acs.org/doi/10.1021/acs.jcim.0c00195>`_.
    """

    def __init__(self, emb_dim, aggr="add", bias=False):
        super(NNConv, self).__init__()
        self.aggr = aggr
        self.emb_dim = emb_dim

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        self.edge_nn = torch.nn.Linear(emb_dim, emb_dim * emb_dim)
        self.gru = nn.GRU(emb_dim, emb_dim)

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))
            zeros(self.bias)
        else:
            self.bias = None

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        x = self.linear(x)
        out = self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

        if self.bias is not None:
            out = out + self.bias

        _, out = self.gru(out.unsqueeze(0), x.unsqueeze(0))
        out = out.squeeze(0)

        return out

    def message(self, x_j, edge_attr):
        weight = self.edge_nn(edge_attr)
        weight = weight.view(-1, self.emb_dim, self.emb_dim)
        return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)
        self.out = torch.nn.Linear(heads * emb_dim, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_index = edge_index[0]
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
        row, col = edge_index
        deg = scatter(edge_weight, row, dim=0, dim_size=num_nodes, reduce='sum')
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        norm = self.norm(edge_index, x.size(0), x.dtype)
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        x = self.weight_linear(x)
        out = self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)
        return self.out(out)

    def message(self, edge_index, x_i, x_j, edge_attr):
        x_i = x_i.view(-1, self.heads, self.emb_dim)
        x_j = x_j.view(-1, self.heads, self.emb_dim)
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j = x_j + edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])
        out = x_j * alpha.view(-1, self.heads, 1)
        out = out.view(-1, self.heads * self.emb_dim)
        return out


