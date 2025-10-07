from torch_geometric.nn import TransformerConv
import torch

class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphTransformer, self).__init__()
        self.conv1 = TransformerConv(in_channels, out_channels)
        self.conv2 = TransformerConv(out_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        return x
