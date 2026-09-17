import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class PIGNN(nn.Module):
    """
    Physics-Informed Graph Neural Network using GATv2 spatial message passing 
    and an MLP continuous acoustic field decoder.
    """
    def __init__(self, in_channels: int = 3, hidden_channels: int = 64, out_channels: int = 2):
        super().__init__()
        # Graph Message-Passing Encoder
        self.conv1 = GATv2Conv(in_channels, hidden_channels, edge_dim=1)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, edge_dim=1)

        # Continuous Field Decoder (Outputs 2 values: [Intensity, N^2])
        # In src/model.py
        self.decoder = nn.Sequential(
   
    nn.Linear(hidden_channels, hidden_channels),
    nn.ReLU(),
    nn.Linear(hidden_channels, 2)  # Raw linear output
) # Raw output! NO ReLU here!


    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.conv1(x, edge_index, edge_attr))
        h = F.relu(self.conv2(h, edge_index, edge_attr))
        
        # out shape: [num_nodes, 2]
        out = self.decoder(h)
        
        pred_intensity = out[:, 0:1]
        pred_N2 = out[:, 1:2]
        
        return pred_intensity, pred_N2