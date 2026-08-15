import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class PIGNN(nn.Module):
    """
    Physics-Informed Graph Neural Network using GATv2 spatial message passing 
    and an MLP continuous acoustic field decoder.
    """
    def __init__(self, in_channels: int = 3, hidden_channels: int = 64, out_channels: int = 1):
        super().__init__()
        # Graph Message-Passing Encoder
        self.conv1 = GATv2Conv(in_channels, hidden_channels, edge_dim=1)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, edge_dim=1)

        # Continuous Field Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.conv1(x, edge_index, edge_attr))
        h = F.relu(self.conv2(h, edge_index, edge_attr))
        out = self.decoder(h)
        return out, h