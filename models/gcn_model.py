"""
Graph Convolutional Network (GCN) for malaria importation prediction.

Note: We do NOT use edge_weight — edge_attr shape doesn't match edge_index
(self-loops are in edge_index but not in edge_attr). GCNConv normalises by
node degree which is sufficient for this sparse graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MalariaGCN(nn.Module):
    """
    1-2 layer GCN for predicting monthly malaria importation counts per node.
    Lightweight for small dataset (14-19 training months per LOOCV fold).
    """

    def __init__(self, in_channels: int, hidden_channels: int = 32,
                 out_channels: int = 1, num_layers: int = 1,
                 dropout: float = 0.2, edge_dim: int = None):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Layer 1
        self.convs.append(GCNConv(in_channels, hidden_channels,
                                   add_self_loops=False, normalize=True))
        self.norms.append(nn.LayerNorm(hidden_channels))

        # Extra layers (if any)
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels,
                                       add_self_loops=False, normalize=True))
            self.norms.append(nn.LayerNorm(hidden_channels))

        # Skip connection
        self.skip = nn.Linear(in_channels, hidden_channels)

        # Output head
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Skip connection
        residual = self.skip(x)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)   # no edge_weight — see module docstring
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = x + residual
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.softplus(x)   # non-negative count prediction
        return x.squeeze(-1)

    def get_embeddings(self, data):
        """Return node embeddings (before output head)."""
        x, edge_index = data.x, data.edge_index
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
        return x
