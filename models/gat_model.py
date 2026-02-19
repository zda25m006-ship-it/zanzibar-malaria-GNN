"""
Graph Attention Network (GAT) for malaria importation prediction.
Uses attention over travel edges to learn which routes contribute most.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class MalariaGAT(nn.Module):
    """
    Multi-head GAT for predicting malaria importation with interpretable attention.

    Architecture:
        2-layer GATConv with multi-head attention
        Edge features inform attention computation
        Returns attention weights for interpretability
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64,
                 out_channels: int = 1, num_heads: int = 4,
                 dropout: float = 0.3, edge_dim: int = 4):
        super().__init__()

        self.num_heads = num_heads
        self.dropout = dropout

        # GAT layers
        self.conv1 = GATConv(
            in_channels, hidden_channels,
            heads=num_heads, dropout=dropout,
            edge_dim=edge_dim, concat=True
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)

        self.conv2 = GATConv(
            hidden_channels * num_heads, hidden_channels,
            heads=num_heads, dropout=dropout,
            edge_dim=edge_dim, concat=False
        )
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        # Output MLP
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, out_channels)

        # Skip connection
        self.skip = nn.Linear(in_channels, hidden_channels)

        # Store attention weights for interpretability
        self.attention_weights = None

    def forward(self, data, return_attention=False):
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        # Skip connection
        residual = self.skip(x)

        # Layer 1
        if return_attention:
            x, attn1 = self.conv1(x, edge_index, edge_attr=edge_attr,
                                   return_attention_weights=True)
        else:
            x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        if return_attention:
            x, attn2 = self.conv2(x, edge_index, edge_attr=edge_attr,
                                   return_attention_weights=True)
            self.attention_weights = (attn1, attn2)
        else:
            x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.elu(x)

        # Add residual
        x = x + residual

        # Output
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.softplus(x)

        if return_attention:
            return x.squeeze(-1), self.attention_weights
        return x.squeeze(-1)

    def get_attention_weights(self, data):
        """Get attention weights for interpretability."""
        self.eval()
        with torch.no_grad():
            _, attn = self.forward(data, return_attention=True)
        return attn

    def get_embeddings(self, data):
        """Get node embeddings before output layer."""
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        return x
