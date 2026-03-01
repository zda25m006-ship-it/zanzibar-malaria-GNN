"""
Spatio-Temporal Graph Neural Network for malaria importation prediction.
Combines GCN for spatial learning with GRU for temporal dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class SpatialBlock(nn.Module):
    """Graph convolution block for spatial feature extraction."""

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int = 4):
        super().__init__()
        self.conv = GATConv(in_channels, out_channels, heads=2,
                           concat=False, edge_dim=edge_dim,
                           add_self_loops=False)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv(x, edge_index, edge_attr=edge_attr)
        x = self.norm(x)
        return F.relu(x)


class TemporalBlock(nn.Module):
    """GRU block for temporal dynamics modeling."""

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_channels,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )

    def forward(self, x_seq):
        """
        Args:
            x_seq: [N, T, F] - N nodes, T time steps, F features
        Returns:
            output: [N, hidden]
        """
        output, _ = self.gru(x_seq)
        return output[:, -1, :]  # Last time step


class RainfallAttention(nn.Module):
    """Multi-head attention for weighting rainfall influence across regions."""

    def __init__(self, hidden_dim: int, num_heads: int = 2):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        Args:
            x: [N, D] node embeddings
        Returns:
            x: [N, D] attention-refined embeddings
        """
        # Treat nodes as sequence for self-attention
        x_3d = x.unsqueeze(0)  # [1, N, D]
        attn_out, _ = self.attention(x_3d, x_3d, x_3d)
        return self.norm(x + attn_out.squeeze(0))


class MalariaST_GNN(nn.Module):
    """
    Spatio-Temporal GNN combining:
    - GAT for spatial learning on the mobility graph
    - GRU for temporal dynamics across monthly snapshots
    - Multi-head attention for rainfall influence weighting

    Input: Sequence of T graph snapshots
    Output: Next month's predicted importation counts per node
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64,
                 out_channels: int = 1, num_spatial_layers: int = 2,
                 num_temporal_layers: int = 1, dropout: float = 0.3,
                 edge_dim: int = 4, num_attention_heads: int = 2):
        super().__init__()

        self.dropout = dropout
        self.hidden_channels = hidden_channels

        # Spatial blocks
        self.spatial_blocks = nn.ModuleList()
        self.spatial_blocks.append(SpatialBlock(in_channels, hidden_channels, edge_dim))
        for _ in range(num_spatial_layers - 1):
            self.spatial_blocks.append(SpatialBlock(hidden_channels, hidden_channels, edge_dim))

        # Temporal block
        self.temporal = TemporalBlock(hidden_channels, hidden_channels, num_temporal_layers)

        # Rainfall attention
        self.rainfall_attention = RainfallAttention(hidden_channels, num_attention_heads)

        # Output layers
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, out_channels)

        # Skip from input to output
        self.skip_proj = nn.Linear(in_channels, hidden_channels)

    def encode_spatial(self, x, edge_index, edge_attr=None):
        """Encode a single graph snapshot through spatial blocks."""
        for block in self.spatial_blocks:
            x = block(x, edge_index, edge_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, data, target_graph=None):
        """
        Args:
            data: either a single PyG Data object OR a list of Data objects
                  (list = temporal sequence mode, single = LOOCV single-step mode)
            target_graph: unused, kept for API compatibility
        Returns:
            predictions: [N] tensor of predicted counts per node
        """
        # Support both single-graph and sequence-of-graphs input
        if isinstance(data, (list, tuple)):
            graph_sequence = data
        else:
            graph_sequence = [data]  # treat single graph as 1-step sequence

        # Encode each temporal snapshot spatially
        spatial_embeddings = []
        for g in graph_sequence:
            h = self.encode_spatial(g.x, g.edge_index,
                                    g.edge_attr if hasattr(g, 'edge_attr') else None)
            spatial_embeddings.append(h)

        # Stack along time dimension: [N, T, H]
        x_temporal = torch.stack(spatial_embeddings, dim=1)

        # Temporal modeling
        x = self.temporal(x_temporal)  # [N, H]

        # Rainfall attention
        x = self.rainfall_attention(x)  # [N, H]

        # Skip connection from most recent input
        last_graph = graph_sequence[-1]
        skip = self.skip_proj(last_graph.x)
        x = x + skip

        # Output MLP
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.softplus(x)

        return x.squeeze(-1)

    def get_embeddings(self, graph_sequence):
        """Get temporal embeddings for all nodes."""
        spatial_embeddings = []
        for g in graph_sequence:
            h = self.encode_spatial(g.x, g.edge_index,
                                    g.edge_attr if hasattr(g, 'edge_attr') else None)
            spatial_embeddings.append(h)

        x_temporal = torch.stack(spatial_embeddings, dim=1)
        return self.temporal(x_temporal)
