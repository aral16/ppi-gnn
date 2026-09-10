"""GNN encoder + decoder for PPI link prediction.

The encoder is fully inductive: node inputs are structural descriptors plus a
Laplacian positional encoding (no free per-node embedding table), so the model
has to learn transferable neighbourhood rules rather than memorising node ids.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv


class GNNEncoder(nn.Module):
    def __init__(self, num_feats, hidden_dim, out_dim,
                 num_layers=2, dropout=0.3, conv="sage"):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(num_feats, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.dropout = dropout
        Conv = {"sage": SAGEConv, "gcn": GCNConv}[conv]
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            d_in = hidden_dim
            d_out = out_dim if i == num_layers - 1 else hidden_dim
            self.convs.append(Conv(d_in, d_out))
            self.norms.append(nn.LayerNorm(d_out))
        # skip connection from input MLP to the final space
        self.skip = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h0 = self.input(x)
        h = h0
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = self.norms[i](h)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h + self.skip(h0)


class LinkPredictor(nn.Module):
    """Symmetric pair features [z_i * z_j, (z_i - z_j)^2] -> MLP -> logit."""

    def __init__(self, dim, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, z, pairs):
        src, dst = pairs
        zi, zj = z[src], z[dst]
        feat = torch.cat([zi * zj, (zi - zj) ** 2], dim=-1)
        return self.net(feat).squeeze(-1)
