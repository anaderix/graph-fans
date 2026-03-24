"""Simple score network for node feature diffusion on fixed-topology graphs."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding following Vaswani et al."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed scalar timestep into a vector.

        Args:
            t: Timestep tensor, shape [1] or scalar.

        Returns:
            Embedding [dim].
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float() * emb
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
        return emb


class SimpleScoreNetwork(nn.Module):
    """Score network: noisy_features + timestep -> score estimate.

    Architecture: 3-layer GCN with sinusoidal timestep embedding and skip connections.
    Input: noisy node features [n_nodes, n_features] + t (scalar)
    Output: score [n_nodes, n_features] (same shape as input)
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        time_emb_dim: int = 32,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_layers = n_layers
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)

        # Input projection: features + time embedding
        self.input_proj = nn.Linear(n_features + time_emb_dim, hidden_dim)

        # GCN layers with LayerNorm for stability in deeper networks
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Skip connection projections (for residual)
        self.skip_projs = nn.ModuleList()
        for _ in range(n_layers):
            self.skip_projs.append(nn.Linear(hidden_dim, hidden_dim))

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, n_features)
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Noisy node features [n_nodes, n_features].
            t: Timestep [1] or scalar in [0, 1].
            edge_index: Edge index [2, n_edges] in COO format.

        Returns:
            Score estimate [n_nodes, n_features].
        """
        # Time embedding — broadcast to all nodes
        t_emb = self.time_emb(t.view(-1))  # [time_emb_dim]
        t_emb = t_emb.expand(x.size(0), -1)  # [n_nodes, time_emb_dim]

        # Concatenate features with time embedding
        h = torch.cat([x, t_emb], dim=-1)  # [n_nodes, n_features + time_emb_dim]
        h = self.act(self.input_proj(h))  # [n_nodes, hidden_dim]

        # GCN layers with skip connections and LayerNorm
        for conv, norm, skip in zip(self.convs, self.norms, self.skip_projs):
            h_skip = skip(h)
            h = self.act(norm(conv(h, edge_index))) + h_skip

        # Output projection
        return self.output_proj(h)  # [n_nodes, n_features]
