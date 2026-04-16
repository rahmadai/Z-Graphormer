# models/zgraphormer.py

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from models.graphormer_layer import GraphormerLayer


class CentralityEncoding(nn.Module):
    """Binned centrality (electrical degree) encoding per node."""

    def __init__(self, num_bins: int = 10, d_model: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(num_bins, d_model)

    def forward(self, centrality: torch.Tensor) -> torch.Tensor:
        return self.embedding(centrality)


class ZGraphormer(nn.Module):
    """
    Structure-Aware Transformer with Z-bus relative encoding.
    """

    def __init__(
        self,
        in_channels: int = 5,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        num_z_bins: int = 16,
        max_z: float = 5.0,
        num_centrality_bins: int = 10,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_channels, d_model)
        self.centrality_enc = CentralityEncoding(num_centrality_bins, d_model)

        self.layers = nn.ModuleList(
            [
                GraphormerLayer(d_model, num_heads, d_ff, dropout, num_z_bins, max_z)
                for _ in range(num_layers)
            ]
        )

        self.voltage_head = nn.Linear(d_model, 1)
        self.security_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, batch, return_attention: bool = False):
        """
        Args:
            batch: torch_geometric.data.Batch with attributes:
                - x: [total_N, in_channels] node features
                - centrality: [total_N] binned centrality indices
                - z_matrix: list of [N_i, N_i] tensors (impedance magnitudes)
                - batch: [total_N] assignment vector
        Returns:
            volt: [total_N, 1] predicted voltage magnitudes
            sec:  [num_graphs, 1] predicted security logits
        """
        x = batch.x
        centrality = batch.centrality
        z_matrices = batch.z_matrix
        assign = batch.batch

        # Input projection + centrality
        h = self.in_proj(x) + self.centrality_enc(centrality)

        # Build padded batch for batched attention
        graph_ids = torch.unique(assign, sorted=True)
        sizes = [(assign == gid).sum().item() for gid in graph_ids]
        max_n = max(sizes)
        B = len(graph_ids)

        h_pad = torch.zeros(B, max_n, h.shape[-1], device=h.device)
        z_pad = torch.zeros(B, max_n, max_n, device=h.device)
        key_mask = torch.ones(B, max_n, dtype=torch.bool, device=h.device)

        for i, gid in enumerate(graph_ids):
            mask = assign == gid
            n_i = sizes[i]
            h_pad[i, :n_i] = h[mask]
            z_pad[i, :n_i, :n_i] = z_matrices[i].to(h.device)
            key_mask[i, :n_i] = False  # False = valid token

        # Run all layers on the padded batch — single GPU call
        all_attns = []
        for layer in self.layers:
            if return_attention:
                h_pad, attn = layer(h_pad, z_pad, key_mask=key_mask, return_attention=True)
                all_attns.append(attn.detach().cpu())
            else:
                h_pad = layer(h_pad, z_pad, key_mask=key_mask)

        # Unpad node representations
        h_list = [h_pad[i, :sizes[i]] for i in range(B)]
        h = torch.cat(h_list, dim=0)  # [total_N, d_model]

        volt = self.voltage_head(h)  # [total_N, 1]

        # Security: global mean pool per graph (mask-aware, no Python loop)
        valid_counts = (~key_mask).sum(dim=1).clamp(min=1)  # [B]
        h_mean = (h_pad * (~key_mask).unsqueeze(-1)).sum(dim=1) / valid_counts.unsqueeze(-1)  # [B, d_model]
        sec = self.security_head(h_mean)  # [B, 1]

        if return_attention:
            return volt, sec, all_attns
        return volt, sec
