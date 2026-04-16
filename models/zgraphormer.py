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
        """
        Args:
            centrality: [N] discrete bin indices (0-9)
        Returns:
            enc: [N, d_model]
        """
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

        # Process each graph independently
        graph_ids = torch.unique(assign, sorted=True)
        outputs = []
        all_attns = []  # list of list of tensors per graph per layer
        for gid in graph_ids:
            mask = assign == gid
            h_g = h[mask]  # [N_i, d_model]
            z_g = z_matrices[gid] if isinstance(z_matrices, list) else z_matrices[mask][:, mask]
            graph_attns = []
            for layer in self.layers:
                if return_attention:
                    h_g, attn = layer(h_g, z_g, return_attention=True)
                    graph_attns.append(attn.detach().cpu())
                else:
                    h_g = layer(h_g, z_g)
            outputs.append(h_g)
            if return_attention:
                all_attns.append(graph_attns)

        h = torch.cat(outputs, dim=0)  # [total_N, d_model]

        volt = self.voltage_head(h)  # [total_N, 1]
        # Security: global mean pool per graph
        sec = torch.zeros(len(graph_ids), 1, device=h.device)
        for i, gid in enumerate(graph_ids):
            mask = assign == gid
            sec[i] = self.security_head(h[mask].mean(dim=0))

        if return_attention:
            return volt, sec, all_attns
        return volt, sec
