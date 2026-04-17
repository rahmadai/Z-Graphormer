import torch
import torch.nn as nn

from models.elecformer_block import ElecFormerBlock


class ElecFormer(nn.Module):
    """
    Evoformer-inspired architecture for inductive power flow generalization.
    """

    def __init__(
        self,
        in_channels: int = 9,
        d_node: int = 128,
        d_pair: int = 64,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_node = d_node
        self.d_pair = d_pair
        self.node_proj = nn.Linear(in_channels, d_node)

        # Pair init: [z_norm, log(z_norm), z_norm^2] -> d_pair
        self.pair_init = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, d_pair),
        )

        self.blocks = nn.ModuleList(
            [
                ElecFormerBlock(d_node, d_pair, num_heads, d_ff)
                for _ in range(num_layers)
            ]
        )

        self.voltage_head = nn.Linear(d_node, 1)
        self.security_head = nn.Sequential(
            nn.Linear(d_node, d_node // 2),
            nn.ReLU(),
            nn.Linear(d_node // 2, 1),
        )

    def forward(self, batch, return_attention: bool = False):
        """
        Args:
            batch: PyG Batch with:
                - x: [total_N, in_channels]
                - z_matrix: list of [N_i, N_i]
                - batch: assignment vector
        Returns:
            volt: [total_N, 1]
            sec: [num_graphs, 1]
        """
        x = batch.x
        z_matrices = batch.z_matrix
        assign = batch.batch

        # Project nodes once
        h = self.node_proj(x)  # [total_N, d_node]

        # Build padded batch
        graph_ids = torch.unique(assign, sorted=True)
        sizes = [(assign == gid).sum().item() for gid in graph_ids]
        max_n = max(sizes)
        B = len(graph_ids)
        device = x.device

        s_pad = torch.zeros(B, max_n, self.d_node, device=device)
        z_pad = torch.zeros(B, max_n, max_n, 3, device=device)
        key_mask = torch.ones(B, max_n, dtype=torch.bool, device=device)

        for i, gid in enumerate(graph_ids):
            mask = assign == gid
            n_i = sizes[i]
            s_pad[i, :n_i] = h[mask]

            z_g = z_matrices[i].to(device)  # [N_i, N_i]
            z_median = z_g.median()
            z_norm = z_g / (z_median + 1e-8)
            z_features = torch.stack(
                [
                    z_norm,
                    torch.log(z_norm + 1e-8),
                    z_norm ** 2,
                ],
                dim=-1,
            )  # [N_i, N_i, 3]
            z_pad[i, :n_i, :n_i] = z_features
            key_mask[i, :n_i] = False

        z_pad = self.pair_init(z_pad)  # [B, max_n, max_n, d_pair]

        # Run ElecFormer blocks
        all_attns = []
        for block in self.blocks:
            if return_attention:
                s_pad, z_pad, attn = block(s_pad, z_pad, key_mask, return_attention=True)
                all_attns.append(attn)
            else:
                s_pad, z_pad = block(s_pad, z_pad, key_mask)

        # Unpad node representations
        s_list = [s_pad[i, :sizes[i]] for i in range(B)]
        s = torch.cat(s_list, dim=0)  # [total_N, d_node]

        volt = self.voltage_head(s)  # [total_N, 1]

        # Security: mask-aware mean pool
        valid_counts = (~key_mask).sum(dim=1).clamp(min=1)
        s_mean = (s_pad * (~key_mask).unsqueeze(-1)).sum(dim=1) / valid_counts.unsqueeze(-1)
        sec = self.security_head(s_mean)  # [B, 1]

        if return_attention:
            return volt, sec, all_attns
        return volt, sec
