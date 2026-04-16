import math
import torch
import torch.nn as nn
from models.zbus_encoding import ZBusRelativeEncoding


class GraphormerAttention(nn.Module):
    """Multi-head self-attention with Z-bus relative bias."""

    def __init__(self, d_model: int = 128, num_heads: int = 8, dropout: float = 0.1, num_z_bins: int = 16, max_z: float = 5.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.z_encoder = ZBusRelativeEncoding(num_heads=num_heads, num_bins=num_z_bins, max_z=max_z)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, z_matrix: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: [N, d_model]
            z_matrix: [N, N] impedance magnitude matrix
        Returns:
            out: [N, d_model]
        """
        N, _ = x.shape
        q = self.q_proj(x)  # [N, d_model]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [N, num_heads, head_dim]
        q = q.view(N, self.num_heads, self.head_dim)
        k = k.view(N, self.num_heads, self.head_dim)
        v = v.view(N, self.num_heads, self.head_dim)

        # Attention scores: [N, num_heads, N]
        scores = torch.einsum("nhd,mhd->nhm", q, k) * self.scale  # [N, num_heads, N]

        # Add Z-bus relative bias
        src = torch.arange(N, device=x.device).repeat_interleave(N)
        dst = torch.arange(N, device=x.device).repeat(N)
        z_vals = z_matrix[src, dst]  # [N*N]
        bias = self.z_encoder(z_vals)  # [N*N, num_heads]
        bias = bias.view(N, N, self.num_heads).permute(0, 2, 1)  # [N, num_heads, N]
        scores = scores + bias

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("nhm,mhd->nhd", attn, v)  # [N, num_heads, head_dim]
        out = out.reshape(N, self.d_model)
        out = self.out_proj(out)
        if return_attention:
            return out, attn
        return out


class GraphormerLayer(nn.Module):
    """Single Graphormer block: pre-norm attention + FFN."""

    def __init__(self, d_model: int = 128, num_heads: int = 8, d_ff: int = 512, dropout: float = 0.1, num_z_bins: int = 16, max_z: float = 5.0):
        super().__init__()
        self.attn = GraphormerAttention(d_model, num_heads, dropout, num_z_bins, max_z)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, z_matrix: torch.Tensor, return_attention: bool = False):
        if return_attention:
            attn_out, attn = self.attn(self.norm1(x), z_matrix, return_attention=True)
            x = x + attn_out
            x = x + self.ffn(self.norm2(x))
            return x, attn
        x = x + self.attn(self.norm1(x), z_matrix)
        x = x + self.ffn(self.norm2(x))
        return x
