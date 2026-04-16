# models/graphormer_layer.py

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

    def forward(self, x: torch.Tensor, z_matrix: torch.Tensor, key_mask: torch.Tensor = None, return_attention: bool = False):
        """
        Args:
            x: [B, N, d_model]
            z_matrix: [B, N, N] impedance magnitude matrix
            key_mask: [B, N] bool, True for padded positions
        Returns:
            out: [B, N, d_model]
        """
        B, N, _ = x.shape
        q = self.q_proj(x)  # [B, N, d_model]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [B, num_heads, N, head_dim]
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: [B, num_heads, N, N]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Add Z-bus relative bias
        z_vals = z_matrix.reshape(-1)  # [B*N*N]
        bias = self.z_encoder(z_vals)  # [B*N*N, num_heads]
        bias = bias.view(B, N, N, self.num_heads).permute(0, 3, 1, 2)  # [B, num_heads, N, N]
        scores = scores + bias

        if key_mask is not None:
            # key_mask: [B, N] -> [B, 1, 1, N]
            scores = scores.masked_fill(key_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, 0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, num_heads, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, self.d_model)
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

    def forward(self, x: torch.Tensor, z_matrix: torch.Tensor, key_mask: torch.Tensor = None, return_attention: bool = False):
        if return_attention:
            attn_out, attn = self.attn(self.norm1(x), z_matrix, key_mask=key_mask, return_attention=True)
            x = x + attn_out
            x = x + self.ffn(self.norm2(x))
            return x, attn
        x = x + self.attn(self.norm1(x), z_matrix, key_mask=key_mask)
        x = x + self.ffn(self.norm2(x))
        return x
