import torch
import torch.nn as nn


class OuterProductUpdate(nn.Module):
    """
    S_i, S_j -> ΔZ_ij
    Nodes inform pair representation via outer product.
    """
    def __init__(self, d_node, d_pair, d_hidden=32):
        super().__init__()
        self.norm = nn.LayerNorm(d_node)
        self.proj_a = nn.Linear(d_node, d_hidden)
        self.proj_b = nn.Linear(d_node, d_hidden)
        self.out = nn.Linear(d_hidden * d_hidden, d_pair)

    def forward(self, s):
        # s: [B, N, d_node]
        s = self.norm(s)
        a = self.proj_a(s)  # [B, N, d_hidden]
        b = self.proj_b(s)  # [B, N, d_hidden]
        outer = torch.einsum('bid,bje->bijde', a, b)
        B, N, _, _, _ = outer.shape
        outer = outer.reshape(B, N, N, -1)
        return self.out(outer)  # [B, N, N, d_pair]


class TriangleMultiplicativeUpdate(nn.Module):
    """
    KVL analog: Z_ij updated via all triangle paths through k.
    Outgoing: i->k->j paths
    Incoming: k->i, k->j paths
    """
    def __init__(self, d_pair, outgoing=True):
        super().__init__()
        self.outgoing = outgoing
        self.norm_in = nn.LayerNorm(d_pair)
        self.proj_a = nn.Linear(d_pair, d_pair)
        self.proj_b = nn.Linear(d_pair, d_pair)
        self.gate_a = nn.Linear(d_pair, d_pair)
        self.gate_b = nn.Linear(d_pair, d_pair)
        self.gate_out = nn.Linear(d_pair, d_pair)
        self.proj_out = nn.Linear(d_pair, d_pair)
        self.norm_out = nn.LayerNorm(d_pair)

    def forward(self, z, key_mask=None):
        # z: [B, N, N, d_pair]
        z_normed = self.norm_in(z)
        a = torch.sigmoid(self.gate_a(z_normed)) * self.proj_a(z_normed)
        b = torch.sigmoid(self.gate_b(z_normed)) * self.proj_b(z_normed)

        if key_mask is not None:
            if self.outgoing:
                mask = key_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
            else:
                mask = key_mask.unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1]
            a = a.masked_fill(mask, 0.0)
            b = b.masked_fill(mask, 0.0)

        if self.outgoing:
            x = torch.einsum('bikd,bjkd->bijd', a, b)
        else:
            x = torch.einsum('bkid,bkjd->bijd', a, b)

        x = self.norm_out(x)
        g = torch.sigmoid(self.gate_out(z_normed))
        return g * self.proj_out(x)


class TriangleAttention(nn.Module):
    """
    Pair tokens attend to other pairs sharing a start node (row)
    or end node (column).
    """
    def __init__(self, d_pair, num_heads=4, row_wise=True):
        super().__init__()
        self.row_wise = row_wise
        self.attn = nn.MultiheadAttention(d_pair, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_pair)

    def forward(self, z, key_mask=None):
        # z: [B, N, N, d_pair]
        B, N, _, d = z.shape
        z_normed = self.norm(z)

        if self.row_wise:
            z_in = z_normed.reshape(B * N, N, d)
        else:
            z_in = z_normed.permute(0, 2, 1, 3).reshape(B * N, N, d)

        if key_mask is not None:
            attn_mask = key_mask.unsqueeze(1).expand(B, N, N).reshape(B * N, N)
        else:
            attn_mask = None

        out, _ = self.attn(z_in, z_in, z_in, key_padding_mask=attn_mask)
        out = out.reshape(B, N, N, d)

        if not self.row_wise:
            out = out.permute(0, 2, 1, 3)

        return out


class ZWeightedNodeAttention(nn.Module):
    """
    Pair repr Z_ij biases attention from node j to node i.
    """
    def __init__(self, d_node, d_pair, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_node // num_heads
        self.q = nn.Linear(d_node, d_node)
        self.k = nn.Linear(d_node, d_node)
        self.v = nn.Linear(d_node, d_node)
        self.b = nn.Linear(d_pair, num_heads)
        self.g = nn.Linear(d_node, d_node)
        self.o = nn.Linear(d_node, d_node)
        self.norm = nn.LayerNorm(d_node)

    def forward(self, s, z, key_mask=None, return_attention=False):
        # s: [B, N, d_node]
        # z: [B, N, N, d_pair]
        B, N, _ = s.shape
        s_normed = self.norm(s)

        q = self.q(s_normed).view(B, N, self.num_heads, self.head_dim)
        k = self.k(s_normed).view(B, N, self.num_heads, self.head_dim)
        v = self.v(s_normed).view(B, N, self.num_heads, self.head_dim)

        scores = torch.einsum('bnhd,bmhd->bnmh', q, k) / (self.head_dim ** 0.5)
        bias = self.b(z)  # [B, N, N, num_heads]
        scores = scores + bias

        if key_mask is not None:
            scores = scores.masked_fill(key_mask.unsqueeze(1).unsqueeze(-1), float('-inf'))

        attn = torch.softmax(scores, dim=2)
        attn = torch.nan_to_num(attn, 0.0)
        out = torch.einsum('bnmh,bmhd->bnhd', attn, v).reshape(B, N, -1)

        gate = torch.sigmoid(self.g(s_normed))
        out = gate * self.o(out)
        if return_attention:
            return out, attn
        return out


class ElecFormerBlock(nn.Module):
    def __init__(self, d_node=128, d_pair=64, num_heads=8, d_ff=512):
        super().__init__()
        self.outer_product = OuterProductUpdate(d_node, d_pair)
        self.tri_mul_out = TriangleMultiplicativeUpdate(d_pair, outgoing=True)
        self.tri_mul_in = TriangleMultiplicativeUpdate(d_pair, outgoing=False)
        self.tri_attn_row = TriangleAttention(d_pair, row_wise=True)
        self.tri_attn_col = TriangleAttention(d_pair, row_wise=False)
        self.pair_transition = nn.Sequential(
            nn.LayerNorm(d_pair),
            nn.Linear(d_pair, d_pair * 4),
            nn.ReLU(),
            nn.Linear(d_pair * 4, d_pair),
        )
        self.node_attn = ZWeightedNodeAttention(d_node, d_pair, num_heads)
        self.node_transition = nn.Sequential(
            nn.LayerNorm(d_node),
            nn.Linear(d_node, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_node),
        )

    def forward(self, s, z, key_mask=None, return_attention=False):
        z = z + self.outer_product(s)
        z = z + self.tri_mul_out(z, key_mask)
        z = z + self.tri_mul_in(z, key_mask)
        z = z + self.tri_attn_row(z, key_mask)
        z = z + self.tri_attn_col(z, key_mask)
        z = z + self.pair_transition(z)

        if return_attention:
            attn_out, attn = self.node_attn(s, z, key_mask, return_attention=True)
            s = s + attn_out
            s = s + self.node_transition(s)
            return s, z, attn

        s = s + self.node_attn(s, z, key_mask)
        s = s + self.node_transition(s)
        return s, z
