import math
import torch
import torch.nn as nn


class ZBusRelativeEncoding(nn.Module):
    """
    Converts absolute Z-bus impedance magnitudes to relative attention bias.
    Uses log-spaced bins to discretize |Z_ij| values.
    """

    def __init__(self, num_heads: int = 8, num_bins: int = 16, max_z: float = 5.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_bins = num_bins
        self.max_z = max_z

        # Log-spaced bins from a small epsilon to max_z
        # bin_edges: [num_bins + 1]
        self.register_buffer(
            "bin_edges",
            torch.logspace(
                math.log10(0.001), math.log10(max_z), steps=num_bins + 1
            ),
        )

        # Embedding lookup: each bin gets a vector of size num_heads
        self.embedding = nn.Embedding(num_bins, num_heads)

    def _digitize(self, z_vals: torch.Tensor) -> torch.Tensor:
        """Assign each |Z_ij| to a bin index."""
        # z_vals: [E]
        # Expand for broadcasting: [E, 1] vs [num_bins + 1]
        expanded = z_vals.unsqueeze(-1)  # [E, 1]
        edges = self.bin_edges.unsqueeze(0)  # [1, num_bins + 1]

        # Find which bin each value falls into
        # True where expanded >= edges
        ge = expanded >= edges  # [E, num_bins + 1]
        # Sum across edges to get bin index
        bin_idx = ge.sum(dim=-1) - 1  # [E]
        bin_idx = bin_idx.clamp(min=0, max=self.num_bins - 1)
        return bin_idx

    def forward(self, z_vals: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_vals: [|Z_ij|] values, shape [E]
        Returns:
            bias: [E, num_heads]
        """
        # Cap values at max_z to avoid overflow
        z_vals = z_vals.clamp(max=self.max_z)
        bin_idx = self._digitize(z_vals)
        bias = self.embedding(bin_idx)  # [E, num_heads]
        return bias
