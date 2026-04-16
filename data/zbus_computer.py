# data/zbus_computer.py

import numpy as np


def compute_zbus_magnitude(net):
    """
    Compute the magnitude of the Z-bus matrix from a pandapower net.
    Reuses the existing power flow result; does NOT call runpp again.
    Returns:
        z_mag: [N, N] numpy array of |Z_ij| in per-unit
    """
    ybus = net._ppc["internal"]["Ybus"].toarray()
    zbus = np.linalg.inv(ybus)
    z_mag = np.abs(zbus).astype(np.float32)
    return z_mag


def compute_centrality(net, num_bins: int = 10):
    """
    Compute binned electrical centrality based on row-sum |Ybus|.
    Reuses the existing power flow result; does NOT call runpp again.
    Returns:
        centrality: [N] integer bin indices (0 to num_bins-1)
    """
    ybus = net._ppc["internal"]["Ybus"].toarray()
    row_sums = np.abs(ybus).sum(axis=1)
    edges = np.linspace(row_sums.min(), row_sums.max(), num_bins + 1)
    bins = np.digitize(row_sums, edges[1:-1])
    bins = np.clip(bins, 0, num_bins - 1)
    return bins.astype(np.int64)
