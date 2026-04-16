import numpy as np
import pandapower as pp


def compute_zbus_magnitude(net):
    """
    Compute the magnitude of the Z-bus matrix from a pandapower net.
    Returns:
        z_mag: [N, N] numpy array of |Z_ij| in per-unit
    """
    # Get the admittance matrix Ybus (complex)
    # pandapower stores Ybus in net._ppc["internal"]["Ybus"] after a power flow
    pp.runpp(net, verbose=False, init="auto")
    ybus = net._ppc["internal"]["Ybus"].toarray()

    # Invert to get Zbus
    zbus = np.linalg.inv(ybus)
    z_mag = np.abs(zbus)
    return z_mag


def compute_centrality(net, num_bins: int = 10):
    """
    Compute binned electrical centrality based on row-sum |Ybus|.
    Returns:
        centrality: [N] integer bin indices (0 to num_bins-1)
    """
    pp.runpp(net, verbose=False, init="auto")
    ybus = net._ppc["internal"]["Ybus"].toarray()
    row_sums = np.abs(ybus).sum(axis=1)
    # Digitize into bins
    edges = np.linspace(row_sums.min(), row_sums.max(), num_bins + 1)
    bins = np.digitize(row_sums, edges[1:-1])
    bins = np.clip(bins, 0, num_bins - 1)
    return bins.astype(np.int64)
