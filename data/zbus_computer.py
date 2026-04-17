import numpy as np

# Global centrality edges covering all IEEE standard cases (14 to 118 bus)
GLOBAL_CENTRALITY_EDGES = np.logspace(-3, 3, 11).astype(np.float32)


def compute_zbus_magnitude(net):
    """
    Compute |Z-bus| from an already-solved pandapower net.
    Reuses net._ppc['internal']['Ybus']; does NOT call runpp.
    """
    ybus = net._ppc["internal"]["Ybus"].toarray()
    zbus = np.linalg.inv(ybus)
    z_mag = np.abs(zbus).astype(np.float32)
    return z_mag


def compute_centrality(net, num_bins=10, global_edges=None):
    """
    Compute binned electrical centrality with globally fixed edges.
    Reuses net._ppc['internal']['Ybus']; does NOT call runpp.
    """
    if global_edges is None:
        global_edges = GLOBAL_CENTRALITY_EDGES
    ybus = net._ppc["internal"]["Ybus"].toarray()
    row_sums = np.abs(ybus).sum(axis=1)
    bins = np.digitize(row_sums, global_edges[1:-1])
    bins = np.clip(bins, 0, num_bins - 1)
    return bins.astype(np.int64)
