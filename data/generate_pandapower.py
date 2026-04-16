# data/generate_pandapower.py

import os
import argparse
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import pandapower as pp
import pandapower.networks as pn
from tqdm import tqdm

from data.zbus_computer import compute_zbus_magnitude, compute_centrality


SYSTEMS = {
    "case14": pn.case14,
    "case30": pn.case30,
    "case39": pn.case39,
    "case57": pn.case57,
    "case118": pn.case118,
}


def generate_sample(base_net, contingency_prob: float = 0.3, load_scale_range=(0.6, 1.4)):
    """
    Generate one power flow sample with random load scaling and possible N-1 contingency.
    Returns:
        net (after power flow), converged (bool), features (np.ndarray)
    """
    import copy
    net = copy.deepcopy(base_net)

    # Random load scaling
    scale = np.random.uniform(*load_scale_range)
    net.load["p_mw"] *= scale
    net.load["q_mvar"] *= scale

    # Optional N-1 contingency
    line_out = False
    if np.random.rand() < contingency_prob and len(net.line) > 0:
        line_idx = np.random.randint(len(net.line))
        net.line.loc[line_idx, 'in_service'] = False
        line_out = True

    # Run power flow
    try:
        pp.runpp(net, verbose=False, init="auto")
        converged = net.converged
    except Exception:
        converged = False

    if not converged:
        return None

    # Extract node features
    # For each bus: [V_pu, delta_deg, P_gen, Q_gen, type_onehot...]
    # Simplify: [V, delta, P, Q, type] where type is one-hot encoded in a scalar for simplicity
    # Actually let's use 5-dim: [V, delta, P, Q, type_idx] and handle type in model if needed
    n_bus = len(net.bus)
    res_bus = net.res_bus
    # Map bus index to results
    vm = res_bus.vm_pu.values
    va = res_bus.va_degree.values

    # Compute injected P, Q per bus
    p_inj = np.zeros(n_bus)
    q_inj = np.zeros(n_bus)

    # Add generation
    for _, gen in net.gen.iterrows():
        bus = gen.bus
        p_inj[bus] += gen.p_mw
        # Q from results
    for _, sgen in net.sgen.iterrows():
        bus = sgen.bus
        p_inj[bus] += sgen.p_mw

    # Subtract load
    for _, load in net.load.iterrows():
        bus = load.bus
        p_inj[bus] -= load.p_mw
        q_inj[bus] -= load.q_mvar

    # Gen Q from res_gen if available
    if len(net.res_gen) > 0:
        for idx, q in net.res_gen.q_mvar.items():
            bus = net.gen.bus.loc[idx]
            q_inj[bus] += q

    bus_type = net.bus.in_service.values.astype(np.float32)  # placeholder; use 0/1 for now
    # Better: encode bus type (ref=3, pv=2, pq=1) normalized
    type_map = {"ref": 3, "pv": 2, "pq": 1}
    type_vals = net.bus["type"].map(type_map).fillna(1).values.astype(np.float32) / 3.0

    features = np.stack([vm, va, p_inj, q_inj, type_vals], axis=1).astype(np.float32)

    # Security label
    vm_secure = np.all((vm >= 0.95) & (vm <= 1.05))
    line_load = net.res_line.loading_percent.values if len(net.res_line) > 0 else np.array([])
    line_secure = np.all(line_load < 100.0) if len(line_load) > 0 else True
    secure = float(vm_secure and line_secure and converged)

    # Z-bus and centrality
    z_mag = compute_zbus_magnitude(net)
    centrality = compute_centrality(net, num_bins=10)

    # Sanity check: dimensions must match number of buses
    if z_mag.shape[0] != n_bus or centrality.shape[0] != n_bus or features.shape[0] != n_bus:
        return None

    data = Data(
        x=torch.from_numpy(features),
        z_matrix=torch.from_numpy(z_mag).float(),
        centrality=torch.from_numpy(centrality).long(),
        y_volt=torch.from_numpy(vm).float().unsqueeze(1),  # target voltage
        y_sec=torch.tensor([secure], dtype=torch.float),
        n_bus=n_bus,
    )
    return data


class PowerFlowDataset(Dataset):
    def __init__(self, root, system_names, num_samples, transform=None, pre_transform=None):
        self.system_names = system_names
        self.num_samples = num_samples
        super().__init__(root, transform, pre_transform)

    @property
    def processed_file_names(self):
        return [f"{name}_data.pt" for name in self.system_names]

    def process(self):
        for name in self.system_names:
            net_fn = SYSTEMS[name]
            net = net_fn()
            data_list = []
            for _ in tqdm(range(self.num_samples), desc=f"Generating {name}"):
                data = generate_sample(net)
                if data is not None:
                    data_list.append(data)
            path = os.path.join(self.processed_dir, f"{name}_data.pt")
            torch.save(data_list, path)

    def len(self):
        if not hasattr(self, '_data_cache'):
            self._data_cache = []
            for name in self.system_names:
                path = os.path.join(self.processed_dir, f"{name}_data.pt")
                if os.path.exists(path):
                    self._data_cache.extend(torch.load(path, weights_only=False))
        return len(self._data_cache)

    def get(self, idx):
        if not hasattr(self, '_data_cache'):
            self.len()  # trigger caching
        return self._data_cache[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/processed")
    parser.add_argument("--systems", nargs="+", default=["case14", "case30"])
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    dataset = PowerFlowDataset(root=args.root, system_names=args.systems, num_samples=args.num_samples)
    dataset.process()
    print(f"Dataset generated at {args.root}. Total samples: {len(dataset)}")


if __name__ == "__main__":
    main()
