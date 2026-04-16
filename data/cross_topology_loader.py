# data/cross_topology_loader.py

from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch


def collate_variable_n(data_list):
    """
    Custom collate for variable-size Z-matrices.
    PyG Batch will handle x and assignment vectors; we ensure z_matrix is stored
    as a list so it does not try to stack tensors of different sizes.
    """
    z_matrices = [d.z_matrix for d in data_list]
    # Create shallow copies so we don't mutate the cached Data objects
    copied = []
    for d in data_list:
        c = d.clone()
        c.z_matrix = None
        copied.append(c)

    batch = Batch.from_data_list(copied)
    batch.z_matrix = z_matrices
    return batch


def get_dataloader(dataset, batch_size: int = 32, shuffle: bool = True, num_workers: int = 4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=collate_variable_n,
    )
