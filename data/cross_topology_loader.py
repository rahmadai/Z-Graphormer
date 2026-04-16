from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch


def collate_variable_n(data_list):
    """
    Custom collate for variable-size Z-matrices.
    PyG Batch will handle x and assignment vectors; we ensure z_matrix is stored
    as a list so it does not try to stack tensors of different sizes.
    """
    # Temporarily remove z_matrix from each Data object, batch the rest, then reattach
    z_matrices = [d.z_matrix for d in data_list]
    for d in data_list:
        d.z_matrix = None

    batch = Batch.from_data_list(data_list)
    batch.z_matrix = z_matrices
    return batch


def get_dataloader(dataset, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_variable_n,
    )
