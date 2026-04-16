import math
import torch
import torch.nn as nn
from models.zbus_encoding import ZBusRelativeEncoding
from models.zgraphormer import ZGraphormer
from torch_geometric.data import Data, Batch
from data.cross_topology_loader import collate_variable_n


def test_zbus_invariance():
    """Same impedance = same bias regardless of grid size"""
    encoder = ZBusRelativeEncoding(num_heads=8)

    z_14 = torch.tensor([0.5])  # IEEE 14 context
    z_118 = torch.tensor([0.5])  # IEEE 118 context

    bias_14 = encoder(z_14)
    bias_118 = encoder(z_118)

    assert torch.allclose(bias_14, bias_118)
    print("PASS: test_zbus_invariance")


def test_variable_topology():
    """Model handles mixed batch sizes (14+30+118) without error"""
    model = ZGraphormer(d_model=64, num_heads=4, num_layers=2)

    data_list = [
        Data(x=torch.randn(14, 5), z_matrix=torch.randn(14, 14).abs(), centrality=torch.randint(0, 10, (14,))),
        Data(x=torch.randn(30, 5), z_matrix=torch.randn(30, 30).abs(), centrality=torch.randint(0, 10, (30,))),
        Data(x=torch.randn(118, 5), z_matrix=torch.randn(118, 118).abs(), centrality=torch.randint(0, 10, (118,))),
    ]
    batch = collate_variable_n(data_list)

    volt, sec = model(batch)
    assert volt.shape[0] == 14 + 30 + 118  # Total nodes
    assert sec.shape[0] == 3  # 3 graph predictions
    print("PASS: test_variable_topology")


def test_electrical_attention():
    """Z-bus encoding produces different biases for different electrical distances"""
    encoder = ZBusRelativeEncoding(num_heads=8)

    # Close vs far electrical distances
    z_close = torch.tensor([0.1])
    z_far = torch.tensor([2.0])

    bias_close = encoder(z_close)
    bias_far = encoder(z_far)

    # They should map to different bins and produce different embeddings
    assert not torch.allclose(bias_close, bias_far), "Different Z values should produce different biases"

    # Sanity check: identical Z values produce identical biases
    z_same = torch.tensor([0.1, 0.1])
    bias_same = encoder(z_same)
    assert torch.allclose(bias_same[0], bias_same[1]), "Identical Z values should produce identical biases"
    print("PASS: test_electrical_attention")


def test_train_step():
    """A single training step runs without error."""
    model = ZGraphormer(d_model=32, num_heads=2, num_layers=1)
    data_list = [
        Data(
            x=torch.randn(14, 5),
            z_matrix=torch.randn(14, 14).abs(),
            centrality=torch.randint(0, 10, (14,)),
            y_volt=torch.randn(14, 1),
            y_sec=torch.tensor([1.0]),
        ),
    ]
    batch = collate_variable_n(data_list)
    volt, sec = model(batch)

    loss = nn.MSELoss()(volt, batch.y_volt) + nn.BCEWithLogitsLoss()(sec, batch.y_sec.unsqueeze(1))
    loss.backward()
    print("PASS: test_train_step")


if __name__ == "__main__":
    test_zbus_invariance()
    test_variable_topology()
    test_electrical_attention()
    test_train_step()
