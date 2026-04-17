import argparse
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryF1Score

from models.elecformer import ElecFormer
from data.generate_pandapower import PowerFlowDataset
from data.cross_topology_loader import get_dataloader


def evaluate_zero_shot(model, loader, device):
    model.eval()
    all_v = []
    all_v_pred = []
    all_s = []
    all_s_pred = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            volt, sec = model(batch)
            all_v.append(batch.y_volt)
            all_v_pred.append(volt)
            all_s.append(batch.y_sec.unsqueeze(1))
            all_s_pred.append(torch.sigmoid(sec))

    v_true = torch.cat(all_v, dim=0)
    v_pred = torch.cat(all_v_pred, dim=0)
    s_true = torch.cat(all_s, dim=0)
    s_pred = torch.cat(all_s_pred, dim=0)

    mae = torch.mean(torch.abs(v_true - v_pred)).item()
    max_err = torch.max(torch.abs(v_true - v_pred)).item()
    f1 = BinaryF1Score().to(device)(s_pred, s_true.int()).item()
    return {"mae": mae, "max_err": max_err, "f1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/processed")
    parser.add_argument("--test_systems", nargs="+", default=["case39"])
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--checkpoint", default="data/processed/best_model.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = ElecFormer(in_channels=9, d_node=128, d_pair=64, num_heads=8, num_layers=4, d_ff=512).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    for sys_name in args.test_systems:
        dataset = PowerFlowDataset(root=args.data_root, system_names=[sys_name], num_samples=args.num_samples)
        dataset.process()
        loader = get_dataloader(dataset, batch_size=16, shuffle=False, num_workers=0)
        metrics = evaluate_zero_shot(model, loader, device)
        print(f"{sys_name}: MAE={metrics['mae']:.5f} p.u., MaxErr={metrics['max_err']:.5f} p.u., F1={metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
