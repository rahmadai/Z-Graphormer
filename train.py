import os
import csv
import argparse
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from models.zgraphormer import ZGraphormer
from data.generate_pandapower import PowerFlowDataset
from data.cross_topology_loader import get_dataloader


def train_epoch(model, loader, optimizer, device, alpha=0.5):
    model.train()
    total_loss = 0.0
    total_vloss = 0.0
    total_sloss = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        batch = batch.to(device)
        volt, sec = model(batch)

        v_loss = nn.MSELoss()(volt, batch.y_volt)
        s_loss = nn.BCEWithLogitsLoss()(sec, batch.y_sec.unsqueeze(1))
        loss = alpha * v_loss + (1 - alpha) * s_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_vloss += v_loss.item()
        total_sloss += s_loss.item()

    n = len(loader)
    return total_loss / n, total_vloss / n, total_sloss / n


def evaluate(model, loader, device, alpha=0.5):
    model.eval()
    total_loss = 0.0
    total_vloss = 0.0
    total_sloss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val  ", leave=False):
            batch = batch.to(device)
            volt, sec = model(batch)
            v_loss = nn.MSELoss()(volt, batch.y_volt)
            s_loss = nn.BCEWithLogitsLoss()(sec, batch.y_sec.unsqueeze(1))
            loss = alpha * v_loss + (1 - alpha) * s_loss
            total_loss += loss.item()
            total_vloss += v_loss.item()
            total_sloss += s_loss.item()
    n = len(loader)
    return total_loss / n, total_vloss / n, total_sloss / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/processed")
    parser.add_argument("--systems", nargs="+", default=["case14", "case30"])
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.data_root, exist_ok=True)
    dataset = PowerFlowDataset(root=args.data_root, system_names=args.systems, num_samples=args.num_samples)
    if not all(os.path.exists(os.path.join(dataset.processed_dir, f"{s}_data.pt")) for s in args.systems):
        print("Generating dataset...")
        dataset.process()

    n = len(dataset)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    test_size = n - train_size - val_size
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = ZGraphormer(d_model=128, num_heads=8, num_layers=4, d_ff=512).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    metrics_path = os.path.join(args.data_root, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_v", "train_s", "val_loss", "val_v", "val_s"])

        best_val = float("inf")
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            t_loss, t_v, t_s = train_epoch(model, train_loader, optimizer, device, alpha=args.alpha)
            v_loss, v_v, v_s = evaluate(model, val_loader, device, alpha=args.alpha)
            elapsed = time.time() - epoch_start
            print(f"Epoch {epoch:03d}/{args.epochs} | {elapsed:.1f}s | Train: {t_loss:.4f} (V={t_v:.4f}, S={t_s:.4f}) | Val: {v_loss:.4f} (V={v_v:.4f}, S={v_s:.4f})")
            writer.writerow([epoch, f"{t_loss:.6f}", f"{t_v:.6f}", f"{t_s:.6f}", f"{v_loss:.6f}", f"{v_v:.6f}", f"{v_s:.6f}"])
            f.flush()
            if v_loss < best_val:
                best_val = v_loss
                torch.save(model.state_dict(), os.path.join(args.data_root, "best_model.pt"))

    test_loss, test_v, test_s = evaluate(model, test_loader, device, alpha=args.alpha)
    print(f"Test Loss: {test_loss:.4f} (V={test_v:.4f}, S={test_s:.4f})")


if __name__ == "__main__":
    main()
