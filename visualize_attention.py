import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.zgraphormer import ZGraphormer
from data.generate_pandapower import PowerFlowDataset
from data.cross_topology_loader import get_dataloader


def plot_sample(model, batch, graph_idx: int = 0, layer_idx: int = -1, save_path: str = "viz.png"):
    model.eval()
    with torch.no_grad():
        volt, sec, all_attns = model(batch, return_attention=True)

    # Identify nodes belonging to the chosen graph
    assign = batch.batch.cpu()
    graph_ids = torch.unique(assign, sorted=True)
    gid = graph_ids[graph_idx]
    mask = assign == gid
    node_offset = mask.nonzero(as_tuple=True)[0][0].item()
    n_nodes = mask.sum().item()

    # Extract Z-matrix for this graph
    z_mat = batch.z_matrix[graph_idx].cpu().numpy()  # [N, N]

    # Extract attention for chosen layer, averaged over heads
    attn = all_attns[graph_idx][layer_idx].mean(dim=1).cpu().numpy()  # [N, N]

    # Extract predictions / targets for this graph
    v_pred = volt[mask].cpu().numpy().squeeze()
    v_true = batch.y_volt[mask].cpu().numpy().squeeze()

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    # 1. Z-bus magnitude heatmap
    ax = axes[0]
    im = ax.imshow(z_mat, cmap="viridis", aspect="auto")
    ax.set_title(f"Z-bus Magnitude | Graph {graph_idx} ({n_nodes} buses)")
    ax.set_xlabel("Bus j")
    ax.set_ylabel("Bus i")
    plt.colorbar(im, ax=ax)

    # 2. Attention heatmap
    ax = axes[1]
    im = ax.imshow(attn, cmap="hot", aspect="auto", vmin=0, vmax=1)
    ax.set_title(f"Attention Weights (Layer {layer_idx}, avg heads)")
    ax.set_xlabel("Bus j")
    ax.set_ylabel("Bus i")
    plt.colorbar(im, ax=ax)

    # 3. Voltage scatter
    ax = axes[2]
    ax.scatter(v_true, v_pred, alpha=0.7, edgecolors="k")
    vmin, vmax = v_true.min(), v_true.max()
    ax.plot([vmin, vmax], [vmin, vmax], "r--", lw=2, label="Perfect")
    ax.set_xlabel("True Voltage (p.u.)")
    ax.set_ylabel("Predicted Voltage (p.u.)")
    ax.set_title(f"Voltage Prediction | MAE={np.abs(v_true-v_pred).mean():.4f}")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)

    # 4. Attention vs Z-ij scatter (flatten upper triangle excluding diagonal)
    ax = axes[3]
    triu_idx = np.triu_indices(n_nodes, k=1)
    z_vals = z_mat[triu_idx]
    a_vals = attn[triu_idx]
    ax.scatter(z_vals, a_vals, alpha=0.5, s=20)
    ax.set_xlabel("|Z_ij| (p.u.)")
    ax.set_ylabel("Attention Weight")
    ax.set_title("Attention vs Electrical Distance")
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved visualization to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/processed")
    parser.add_argument("--system", default="case14")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--checkpoint", default="data/processed/best_model.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--graph_idx", type=int, default=0)
    parser.add_argument("--layer_idx", type=int, default=-1)
    parser.add_argument("--out", default="viz.png")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = ZGraphormer(d_model=128, num_heads=8, num_layers=4, d_ff=512).to(device)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        print(f"Warning: checkpoint {args.checkpoint} not found. Using random weights.")

    dataset = PowerFlowDataset(root=args.data_root, system_names=[args.system], num_samples=args.num_samples)
    if not os.path.exists(os.path.join(dataset.processed_dir, f"{args.system}_data.pt")):
        dataset.process()
    loader = get_dataloader(dataset, batch_size=4, shuffle=False)

    batch = next(iter(loader)).to(device)
    plot_sample(model, batch, graph_idx=args.graph_idx, layer_idx=args.layer_idx, save_path=args.out)


if __name__ == "__main__":
    main()
