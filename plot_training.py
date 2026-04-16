import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd


def plot_curves(csv_path: str, save_path: str = "training_curves.png"):
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Total loss
    ax = axes[0]
    ax.plot(df["epoch"], df["train_loss"], label="Train", marker="o")
    ax.plot(df["epoch"], df["val_loss"], label="Val", marker="s")
    ax.set_title("Total Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)

    # Voltage MSE
    ax = axes[1]
    ax.plot(df["epoch"], df["train_v"], label="Train", marker="o")
    ax.plot(df["epoch"], df["val_v"], label="Val", marker="s")
    ax.set_title("Voltage MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)

    # Security BCE
    ax = axes[2]
    ax.plot(df["epoch"], df["train_s"], label="Train", marker="o")
    ax.plot(df["epoch"], df["val_s"], label="Val", marker="s")
    ax.set_title("Security BCE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved training curves to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/processed/metrics.csv")
    parser.add_argument("--out", default="training_curves.png")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"Metrics CSV not found at {args.csv}. Run training first.")
    plot_curves(args.csv, args.out)


if __name__ == "__main__":
    main()
