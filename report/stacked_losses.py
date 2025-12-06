#!/usr/bin/env python3
"""Plot a stacked area chart of KD loss components from an extracted CSV.

Example:
python3 report/stacked_losses.py --csv report/logs/crd_sw_a1.csv --out report/graphs/stacked_losses.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TAGS = ["train_loss_cls", "train_loss_div", "train_loss_kd"]


def load_components(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    available = set(df["tag"].unique())
    missing = [t for t in TAGS if t not in available]
    if missing:
        raise ValueError(f"Missing tags in {csv_path}: {missing}. Available: {sorted(available)}")

    subset = df[df["tag"].isin(TAGS)].copy()
    pivot = subset.pivot_table(index="step", columns="tag", values="value", aggfunc="mean")
    pivot = pivot.reset_index().sort_values("step")
    return pivot


def plot_stacked(pivot: pd.DataFrame, out_file: Path) -> None:
    steps = pivot["step"].to_numpy()
    layers = [pivot[t].to_numpy() for t in TAGS]
    labels = ["CE (cls)", "KL (div)", "SW-CRD (kd)"]
    colors = ["#3FA85B", "#6CCF70",  "#A8E6A1"]

    plt.figure(figsize=(7.5, 3.5))
    plt.stackplot(steps, layers, labels=labels, colors=colors, alpha=0.85)
    plt.xlabel("Epoch / Step")
    plt.ylabel("Loss components")
    plt.title("KD loss decomposition (stacked)")
    plt.legend(loc="upper right")
    plt.tight_layout()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved stacked plot to {out_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot stacked KD loss components (cls/div/kd) from CSV.")
    parser.add_argument("--csv", required=True, type=Path, help="CSV exported by extract_log.py")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("report/graphs/stacked_losses.png"),
        help="Output image path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    pivot = load_components(args.csv)
    plot_stacked(pivot, args.out)


if __name__ == "__main__":
    main()
