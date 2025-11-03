#!/usr/bin/env python3
"""Plot training/testing loss and accuracy curves from TensorBoard scalar CSVs."""

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def load_curves(csv_path: Path, tags: List[str]) -> dict:
    df = pd.read_csv(csv_path)
    curves = {}
    for tag in tags:
        curve = df[df["tag"] == tag].copy()
        if curve.empty:
            raise ValueError(f"Tag '{tag}' not found in {csv_path}")
        curve.sort_values("step", inplace=True)
        curve["run"] = csv_path.stem
        curves[tag] = curve
    return curves


def plot_metric(curves: List[pd.DataFrame], tag: str, ylabel: str, out_file: Path) -> None:
    plt.figure(figsize=(7, 4))
    for curve in curves:
        plt.plot(curve["step"], curve["value"], label=curve["run"].iloc[0])
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Epoch")
    plt.legend()
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Draw training curves from CSV exports.")
    parser.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="Paths to CSV files exported by report/extract_log.py (exactly three).",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("report/plots"), help="Directory to store the generated plots.")
    args = parser.parse_args()

    if len(args.csv) != 3:
        raise ValueError("Provide exactly three CSV files via --csv.")

    csv_paths = [Path(p) for p in args.csv]
    for path in csv_paths:
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

    tags = ["train_loss", "train_acc", "test_loss", "test_acc"]
    curves_by_tag = {tag: [] for tag in tags}
    for path in csv_paths:
        curves = load_curves(path, tags)
        for tag in tags:
            curves_by_tag[tag].append(curves[tag])

    metrics = [
        ("train_loss", "Training Loss", "training_loss_vs_epoch.png"),
        ("train_acc", "Training Accuracy", "training_accuracy_vs_epoch.png"),
        ("test_loss", "Test Loss", "test_loss_vs_epoch.png"),
        ("test_acc", "Test Accuracy", "test_accuracy_vs_epoch.png"),
    ]

    for tag, ylabel, filename in metrics:
        plot_metric(curves_by_tag[tag], tag=tag, ylabel=ylabel, out_file=args.out_dir / filename)


if __name__ == "__main__":
    main()
