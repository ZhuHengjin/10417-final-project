#!/usr/bin/env python3
"""Summarize best accuracies and losses from TensorBoard scalar CSVs."""

import argparse
from pathlib import Path

import pandas as pd

METRIC_SPECS = [
    ("train_loss", "Training Loss", "min"),
    ("test_loss", "Test Loss", "min"),
    ("train_acc", "Training Accuracy", "max"),
    ("test_acc", "Test Accuracy", "max"),
    ("test_acc_top5", "Test Top-5 Accuracy", "max"),
]


def summarize(csv_path: Path):
    df = pd.read_csv(csv_path)
    summary = []
    for tag, display, mode in METRIC_SPECS:
        subset = df[df["tag"] == tag]
        if subset.empty:
            summary.append((display, None, None))
            continue
        if mode == "max":
            idx = subset["value"].idxmax()
        else:
            idx = subset["value"].idxmin()
        row = subset.loc[idx]
        summary.append((display, row["value"], int(row["step"])))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Print best accuracies and losses from exported CSV.")
    parser.add_argument("--csv", type=Path, required=True, help="CSV exported by report/extract_log.py")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    summary = summarize(args.csv)
    print(f"Results for {args.csv}:")
    for label, value, step in summary:
        if value is None:
            print(f"  {label}: N/A (tag missing)")
        else:
            print(f"- {label}: {value:.4f} at epoch {step}")


if __name__ == "__main__":
    main()
