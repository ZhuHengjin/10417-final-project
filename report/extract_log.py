#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def read_scalars(run_dir: Path, tags):
    ea = event_accumulator.EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))

    records = []
    for tag in tags:
        if tag not in available:
            continue
        for event in ea.Scalars(tag):
            records.append(
                {
                    "tag": tag,
                    "step": event.step,
                    "wall_time": event.wall_time,
                    "value": event.value,
                }
            )

    if not records:
        raise ValueError(
            f"No matching scalar tags found in {run_dir}. "
            f"Available tags: {sorted(available)}"
        )
    return pd.DataFrame.from_records(records)


def main():
    parser = argparse.ArgumentParser(
        description="Extract TensorBoard scalars from a single run directory."
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Directory that contains the TensorBoard event file(s) for one run "
             "(e.g. logs/kd/S:resnet32_T:ResNet50_...).",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=["train_loss", "train_acc", "test_loss", "test_acc", "test_acc_top5"],
        help="Scalar tags to export. Only tags found in the event file will be kept.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("run_scalars.csv"),
        help="Path of the CSV file to write.",
    )
    args = parser.parse_args()

    if not args.run_dir.exists():
        raise FileNotFoundError(f"Run directory {args.run_dir} does not exist")

    if not any(args.run_dir.glob("events.out.tfevents.*")):
        raise FileNotFoundError(
            f"No TensorBoard event files found under {args.run_dir}. "
            "Did you point to the right run folder?"
        )

    df = read_scalars(args.run_dir, args.tags)
    df.sort_values(["tag", "step"], inplace=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
