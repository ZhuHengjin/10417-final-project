#!/usr/bin/env python3
"""Two-phase UMAP visualization from checkpoints.

Phase 1 (`mode=predict`): load a checkpoint, extract features, run UMAP, and save
the 2D embedding plus labels to CSV (no plot).
Phase 2 (`mode=plot`): read a CSV with columns [x, y, label] and render the plot.

CLI examples:
- Predict: python3 report/visualize_umap.py --mode predict \
  --ckpt save/student_model/S:resnet8x4_T:resnet32x4_cifar100_crd_sw_r:1_a:1.0_b:0.8_sw:1.0_tau:0.5_1/ckpt_epoch_200.pth \
  --model resnet8x4 --role student --csv-out crd_sw_1.0_0.5.csv

- Plot (with optional rotation): python report/visualize_umap.py --mode plot --csv tmp.csv --out umap.png --rotate-deg 30
"""

import argparse
import csv
import os
import sys
from typing import Optional, Set, Tuple

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Avoid the need for a display (works in SSH/CLI).
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import umap  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import model_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UMAP visualization with two phases: predict -> plot")
    parser.add_argument("--mode", choices=["predict", "plot"], default="predict", help="Phase to run.")
    parser.add_argument(
        "--ckpt",
        help="Path to the checkpoint (required for predict mode).",
    )
    parser.add_argument("--model", choices=model_dict.keys(), help="Architecture name (required for predict mode).")
    parser.add_argument("--role", choices=["student", "teacher"], default="student", help="Checkpoint role.")
    parser.add_argument("--split", choices=["train", "test"], default="test", help="Dataset split to embed.")
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated class ids to include (e.g., '0,1,2'); use all if omitted.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=5000, help="Subsample to keep UMAP fast.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv-out", help="Output CSV path for predict mode (default: umap_{role}.csv)")
    parser.add_argument("--csv", help="Input CSV path for plot mode.")
    parser.add_argument("--out", default=None, help="Output path for the figure; defaults to umap_{role}.png")
    parser.add_argument("--rotate-deg", type=float, default=0.0, help="Rotation angle (degrees) applied in plot mode")
    return parser.parse_args()


def build_loader(split: str, batch_size: int, num_workers: int) -> DataLoader:
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    dataset = datasets.CIFAR100(root="./data", train=split == "train", download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def load_model(model_name: str, ckpt_path: str, device: str) -> torch.nn.Module:
    model = model_dict[model_name](num_classes=100)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    key = "model" if "model" in state else "state_dict"
    model.load_state_dict(state[key])
    model.to(device)
    model.eval()
    return model


def collect_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    max_samples: int,
    seed: int,
    allowed_classes: Optional[Set[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    # torch.randperm only supports CPU generators, so always keep this one on CPU.
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    features = []
    labels = []
    seen = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images, is_feat=True)
            feat_list, _ = outputs
            feat = feat_list[-1]
            if feat.dim() > 2:
                feat = torch.flatten(feat, 1)  # flatten spatial dims if present
            if allowed_classes is not None:
                mask = torch.tensor([t.item() in allowed_classes for t in targets], device=targets.device)
                if mask.sum() == 0:
                    continue
                feat = feat[mask]
                targets = targets[mask]
            features.append(feat.cpu())
            labels.append(targets)
            seen += targets.size(0)
            if max_samples and seen >= max_samples:
                break

    features_tensor = torch.cat(features, dim=0)
    labels_tensor = torch.cat(labels, dim=0)

    if max_samples and features_tensor.size(0) > max_samples:
        perm = torch.randperm(features_tensor.size(0), generator=rng)[:max_samples]
        features_tensor = features_tensor[perm]
        labels_tensor = labels_tensor[perm]

    feats_np = features_tensor.numpy()
    labels_np = labels_tensor.numpy()
    feats_np = (feats_np - feats_np.mean(axis=0)) / (feats_np.std(axis=0) + 1e-6)
    return feats_np, labels_np


def run_umap(
    features: np.ndarray, n_neighbors: int, min_dist: float, metric: str, seed: int
) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    return reducer.fit_transform(features)


def plot_embedding(embedding: np.ndarray, labels: np.ndarray, out_path: str, role: str) -> None:
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap=plt.cm.get_cmap("tab20", np.unique(labels).size),
        s=6,
        alpha=0.8,
        linewidths=0,
    )
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(f"{role.capitalize()} features (final hidden state)")
    cbar = plt.colorbar(scatter, fraction=0.046, pad=0.04)
    cbar.set_label("Class")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved UMAP plot to {out_path}")


def save_csv(embedding: np.ndarray, labels: np.ndarray, csv_path: str) -> None:
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "label"])
        for (x, y), lbl in zip(embedding, labels):
            writer.writerow([float(x), float(y), int(lbl)])
    print(f"Saved UMAP embedding to {csv_path}")


def load_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys, labels = [], [], []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            labels.append(int(row["label"]))
    emb = np.vstack([xs, ys]).T
    labs = np.array(labels)
    return emb, labs


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == "predict":
        if not args.ckpt or not args.model:
            print("predict mode requires --ckpt and --model", file=sys.stderr)
            sys.exit(1)
        if not os.path.isfile(args.ckpt):
            print(f"Checkpoint not found: {args.ckpt}", file=sys.stderr)
            sys.exit(1)
        csv_out = args.csv_out or f"umap_{args.role}.csv"

        allowed_classes = None
        if args.classes:
            allowed_classes = set(int(x) for x in args.classes.split(",") if x.strip() != "")

        loader = build_loader(args.split, args.batch_size, args.num_workers)
        model = load_model(args.model, args.ckpt, args.device)
        features, labels = collect_features(
            model, loader, args.device, args.max_samples, args.seed, allowed_classes=allowed_classes
        )
        embedding = run_umap(features, args.n_neighbors, args.min_dist, args.metric, args.seed)
        save_csv(embedding, labels, csv_out)
    else:
        if not args.csv:
            print("plot mode requires --csv", file=sys.stderr)
            sys.exit(1)
        emb, labs = load_csv(args.csv)
        if args.rotate_deg:
            theta = np.deg2rad(args.rotate_deg)
            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            emb = emb @ rot.T
        out_path = args.out or f"umap_{args.role}.png"
        plot_embedding(emb, labs, out_path, args.role)


if __name__ == "__main__":
    main()
