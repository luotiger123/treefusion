# -*- coding: utf-8 -*-
"""
TreeFusion v1.0 evaluation script.

Author:
    Taige Luo et al.

Description:
    This script evaluates a trained TreeFusion model on validation and/or
    test splits. It reports standard canopy height regression metrics:
    MAE, RMSE, and R2 in meters.

Checkpoint compatibility:
    This script is designed to work with checkpoints saved by the original
    TreeDinoWithHeightAttention training script. The original model class name
    is retained for compatibility with existing checkpoint keys.

Expected input:
    x: [B, 8, H, W]
    channels = [RGB-NIR(4), SAR VV/VH(2), nDSM + Rough CHM(2)]

Expected output:
    pred: [B, 1, H, W]
    predicted canopy height map, normalized by --scale during training.

Example:
    python test.py ^
      --project-dir D:\pythonc\TreeFusion ^
      --data-root D:\pythonc\TreeFusion\FinalData ^
      --split-dir D:\pythonc\TreeFusion\FinalData\splits_811 ^
      --checkpoint D:\pythonc\TreeFusion\checkFinalAll\latest.pth ^
      --eval-splits val test ^
      --batch-size 4 ^
      --num-workers 2 ^
      --strict

    # Evaluate TreeFusion v1.0 on validation and test splits.
    # --data-root: root directory of the multimodal dataset
    # --split-dir: directory containing train.csv, val.csv, and test.csv
    # --checkpoint: path to the trained model checkpoint
    # --eval-splits: dataset splits to evaluate

"""

import os
import sys
import csv
import math
import random
import argparse
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


# ============================================================
# Argument parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate TreeFusion v1.0 on validation and test splits."
    )

    # Project and data paths.
    parser.add_argument(
        "--project-dir",
        type=str,
        default=r"D:\pythonc\TreeFusion",
        help="Root directory of the TreeFusion project."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help=(
            "Root directory of the training/evaluation dataset. "
            "If not provided, it defaults to <project-dir>/FinalData."
        )
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default=None,
        help=(
            "Directory containing train.csv, val.csv, and test.csv. "
            "If not provided, it defaults to <data-root>/splits_811."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained TreeFusion checkpoint."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory for saving evaluation results. "
            "If not provided, it defaults to <project-dir>/evaluation_results."
        )
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="val_test_metrics.csv",
        help="Output CSV filename."
    )

    # Evaluation settings.
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        default=["val", "test"],
        choices=["train", "val", "test"],
        help="Dataset splits to evaluate."
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=256,
        help="Input patch size used by the dataset."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of DataLoader workers."
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=50.0,
        help=(
            "Height normalization scale. "
            "For example, 50 means normalized value 1.0 equals 50 meters."
        )
    )

    # Model settings.
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of attention heads used by TreeDinoWithHeightAttention."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict=True when loading model weights."
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision during evaluation."
    )

    # Reproducibility and runtime.
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for evaluation."
    )

    return parser.parse_args()


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducible evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset utilities
# ============================================================

def load_keys_csv(csv_path: str):
    """
    Load sample keys from a split CSV.

    The split CSV is expected to contain one column named 'key'.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    keys = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if "key" not in reader.fieldnames:
            raise ValueError(
                f"Split CSV must contain a 'key' column: {csv_path}"
            )

        for row in reader:
            keys.append(row["key"])

    return keys


def build_full_dataset(data_root: str, target_size: int):
    """
    Build the full multimodal TreeFusion dataset.

    Expected directory structure:
        <data-root>/
            RGB/
            SAR_clips/
                VV/
                VH/
            nDSM_clips/
            ETH_chms/
            CHM-VHR/
            splits_811/
                train.csv
                val.csv
                test.csv
    """
    from datasat_multi.datasatKD_8Chanel import MultiModalTreeCHMFullDataset

    return MultiModalTreeCHMFullDataset(
        optical_dir=os.path.join(data_root, "RGB"),
        sar_vv_dir=os.path.join(data_root, "SAR_clips", "VV"),
        sar_vh_dir=os.path.join(data_root, "SAR_clips", "VH"),
        ndsm_dir=os.path.join(data_root, "nDSM_clips"),
        roughchm_dir=os.path.join(data_root, "ETH_chms"),
        vhr_chm_dir=os.path.join(data_root, "CHM-VHR"),
        target_size=target_size,
    )


def subset_from_keys(full_dataset, keys):
    """
    Create a dataset subset using sample keys.

    Returns
    -------
    subset : torch.utils.data.Subset
        Dataset subset containing matched samples.
    matched_count : int
        Number of matched samples.
    missing_count : int
        Number of keys not found in the dataset.
    """
    key_to_idx = {sample["key"]: i for i, sample in enumerate(full_dataset.samples)}
    indices = [key_to_idx[k] for k in keys if k in key_to_idx]

    matched_count = len(indices)
    missing_count = len(keys) - matched_count

    return Subset(full_dataset, indices), matched_count, missing_count


# ============================================================
# Checkpoint utilities
# ============================================================

def remove_module_prefix(state_dict):
    """
    Remove 'module.' prefix from DataParallel checkpoints if present.
    """
    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[len("module."):]] = value
        else:
            cleaned[key] = value

    return cleaned


def extract_state_dict(checkpoint):
    """
    Extract model state_dict from different checkpoint formats.

    Supported formats:
        1. {"model": state_dict, ...}
        2. {"state_dict": state_dict, ...}
        3. raw state_dict
    """
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    return checkpoint


def print_checkpoint_info(checkpoint):
    """Print optional checkpoint metadata if available."""
    if not isinstance(checkpoint, dict):
        return

    if "epoch" in checkpoint:
        print(f"[INFO] Checkpoint epoch: {checkpoint['epoch']}")

    if "best_mae" in checkpoint:
        try:
            print(f"[INFO] Checkpoint best MAE: {float(checkpoint['best_mae']):.4f} m")
        except Exception:
            print(f"[INFO] Checkpoint best MAE: {checkpoint['best_mae']}")

    if "global_step" in checkpoint:
        print(f"[INFO] Checkpoint global step: {checkpoint['global_step']}")


def load_model_checkpoint(model, checkpoint_path: str, device, strict: bool):
    """
    Load trained weights into the TreeFusion model.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    print_checkpoint_info(checkpoint)

    state_dict = extract_state_dict(checkpoint)
    state_dict = remove_module_prefix(state_dict)

    load_result = model.load_state_dict(state_dict, strict=strict)

    print("[INFO] Model weights loaded.")

    if not strict:
        missing_keys, unexpected_keys = load_result

        print(f"[WARN] Missing keys: {len(missing_keys)}")
        print(f"[WARN] Unexpected keys: {len(unexpected_keys)}")

        if len(missing_keys) > 0:
            print("[WARN] First missing keys:")
            for key in missing_keys[:10]:
                print(f"       {key}")

        if len(unexpected_keys) > 0:
            print("[WARN] First unexpected keys:")
            for key in unexpected_keys[:10]:
                print(f"       {key}")

    return model


# ============================================================
# Metrics
# ============================================================

@torch.no_grad()
def evaluate_split(
    model,
    loader,
    device,
    scale: float,
    use_amp: bool,
    split_name: str,
):
    """
    Evaluate one dataset split.

    Metrics are computed in meters:
        MAE  = mean absolute error
        RMSE = root mean squared error
        R2   = coefficient of determination

    Notes:
        The model output and ground truth are assumed to be normalized.
        Both are multiplied by --scale before metric calculation.
    """
    model.eval()

    n = 0
    sum_abs_error = 0.0
    sum_squared_error = 0.0
    sum_y = 0.0
    sum_y2 = 0.0

    if device.type == "cuda" and use_amp:
        amp_context = torch.amp.autocast("cuda")
    else:
        amp_context = nullcontext()

    pbar = tqdm(loader, desc=f"Evaluating {split_name}", dynamic_ncols=True)

    for batch in pbar:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        with amp_context:
            pred = model(x)

        pred_m = (pred.float() * scale).reshape(-1)
        y_m = (y.float() * scale).reshape(-1)

        diff = pred_m - y_m

        sum_abs_error += torch.abs(diff).sum().item()
        sum_squared_error += torch.sum(diff * diff).item()

        sum_y += y_m.sum().item()
        sum_y2 += torch.sum(y_m * y_m).item()

        n += y_m.numel()

    if n == 0:
        return {
            "mae_m": float("nan"),
            "rmse_m": float("nan"),
            "r2": float("nan"),
            "num_pixels": 0,
            "num_batches": len(loader),
        }

    mae = sum_abs_error / n
    rmse = math.sqrt(sum_squared_error / n)

    sse = sum_squared_error
    mean_y = sum_y / n
    sst = sum_y2 - n * (mean_y ** 2)

    if sst <= 1e-12:
        r2 = float("nan")
    else:
        r2 = 1.0 - sse / sst

    return {
        "mae_m": mae,
        "rmse_m": rmse,
        "r2": r2,
        "num_pixels": n,
        "num_batches": len(loader),
    }


# ============================================================
# Output utilities
# ============================================================

def save_metrics_csv(output_path: str, results: dict):
    """Save evaluation metrics to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "split",
            "mae_m",
            "rmse_m",
            "r2",
            "num_pixels",
            "num_batches",
        ])

        for split_name, metrics in results.items():
            writer.writerow([
                split_name,
                metrics["mae_m"],
                metrics["rmse_m"],
                metrics["r2"],
                metrics["num_pixels"],
                metrics["num_batches"],
            ])


def print_results(results: dict, output_csv: str):
    """Print evaluation results."""
    print("\n" + "=" * 80)
    print("TreeFusion v1.0 Evaluation Results")
    print("=" * 80)

    for split_name, metrics in results.items():
        print(
            f"{split_name.upper():>5} | "
            f"MAE = {metrics['mae_m']:.3f} m | "
            f"RMSE = {metrics['rmse_m']:.3f} m | "
            f"R2 = {metrics['r2']:.4f} | "
            f"Pixels = {metrics['num_pixels']:,}"
        )

    print("=" * 80)
    print(f"[DONE] Metrics saved to: {output_csv}")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    project_dir = Path(args.project_dir).resolve()

    if args.data_root is None:
        data_root = project_dir / "FinalData"
    else:
        data_root = Path(args.data_root).resolve()

    if args.split_dir is None:
        split_dir = data_root / "splits_811"
    else:
        split_dir = Path(args.split_dir).resolve()

    if args.output_dir is None:
        output_dir = project_dir / "evaluation_results"
    else:
        output_dir = Path(args.output_dir).resolve()

    output_csv = output_dir / args.output_csv

    # Add project root to Python path for local imports.
    os.chdir(project_dir)
    sys.path.insert(0, str(project_dir))

    set_seed(args.seed)

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_amp = (not args.no_amp) and (device.type == "cuda")

    print("[INFO] TreeFusion v1.0 evaluation")
    print(f"[INFO] Project directory : {project_dir}")
    print(f"[INFO] Data root         : {data_root}")
    print(f"[INFO] Split directory   : {split_dir}")
    print(f"[INFO] Checkpoint        : {args.checkpoint}")
    print(f"[INFO] Output CSV        : {output_csv}")
    print(f"[INFO] Device            : {device}")
    print(f"[INFO] AMP enabled       : {use_amp}")
    print(f"[INFO] Evaluation splits : {args.eval_splits}")
    print(f"[INFO] Scale             : {args.scale}")

    # Import after project_dir is added to sys.path.
    from models.TreeDinoHeightAtt import TreeDinoWithHeightAttention

    # Build dataset once, then create split-specific subsets.
    print("\n[INFO] Building dataset...")
    full_dataset = build_full_dataset(
        data_root=str(data_root),
        target_size=args.target_size,
    )

    loaders = {}

    for split_name in args.eval_splits:
        split_csv = split_dir / f"{split_name}.csv"
        keys = load_keys_csv(str(split_csv))

        subset, matched_count, missing_count = subset_from_keys(
            full_dataset,
            keys,
        )

        print(
            f"[INFO] Split {split_name}: "
            f"keys={len(keys)} | matched={matched_count} | missing={missing_count}"
        )

        if matched_count == 0:
            raise RuntimeError(
                f"No matched samples for split '{split_name}'. "
                f"Please check the split CSV and dataset keys."
            )

        loaders[split_name] = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    # Build model.
    print("\n[INFO] Building model...")
    model = TreeDinoWithHeightAttention(
        num_heads=args.num_heads,
    ).to(device)

    # Load checkpoint.
    model = load_model_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
        strict=args.strict,
    )

    # Evaluate selected splits.
    print("\n[INFO] Starting evaluation...")
    results = {}

    for split_name, loader in loaders.items():
        metrics = evaluate_split(
            model=model,
            loader=loader,
            device=device,
            scale=args.scale,
            use_amp=use_amp,
            split_name=split_name,
        )
        results[split_name] = metrics

    # Save and print results.
    save_metrics_csv(
        output_path=str(output_csv),
        results=results,
    )

    print_results(
        results=results,
        output_csv=str(output_csv),
    )


if __name__ == "__main__":
    main()