# -*- coding: utf-8 -*-
"""
TreeFusion v1.0 training script.

Author:
    Taige Luo et al.

Description:
    Single-GPU training script for TreeDinoWithHeightAttention.

Main features:
    - Standard resume training.
    - DINOv3 Transformer backbone can be frozen while patch_embed remains trainable.
    - Batch-level training status is printed every fixed number of steps.
    - Train/validation/test metrics are saved after each epoch.
    - Checkpoints save and restore:
        model state_dict,
        optimizer state_dict,
        AMP scaler,
        best validation MAE,
        global step,
        random number generator states.

Metrics:
    LOSS, MAE, RMSE, and R2 are reported.
    MAE and RMSE are reported in meters.

Data assumption:
    All modalities share the same filename key:
        {key}.tif

    Split CSV files contain a column named:
        key
"""

import os
import csv
import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.TreeDinoHeightAtt import TreeDinoWithHeightAttention
from models.train_utils import (
    set_seed,
    freeze_dino_backbone_except_embed,
    gradient_mask_loss,
    WarmupCosineScheduler,
    eval_metrics_epoch,
    save_checkpoint,
    load_checkpoint,
)
from datasat_multi.datasatKD_8Chanel import MultiModalTreeCHMFullDataset


# ============================================================
# Argument parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TreeFusion v1.0 for very-high-resolution canopy height mapping."
    )

    # Project and data paths.
    parser.add_argument(
        "--project-dir",
        type=str,
        default=".",
        help="Root directory of the TreeFusion project. Use '.' when running from the repository root."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory of the multimodal training dataset. Default: <project-dir>/FinalData."
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default=None,
        help="Directory containing train.csv, val.csv, and test.csv. Default: <data-root>/splits_811."
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for saving training logs and metric CSV files. Default: <project-dir>/logs_FinalAll."
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Directory for saving checkpoints. Default: <project-dir>/checkFinalAll."
    )
    parser.add_argument(
        "--metrics-filename",
        type=str,
        default="epoch_metrics_rgb_val_test.csv",
        help="Filename for saving epoch-level metrics."
    )

    # Model settings.
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of attention heads used in TreeDinoWithHeightAttention."
    )
    parser.add_argument(
        "--freeze-dino",
        action="store_true",
        default=True,
        help="Freeze the DINOv3 Transformer backbone and keep patch_embed trainable."
    )
    parser.add_argument(
        "--no-freeze-dino",
        dest="freeze_dino",
        action="store_false",
        help="Do not freeze the DINOv3 Transformer backbone."
    )

    # Training hyperparameters.
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Total number of training epochs."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Training batch size."
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=4,
        help="Validation and test batch size."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-6,
        help="Base learning rate."
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW."
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=3,
        help="Number of warmup epochs for the warmup cosine scheduler."
    )
    parser.add_argument(
        "--lambda-grad",
        type=float,
        default=0.05,
        help="Weight for masked gradient loss."
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=50.0,
        help="Height normalization scale. For example, 50 means 1.0 equals 50 meters."
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=256,
        help="Input patch size."
    )

    # Evaluation settings.
    parser.add_argument(
        "--train-eval-fraction",
        type=float,
        default=1.0,
        help="Fraction of training batches used for epoch-level training evaluation."
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=1.0,
        help="Fraction of validation batches used for evaluation."
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=1.0,
        help="Fraction of test batches used for evaluation."
    )

    # Runtime settings.
    parser.add_argument(
        "--num-workers-train",
        type=int,
        default=4,
        help="Number of DataLoader workers for training."
    )
    parser.add_argument(
        "--num-workers-eval",
        type=int,
        default=2,
        help="Number of DataLoader workers for validation and test."
    )
    parser.add_argument(
        "--print-every-steps",
        type=int,
        default=500,
        help="Print batch-level training status every N global steps."
    )
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
        help="Device used for training."
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision."
    )

    # Resume settings.
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from a checkpoint."
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default=None,
        help="Path to the checkpoint used for resume. Default: <ckpt-dir>/latest.pth."
    )

    return parser.parse_args()


# ============================================================
# Dataset utilities
# ============================================================

def load_keys_csv(path):
    """Load sample keys from a split CSV file."""
    with open(path, "r", encoding="utf-8") as f:
        return [row["key"] for row in csv.DictReader(f)]


def build_full_dataset(data_root, target_size=256):
    """
    Build the full multimodal TreeFusion dataset.

    Expected directory structure:
        data_root/
            RGB/
            SAR_clips/
                VV/
                VH/
            nDSM_clips/
            ETH_chms/
            CHM-VHR/
    """
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
    Build a subset from sample keys.

    Returns
    -------
    subset : torch.utils.data.Subset
        Dataset subset with matched samples.

    matched_count : int
        Number of matched keys.

    missing_count : int
        Number of keys not found in the full dataset.
    """
    key_to_idx = {s["key"]: i for i, s in enumerate(full_dataset.samples)}
    idx = [key_to_idx[k] for k in keys if k in key_to_idx]

    return Subset(full_dataset, idx), len(idx), (len(keys) - len(idx))


# ============================================================
# Path utilities
# ============================================================

def resolve_paths(args):
    """Resolve project, data, split, log, and checkpoint directories."""
    project_dir = Path(args.project_dir).resolve()

    if args.data_root is None:
        data_root = project_dir / "FinalData"
    else:
        data_root = Path(args.data_root).resolve()

    if args.split_dir is None:
        split_dir = data_root / "splits_811"
    else:
        split_dir = Path(args.split_dir).resolve()

    if args.log_dir is None:
        log_dir = project_dir / "logs_FinalAll"
    else:
        log_dir = Path(args.log_dir).resolve()

    if args.ckpt_dir is None:
        ckpt_dir = project_dir / "checkFinalAll"
    else:
        ckpt_dir = Path(args.ckpt_dir).resolve()

    if args.resume_path is None:
        resume_path = ckpt_dir / "latest.pth"
    else:
        resume_path = Path(args.resume_path).resolve()

    metrics_csv = log_dir / args.metrics_filename

    return {
        "project_dir": project_dir,
        "data_root": data_root,
        "split_dir": split_dir,
        "log_dir": log_dir,
        "ckpt_dir": ckpt_dir,
        "resume_path": resume_path,
        "metrics_csv": metrics_csv,
    }


def get_device(device_arg):
    """Get training device."""
    if device_arg == "cuda":
        return torch.device("cuda")

    if device_arg == "cpu":
        return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Main training process
# ============================================================

def main():
    args = parse_args()
    paths = resolve_paths(args)

    project_dir = paths["project_dir"]
    data_root = paths["data_root"]
    split_dir = paths["split_dir"]
    log_dir = paths["log_dir"]
    ckpt_dir = paths["ckpt_dir"]
    resume_path = paths["resume_path"]
    metrics_csv = paths["metrics_csv"]

    os.chdir(project_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    device = get_device(args.device)
    use_amp = (not args.no_amp) and (device.type == "cuda")

    print("[INFO] TreeFusion v1.0 training")
    print(f"[INFO] Project directory : {project_dir}")
    print(f"[INFO] Data root         : {data_root}")
    print(f"[INFO] Split directory   : {split_dir}")
    print(f"[INFO] Log directory     : {log_dir}")
    print(f"[INFO] Checkpoint dir    : {ckpt_dir}")
    print(f"[INFO] Resume path       : {resume_path}")
    print(f"[INFO] Metrics CSV       : {metrics_csv}")
    print(f"[INFO] Device            : {device}")
    print(f"[INFO] AMP enabled       : {use_amp}")
    print(f"[INFO] Freeze DINO       : {args.freeze_dino}")

    set_seed(args.seed)

    # --------------------------------------------------------
    # Initialize metrics CSV
    # --------------------------------------------------------
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "epoch", "lr",
                "train_loss", "train_mae_m", "train_rmse_m", "train_r2",
                "val_loss",   "val_mae_m",   "val_rmse_m",   "val_r2",
                "test_loss",  "test_mae_m",  "test_rmse_m",  "test_r2",
            ])

    # --------------------------------------------------------
    # Load split files
    # --------------------------------------------------------
    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"
    test_csv = split_dir / "test.csv"

    train_keys = load_keys_csv(train_csv)
    val_keys = load_keys_csv(val_csv)
    test_keys = load_keys_csv(test_csv)

    print(
        f"[INFO] Split keys: "
        f"train={len(train_keys)} | "
        f"val={len(val_keys)} | "
        f"test={len(test_keys)}"
    )

    # --------------------------------------------------------
    # Build full dataset and split-specific subsets
    # --------------------------------------------------------
    full_dataset = build_full_dataset(
        data_root=str(data_root),
        target_size=args.target_size,
    )

    train_set, ntr, miss_tr = subset_from_keys(full_dataset, train_keys)
    val_set, nva, miss_va = subset_from_keys(full_dataset, val_keys)
    test_set, nte, miss_te = subset_from_keys(full_dataset, test_keys)

    print(
        f"[INFO] Matched samples: "
        f"train={ntr} missing={miss_tr} | "
        f"val={nva} missing={miss_va} | "
        f"test={nte} missing={miss_te}"
    )

    # --------------------------------------------------------
    # Build DataLoaders
    # --------------------------------------------------------
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers_train,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers_eval,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers_eval,
        pin_memory=(device.type == "cuda"),
    )

    # --------------------------------------------------------
    # Build model and optimizer
    # --------------------------------------------------------
    model = TreeDinoWithHeightAttention(num_heads=args.num_heads).to(device)

    if args.freeze_dino:
        freeze_dino_backbone_except_embed(model.visualEncoder.backbone)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    print(f"[INFO] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = WarmupCosineScheduler(
        base_lr=args.lr,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
    )

    criterion = nn.L1Loss()

    scaler = (
        torch.amp.GradScaler("cuda")
        if device.type == "cuda" and use_amp
        else torch.amp.GradScaler()
    )

    best_mae = float("inf")
    global_step = 0
    start_epoch = 1

    # --------------------------------------------------------
    # Resume training if requested
    # --------------------------------------------------------
    if args.resume:
        if os.path.exists(resume_path):
            start_epoch, best_mae, global_step = load_checkpoint(
                str(resume_path),
                model,
                optimizer,
                scaler,
                device,
            )
        else:
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    if start_epoch > args.epochs:
        print(f"Training already finished: start_epoch={start_epoch} > epochs={args.epochs}")
        print(f"Metrics CSV: {metrics_csv}")
        return

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        lr = scheduler.get_lr(epoch)

        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()

        total_l1 = 0.0
        total_g = 0.0
        total_loss = 0.0
        steps = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}",
            dynamic_ncols=True,
        )

        for i, batch in enumerate(pbar, start=1):
            global_step += 1

            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            amp_ctx = (
                torch.amp.autocast("cuda")
                if device.type == "cuda" and use_amp
                else torch.amp.autocast("cpu", enabled=False)
            )

            with amp_ctx:
                pred = model(x)
                loss_l1 = criterion(pred, y)

                if args.lambda_grad > 0:
                    loss_g = gradient_mask_loss(
                        pred,
                        y,
                        scale=args.scale,
                    )
                    loss = loss_l1 + args.lambda_grad * loss_g
                else:
                    loss_g = torch.tensor(0.0, device=device)
                    loss = loss_l1

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            l1v = float(loss_l1.item())
            gv = float(loss_g.item())
            tv = float(loss.item())

            total_l1 += l1v
            total_g += gv
            total_loss += tv
            steps += 1

            if (global_step % args.print_every_steps == 0) or (i == len(train_loader)):
                pred_mean = float(pred.detach().mean().item())
                pred_std = float(pred.detach().std().item())

                pbar.set_postfix({
                    "step": f"{global_step}",
                    "L1": f"{l1v:.4f}",
                    "Grad": f"{gv:.4f}",
                    "Total": f"{tv:.4f}",
                    "mean": f"{pred_mean:.3f}",
                    "std": f"{pred_std:.3f}",
                    "lr": f"{lr:.2e}",
                })

        avg_l1 = total_l1 / max(1, steps)
        avg_g = total_g / max(1, steps)
        avg_tot = total_loss / max(1, steps)

        # ----------------------------------------------------
        # Evaluate train, validation, and test splits
        # ----------------------------------------------------
        train_metrics = eval_metrics_epoch(
            model,
            train_loader,
            device,
            scale=args.scale,
            lambda_grad=args.lambda_grad,
            frac=args.train_eval_fraction,
        )

        val_metrics = eval_metrics_epoch(
            model,
            val_loader,
            device,
            scale=args.scale,
            lambda_grad=args.lambda_grad,
            frac=args.val_fraction,
        )

        test_metrics = eval_metrics_epoch(
            model,
            test_loader,
            device,
            scale=args.scale,
            lambda_grad=args.lambda_grad,
            frac=args.test_fraction,
        )

        print(
            f"\nEpoch {epoch:03d} | lr={lr:.2e} | "
            f"Train(loss={train_metrics['loss']:.5f}, "
            f"MAE={train_metrics['mae_m']:.3f} m, "
            f"RMSE={train_metrics['rmse_m']:.3f} m, "
            f"R2={train_metrics['r2']:.4f}) | "
            f"Val(loss={val_metrics['loss']:.5f}, "
            f"MAE={val_metrics['mae_m']:.3f} m, "
            f"RMSE={val_metrics['rmse_m']:.3f} m, "
            f"R2={val_metrics['r2']:.4f}) | "
            f"Test(loss={test_metrics['loss']:.5f}, "
            f"MAE={test_metrics['mae_m']:.3f} m, "
            f"RMSE={test_metrics['rmse_m']:.3f} m, "
            f"R2={test_metrics['r2']:.4f}) | "
            f"TrainAvgBatch(L1={avg_l1:.5f}, Grad={avg_g:.5f}, Total={avg_tot:.5f})"
        )

        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                epoch, lr,
                train_metrics["loss"], train_metrics["mae_m"], train_metrics["rmse_m"], train_metrics["r2"],
                val_metrics["loss"],   val_metrics["mae_m"],   val_metrics["rmse_m"],   val_metrics["r2"],
                test_metrics["loss"],  test_metrics["mae_m"],  test_metrics["rmse_m"],  test_metrics["r2"],
            ])

        # ----------------------------------------------------
        # Save latest checkpoint
        # ----------------------------------------------------
        save_checkpoint(
            os.path.join(ckpt_dir, "latest.pth"),
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            best_mae=best_mae,
            global_step=global_step,
        )

        # ----------------------------------------------------
        # Save best checkpoint based on validation MAE
        # ----------------------------------------------------
        if val_metrics["mae_m"] < best_mae:
            best_mae = val_metrics["mae_m"]

            save_checkpoint(
                os.path.join(ckpt_dir, f"best_{best_mae:.3f}m.pth"),
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                best_mae=best_mae,
                global_step=global_step,
            )

    print(f"Training finished. Best validation MAE = {best_mae:.3f} m")
    print(f"Metrics saved to: {metrics_csv}")


if __name__ == "__main__":
    main()