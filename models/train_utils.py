# -*- coding: utf-8 -*-
"""
TreeFusion v1.0 training utilities.

Author:
    Taige Luo et al.

Description:
    Utility functions used by the TreeFusion training script, including
    reproducibility setup, DINOv3 backbone freezing, gradient loss,
    learning-rate scheduling, evaluation metrics, and checkpoint I/O.

Notes:
    This file is separated from the main training script to keep the training
    workflow readable. It does not define model architectures.
"""

import random
import math

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn


# ============================================================
# Reproducibility utilities
# ============================================================

def set_seed(seed=42):
    """Set random seeds for reproducible training and evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze_dino_backbone_except_embed(dino_model):
    """
    Freeze the DINOv3 Transformer backbone while keeping patch_embed trainable.

    This setting allows the model to adapt the input embedding layer while
    keeping most pretrained DINOv3 parameters fixed.
    """
    for name, param in dino_model.named_parameters():
        if "patch_embed" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    print("DINOv3 Transformer backbone frozen; patch_embed remains trainable.")


# ============================================================
# Sobel gradient loss
# ============================================================

def compute_gradients(tensor):
    """
    Compute Sobel gradients along x and y directions.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor with shape [B, 1, H, W].

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Sobel gradients in x and y directions.
    """
    sobel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        device=tensor.device,
        dtype=tensor.dtype
    ).unsqueeze(0).unsqueeze(0)

    sobel_y = torch.tensor(
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]],
        device=tensor.device,
        dtype=tensor.dtype
    ).unsqueeze(0).unsqueeze(0)

    gx = F.conv2d(tensor, sobel_x, padding=1)
    gy = F.conv2d(tensor, sobel_y, padding=1)

    return gx, gy


def gradient_mask_loss(pred, gt, scale, h_thresh=1.0, grad_thresh=0.02):
    """
    Compute masked gradient loss for canopy height regression.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted CHM with shape [B, 1, H, W]. Values are normalized to [0, 1].

    gt : torch.Tensor
        Ground-truth CHM with shape [B, 1, H, W]. Values are normalized to [0, 1].

    scale : float
        Height normalization scale. For example, scale=50 means that
        normalized value 1.0 corresponds to 50 meters.

    h_thresh : float, default=1.0
        Height threshold in meters. Pixels below this threshold are ignored.

    grad_thresh : float, default=0.02
        Gradient difference threshold in normalized units.

    Returns
    -------
    torch.Tensor
        Masked gradient loss.
    """
    mask_h = gt > (h_thresh / scale)

    gx_p, gy_p = compute_gradients(pred)
    gx_g, gy_g = compute_gradients(gt)

    gdiff = torch.abs(gx_p - gx_g) + torch.abs(gy_p - gy_g)
    mask = (mask_h & (gdiff > grad_thresh)).float()

    if mask.sum() < 1e-6:
        return torch.tensor(0.0, device=pred.device)

    return (gdiff * mask).sum() / (mask.sum() + 1e-6)


# ============================================================
# Warmup cosine learning-rate scheduler
# ============================================================

class WarmupCosineScheduler:
    """
    Warmup + cosine decay learning-rate scheduler.

    The learning rate increases linearly during warmup epochs and then follows
    a cosine decay schedule.
    """

    def __init__(self, base_lr, total_epochs, warmup_epochs=0):
        self.base_lr = base_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs

    def get_lr(self, epoch):
        """Return the learning rate for the current epoch."""
        if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
            return self.base_lr * epoch / self.warmup_epochs

        if self.total_epochs == self.warmup_epochs:
            return self.base_lr

        progress = (epoch - self.warmup_epochs) / (
            self.total_epochs - self.warmup_epochs
        )
        progress = min(max(progress, 0.0), 1.0)

        return self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))


# ============================================================
# Evaluation metrics
# ============================================================

@torch.no_grad()
def eval_metrics_epoch(model, loader, device, scale, lambda_grad=0.0, frac=1.0):
    """
    Evaluate one epoch on a given DataLoader.

    Metrics:
        loss   : normalized training-style loss
        mae_m  : mean absolute error in meters
        rmse_m : root mean squared error in meters
        r2     : coefficient of determination

    Parameters
    ----------
    model : torch.nn.Module
        TreeFusion model.

    loader : torch.utils.data.DataLoader
        Evaluation DataLoader.

    device : torch.device
        Device used for evaluation.

    scale : float
        Height normalization scale.

    lambda_grad : float, default=0.0
        Weight for the masked gradient loss.

    frac : float, default=1.0
        Fraction of batches used for evaluation.
    """
    model.eval()
    l1 = nn.L1Loss(reduction="mean")

    loss_sum = 0.0
    loss_batches = 0

    n = 0
    sum_abs = 0.0
    sum_sq = 0.0
    sum_y = 0.0
    sum_y2 = 0.0

    amp_ctx = (
        torch.amp.autocast("cuda")
        if device.type == "cuda"
        else torch.amp.autocast("cpu")
    )

    for batch in loader:
        if frac < 1.0 and random.random() > frac:
            continue

        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        with amp_ctx:
            pred = model(x)
            loss_l1 = l1(pred, y)

            if lambda_grad > 0:
                lg = gradient_mask_loss(pred, y, scale=scale)
                loss = loss_l1 + lambda_grad * lg
            else:
                loss = loss_l1

        loss_sum += float(loss.item())
        loss_batches += 1

        pred_m = (pred.float() * scale).reshape(-1)
        y_m = (y.float() * scale).reshape(-1)
        diff = pred_m - y_m

        sum_abs += torch.abs(diff).sum().item()
        sum_sq += (diff * diff).sum().item()

        sum_y += y_m.sum().item()
        sum_y2 += (y_m * y_m).sum().item()
        n += y_m.numel()

    if n == 0:
        return {
            "loss": float("nan"),
            "mae_m": float("nan"),
            "rmse_m": float("nan"),
            "r2": float("nan"),
        }

    mae = sum_abs / n
    rmse = math.sqrt(sum_sq / n)

    sse = sum_sq
    mean_y = sum_y / n
    sst = sum_y2 - n * (mean_y ** 2)
    r2 = float("nan") if sst <= 1e-12 else (1.0 - sse / sst)

    avg_loss = loss_sum / max(1, loss_batches)

    return {
        "loss": avg_loss,
        "mae_m": mae,
        "rmse_m": rmse,
        "r2": r2,
    }


# ============================================================
# Checkpoint and resume utilities
# ============================================================

def _get_rng_state():
    """Get random number generator states for reproducible resume training."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def _set_rng_state(state):
    """Restore random number generator states."""
    if state is None:
        return

    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])

        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])

    except Exception as e:
        print(f"[WARN] Failed to restore RNG state. Ignored. Error: {e}")


def save_checkpoint(path, epoch, model, optimizer, scaler, best_mae, global_step):
    """
    Save a training checkpoint.

    The checkpoint includes model, optimizer, AMP scaler, best validation MAE,
    global step, and RNG states.
    """
    ckpt = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_mae": float(best_mae),
        "global_step": int(global_step),
        "rng_state": _get_rng_state(),
    }

    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer, scaler, device):
    """
    Load a training checkpoint for standard resume training.

    This function assumes the checkpoint was saved by save_checkpoint().
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])

    best_mae = float(ckpt["best_mae"])
    global_step = int(ckpt["global_step"])
    epoch = int(ckpt["epoch"])

    _set_rng_state(ckpt.get("rng_state", None))

    start_epoch = epoch + 1

    print(
        f"Resume successful: {path} | "
        f"checkpoint epoch={epoch} -> start epoch={start_epoch} | "
        f"best MAE={best_mae:.4f} m | "
        f"global step={global_step}"
    )

    return start_epoch, best_mae, global_step