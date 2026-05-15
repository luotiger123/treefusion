# -*- coding: utf-8 -*-
"""
TreeFusion v1.0 vegetation segmentation model.

This file defines a lightweight RGB-NIR vegetation segmentation model based on
the DINOv3 optical encoder. The model predicts pixel-level vegetation masks
from four-channel RGB-NIR imagery.

Author:
    Taige Luo et al.

Notes:
    The class names and internal module names are kept unchanged for
    compatibility with existing scripts and checkpoints.
"""

import torch
from torch import nn

from models.OpticalEncoder import DinoV3dpt24


class SegmentationHead(nn.Module):
    """
    Lightweight segmentation head for pixel-level vegetation classification.

    Parameters
    ----------
    in_ch : int, default=256
        Number of input feature channels.

    num_classes : int, default=2
        Number of output classes. For vegetation segmentation, the default
        setting uses two classes: background and vegetation.

    dropout : float, default=0.1
        Dropout ratio used in the segmentation head.

    Input
    -----
    x : torch.Tensor
        Feature tensor with shape [B, in_ch, H, W].

    Output
    ------
    torch.Tensor
        Segmentation logits with shape [B, num_classes, H, W].
    """

    def __init__(self, in_ch=256, num_classes=2, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, num_classes, 1)
        )

    def forward(self, x):
        # Return raw logits. Softmax or argmax should be applied outside
        # the model during inference or evaluation.
        return self.net(x)


class TreeDinoSegLite(nn.Module):
    """
    Lightweight DINOv3-based RGB-NIR segmentation model.

    This model uses the RGB-NIR optical encoder from TreeFusion and a compact
    convolutional segmentation head for vegetation mask prediction.

    Input
    -----
    x : torch.Tensor
        RGB-NIR input tensor with shape [B, 4, H, W].

    Output
    ------
    torch.Tensor
        Segmentation logits with shape [B, 2, H, W] by default.

    Notes
    -----
    DinoV3dpt24 internally expands the original DINOv3 patch embedding from
    3-channel RGB input to 4-channel RGB-NIR input. The NIR channel weights
    are initialized using the mean of the pretrained RGB patch-embedding
    weights.
    """

    def __init__(self, num_classes=2, dropout=0.1):
        super().__init__()

        # RGB-NIR optical encoder. Output feature shape: [B, 256, H, W].
        self.visualEncoder = DinoV3dpt24()

        # Final feature normalization before segmentation.
        self.final_norm = nn.GroupNorm(1, 256)

        # Pixel-level segmentation head.
        self.segHead = SegmentationHead(
            in_ch=256,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            RGB-NIR input tensor with shape [B, 4, H, W].

        Returns
        -------
        torch.Tensor
            Segmentation logits with shape [B, num_classes, H, W].
        """
        assert x.dim() == 4 and x.shape[1] == 4, \
            f"Expected RGB-NIR input (B,4,H,W), got {tuple(x.shape)}"

        feat = self.visualEncoder(x)
        feat = self.final_norm(feat)
        logits = self.segHead(feat)

        return logits


if __name__ == "__main__":
    # Simple sanity check for model input and output shapes.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = TreeDinoSegLite(num_classes=2, dropout=0.1).to(device)
    model.eval()

    B, H, W = 4, 256, 256

    # Dummy RGB-NIR input.
    x_rgbn = torch.randn(B, 4, H, W, device=device)

    with torch.no_grad():
        logits = model(x_rgbn)

    print("Input shape:", x_rgbn.shape)
    print("Logits shape:", logits.shape)
    print("Logits min/max:", logits.min().item(), logits.max().item())

    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1)

    print("Probability shape:", probs.shape)
    print("Predicted mask shape:", pred.shape)
    print("Unique labels in predicted mask:", torch.unique(pred).tolist())

    print("TreeDinoSegLite RGB-NIR forward pass completed successfully.")