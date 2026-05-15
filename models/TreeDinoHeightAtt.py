# -*- coding: utf-8 -*-
"""
TreeFusion v1.0

Multimodal deep learning model for very-high-resolution urban tree canopy
height mapping.

This file keeps the original model class names and module attribute names
for checkpoint compatibility. Please do not rename the classes or internal
modules unless you also convert the checkpoint keys.

Author:
    Taige Luo et al.

Model:
    TreeDinoWithHeightAttention
    TreeDinoWithHeightAttention7

Input:
    TreeDinoWithHeightAttention:
        x: [B, 8, H, W]
        channels = [RGBN(4), SAR VV/VH(2), nDSM + Rough CHM(2)]

    TreeDinoWithHeightAttention7:
        x: [B, 7, H, W]
        channels = [RGBN(4), SAR VV/VH(2), Rough CHM(1)]

Output:
    Predicted canopy height map with shape [B, 1, H, W].
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import List

from models.SarEncoder import SARConvNeXtEncoder, LightSAREncoder
from models.OpticalEncoder import DinoV3dpt24, SARFusionBlock, RegressionHead
from models.utils import *


# -----------------------------------------------------------
# Main TreeFusion model with height-prior attention.
#
# NOTE:
# The class name and all internal module names are intentionally
# kept unchanged to maintain compatibility with previously trained
# checkpoints.
# -----------------------------------------------------------
class TreeDinoWithHeightAttention(nn.Module):
    def __init__(self, num_heads=4):
        """
        Initialize the 8-channel TreeFusion model.

        Parameters
        ----------
        num_heads : int, default=4
            Number of attention heads used in the bidirectional
            RGB-SAR cross-attention module.
        """
        super().__init__()

        # Optical encoder for RGB-NIR imagery.
        self.visualEncoder = DinoV3dpt24()

        # SAR encoder for VV/VH backscatter.
        self.sarEncoder = SARConvNeXtEncoder()

        # Fusion block for merging multi-scale SAR features.
        self.fusion = SARFusionBlock()

        # Resolution-matched bidirectional cross-attention between
        # optical and SAR features.
        self.rgb_sar_cross = BiModalCrossAttention(
            dim=256,
            num_heads=num_heads,
            pool_levels=[8, 16],
            alpha=0.25,
            beta=0.25
        )

        # Global context module for height-prior modulation.
        self.height_ctx = HeightGlobalContext(
            in_dim=256,
            hidden=128,
            out_dim=256
        )

        # Modulation strength for height-prior context.
        self.height_ctx_strength = 0.15

        # Align concatenated optical and SAR features back to 256 channels.
        self.fused_align = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Height encoder for nDSM and rough CHM.
        self.heightEncoder = HeightEncoder(in_ch=2, out_ch=256)

        # Final normalization before dense regression.
        self.final_norm = nn.GroupNorm(1, 256)

        # Regression head for canopy height prediction.
        self.regressionHead = RegressionHead()

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape [B, 8, H, W].
            Channel order:
                0:3  -> RGB-NIR
                4:5  -> SAR VV/VH
                6:7  -> nDSM and rough CHM

        Returns
        -------
        torch.Tensor
            Predicted canopy height map with shape [B, 1, H, W].
        """
        rgbn = x[:, :4]
        sar = x[:, 4:6]
        h2d = x[:, 6:8]

        # 1. Extract optical and SAR features.
        f_opt = self.visualEncoder(rgbn)
        f_sars = self.sarEncoder(sar)
        f_sar = self.fusion(f_sars)

        # 2. Apply bidirectional RGB-SAR cross-attention.
        f_opt_refined, f_sar_refined = self.rgb_sar_cross(f_opt, f_sar)

        # 3. Fuse refined optical and SAR features.
        fused = torch.cat([f_opt_refined, f_sar_refined], dim=1)
        fused = self.fused_align(fused)

        # 4. Encode height priors and apply global context modulation.
        height_feat = self.heightEncoder(h2d)
        gamma, beta = self.height_ctx(height_feat)

        fused = fused * (1 + self.height_ctx_strength * gamma) \
                + self.height_ctx_strength * beta

        fused = self.final_norm(fused)

        # 5. Predict canopy height.
        out = self.regressionHead(fused)

        return out

if __name__ == '__main__':
    # Simple sanity check for model input and output shapes.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = TreeDinoWithHeightAttention(num_heads=4).to(device)

    dummy = torch.randn(18, 8, 256, 256).to(device)

    with torch.no_grad():
        out = model(dummy)

    print("Output shape:", out.shape)