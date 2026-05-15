# -*- coding: utf-8 -*-
"""
TreeFusion v1.0 SAR encoder modules.

This file defines the SAR feature encoders used in TreeFusion, including
a ConvNeXt-style SAR encoder and a lightweight convolutional SAR encoder.

Author:
    Taige Luo et al.

Notes:
    Class names and internal module names are kept unchanged for checkpoint
    compatibility. Do not rename these modules unless checkpoint keys are
    also converted.
"""

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """
    LayerNorm wrapper for BCHW feature maps.

    PyTorch LayerNorm is applied on the channel dimension after converting
    the feature layout from BCHW to BHWC, then converted back to BCHW.
    """

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class ConvNeXtBlock(nn.Module):
    """
    Basic ConvNeXt-style block for SAR feature extraction.

    The block contains:
        1. Depthwise convolution
        2. Channel-wise LayerNorm
        3. Pointwise MLP
        4. Residual connection
    """

    def __init__(self, dim):
        super().__init__()

        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=7,
            padding=3,
            groups=dim
        )

        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        shortcut = x

        x = self.dwconv(x)

        # Convert BCHW to BHWC for channel-wise LayerNorm and linear layers.
        x = x.permute(0, 2, 3, 1)

        # Use the internal LayerNorm directly because the tensor is already BHWC.
        x = self.norm.norm(x)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # Convert back to BCHW.
        x = x.permute(0, 3, 1, 2)

        return x + shortcut


class SARConvNeXtEncoder(nn.Module):
    """
    ConvNeXt-style SAR encoder.

    This encoder takes two-channel SAR input, usually VV and VH backscatter,
    and outputs multi-scale SAR features for later fusion with optical
    features.

    Parameters
    ----------
    in_chans : int, default=2
        Number of input SAR channels. The default setting uses VV and VH.

    dims : tuple of int, default=(64, 128, 256, 512)
        Channel dimensions for the four encoder stages.

    Input
    -----
    x : torch.Tensor
        SAR input tensor with shape [B, 2, H, W].

    Output
    ------
    list[torch.Tensor]
        Multi-scale SAR feature maps from four encoder stages.
    """

    def __init__(self, in_chans=2, dims=(64, 128, 256, 512)):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()

        # Stage 0: stem layer.
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0])
        )
        self.downsample_layers.append(stem)

        # Stages 1-3: downsampling followed by ConvNeXt blocks.
        for i in range(3):
            downsample = nn.Sequential(
                LayerNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            )

            self.downsample_layers.append(downsample)

            stage = nn.Sequential(
                *[ConvNeXtBlock(dims[i + 1]) for _ in range(3)]
            )
            self.stages.append(stage)

        self.out_dims = dims

        self.initialize_sar_encoder()

    def forward(self, x):
        features = []

        for i in range(4):
            x = self.downsample_layers[i](x)

            if i > 0:
                x = self.stages[i - 1](x)

            features.append(x)

        return features

    def initialize_sar_encoder(self):
        """
        Initialize SAR encoder weights.

        Conv2d layers are initialized with Kaiming normal initialization.
        LayerNorm parameters are initialized with bias = 0 and weight = 1.
        """
        print("Initializing SAR encoder weights with Kaiming normal initialization.")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)

            elif isinstance(m, LayerNorm2d):
                nn.init.constant_(m.norm.bias, 0)
                nn.init.constant_(m.norm.weight, 1)

        print("SAR encoder initialization completed.")


class LightSAREncoder(nn.Module):
    """
    Lightweight convolutional SAR encoder.

    This module provides a simple alternative SAR encoder with three
    convolutional layers. It is mainly used for ablation studies or
    lightweight baseline experiments.

    Parameters
    ----------
    in_ch : int, default=2
        Number of SAR input channels.

    out_ch : int, default=256
        Number of output feature channels.

    Input
    -----
    x : torch.Tensor
        SAR input tensor with shape [B, in_ch, H, W].

    Output
    ------
    torch.Tensor
        SAR feature tensor with shape [B, out_ch, H, W].
    """

    def __init__(self, in_ch=2, out_ch=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.encoder(x)