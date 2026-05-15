import os
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from torch import nn
from torchvision.transforms import v2

from dinov3.eval.depth.models.encoder import DinoVisionTransformerWrapper
from dinov3.eval.depth.models.dpt_head import DPTHead
from dinov3.eval.detection.models import backbone
from models.SarEncoder import SARConvNeXtEncoder


# ============================================================
# Project paths
# ============================================================
# This file is expected to be located at:
#   TreeFusion/models/OpticalEncoder.py
#
# Therefore, PROJECT_ROOT points to:
#   TreeFusion/
#
# The local DINOv3 source should be available under the TreeFusion
# project root, where hubconf.py is located.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Local torch.hub source directory for DINOv3.
Repo_Dir = str(PROJECT_ROOT)

# Pretrained DINOv3 checkpoints.
DINO_VITL16_WEIGHT = str(
    PROJECT_ROOT / "weights" / "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
)

DINO_VIT7B_WEIGHT = str(
    PROJECT_ROOT / "weights" / "7b.pth"
)


def make_transform(resize_size: int = 256):
    """Build the default image transform for DINOv3 optical inputs."""
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.430, 0.411, 0.296),
        std=(0.213, 0.156, 0.143),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


class RegressionHead(nn.Module):
    """Dense regression head for single-channel canopy height prediction."""

    def __init__(self):
        super(RegressionHead, self).__init__()
        self.regression_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Output one continuous height channel.
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.initialize_regression_head()

    def forward(self, x):
        return self.regression_head(x)

    def initialize_regression_head(self):
        """Initialize convolution and batch-normalization layers."""
        print("Initializing regression head weights with Xavier normal initialization.")
        for m in self.regression_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class RegressionHead512(nn.Module):
    """Dense regression head for 512-channel input features."""

    def __init__(self):
        super(RegressionHead512, self).__init__()
        self.regression_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Output one continuous height channel.
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.initialize_regression_head()

    def forward(self, x):
        return self.regression_head(x)

    def initialize_regression_head(self):
        """Initialize convolution and batch-normalization layers."""
        print("Initializing regression head weights with Xavier normal initialization.")
        for m in self.regression_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DinoV3dpt24(nn.Module):
    """
    DINOv3 ViT-L/16 optical encoder with a DPT dense prediction head.

    The original DINOv3 patch embedding is expanded from RGB input
    to RGB-NIR input by initializing the NIR channel with the mean
    of the RGB patch-embedding weights.
    """

    def __init__(self):
        super(DinoV3dpt24, self).__init__()
        print("Loading DINOv3 ViT-L/16 backbone weights.")

        self.backbone = torch.hub.load(
            Repo_Dir,
            'dinov3_vitl16',
            source='local',
            weights=DINO_VITL16_WEIGHT
        )

        self.expand_dino_patch_embed_to_rgbn(self.backbone)

        self.backbone_wra = DinoVisionTransformerWrapper(
            self.backbone,
            backbone_out_layers=[11, 15, 19, 23]
        )

        # self.freeze_dino_backbone_except_embed(self.backbone)
        self.densehead = DPTHead()

    def forward(self, x):
        features = self.backbone_wra(x)
        output = self.densehead(features)
        return output

    def expand_dino_patch_embed_to_rgbn(self, dino_model):
        """
        Expand DINOv3 patch embedding from 3 input channels to 4 input channels.

        RGB weights are copied directly. The NIR channel is initialized as
        the mean of the RGB patch-embedding weights.
        """
        pe = dino_model.patch_embed.proj  # Conv2d(3 -> 1024, 16, 16)
        assert isinstance(pe, torch.nn.Conv2d)

        new_pe = torch.nn.Conv2d(
            in_channels=4,
            out_channels=pe.out_channels,
            kernel_size=pe.kernel_size,
            stride=pe.stride,
            padding=pe.padding,
            bias=(pe.bias is not None)
        )

        with torch.no_grad():
            # Copy pretrained RGB weights.
            new_pe.weight[:, :3] = pe.weight

            # Initialize the NIR channel using the mean RGB weights.
            new_pe.weight[:, 3] = pe.weight.mean(dim=1)

            if pe.bias is not None:
                new_pe.bias.copy_(pe.bias)

        dino_model.patch_embed.proj = new_pe
        print("Expanded patch_embed from RGB to RGB-NIR. "
              "The NIR channel was initialized using the mean RGB weights.")
        return dino_model


class DinoV3dpt24_RGB(nn.Module):
    """RGB-only DINOv3 ViT-L/16 encoder with a DPT dense prediction head."""

    def __init__(self):
        super(DinoV3dpt24_RGB, self).__init__()
        print("Loading DINOv3 ViT-L/16 backbone weights for RGB-only input.")

        self.backbone = torch.hub.load(
            Repo_Dir,
            'dinov3_vitl16',
            source='local',
            weights=DINO_VITL16_WEIGHT
        )

        # Keep the original RGB patch embedding without channel expansion.
        self.backbone_wra = DinoVisionTransformerWrapper(
            self.backbone,
            backbone_out_layers=[11, 15, 19, 23]
        )

        self.densehead = DPTHead()

    def forward(self, x):
        # Enforce RGB-only input.
        # This check can be removed if more flexible input handling is needed.
        assert x.dim() == 4 and x.shape[1] == 3, \
            f"Expected RGB input (B,3,H,W), got {tuple(x.shape)}"

        features = self.backbone_wra(x)
        output = self.densehead(features)
        return output


# def freeze_dino_backbone_except_embed(self, dino_model):
#     """Freeze all DINO Transformer parameters except the patch embedding layer."""
#     for name, param in dino_model.named_parameters():
#         if "patch_embed" in name:
#             param.requires_grad = True
#         else:
#             param.requires_grad = False
#     print("Frozen Transformer layers and kept patch embedding trainable.")


class DinoV3dpt24_7b(nn.Module):
    """
    DINOv3 ViT-7B optical encoder with a DPT dense prediction head.

    This class manually loads the 7B checkpoint and synchronizes the
    backbone parameters across distributed ranks when distributed training
    is enabled.
    """

    def __init__(self):
        super(DinoV3dpt24_7b, self).__init__()
        print("Loading DINOv3 ViT-7B backbone.")

        self.backbone = torch.hub.load(
            Repo_Dir,
            'dinov3_vit7b16',
            source='local',
            weights=DINO_VIT7B_WEIGHT
        )

        self.backbone = torch.hub.load(
            Repo_Dir,
            'dinov3_vit7b16',
            source='local',
            pretrained=False,
        )

        # Manually load the 7B checkpoint on rank 0.
        if not dist.is_initialized() or dist.get_rank() == 0:
            print("Loading 7B checkpoint to GPU.")
            checkpoint = torch.load(
                DINO_VIT7B_WEIGHT,
                map_location='cuda'
            )

            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            msg = self.backbone.load_state_dict(state_dict, strict=False)
            print(f"Loaded backbone weights with message: {msg}")

            del checkpoint
            torch.cuda.empty_cache()

        # Broadcast backbone parameters from rank 0 to other distributed ranks.
        if dist.is_initialized():
            dist.barrier()
            for p in self.backbone.parameters():
                dist.broadcast(p.data, src=0)

        self.expand_dino_patch_embed_to_rgbn(self.backbone)

        self.backbone_wra = DinoVisionTransformerWrapper(
            self.backbone,
            backbone_out_layers=[11, 15, 19, 23]
        )

        self.freeze_dino_backbone_except_embed(self.backbone)
        self.densehead = DPTHead()

    def forward(self, x):
        features = self.backbone_wra(x)
        output = self.densehead(features)
        return output

    def expand_dino_patch_embed_to_rgbn(self, dino_model):
        """
        Expand DINOv3 patch embedding from 3 input channels to 4 input channels.

        RGB weights are copied directly. The NIR channel is initialized as
        the mean of the RGB patch-embedding weights.
        """
        pe = dino_model.patch_embed.proj  # Conv2d(3 -> 1024, 16, 16)
        assert isinstance(pe, torch.nn.Conv2d)

        new_pe = torch.nn.Conv2d(
            in_channels=4,
            out_channels=pe.out_channels,
            kernel_size=pe.kernel_size,
            stride=pe.stride,
            padding=pe.padding,
            bias=(pe.bias is not None)
        )

        with torch.no_grad():
            # Copy pretrained RGB weights.
            new_pe.weight[:, :3] = pe.weight

            # Initialize the NIR channel using the mean RGB weights.
            new_pe.weight[:, 3] = pe.weight.mean(dim=1)

            if pe.bias is not None:
                new_pe.bias.copy_(pe.bias)

        dino_model.patch_embed.proj = new_pe
        print("Expanded patch_embed from RGB to RGB-NIR. "
              "The NIR channel was initialized using the mean RGB weights.")
        return dino_model

    def freeze_dino_backbone_except_embed(self, dino_model):
        """Freeze all DINO Transformer parameters except the patch embedding layer."""
        for name, param in dino_model.named_parameters():
            if "patch_embed" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("Frozen Transformer layers and kept patch embedding trainable.")


class SARFusionBlock(nn.Module):
    """
    Multi-scale SAR feature fusion block.

    The block first aligns SAR features from multiple encoder stages to the
    same channel dimension, upsamples them to the highest SAR feature
    resolution, concatenates them, and then produces a fused 256-channel
    feature map. The final output is upsampled to 256 x 256 to match the
    optical feature resolution.
    """

    def __init__(self, sar_dims=(64, 128, 256, 512), out_dim=256):
        super().__init__()

        self.align_convs = nn.ModuleList([
            nn.Conv2d(sar_dim, out_dim, kernel_size=1)
            for sar_dim in sar_dims
        ])

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_dim * len(sar_dims), out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, sar_feats):
        upsampled = []

        # Use the highest-resolution SAR feature as the target size.
        target_size = sar_feats[0].shape[-2:]

        # Upsample all SAR features to the target spatial resolution.
        for i, feat in enumerate(sar_feats):
            aligned = self.align_convs[i](feat)
            up = torch.nn.functional.interpolate(
                aligned,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
            upsampled.append(up)

        cat = torch.cat(upsampled, dim=1)
        fused = self.fusion_conv(cat)

        # Upsample the fused SAR feature to match the optical feature resolution.
        fused_upsampled = torch.nn.functional.interpolate(
            fused,
            size=(256, 256),
            mode='bilinear',
            align_corners=False
        )

        return fused_upsampled