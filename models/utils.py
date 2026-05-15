import torch
from torch import nn
import torch.nn.functional as F
from typing import List

# -----------------------------------------------------------
# Height Encoder
# -----------------------------------------------------------
class HeightEncoder(nn.Module):
    """
    将 (nDSM, RoughCHM) 两通道编码到与主干同维度 (默认 512)
    """
    def __init__(self, in_ch=2, out_ch=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, out_ch, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------
# ConvProj (used previously) - 保留若需
# -----------------------------------------------------------
class ConvProj(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        # x: (B,C,H,W)
        B,C,H,W = x.shape
        x = x.permute(0,2,3,1)      # BCHW → BHWC
        x = self.ln(x)
        x = x.permute(0,3,1,2)      # BHWC → BCHW
        return self.conv(x)

class GlobalCrossAttention(nn.Module):
    """
    Q from src, K/V from tgt. No windows, full attention.
    src_feat / tgt_feat: (B, C, H, W)
    """
    def __init__(self, dim, num_heads=4, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0, logit_clamp=6.0):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_clamp = float(logit_clamp)

        self.q_proj = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.k_proj = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.v_proj = nn.Conv2d(dim, dim, 1, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, src_feat, tgt_feat):
        """
        src_feat : 作为 Q
        tgt_feat : 作为 K/V
        """
        B, C, H, W = src_feat.shape

        Q = self.q_proj(src_feat)
        K = self.k_proj(tgt_feat)
        V = self.v_proj(tgt_feat)

        qN = H * W
        Q = Q.view(B, self.num_heads, self.head_dim, qN).permute(0,1,3,2)
        K = K.view(B, self.num_heads, self.head_dim, qN).permute(0,1,3,2)
        V = V.view(B, self.num_heads, self.head_dim, qN).permute(0,1,3,2)

        attn = torch.matmul(Q, K.transpose(-2,-1)) * self.scale
        attn = torch.clamp(attn, -self.logit_clamp, self.logit_clamp)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, V)
        out = out.permute(0,1,3,2).contiguous().view(B, C, H, W)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out



class BiModalCrossAttention(nn.Module):
    """
    Optical ↔ SAR 双向全局 cross-attention（多尺度）。
    optical_feat <-> sar_feat
    returns:
        refined_optical, refined_sar
    """
    def __init__(self, dim=256, num_heads=4,
                 pool_levels=[8,16],
                 alpha=0.25, beta=0.25):
        super().__init__()

        self.pool_levels = pool_levels
        self.alpha = alpha
        self.beta = beta

        self.opt2sar = nn.ModuleList([
            GlobalCrossAttention(dim, num_heads) for _ in pool_levels
        ])
        self.sar2opt = nn.ModuleList([
            GlobalCrossAttention(dim, num_heads) for _ in pool_levels
        ])

        self.proj_opt = nn.Conv2d(dim, dim, 1)
        self.proj_sar = nn.Conv2d(dim, dim, 1)

    def forward(self, opt_feat, sar_feat):
        """
        opt_feat : (B,C,H,W)  optical
        sar_feat : (B,C,H,W)  SAR
        """

        B,C,H,W = opt_feat.shape
        opt2sar_all = []
        sar2opt_all = []

        for i, p in enumerate(self.pool_levels):
            if p > 1:
                opt_low = F.avg_pool2d(opt_feat, p, p)
                sar_low = F.avg_pool2d(sar_feat, p, p)
            else:
                opt_low = opt_feat
                sar_low = sar_feat

            # Optical → SAR
            opt2sar_low = self.opt2sar[i](opt_low, sar_low)
            # SAR → Optical
            sar2opt_low = self.sar2opt[i](sar_low, opt_low)

            # back to original size
            opt2sar_up = F.interpolate(opt2sar_low, size=(H,W), mode="bilinear")
            sar2opt_up = F.interpolate(sar2opt_low, size=(H,W), mode="bilinear")

            opt2sar_all.append(opt2sar_up)
            sar2opt_all.append(sar2opt_up)

        opt2sar = torch.stack(opt2sar_all).mean(0)
        sar2opt = torch.stack(sar2opt_all).mean(0)

        refined_opt = opt_feat + self.alpha * self.proj_opt(sar2opt)
        refined_sar = sar_feat + self.beta * self.proj_sar(opt2sar)

        return refined_opt, refined_sar


# -----------------------------------------------------------
# Height Global Context (unchanged)
# -----------------------------------------------------------
class HeightGlobalContext(nn.Module):
    """
    q_feat : (B,C,H,W)  -> Q
    kv_feat: (B,C,H,W)  -> K,V
    """
    def __init__(self, in_dim=256, hidden=128, out_dim=256):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_dim, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_dim * 2, 1)
        )

    def forward(self, height_feat):
        """
        输入: height_feat (B,256,H,W)
        输出: gamma, beta (B,256,1,1)
        """
        x = self.pool(height_feat)        # (B,256,1,1)
        x = self.mlp(x)                   # (B,512,1,1)
        gamma, beta = torch.chunk(x, 2, dim=1)
        gamma = torch.tanh(gamma)
        beta  = torch.tanh(beta)
        return gamma, beta