"""Convolutional blocks, attention, and gated structural fusion."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        dilation: int = 1,
    ) -> None:
        if padding is None:
            padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class SEBlock(nn.Module):
    """Channel attention (squeeze-and-excitation)."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        s = F.adaptive_avg_pool2d(x, 1)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s), inplace=True)))
        return x * s


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True).values
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 8, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel = SEBlock(channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.spatial(self.channel(x))


class GatedStructuralFusion(nn.Module):
    """Inject a projected structural feature into an RGB feature with a learned gate.

    The gate is produced from both streams (spatial + channel attention), so the model
    can suppress handcrafted features on images where they are misleading.
    """

    def __init__(self, rgb_ch: int, struct_ch: int, out_ch: int | None = None) -> None:
        super().__init__()
        out_ch = out_ch or rgb_ch
        self.rgb_proj = nn.Conv2d(rgb_ch, out_ch, 1, bias=False) if rgb_ch != out_ch else nn.Identity()
        self.struct_proj = ConvBNReLU(struct_ch, out_ch, 3)
        self.gate = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.Sigmoid(),
        )
        self.refine = CBAM(out_ch)

    def forward(self, rgb: Tensor, struct: Tensor) -> Tensor:
        if struct.shape[-2:] != rgb.shape[-2:]:
            struct = F.interpolate(struct, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        rgb = self.rgb_proj(rgb)
        struct = self.struct_proj(struct)
        gate = self.gate(torch.cat([rgb, struct], dim=1))
        return self.refine(rgb + gate * struct)


class CrossAttentionFusion(nn.Module):
    """Optional global cross-attention at the deepest stage (structural query -> RGB)."""

    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, rgb: Tensor, struct: Tensor) -> Tensor:
        b, c, h, w = rgb.shape
        q = struct.flatten(2).transpose(1, 2)
        kv = rgb.flatten(2).transpose(1, 2)
        attn, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv), need_weights=False)
        q = q + attn
        q = q + self.ffn(self.norm2(q))
        out = q.transpose(1, 2).reshape(b, c, h, w)
        return rgb + torch.tanh(self.gate) * out


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch + skip_ch, out_ch, 3)
        self.conv2 = ConvBNReLU(out_ch, out_ch, 3)

    def forward(self, x: Tensor, skip: Tensor | None = None) -> Tensor:
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv2(self.conv1(x))


class ContrastiveHead(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, embed_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.proj(x), dim=1)
