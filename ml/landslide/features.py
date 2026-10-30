"""Ten-channel differentiable structural features computed from RGB only.

The reference architecture uses handcrafted structural channels.  We keep the same
idea but make every channel a torch op, so it runs on CPU/GPU, inside the training
graph, and inside the exported deployment graph without OpenCV or GIS libraries.

Channels (all in [0, 1]):
    0 luminance
    1 Sobel gradient magnitude
    2 gradient orientation cos(2*theta)
    3 gradient orientation sin(2*theta)
    4 Laplacian response
    5 local standard deviation (7x7)
    6 local Shannon entropy (8 bins, 7x7)
    7 morphological top-hat (bright thin structures)
    8 morphological black-hat (dark thin structures)
    9 HSV saturation
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _sobel_kernels(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    return kx, ky


def _laplacian_kernel(device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)


def _dilate(x: Tensor, k: int = 3) -> Tensor:
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)


def _erode(x: Tensor, k: int = 3) -> Tensor:
    return -F.max_pool2d(-x, kernel_size=k, stride=1, padding=k // 2)


def denormalize_rgb(image: Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> Tensor:
    """Undo ImageNet normalization so structural features see physical [0, 1] RGB."""
    m = torch.tensor(mean, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    return (image * s + m).clamp(0.0, 1.0)


class StructuralFeatureExtractor(nn.Module):
    """Stateless 10-channel structural prior.

    The extractor has no learnable parameters, which keeps the contribution
    interpretable and makes the "structural features" ablation exact.
    """

    def __init__(self, window: int = 7, entropy_bins: int = 8) -> None:
        super().__init__()
        self.window = window
        self.entropy_bins = entropy_bins

    def forward(self, rgb01: Tensor) -> Tensor:
        if rgb01.ndim != 4 or rgb01.shape[1] != 3:
            raise ValueError(f"expected [B,3,H,W] RGB, got {tuple(rgb01.shape)}")
        b, _, h, w = rgb01.shape
        r, g, bl = rgb01[:, 0:1], rgb01[:, 1:2], rgb01[:, 2:3]
        gray = (0.299 * r + 0.587 * g + 0.114 * bl).clamp(0.0, 1.0)

        kx, ky = _sobel_kernels(rgb01.device, rgb01.dtype)
        gray_pad = F.pad(gray, (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(gray_pad, kx, padding=0)
        gy = F.conv2d(gray_pad, ky, padding=0)
        energy = gx * gx + gy * gy + 1e-6
        mag = torch.sqrt(energy)
        # cos(2*theta), sin(2*theta) without atan2/cos/sin: export-friendly and
        # numerically stable for the ONNX/tract deployment path.
        orient_cos = (0.5 * ((gx * gx - gy * gy) / energy + 1.0)).clamp(0.0, 1.0)
        orient_sin = (0.5 * ((2.0 * gx * gy) / energy + 1.0)).clamp(0.0, 1.0)

        lap = F.conv2d(gray_pad, _laplacian_kernel(rgb01.device, rgb01.dtype), padding=0)

        pad = self.window // 2
        gray_win = F.pad(gray, (pad, pad, pad, pad), mode="replicate")
        local_mean = F.avg_pool2d(gray_win, self.window, stride=1, padding=0)
        local_sq = F.avg_pool2d(gray_win * gray_win, self.window, stride=1, padding=0)
        local_std = torch.sqrt((local_sq - local_mean * local_mean).clamp_min(0.0))

        bins = torch.clamp((gray * self.entropy_bins).long(), 0, self.entropy_bins - 1).squeeze(1)
        onehot = F.one_hot(bins, self.entropy_bins).permute(0, 3, 1, 2).to(gray.dtype)
        onehot_pad = F.pad(onehot, (pad, pad, pad, pad), mode="replicate")
        probs = F.avg_pool2d(onehot_pad, self.window, stride=1, padding=0)
        entropy = -(probs.clamp_min(1e-6).log() * probs).sum(dim=1, keepdim=True)
        entropy = entropy / math.log(self.entropy_bins)

        opening = _dilate(_erode(gray, 3), 3)
        closing = _erode(_dilate(gray, 3), 3)
        tophat = (gray - opening).clamp_min(0.0)
        blackhat = (closing - gray).clamp_min(0.0)

        mx = rgb01.max(dim=1, keepdim=True).values
        mn = rgb01.min(dim=1, keepdim=True).values
        saturation = (mx - mn) / (mx + 1e-6)

        # Fixed, monotone scalings keep channels comparable without dataset statistics.
        feats = torch.cat(
            [
                gray,
                (mag * 4.0).clamp(0.0, 1.0),
                orient_cos,
                orient_sin,
                (lap.abs() * 4.0).clamp(0.0, 1.0),
                (local_std * 4.0).clamp(0.0, 1.0),
                entropy.clamp(0.0, 1.0),
                (tophat * 6.0).clamp(0.0, 1.0),
                (blackhat * 6.0).clamp(0.0, 1.0),
                saturation.clamp(0.0, 1.0),
            ],
            dim=1,
        )
        assert feats.shape == (b, 10, h, w)
        return feats
