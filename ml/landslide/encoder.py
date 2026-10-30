"""Multi-scale RGB encoders with an optional ImageNet-pretrained backbone."""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .blocks import ConvBNReLU


class TinyEncoder(nn.Module):
    """Small from-scratch encoder used for tests and fully offline CPU runs."""

    def __init__(self, in_ch: int = 3, widths: tuple[int, int, int, int] = (24, 48, 96, 192)) -> None:
        super().__init__()
        w0, w1, w2, w3 = widths
        self.stem = ConvBNReLU(in_ch, w0, 3, stride=2)
        self.stage1 = nn.Sequential(ConvBNReLU(w0, w1, 3, stride=2), ConvBNReLU(w1, w1, 3))
        self.stage2 = nn.Sequential(ConvBNReLU(w1, w2, 3, stride=2), ConvBNReLU(w2, w2, 3))
        self.stage3 = nn.Sequential(ConvBNReLU(w2, w3, 3, stride=2), ConvBNReLU(w3, w3, 3))
        self.channels = (w1, w2, w3, w3)
        self.out_strides = (4, 8, 16, 32)

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return [f1, f2, f3, f3]


class _TorchvisionEncoder(nn.Module):
    """Adapts torchvision RegNet/MobileNet feature extractors to four scales."""

    def __init__(self, name: str, pretrained: bool) -> None:
        super().__init__()
        self.name = name
        self.kind = "regnet" if name.startswith("regnet") else "mobilenet"
        self.body = self._build(name, pretrained)
        channels = self._infer_channels()
        self.channels = tuple(channels)
        self.out_strides = (4, 8, 16, 32)

    @staticmethod
    def _build(name: str, pretrained: bool) -> nn.Module:
        try:
            import torchvision.models as tvm
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "torchvision is required for pretrained encoders; install it or use "
                "--encoder tiny"
            ) from exc

        factory = getattr(tvm, name, None)
        if factory is None:
            raise ValueError(f"unknown torchvision encoder: {name}")

        weights = None
        if pretrained:
            weights_enum = getattr(tvm, name.replace("regnet", "RegNet").upper() + "_Weights", None)
            # torchvision exposes enums such as RegNet_Y_400MF_Weights; resolve by name.
            enum_name = {
                "regnet_y_400mf": "RegNet_Y_400MF_Weights",
                "regnet_y_800mf": "RegNet_Y_800MF_Weights",
                "regnet_y_1_6gf": "RegNet_Y_1_6GF_Weights",
                "regnet_y_3_2gf": "RegNet_Y_3_2GF_Weights",
                "regnet_x_400mf": "RegNet_X_400MF_Weights",
                "mobilenet_v3_small": "MobileNet_V3_Small_Weights",
                "mobilenet_v3_large": "MobileNet_V3_Large_Weights",
            }.get(name)
            if enum_name and hasattr(tvm, enum_name):
                weights = getattr(tvm, enum_name).IMAGENET1K_V1
        try:
            return factory(weights=weights)
        except Exception as exc:  # download/offline failure -> explicit fallback
            warnings.warn(f"could not load pretrained weights for {name}: {exc}; using random init")
            return factory(weights=None)

    @torch.no_grad()
    def _infer_channels(self) -> list[int]:
        was_training = self.training
        self.eval()
        x = torch.zeros(1, 3, 128, 128)
        feats = self._forward_features(x)
        chans = [f.shape[1] for f in feats]
        self.train(was_training)
        return chans

    def _forward_features(self, x: Tensor) -> list[Tensor]:
        if self.kind == "regnet":
            x = self.body.stem(x)
            feats = []
            for stage in self.body.trunk_output:
                x = stage(x)
                feats.append(x)
            return feats[-4:]
        feats: list[Tensor] = []
        x = x
        h = x.shape[-1]
        target_sizes = [h // 4, h // 8, h // 16, h // 32]
        for layer in self.body.features:
            x = layer(x)
            if x.shape[-1] in target_sizes and (not feats or x.shape[-1] < feats[-1].shape[-1]):
                feats.append(x)
        if len(feats) < 4:
            feats.append(x)
        while len(feats) < 4:
            feats.append(feats[-1])
        return feats[:4]

    def forward(self, x: Tensor) -> list[Tensor]:
        return self._forward_features(x)


def build_encoder(name: str = "regnet_y_400mf", pretrained: bool = False) -> nn.Module:
    name = name.lower()
    if name in {"tiny", "scratch"}:
        return TinyEncoder()
    return _TorchvisionEncoder(name, pretrained)


class EncoderProjector(nn.Module):
    """Project every scale to a common width for the decoder."""

    def __init__(self, in_channels: tuple[int, ...], width: int = 96) -> None:
        super().__init__()
        self.projs = nn.ModuleList([nn.Conv2d(c, width, 1) for c in in_channels])
        self.width = width

    def forward(self, feats: list[Tensor]) -> list[Tensor]:
        return [proj(f) for proj, f in zip(self.projs, feats)]


def resize_to(x: Tensor, ref: Tensor) -> Tensor:
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
