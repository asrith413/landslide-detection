"""SegFormer and LR-ASPP research model variants.

Optional model libraries are imported only while constructing the corresponding
model, so the core SA-LSNet remains usable in minimal environments.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .blocks import ContrastiveHead, GatedStructuralFusion
from .features import IMAGENET_MEAN, IMAGENET_STD, StructuralFeatureExtractor

SEGFORMER_B0_CHECKPOINT = "nvidia/mit-b0"


def _build_segformer(
    pretrained: bool,
    checkpoint: str,
    config: Any | None = None,
) -> nn.Module:
    try:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "transformers is required for SegFormer models; install transformers "
            "or select another architecture"
        ) from exc

    if isinstance(config, dict):
        config = SegformerConfig(**config)

    if config is None and not pretrained:
        # SegformerConfig defaults to the MiT-B0 widths and depths and performs no I/O.
        config = SegformerConfig(num_labels=1)
    elif config is not None:
        config.num_labels = 1
        config.id2label = {0: "landslide"}
        config.label2id = {"landslide": 0}

    if pretrained:
        kwargs: dict[str, Any] = {
            "num_labels": 1,
            "ignore_mismatched_sizes": True,
        }
        if config is not None:
            kwargs["config"] = config
        return SegformerForSemanticSegmentation.from_pretrained(checkpoint, **kwargs)
    return SegformerForSemanticSegmentation(config)


class SegFormerB0Baseline(nn.Module):
    """Binary SegFormer-B0 with an optional pretrained MiT backbone."""

    def __init__(
        self,
        pretrained: bool = False,
        checkpoint: str = SEGFORMER_B0_CHECKPOINT,
        config: Any | None = None,
    ) -> None:
        super().__init__()
        self.model = _build_segformer(pretrained, checkpoint, config)

    def forward(self, x: Tensor) -> dict[str, Tensor | None]:
        logits = self.model(pixel_values=x, return_dict=True).logits
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"logits": logits, "embedding": None}


class LRASPPMobileNetV3Baseline(nn.Module):
    """Binary LR-ASPP with an optionally ImageNet-pretrained MobileNetV3-Large."""

    def __init__(self, pretrained_backbone: bool = False) -> None:
        super().__init__()
        try:
            from torchvision.models import MobileNet_V3_Large_Weights
            from torchvision.models.segmentation import lraspp_mobilenet_v3_large
        except (ImportError, OSError, RuntimeError) as exc:  # pragma: no cover
            raise RuntimeError("torchvision segmentation models are required for LR-ASPP") from exc

        weights_backbone = MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
        self.model = lraspp_mobilenet_v3_large(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=1,
        )

    def forward(self, x: Tensor) -> dict[str, Tensor | None]:
        logits = self.model(x)["out"]
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"logits": logits, "embedding": None}


class DepthwiseSeparableStructuralAdapter(nn.Module):
    """Compact projection from the handcrafted prior to one encoder stage."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_norm = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pointwise_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.depthwise_norm(self.depthwise(x)))
        return self.activation(self.pointwise_norm(self.pointwise(x)))


class StructuralSegFormerB0(nn.Module):
    """SegFormer-B0 with gated structural priors at the 1/4 and 1/8 stages."""

    def __init__(
        self,
        pretrained: bool = False,
        checkpoint: str = SEGFORMER_B0_CHECKPOINT,
        config: Any | None = None,
        use_contrastive: bool = True,
        embed_dim: int = 64,
        rgb_mean: tuple[float, float, float] = IMAGENET_MEAN,
        rgb_std: tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        super().__init__()
        base = _build_segformer(pretrained, checkpoint, config)
        self.segformer = base.segformer
        self.decode_head = base.decode_head
        hidden_sizes = tuple(base.config.hidden_sizes)
        if len(hidden_sizes) < 2:
            raise ValueError("SegFormer config must expose at least two encoder stages")

        self.structural = StructuralFeatureExtractor()
        self.adapters = nn.ModuleList(
            [DepthwiseSeparableStructuralAdapter(10, channels) for channels in hidden_sizes[:2]]
        )
        self.fusions = nn.ModuleList(
            [
                GatedStructuralFusion(channels, channels, channels, gate_bias=-2.0)
                for channels in hidden_sizes[:2]
            ]
        )
        self.embedding_head = (
            ContrastiveHead(hidden_sizes[0], embed_dim) if use_contrastive else None
        )
        self.register_buffer("rgb_mean", torch.tensor(rgb_mean).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor(rgb_std).view(1, 3, 1, 1))

    def forward(self, x: Tensor) -> dict[str, Tensor | None]:
        encoder_output = self.segformer(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = list(encoder_output.hidden_states)
        if len(hidden_states) < 2:
            raise RuntimeError("SegFormer encoder did not return its four stage features")

        rgb01 = (x * self.rgb_std + self.rgb_mean).clamp(0.0, 1.0)
        structural = self.structural(rgb01)
        for index, (adapter, fusion) in enumerate(zip(self.adapters, self.fusions)):
            resized = F.interpolate(
                structural,
                size=hidden_states[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            hidden_states[index] = fusion(hidden_states[index], adapter(resized))

        embedding = (
            self.embedding_head(hidden_states[0]) if self.embedding_head is not None else None
        )
        logits = self.decode_head(tuple(hidden_states))
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"logits": logits, "embedding": embedding}


# Short aliases are convenient in experiment code while the descriptive class names
# remain the stable public API.
SegFormerB0 = SegFormerB0Baseline
LRASPPMobileNetV3 = LRASPPMobileNetV3Baseline
CSGSegFormerB0 = StructuralSegFormerB0
