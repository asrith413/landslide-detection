"""SA-LSNet: structure-aware RGB-only landslide segmentation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .baselines import (
    LRASPPMobileNetV3Baseline,
    SegFormerB0Baseline,
    StructuralSegFormerB0,
)
from .blocks import (
    ContrastiveHead,
    ConvBNReLU,
    CrossAttentionFusion,
    GatedStructuralFusion,
    UpBlock,
)
from .encoder import EncoderProjector, build_encoder
from .features import IMAGENET_MEAN, IMAGENET_STD, StructuralFeatureExtractor


class StructureAwareLandslideNet(nn.Module):
    """Configurable model covering all ablation variants.

    Parameters
    ----------
    encoder:
        ``tiny``, ``regnet_y_400mf``, ``regnet_y_800mf``, ``mobilenet_v3_small`` ...
    use_structural:
        Enable the 10-channel structural prior branch.
    fusion:
        ``none`` (plain concat/residual), ``gate`` (default), or ``cross``.
    use_contrastive:
        Add the mask-constrained object-level embedding head.
    """

    def __init__(
        self,
        encoder: str = "regnet_y_400mf",
        pretrained: bool = False,
        use_structural: bool = True,
        fusion: str = "gate",
        use_contrastive: bool = True,
        width: int = 96,
        decoder_channels: int = 64,
        embed_dim: int = 64,
        dropout: float = 0.1,
        rgb_mean: tuple[float, float, float] = IMAGENET_MEAN,
        rgb_std: tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        super().__init__()
        self.use_structural = use_structural
        self.fusion_kind = fusion
        self.use_contrastive = use_contrastive
        self.register_buffer("rgb_mean", torch.tensor(rgb_mean).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor(rgb_std).view(1, 3, 1, 1))

        self.encoder = build_encoder(encoder, pretrained=pretrained)
        enc_channels = tuple(self.encoder.channels)
        self.enc_channels = enc_channels

        self.structural = StructuralFeatureExtractor() if use_structural else None
        if use_structural:
            if fusion not in {"none", "gate", "cross"}:
                raise ValueError(f"unknown fusion: {fusion}")
            self.struct_projs = nn.ModuleList([ConvBNReLU(10, c, 3) for c in enc_channels])
            if fusion in {"gate", "cross"}:
                self.fusions = nn.ModuleList([GatedStructuralFusion(c, c, c) for c in enc_channels])
            else:
                self.fusions = nn.ModuleList([nn.Conv2d(c * 2, c, 1) for c in enc_channels])
            self.deep_cross = CrossAttentionFusion(enc_channels[-1]) if fusion == "cross" else None
        else:
            self.struct_projs = None
            self.fusions = None
            self.deep_cross = None

        self.projector = EncoderProjector(enc_channels, width=width)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.dec3 = UpBlock(width, width, width)
        self.dec2 = UpBlock(width, width, width)
        self.dec1 = UpBlock(width, width, width)
        self.dec0 = nn.Sequential(
            ConvBNReLU(width, decoder_channels, 3),
            ConvBNReLU(decoder_channels, decoder_channels, 3),
        )
        self.head = nn.Conv2d(decoder_channels, 1, 1)
        self.contrastive_head = ContrastiveHead(width, embed_dim) if use_contrastive else None

    def forward(self, x: Tensor) -> dict[str, Tensor | None]:
        feats = self.encoder(x)
        if len(feats) != 4:
            raise RuntimeError(f"encoder must return 4 scales, got {len(feats)}")

        if self.use_structural:
            assert self.structural is not None
            rgb01 = (x * self.rgb_std + self.rgb_mean).clamp(0.0, 1.0)
            struct = self.structural(rgb01)
            fused = []
            for i, (f, proj, fus) in enumerate(zip(feats, self.struct_projs, self.fusions)):
                s = F.interpolate(struct, size=f.shape[-2:], mode="bilinear", align_corners=False)
                s = proj(s)
                if self.fusion_kind in {"gate", "cross"}:
                    y = fus(f, s)
                    if self.fusion_kind == "cross" and i == len(feats) - 1:
                        assert self.deep_cross is not None
                        y = self.deep_cross(y, s)
                    fused.append(y)
                else:
                    fused.append(fus(torch.cat([f, s], dim=1)))
            feats = fused

        p = self.projector(feats)
        y = self.dec3(p[3], p[2])
        y = self.dec2(y, p[1])
        y = self.dec1(y, p[0])
        y = self.dropout(y)
        embedding = self.contrastive_head(y) if self.contrastive_head is not None else None
        y = self.dec0(y)
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.head(y)
        return {"logits": logits, "embedding": embedding}


def build_model(
    config: dict[str, Any],
    pretrained_override: bool | None = None,
) -> nn.Module:
    """Build a model from a JSON config's ``model`` section."""
    m = config.get("model", config)
    architecture = str(m.get("architecture", "sa_lsnet")).lower()
    pretrained = (
        bool(m.get("pretrained", False))
        if pretrained_override is None
        else bool(pretrained_override)
    )

    if architecture == "sa_lsnet":
        return StructureAwareLandslideNet(
            encoder=m.get("encoder", "regnet_y_400mf"),
            pretrained=pretrained,
            use_structural=bool(m.get("use_structural", True)),
            fusion=m.get("fusion", "gate"),
            use_contrastive=bool(m.get("use_contrastive", True)),
            width=int(m.get("width", 96)),
            decoder_channels=int(m.get("decoder_channels", 64)),
            embed_dim=int(m.get("embed_dim", 64)),
            dropout=float(m.get("dropout", 0.1)),
        )

    checkpoint = str(m.get("checkpoint", m.get("pretrained_checkpoint", "nvidia/mit-b0")))
    segformer_config = m.get("segformer_config")
    if architecture == "segformer_b0":
        return SegFormerB0Baseline(
            pretrained=pretrained,
            checkpoint=checkpoint,
            config=segformer_config,
        )
    if architecture == "lraspp_mobilenet_v3":
        pretrained_backbone = pretrained
        if pretrained_override is None:
            pretrained_backbone = bool(m.get("pretrained_backbone", pretrained))
        return LRASPPMobileNetV3Baseline(pretrained_backbone=pretrained_backbone)
    if architecture == "csg_segformer_b0":
        return StructuralSegFormerB0(
            pretrained=pretrained,
            checkpoint=checkpoint,
            config=segformer_config,
            use_contrastive=bool(m.get("use_contrastive", True)),
            embed_dim=int(m.get("embed_dim", 64)),
        )
    raise ValueError(f"unknown model architecture: {architecture}")


class DeploymentModel(nn.Module):
    """ONNX-friendly wrapper returning a single probability map."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.model(x)["logits"])
