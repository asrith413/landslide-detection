"""Segmentation and mask-constrained object-level contrastive losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def dice_loss(logits: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (probs * target).sum(dims)
    union = probs.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * inter + eps) / (union + eps)).mean()


def soft_boundary(target: Tensor, k: int = 3) -> Tensor:
    """Approximate boundary band of a binary mask (dilation minus erosion)."""
    dil = F.max_pool2d(target, k, stride=1, padding=k // 2)
    ero = -F.max_pool2d(-target, k, stride=1, padding=k // 2)
    return (dil - ero).clamp(0.0, 1.0)


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: float = 2.0,
        boundary_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        loss = self.bce_weight * bce + self.dice_weight * dice_loss(logits, target)
        if self.boundary_weight > 0:
            band = soft_boundary(target, 3)
            w = 1.0 + 4.0 * band
            bce_b = F.binary_cross_entropy_with_logits(logits, target, weight=w, pos_weight=self.pos_weight)
            loss = loss + self.boundary_weight * bce_b
        return loss


class MaskedObjectContrastiveLoss(nn.Module):
    """Mask-constrained object-level contrastive objective.

    Foreground and background prototypes are computed by mask pooling.  Every pixel
    embedding is then pulled toward its own object prototype and pushed from the other
    prototype.  Images without both classes are skipped.  A prototype-separation term
    prevents collapse when the batch contains very few positive images.
    """

    def __init__(self, temperature: float = 0.1, separation_weight: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature
        self.separation_weight = separation_weight

    def forward(self, embedding: Tensor, mask: Tensor) -> Tensor:
        if embedding.shape[-2:] != mask.shape[-2:]:
            mask = F.interpolate(mask, size=embedding.shape[-2:], mode="nearest")
        b, c, h, w = embedding.shape
        emb = F.normalize(embedding, dim=1)
        m = (mask > 0.5).to(emb.dtype).squeeze(1)  # [B,H,W]
        fg_cnt = m.sum(dim=(1, 2))
        bg_cnt = (1.0 - m).sum(dim=(1, 2))
        valid = (fg_cnt > 8) & (bg_cnt > 8)
        if not valid.any():
            return embedding.sum() * 0.0

        fg_sum = (emb * m[:, None]).sum(dim=(2, 3))
        bg_sum = (emb * (1.0 - m[:, None])).sum(dim=(2, 3))
        fg_proto = fg_sum / fg_cnt.clamp_min(1.0)[:, None]
        bg_proto = bg_sum / bg_cnt.clamp_min(1.0)[:, None]

        # Prototype separation: cosine similarity should be low for distinct objects.
        proto_sim = (F.normalize(fg_proto, dim=1) * F.normalize(bg_proto, dim=1)).sum(dim=1)
        sep = proto_sim[valid].clamp(-1.0, 1.0).mean()

        pix = emb.permute(0, 2, 3, 1).reshape(b, h * w, c)
        sim_fg = (pix * fg_proto[:, None, :]).sum(-1) / self.temperature
        sim_bg = (pix * bg_proto[:, None, :]).sum(-1) / self.temperature
        logits = torch.stack([sim_fg, sim_bg], dim=-1)
        target = (m.reshape(b, h * w) > 0.5).long()
        ce = F.cross_entropy(
            logits.reshape(-1, 2),
            target.reshape(-1),
            reduction="none",
        ).reshape(b, h * w)
        pixel_loss = ce[valid].mean()
        return pixel_loss + self.separation_weight * sep


class TotalLoss(nn.Module):
    def __init__(
        self,
        seg: nn.Module | None = None,
        contrastive: nn.Module | None = None,
        contrastive_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.seg = seg or BCEDiceLoss()
        self.contrastive = contrastive
        self.contrastive_weight = contrastive_weight

    def forward(self, outputs: dict[str, Tensor], target: Tensor) -> dict[str, Tensor]:
        seg = self.seg(outputs["logits"], target)
        total = seg
        con = seg.new_zeros(())
        if self.contrastive is not None and outputs.get("embedding") is not None:
            con = self.contrastive(outputs["embedding"], target)
            total = total + self.contrastive_weight * con
        return {"loss": total, "seg": seg.detach(), "contrastive": con.detach()}
