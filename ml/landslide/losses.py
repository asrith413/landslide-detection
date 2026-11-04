"""Segmentation and mask-constrained object-level contrastive losses."""

from __future__ import annotations

from collections import deque

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
            bce_b = F.binary_cross_entropy_with_logits(
                logits, target, weight=w, pos_weight=self.pos_weight
            )
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
        fg_proto = F.normalize(fg_sum / fg_cnt.clamp_min(1.0)[:, None], dim=1)
        bg_proto = F.normalize(bg_sum / bg_cnt.clamp_min(1.0)[:, None], dim=1)

        # Prototype separation: cosine similarity should be low for distinct objects.
        proto_sim = (fg_proto * bg_proto).sum(dim=1)
        sep = proto_sim[valid].clamp(-1.0, 1.0).mean()

        pix = emb.permute(0, 2, 3, 1).reshape(b, h * w, c)
        sim_fg = (pix * fg_proto[:, None, :]).sum(-1) / self.temperature
        sim_bg = (pix * bg_proto[:, None, :]).sum(-1) / self.temperature
        # Binary class labels are background=0 and foreground=1.
        logits = torch.stack([sim_bg, sim_fg], dim=-1)
        target = (m.reshape(b, h * w) > 0.5).long()
        ce = F.cross_entropy(
            logits.reshape(-1, 2),
            target.reshape(-1),
            reduction="none",
        ).reshape(b, h * w)
        pixel_loss = ce[valid].mean()
        return pixel_loss + self.separation_weight * sep


def _connected_components(mask: Tensor, min_pixels: int) -> list[Tensor]:
    """Return flat indices for 8-connected components in a detached 2-D mask."""
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D component mask, got {tuple(mask.shape)}")
    height, width = mask.shape
    grid = mask.detach().to(device="cpu", dtype=torch.bool).tolist()
    visited = [[False] * width for _ in range(height)]
    components: list[Tensor] = []

    for start_y in range(height):
        for start_x in range(width):
            if not grid[start_y][start_x] or visited[start_y][start_x]:
                continue
            queue = deque([(start_y, start_x)])
            visited[start_y][start_x] = True
            indices: list[int] = []
            while queue:
                y, x = queue.popleft()
                indices.append(y * width + x)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if (
                            0 <= ny < height
                            and 0 <= nx < width
                            and grid[ny][nx]
                            and not visited[ny][nx]
                        ):
                            visited[ny][nx] = True
                            queue.append((ny, nx))
            if len(indices) >= min_pixels:
                components.append(torch.tensor(indices, device=mask.device, dtype=torch.long))
    return components


class HardNegativeRegionContrastiveLoss(nn.Module):
    """Contrast positive objects against the highest-scoring false-positive regions.

    Connected components are found after masks and logits are resized to the embedding
    resolution. Segmentation scores are detached and used only for hard-region
    selection, keeping this objective confined to the training-time embedding branch.
    """

    accepts_segmentation_logits = True

    def __init__(
        self,
        temperature: float = 0.1,
        max_negative_regions: int = 3,
        false_positive_threshold: float = 0.5,
        min_region_pixels: int = 1,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if max_negative_regions < 1:
            raise ValueError("max_negative_regions must be at least one")
        if min_region_pixels < 1:
            raise ValueError("min_region_pixels must be at least one")
        self.temperature = temperature
        self.max_negative_regions = max_negative_regions
        self.false_positive_threshold = false_positive_threshold
        self.min_region_pixels = min_region_pixels

    def forward(
        self,
        embedding: Tensor,
        mask: Tensor,
        segmentation_logits: Tensor | None = None,
    ) -> Tensor:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError(f"expected mask [B,1,H,W], got {tuple(mask.shape)}")
        if embedding.ndim != 4 or embedding.shape[0] != mask.shape[0]:
            raise ValueError("embedding and mask must be batched tensors with matching batches")

        size = embedding.shape[-2:]
        resized_mask = F.interpolate(mask.float(), size=size, mode="nearest") > 0.5
        score_maps: Tensor | None = None
        if segmentation_logits is not None:
            if segmentation_logits.ndim == 3:
                segmentation_logits = segmentation_logits.unsqueeze(1)
            if (
                segmentation_logits.ndim != 4
                or segmentation_logits.shape[0] != embedding.shape[0]
                or segmentation_logits.shape[1] != 1
            ):
                raise ValueError(
                    "segmentation_logits must have shape [B,1,H,W] and match the batch"
                )
            score_maps = torch.sigmoid(
                F.interpolate(
                    segmentation_logits.detach().float(),
                    size=size,
                    mode="bilinear",
                    align_corners=False,
                )
            )

        normalized = F.normalize(embedding, dim=1)
        losses: list[Tensor] = []
        for batch_index in range(embedding.shape[0]):
            positive_mask = resized_mask[batch_index, 0]
            positive_components = _connected_components(positive_mask, self.min_region_pixels)
            if not positive_components:
                continue

            pixels = normalized[batch_index].flatten(1).transpose(0, 1)
            positive_regions = [
                (indices, F.normalize(pixels[indices].mean(dim=0), dim=0))
                for indices in positive_components
            ]

            if score_maps is not None:
                score_map = score_maps[batch_index, 0]
                negative_mask = (~positive_mask) & (score_map >= self.false_positive_threshold)
            else:
                score_map = None
                negative_mask = ~positive_mask

            negative_components = _connected_components(negative_mask, self.min_region_pixels)
            if not negative_components:
                continue

            if score_map is not None:
                flat_scores = score_map.flatten()
                ranked = sorted(
                    negative_components,
                    key=lambda indices: float(flat_scores[indices].mean()),
                    reverse=True,
                )
            else:
                detached_positive = torch.stack(
                    [prototype.detach() for _, prototype in positive_regions]
                )

                def similarity(
                    indices: Tensor,
                    pixel_features: Tensor = pixels,
                    positive_prototypes: Tensor = detached_positive,
                ) -> float:
                    prototype = F.normalize(pixel_features[indices].detach().mean(dim=0), dim=0)
                    return float((positive_prototypes @ prototype).max())

                ranked = sorted(negative_components, key=similarity, reverse=True)

            selected = ranked[: self.max_negative_regions]
            negative_prototypes = torch.stack(
                [F.normalize(pixels[indices].mean(dim=0), dim=0) for indices in selected]
            )
            for indices, positive_prototype in positive_regions:
                queries = pixels[indices]
                positive_score = (queries * positive_prototype).sum(dim=1, keepdim=True)
                negative_scores = queries @ negative_prototypes.transpose(0, 1)
                scores = torch.cat([positive_score, negative_scores], dim=1)
                scores = scores / self.temperature
                losses.append((torch.logsumexp(scores, dim=1) - scores[:, 0]).mean())

        if not losses:
            return embedding.sum() * 0.0
        return torch.stack(losses).mean()


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

    def forward(self, outputs: dict[str, Tensor | None], target: Tensor) -> dict[str, Tensor]:
        logits = outputs["logits"]
        if logits is None:
            raise ValueError("model outputs must include segmentation logits")
        seg = self.seg(logits, target)
        total = seg
        con = seg.new_zeros(())
        if self.contrastive is not None and outputs.get("embedding") is not None:
            embedding = outputs["embedding"]
            assert embedding is not None
            if getattr(self.contrastive, "accepts_segmentation_logits", False):
                con = self.contrastive(
                    embedding,
                    target,
                    segmentation_logits=logits,
                )
            else:
                con = self.contrastive(embedding, target)
            total = total + self.contrastive_weight * con
        return {"loss": total, "seg": seg.detach(), "contrastive": con.detach()}
