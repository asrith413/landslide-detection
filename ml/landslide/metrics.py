"""Segmentation metrics, including Boundary IoU, accumulated over a dataset."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


def _binary(pred: np.ndarray, thr: float = 0.5) -> np.ndarray:
    return (pred >= thr).astype(np.uint8)


def _boundary(mask: np.ndarray, width: int) -> np.ndarray:
    """Boundary band of a binary mask using max/min pooling (numpy)."""
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    k = 2 * width + 1
    dil = F.max_pool2d(t, k, stride=1, padding=width)
    ero = -F.max_pool2d(-t, k, stride=1, padding=width)
    band = (dil - ero) > 0.5
    return band[0, 0].numpy().astype(np.uint8)


def boundary_iou(pred: np.ndarray, target: np.ndarray, dilation_ratio: float = 0.02) -> float:
    pred = _binary(pred)
    target = _binary(target)
    h, w = target.shape[-2:]
    width = max(1, int(round(dilation_ratio * float(np.hypot(h, w)))))
    bp = _boundary(pred, width)
    bt = _boundary(target, width)
    inter = np.logical_and(bp, bt).sum()
    union = np.logical_or(bp, bt).sum()
    if union == 0:
        return 1.0 if not pred.any() and not target.any() else 0.0
    return float(inter / union)


@dataclass
class SegmentationMetrics:
    threshold: float = 0.5
    dilation_ratio: float = 0.02
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    boundary_sum: float = 0.0
    boundary_count: int = 0
    per_image_iou: list[float] = field(default_factory=list)

    def update(self, logits: torch.Tensor, target: torch.Tensor, compute_boundary: bool = True) -> None:
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        gt = (target.detach().cpu().numpy() > 0.5).astype(np.uint8)
        pred = (probs >= self.threshold).astype(np.uint8)
        self.tp += int(np.logical_and(pred, gt).sum())
        self.fp += int(np.logical_and(pred, ~gt.astype(bool)).sum())
        self.fn += int(np.logical_and(~pred.astype(bool), gt).sum())
        self.tn += int(np.logical_and(~pred.astype(bool), ~gt.astype(bool)).sum())
        for p, g in zip(pred, gt):
            inter = np.logical_and(p, g).sum()
            union = np.logical_or(p, g).sum()
            self.per_image_iou.append(float(inter / union) if union else 1.0)
            if compute_boundary:
                self.boundary_sum += boundary_iou(p, g, self.dilation_ratio)
                self.boundary_count += 1

    def compute(self) -> dict[str, float]:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        eps = 1e-9
        iou = tp / (tp + fp + fn + eps)
        dice = 2 * tp / (2 * tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        return {
            "iou": float(iou),
            "dice": float(dice),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "boundary_iou": float(self.boundary_sum / max(self.boundary_count, 1)),
            "miou": float(np.mean(self.per_image_iou)) if self.per_image_iou else 0.0,
            "pixels": int(tp + fp + fn + tn),
        }
