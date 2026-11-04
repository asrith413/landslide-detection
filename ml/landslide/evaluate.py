#!/usr/bin/env python3
"""Evaluate a checkpoint with segmentation, boundary, and efficiency metrics."""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .data import build_dataloader
from .metrics import SegmentationMetrics
from .model import build_model
from .research_data import build_research_dataloader, canonicalize_split
from .utils import count_parameters, model_size_mb, peak_memory_mb, save_json

_MANIFEST_KINDS = {"manifest", "event_manifest", "events"}
_HR_GLDD_KINDS = {"hrgldd", "hr_gldd"}


def _resolve_device(device: str | torch.device) -> torch.device:
    resolved = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if str(device) == "auto"
        else torch.device(device)
    )
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {resolved} was requested but torch.cuda.is_available() is False"
        )
    return resolved


def _kind(data_cfg: Mapping[str, Any]) -> str:
    raw = data_cfg.get("type", data_cfg.get("kind", data_cfg.get("dataset")))
    if raw is None:
        return "manifest" if "manifest" in data_cfg or "manifest_path" in data_cfg else "legacy"
    normalized = str(raw).strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized in _MANIFEST_KINDS:
        return "manifest"
    if normalized in _HR_GLDD_KINDS:
        return "hr_gldd"
    return "legacy"


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was added
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint {path} must contain a mapping")
    return payload


def load_checkpoint(path: Path, device: torch.device):
    """Load a training checkpoint without downloading pretrained weights."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    checkpoint = _torch_load(path)
    if not isinstance(checkpoint.get("config"), Mapping):
        raise KeyError(f"checkpoint {path} does not contain a mapping 'config'")
    if "model_state" not in checkpoint:
        raise KeyError(f"checkpoint {path} does not contain 'model_state'")
    cfg = copy.deepcopy(dict(checkpoint["config"]))
    model = build_model(cfg, pretrained_override=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, cfg, checkpoint


def _event_names(value: Any, batch_size: int) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value] * batch_size
    if isinstance(value, torch.Tensor):
        names = [str(item) for item in value.detach().cpu().tolist()]
    elif isinstance(value, np.ndarray):
        names = [str(item) for item in value.tolist()]
    elif isinstance(value, Sequence):
        names = [str(item) for item in value]
    else:
        names = [str(value)] * batch_size
    if len(names) != batch_size:
        raise ValueError(f"event label count {len(names)} does not match batch size {batch_size}")
    return names


@torch.no_grad()
def evaluate_loader(
    model,
    loader,
    device,
    threshold: float = 0.5,
    amp: bool = False,
    compute_boundary: bool = True,
) -> dict:
    """Evaluate aggregate and, when available, event-specific segmentation metrics."""
    dev = _resolve_device(device)
    model.to(dev).eval()
    aggregate = SegmentationMetrics(threshold=float(threshold))
    per_event: dict[str, SegmentationMetrics] = {}
    non_blocking = bool(getattr(loader, "pin_memory", False) and dev.type == "cuda")

    for batch in loader:
        image = batch["image"].to(dev, non_blocking=non_blocking)
        target = batch["mask"].to(dev, non_blocking=non_blocking)
        with torch.autocast(device_type=dev.type, enabled=amp and dev.type == "cuda"):
            output = model(image)
        logits = output["logits"] if isinstance(output, Mapping) else output
        if not isinstance(logits, torch.Tensor):
            raise TypeError("model must return logits as a tensor or under output['logits']")
        aggregate.update(logits, target, compute_boundary=compute_boundary)

        names = _event_names(batch.get("event"), image.shape[0])
        if names is not None:
            for index, name in enumerate(names):
                event_metrics = per_event.setdefault(
                    name,
                    SegmentationMetrics(threshold=float(threshold)),
                )
                event_metrics.update(
                    logits[index : index + 1],
                    target[index : index + 1],
                    compute_boundary=compute_boundary,
                )

    result: dict[str, Any] = aggregate.compute()
    if per_event:
        result["per_event"] = {
            name: per_event[name].compute() for name in sorted(per_event, key=str.casefold)
        }
    return result


class _TTAModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.model(image)["logits"]
        for flip in (2, 3):
            flipped = torch.flip(image, dims=[flip])
            logits = logits + torch.flip(self.model(flipped)["logits"], dims=[flip])
        return {"logits": logits / 3.0}


def _evaluation_loader(
    checkpoint_cfg: Mapping[str, Any],
    data_root: Path | str | None,
    split: str,
    limit: int | None,
    data_cfg: Mapping[str, Any] | None,
):
    checkpoint_data = checkpoint_cfg.get("data", {})
    if not isinstance(checkpoint_data, Mapping):
        checkpoint_data = {}
    selected = dict(checkpoint_data)
    if data_cfg is not None:
        selected.update(dict(data_cfg))
    kind = _kind(selected)
    normalized_split = (
        canonicalize_split(split)
        if kind != "legacy"
        else ("val" if str(split).strip().casefold() == "validation" else str(split))
    )

    if kind in {"manifest", "hr_gldd"}:
        research_cfg = dict(selected)
        research_cfg.pop("limit", None)
        loader = build_research_dataloader(
            research_cfg,
            normalized_split,
            batch_size=int(research_cfg.get("eval_batch_size", 1)),
            augment=False,
            num_workers=int(research_cfg.get("eval_num_workers", 0)),
            limit=limit,
            shuffle=False,
        )
        return loader, kind, normalized_split

    root = data_root if data_root is not None else selected.get("root", selected.get("data_root"))
    if root is None:
        raise KeyError("Landslide4Sense evaluation requires data_root")
    loader = build_dataloader(
        root,
        normalized_split,
        batch_size=1,
        augment=False,
        num_workers=0,
        limit=limit,
        shuffle=False,
    )
    return loader, kind, normalized_split


@torch.no_grad()
def run(
    checkpoint: Path,
    data_root: Path,
    split: str = "test",
    device: str = "auto",
    tta: bool = False,
    limit: int | None = None,
    out: Path | None = None,
    data_cfg: Mapping[str, Any] | None = None,
    threshold: float = 0.5,
    amp: bool = False,
    compute_boundary: bool = True,
) -> dict:
    """Evaluate a checkpoint while retaining the original positional API."""
    dev = _resolve_device(device)
    model, cfg, checkpoint_payload = load_checkpoint(checkpoint, dev)
    loader, data_kind, normalized_split = _evaluation_loader(cfg, data_root, split, limit, data_cfg)
    evaluation_model: nn.Module = _TTAModel(model) if tta else model
    result = evaluate_loader(
        evaluation_model,
        loader,
        dev,
        threshold=threshold,
        amp=amp,
        compute_boundary=compute_boundary,
    )

    non_blocking = bool(getattr(loader, "pin_memory", False) and dev.type == "cuda")
    latencies: list[float] = []
    sample: torch.Tensor | None = None
    for batch in loader:
        image = batch["image"].to(dev, non_blocking=non_blocking)
        if sample is None:
            sample = image
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        started = time.perf_counter()
        with torch.autocast(device_type=dev.type, enabled=amp and dev.type == "cuda"):
            evaluation_model(image)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        latencies.append((time.perf_counter() - started) * 1000.0 / image.shape[0])
    if sample is None:
        raise ValueError(f"evaluation split {normalized_split!r} is empty")

    for _ in range(3):
        with torch.autocast(device_type=dev.type, enabled=amp and dev.type == "cuda"):
            model(sample)
    repetitions = 20
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    started = time.perf_counter()
    for _ in range(repetitions):
        with torch.autocast(device_type=dev.type, enabled=amp and dev.type == "cuda"):
            model(sample)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    result.update(
        {
            "split": normalized_split,
            "data_type": data_kind,
            "checkpoint": str(checkpoint),
            "best_epoch": checkpoint_payload.get("best_epoch", checkpoint_payload.get("epoch")),
            "parameters": count_parameters(model),
            "model_size_mb": model_size_mb(model),
            "latency_ms_per_image": float(np.mean(latencies)),
            "latency_ms_mean_20": (time.perf_counter() - started) * 1000.0 / repetitions,
            "peak_rss_mb": peak_memory_mb(),
            "device": str(dev),
            "input_size": list(sample.shape),
            "tta": tta,
            "threshold": float(threshold),
            "config": cfg,
        }
    )
    print(json.dumps({key: value for key, value in result.items() if key != "config"}, indent=2))
    if out:
        save_json(result, out)
    return result


def evaluate_external_hr_gldd(
    checkpoint: str | Path,
    root: str | Path,
    split: str = "test",
    device: str | torch.device = "auto",
    threshold: float = 0.5,
    amp: bool = False,
    compute_boundary: bool = True,
    batch_size: int = 1,
    num_workers: int = 0,
    limit: int | None = None,
    out: str | Path | None = None,
) -> dict:
    """Evaluate a frozen checkpoint on official HR-GLDD arrays without fine-tuning."""
    dev = _resolve_device(device)
    model, cfg, checkpoint_payload = load_checkpoint(Path(checkpoint), dev)
    normalized_split = canonicalize_split(split)
    loader = build_research_dataloader(
        {"type": "hr_gldd", "root": str(root)},
        normalized_split,
        batch_size=batch_size,
        augment=False,
        num_workers=num_workers,
        limit=limit,
        shuffle=False,
    )
    result = evaluate_loader(
        model,
        loader,
        dev,
        threshold=threshold,
        amp=amp,
        compute_boundary=compute_boundary,
    )
    result.update(
        {
            "split": normalized_split,
            "data_type": "hr_gldd_external",
            "checkpoint": str(checkpoint),
            "best_epoch": checkpoint_payload.get("best_epoch", checkpoint_payload.get("epoch")),
            "device": str(dev),
            "threshold": float(threshold),
            "fine_tuned": False,
            "config": cfg,
        }
    )
    if out is not None:
        save_json(result, Path(out))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", "--data-root", dest="data_root", type=Path)
    parser.add_argument("--manifest", type=Path, help="override with an event CSV manifest")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    data_override: dict[str, Any] | None = None
    if args.manifest is not None:
        data_override = {
            "type": "manifest",
            "manifest": str(args.manifest),
            "data_root": str(args.data_root or args.manifest.parent),
        }
    elif args.data_root is not None:
        data_override = {
            "root": str(args.data_root),
            "data_root": str(args.data_root),
        }
    legacy_root = args.data_root or Path("data/processed/landslide4sense")
    run(
        args.checkpoint,
        legacy_root,
        args.split,
        args.device,
        args.tta,
        args.limit,
        args.out,
        data_cfg=data_override,
    )


if __name__ == "__main__":
    main()
