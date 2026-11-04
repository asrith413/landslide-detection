#!/usr/bin/env python3
"""Train SA-LSNet or an ablation variant with a reproducible JSON config."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import random
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .data import build_dataloader
from .deployment import collect_runtime_report
from .losses import (
    BCEDiceLoss,
    HardNegativeRegionContrastiveLoss,
    MaskedObjectContrastiveLoss,
    TotalLoss,
)
from .metrics import SegmentationMetrics
from .model import build_model
from .research_data import build_research_dataloader
from .utils import count_parameters, load_config, merge_dicts, save_json, seed_everything

CHECKPOINT_SCHEMA_VERSION = 1

DEFAULTS: dict[str, Any] = {
    "experiment": "sa_lsnet",
    "seed": 42,
    "data": {
        "root": "data/processed/landslide4sense",
        "batch_size": 16,
        "num_workers": 0,
        "limit": 0,
    },
    "model": {
        "encoder": "regnet_y_400mf",
        "pretrained": False,
        "use_structural": True,
        "fusion": "gate",
        "use_contrastive": True,
        "width": 96,
        "decoder_channels": 64,
        "embed_dim": 64,
        "dropout": 0.1,
    },
    "train": {
        "epochs": 30,
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "warmup_epochs": 2,
        "amp": True,
        "grad_clip": 1.0,
        "gradient_accumulation": 1,
        "empty_cache_each_epoch": False,
        "patience": 8,
        "contrastive_loss": "legacy",
        "contrastive_weight": 0.1,
        "contrastive_warmup_epochs": 0,
        "pos_weight": 2.0,
        "boundary_weight": 0.0,
        "bce_weight": 0.5,
        "dice_weight": 0.5,
    },
    "output": "checkpoints/sa_lsnet",
}

_MANIFEST_KINDS = {"manifest", "event_manifest", "events"}
_HR_GLDD_KINDS = {"hrgldd", "hr_gldd"}
_LEGACY_KINDS = {"landslide4sense", "landslide_4_sense", "legacy", "npz"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _resolved_config(cfg: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dictionary")
    merged = merge_dicts(copy.deepcopy(DEFAULTS), copy.deepcopy(cfg))
    source_data = cfg.get("data", {})
    if (
        isinstance(source_data, Mapping)
        and _data_kind(source_data) != "legacy"
        and "root" not in source_data
        and "data_root" not in source_data
    ):
        # The legacy default root must not override manifest-relative paths or
        # masquerade as an HR-GLDD root.
        merged["data"].pop("root", None)
    normalized = _jsonable(merged)
    if not isinstance(normalized, dict):  # pragma: no cover - guarded above
        raise TypeError("cfg must resolve to a dictionary")
    return normalized


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


def _normalize_kind(value: Any) -> str:
    return str(value).strip().casefold().replace("-", "_").replace(" ", "_")


def _data_kind(data_cfg: Mapping[str, Any]) -> str:
    raw = data_cfg.get("type", data_cfg.get("kind", data_cfg.get("dataset")))
    if raw is None:
        return "manifest" if "manifest" in data_cfg or "manifest_path" in data_cfg else "legacy"
    kind = _normalize_kind(raw)
    if kind in _MANIFEST_KINDS:
        return "manifest"
    if kind in _HR_GLDD_KINDS:
        return "hr_gldd"
    if kind in _LEGACY_KINDS:
        return "legacy"
    raise ValueError(f"unknown data type {raw!r}")


def _normalize_split(split: Any) -> str:
    value = str(split).strip().casefold()
    return "val" if value == "validation" else value


def _limit(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("data limits must be non-negative integers")
    return value or None


def cosine_with_warmup(optimizer: torch.optim.Optimizer, warmup: int, total: int) -> LambdaLR:
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, fn)


def _build_loader(
    data_cfg: Mapping[str, Any],
    kind: str,
    split: str,
    *,
    batch_size: int,
    augment: bool,
    num_workers: int,
    limit: int | None,
    seed: int,
    pin_memory: bool,
    shuffle: bool | None,
):
    split = _normalize_split(split)
    if kind == "legacy":
        root = data_cfg.get("root", data_cfg.get("data_root"))
        if root is None:
            raise KeyError("Landslide4Sense data config requires root")
        loader = build_dataloader(
            root,
            split,
            batch_size=batch_size,
            augment=augment,
            num_workers=num_workers,
            limit=limit,
            seed=seed,
            shuffle=shuffle,
        )
        # The legacy factory predates configurable pinning. DataLoader reads this
        # attribute when each iterator is created, so setting it here is sufficient.
        loader.pin_memory = pin_memory
        return loader

    research_cfg = dict(data_cfg)
    research_cfg.pop("limit", None)
    research_cfg["pin_memory"] = pin_memory
    return build_research_dataloader(
        research_cfg,
        split,
        batch_size=batch_size,
        augment=augment,
        num_workers=num_workers,
        limit=limit,
        seed=seed,
        shuffle=shuffle,
    )


def _build_training_loaders(
    cfg: Mapping[str, Any],
    device: torch.device,
    val_limit: int | None,
):
    data_cfg = cfg["data"]
    if not isinstance(data_cfg, Mapping):
        raise TypeError("cfg.data must be a mapping")
    kind = _data_kind(data_cfg)
    allow_external = bool(
        data_cfg.get("allow_external_training", cfg.get("allow_external_training", False))
    )
    if kind == "hr_gldd" and not allow_external:
        raise ValueError(
            "HR-GLDD is an external evaluation dataset and cannot be used for primary "
            "scientific training unless data.allow_external_training=true"
        )

    batch_size = int(data_cfg.get("batch_size", 16))
    num_workers = int(data_cfg.get("num_workers", 0))
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    seed = int(cfg["seed"])
    pin_memory = bool(data_cfg.get("pin_memory", device.type == "cuda"))
    train_limit = _limit(data_cfg.get("train_limit", data_cfg.get("limit")))
    configured_val_limit = data_cfg.get("val_limit", data_cfg.get("validation_limit"))
    effective_val_limit = _limit(val_limit if val_limit is not None else configured_val_limit)
    train_split = _normalize_split(data_cfg.get("train_split", "train"))
    val_split = _normalize_split(data_cfg.get("val_split", data_cfg.get("validation_split", "val")))

    train_loader = _build_loader(
        data_cfg,
        kind,
        train_split,
        batch_size=batch_size,
        augment=bool(data_cfg.get("train_augment", True)),
        num_workers=num_workers,
        limit=train_limit,
        seed=seed,
        pin_memory=pin_memory,
        shuffle=data_cfg.get("train_shuffle", data_cfg.get("shuffle")),
    )
    val_loader = _build_loader(
        data_cfg,
        kind,
        val_split,
        batch_size=int(data_cfg.get("val_batch_size", batch_size)),
        augment=False,
        num_workers=num_workers,
        limit=effective_val_limit,
        seed=seed,
        pin_memory=pin_memory,
        shuffle=bool(data_cfg.get("val_shuffle", False)),
    )
    if len(train_loader.dataset) == 0:
        raise ValueError("empty training split; run prepare_landslide4sense.py or fix data config")
    if len(val_loader.dataset) == 0:
        raise ValueError("empty validation split")
    return train_loader, val_loader, kind, pin_memory


def _manifest_sha256(loader: Any) -> str | None:
    report = getattr(loader.dataset, "validation_report", None)
    if isinstance(report, Mapping):
        value = report.get("manifest_sha256")
        if value:
            return str(value)
    return None


def _contrastive_loss(train_cfg: Mapping[str, Any], model_cfg: Mapping[str, Any]):
    if not bool(model_cfg.get("use_contrastive", True)):
        return None

    specification = train_cfg.get("contrastive_loss", "legacy")
    params: dict[str, Any] = {}
    if isinstance(specification, Mapping):
        params.update(specification)
        name = params.pop("type", params.pop("kind", params.pop("name", "legacy")))
    else:
        name = specification
    extra = train_cfg.get("contrastive_loss_params", {})
    if isinstance(extra, Mapping):
        params.update(extra)
    normalized = _normalize_kind(name)
    if normalized in {"none", "off", "disabled"}:
        return None
    if normalized in {"legacy", "masked_object", "masked_object_contrastive"}:
        return MaskedObjectContrastiveLoss()
    if normalized != "hard_negative_region":
        raise ValueError(f"unknown contrastive loss {name!r}")

    def option(key: str, default: Any) -> Any:
        return params.get(key, train_cfg.get(key, train_cfg.get(f"contrastive_{key}", default)))

    return HardNegativeRegionContrastiveLoss(
        temperature=float(option("temperature", 0.1)),
        max_negative_regions=int(option("max_negative_regions", 3)),
        false_positive_threshold=float(option("false_positive_threshold", 0.5)),
        min_region_pixels=int(option("min_region_pixels", 1)),
    )


def _build_criterion(cfg: Mapping[str, Any], device: torch.device) -> TotalLoss:
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    seg = BCEDiceLoss(
        bce_weight=float(train_cfg.get("bce_weight", 0.5)),
        dice_weight=float(train_cfg.get("dice_weight", 0.5)),
        pos_weight=float(train_cfg.get("pos_weight", 2.0)),
        boundary_weight=float(train_cfg.get("boundary_weight", 0.0)),
    )
    criterion = TotalLoss(
        seg=seg,
        contrastive=_contrastive_loss(train_cfg, model_cfg),
        contrastive_weight=float(train_cfg.get("contrastive_weight", 0.1)),
    )
    return criterion.to(device)


def train_epoch(
    model: nn.Module,
    loader,
    criterion: TotalLoss,
    optimizer,
    scheduler,
    device: torch.device,
    scaler,
    grad_clip: float,
    amp: bool,
    gradient_accumulation: int = 1,
    non_blocking: bool = True,
) -> dict[str, float]:
    if gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be at least one")
    model.train()
    total_loss = 0.0
    total_seg = 0.0
    total_con = 0.0
    n = 0
    total_batches = len(loader)
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader):
        x = batch["image"].to(device, non_blocking=non_blocking)
        y = batch["mask"].to(device, non_blocking=non_blocking)
        group_start = (batch_index // gradient_accumulation) * gradient_accumulation
        group_size = min(gradient_accumulation, total_batches - group_start)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            out = model(x)
            losses = criterion(out, y)
        scaler.scale(losses["loss"] / group_size).backward()

        should_step = (
            batch_index + 1
        ) % gradient_accumulation == 0 or batch_index + 1 == total_batches
        if should_step:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        bs = x.shape[0]
        total_loss += float(losses["loss"].detach()) * bs
        total_seg += float(losses["seg"].detach()) * bs
        total_con += float(losses["contrastive"].detach()) * bs
        n += bs
    return {
        "loss": total_loss / max(n, 1),
        "seg_loss": total_seg / max(n, 1),
        "contrastive_loss": total_con / max(n, 1),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: TotalLoss,
    device: torch.device,
    amp: bool,
    compute_boundary: bool = True,
    non_blocking: bool = True,
) -> tuple[dict[str, float | int], dict[str, float]]:
    model.eval()
    metrics = SegmentationMetrics()
    total_loss = 0.0
    total_seg = 0.0
    total_con = 0.0
    n = 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=non_blocking)
        y = batch["mask"].to(device, non_blocking=non_blocking)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            out = model(x)
            losses = criterion(out, y)
        metrics.update(out["logits"], y, compute_boundary=compute_boundary)
        bs = x.shape[0]
        total_loss += float(losses["loss"].detach()) * bs
        total_seg += float(losses["seg"].detach()) * bs
        total_con += float(losses["contrastive"].detach()) * bs
        n += bs
    result = metrics.compute()
    result["loss"] = total_loss / max(n, 1)
    return result, {
        "seg": total_seg / max(n, 1),
        "contrastive": total_con / max(n, 1),
    }


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: Any) -> None:
    if not isinstance(state, Mapping):
        raise TypeError("incompatible checkpoint: rng_state must be a mapping")
    missing = [name for name in ("python", "numpy", "torch", "cuda") if name not in state]
    if missing:
        raise ValueError(f"incompatible checkpoint: missing RNG states {missing}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def _loader_state(loader: Any) -> dict[str, Any]:
    state: dict[str, Any] = {}
    generator = getattr(loader, "generator", None)
    if isinstance(generator, torch.Generator):
        state["generator"] = generator.get_state()
    sampler_generator = getattr(getattr(loader, "sampler", None), "generator", None)
    if isinstance(sampler_generator, torch.Generator) and sampler_generator is not generator:
        state["sampler_generator"] = sampler_generator.get_state()
    rng = getattr(loader.dataset, "rng", None)
    if isinstance(rng, np.random.Generator):
        state["dataset_numpy_generator"] = copy.deepcopy(rng.bit_generator.state)
    return state


def _restore_loader_state(loader: Any, state: Any) -> None:
    if not isinstance(state, Mapping):
        return
    generator = getattr(loader, "generator", None)
    if isinstance(generator, torch.Generator) and "generator" in state:
        generator.set_state(state["generator"].cpu())
    sampler_generator = getattr(getattr(loader, "sampler", None), "generator", None)
    if isinstance(sampler_generator, torch.Generator) and "sampler_generator" in state:
        sampler_generator.set_state(state["sampler_generator"].cpu())
    rng = getattr(loader.dataset, "rng", None)
    if isinstance(rng, np.random.Generator) and "dataset_numpy_generator" in state:
        rng.bit_generator.state = copy.deepcopy(state["dataset_numpy_generator"])


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was added
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"incompatible checkpoint: {path} does not contain a mapping")
    return checkpoint


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    cfg: Mapping[str, Any],
    manifest_sha256: str | None,
) -> None:
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "incompatible checkpoint: expected schema_version "
            f"{CHECKPOINT_SCHEMA_VERSION}, got {checkpoint.get('schema_version')!r}"
        )
    required = {
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "epoch",
        "best_dice",
        "best_epoch",
        "patience_counter",
        "history",
        "config",
        "manifest_sha256",
        "rng_state",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"incompatible checkpoint: missing fields {missing}")
    checkpoint_cfg = checkpoint["config"]
    if not isinstance(checkpoint_cfg, Mapping):
        raise TypeError("incompatible checkpoint: config must be a mapping")
    if checkpoint_cfg.get("experiment") != cfg.get("experiment"):
        raise ValueError(
            "incompatible checkpoint: experiment differs "
            f"({checkpoint_cfg.get('experiment')!r} != {cfg.get('experiment')!r})"
        )
    if _jsonable(checkpoint_cfg.get("model")) != _jsonable(cfg.get("model")):
        raise ValueError("incompatible checkpoint: model config differs")
    if checkpoint.get("manifest_sha256") != manifest_sha256:
        raise ValueError(
            "incompatible checkpoint: data manifest SHA-256 differs "
            f"({checkpoint.get('manifest_sha256')!r} != {manifest_sha256!r})"
        )
    history = checkpoint["history"]
    epoch = checkpoint["epoch"]
    if not isinstance(history, list) or not history:
        raise ValueError("incompatible checkpoint: history must contain completed epochs")
    if int(history[-1].get("epoch", -1)) != int(epoch):
        raise ValueError("incompatible checkpoint: history does not end at checkpoint epoch")


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: Any,
    epoch: int,
    best_dice: float,
    best_epoch: int,
    patience_counter: int,
    history: list[dict[str, Any]],
    cfg: Mapping[str, Any],
    manifest_sha256: str | None,
    train_loader: Any,
    val: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": int(epoch),
        "best_dice": float(best_dice),
        "best_epoch": int(best_epoch),
        "patience_counter": int(patience_counter),
        "history": copy.deepcopy(history),
        "config": copy.deepcopy(dict(cfg)),
        "manifest_sha256": manifest_sha256,
        "rng_state": _capture_rng_state(),
        "train_loader_state": _loader_state(train_loader),
        "val": copy.deepcopy(dict(val)),
    }


def _save_history(history: list[dict[str, Any]], out_dir: Path) -> None:
    save_json(_jsonable(history), out_dir / "history.json")
    if not history:
        return
    fieldnames: list[str] = []
    for row in history:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _resume_path(cfg: Mapping[str, Any], resume: str | Path | None, out_dir: Path) -> Path | None:
    value: Any = resume
    if value is None:
        value = cfg["train"].get("resume")
    if value is None or value is False or value == "":
        return None
    if value is True or str(value).strip().casefold() in {"auto", "last"}:
        return out_dir / "last.pt"
    return Path(value).expanduser()


def _summary(
    cfg: Mapping[str, Any],
    model: nn.Module,
    device: torch.device,
    history: list[dict[str, Any]],
    best_epoch: int,
    best_dice: float,
    wall_seconds: float,
) -> dict[str, Any]:
    return {
        "experiment": cfg["experiment"],
        "best_epoch": int(best_epoch),
        "best_val_dice": float(best_dice),
        "epochs_run": len(history),
        "last_epoch": int(history[-1]["epoch"]) if history else 0,
        "wall_seconds": round(wall_seconds, 1),
        "parameters": count_parameters(model),
        "device": str(device),
    }


def run_training(
    cfg: dict,
    device: str | torch.device = "auto",
    resume: str | Path | None = None,
    val_limit: int | None = None,
    compute_boundary: bool = True,
) -> dict:
    """Run a complete training job and return its persisted summary."""
    cfg = _resolved_config(cfg)
    dev = _resolve_device(device)
    runtime_report = collect_runtime_report(require_cuda=False)
    seed_everything(int(cfg["seed"]))

    train_cfg = cfg["train"]
    epochs = int(train_cfg["epochs"])
    if epochs < 1:
        raise ValueError("train.epochs must be at least one")
    gradient_accumulation = int(train_cfg.get("gradient_accumulation", 1))
    if gradient_accumulation < 1:
        raise ValueError("train.gradient_accumulation must be at least one")

    out_dir = Path(cfg["output"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_path = _resume_path(cfg, resume, out_dir)
    train_loader, val_loader, _, pin_memory = _build_training_loaders(cfg, dev, val_limit)
    manifest_sha256 = _manifest_sha256(train_loader)

    model = (
        build_model(cfg, pretrained_override=False).to(dev)
        if resume_path
        else build_model(cfg).to(dev)
    )
    criterion = _build_criterion(cfg, dev)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    updates_per_epoch = max(math.ceil(len(train_loader) / gradient_accumulation), 1)
    total_steps = epochs * updates_per_epoch
    warmup_steps = int(train_cfg.get("warmup_epochs", 0)) * updates_per_epoch
    scheduler = cosine_with_warmup(optimizer, warmup_steps, total_steps)
    amp = bool(train_cfg.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and dev.type == "cuda")

    history: list[dict[str, Any]] = []
    best_dice = -1.0
    best_epoch = -1
    patience_counter = 0
    start_epoch = 1

    if resume_path is not None:
        checkpoint = _load_checkpoint(resume_path)
        _validate_checkpoint(checkpoint, cfg, manifest_sha256)
        try:
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            scaler.load_state_dict(checkpoint["scaler_state"])
        except (KeyError, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"incompatible checkpoint: could not restore training state: {exc}"
            ) from exc
        history = copy.deepcopy(checkpoint["history"])
        best_dice = float(checkpoint["best_dice"])
        best_epoch = int(checkpoint["best_epoch"])
        patience_counter = int(checkpoint["patience_counter"])
        start_epoch = int(checkpoint["epoch"]) + 1
        _restore_loader_state(train_loader, checkpoint.get("train_loader_state"))
        _restore_rng_state(checkpoint["rng_state"])

    save_json(cfg, out_dir / "config.json")
    save_json(runtime_report, out_dir / "runtime.json")
    if history:
        _save_history(history, out_dir)

    print(f"device={dev} torch={torch.__version__}")
    print(f"parameters={count_parameters(model):,}")
    started = time.time()
    if start_epoch > epochs:
        summary = _summary(cfg, model, dev, history, best_epoch, best_dice, time.time() - started)
        save_json(summary, out_dir / "summary.json")
        print(f"already complete: checkpoint epoch {start_epoch - 1} >= requested {epochs}")
        return summary

    non_blocking = bool(
        train_cfg.get(
            "non_blocking",
            cfg["data"].get("non_blocking", pin_memory and dev.type == "cuda"),
        )
    )
    configured_contrastive_weight = float(train_cfg.get("contrastive_weight", 0.1))
    contrastive_warmup_epochs = int(train_cfg.get("contrastive_warmup_epochs", 0))
    if contrastive_warmup_epochs < 0:
        raise ValueError("train.contrastive_warmup_epochs must be non-negative")

    for epoch in range(start_epoch, epochs + 1):
        epoch_started = time.time()
        if criterion.contrastive is None:
            effective_contrastive_weight = 0.0
        elif contrastive_warmup_epochs > 0:
            progress = min(max((epoch - 1) / contrastive_warmup_epochs, 0.0), 1.0)
            effective_contrastive_weight = configured_contrastive_weight * progress
        else:
            effective_contrastive_weight = configured_contrastive_weight
        criterion.contrastive_weight = effective_contrastive_weight

        train_result = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            dev,
            scaler,
            float(train_cfg.get("grad_clip", 1.0)),
            amp,
            gradient_accumulation=gradient_accumulation,
            non_blocking=non_blocking,
        )
        val_result, val_losses = validate(
            model,
            val_loader,
            criterion,
            dev,
            amp,
            compute_boundary=compute_boundary,
            non_blocking=non_blocking,
        )
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "contrastive_weight": effective_contrastive_weight,
            **{f"train_{key}": value for key, value in train_result.items()},
            **{f"val_{key}": value for key, value in val_result.items()},
            "val_seg_loss": val_losses["seg"],
            "val_contrastive_loss": val_losses["contrastive"],
            "seconds": round(time.time() - epoch_started, 2),
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} train_loss={train_result['loss']:.4f} "
            f"val_loss={val_result['loss']:.4f} iou={val_result['iou']:.4f} "
            f"dice={val_result['dice']:.4f} biou={val_result['boundary_iou']:.4f} "
            f"({row['seconds']:.1f}s)"
        )

        improved = float(val_result["dice"]) > best_dice
        if improved:
            best_dice = float(val_result["dice"])
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_dice=best_dice,
            best_epoch=best_epoch,
            patience_counter=patience_counter,
            history=history,
            cfg=cfg,
            manifest_sha256=manifest_sha256,
            train_loader=train_loader,
            val=val_result,
        )
        if improved:
            _atomic_torch_save(payload, out_dir / "best.pt")
        _atomic_torch_save(payload, out_dir / "last.pt")
        _save_history(history, out_dir)

        if bool(train_cfg.get("empty_cache_each_epoch", False)) and dev.type == "cuda":
            torch.cuda.empty_cache()
        if patience_counter >= int(train_cfg.get("patience", 8)):
            print(f"early stopping at epoch {epoch} (best {best_epoch})")
            break

    summary = _summary(cfg, model, dev, history, best_epoch, best_dice, time.time() - started)
    save_json(summary, out_dir / "summary.json")
    print(f"done: best val dice={best_dice:.4f} at epoch {best_epoch}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("ml/configs/sa_lsnet.json"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--val-limit", type=int, help="limit validation patches for smoke runs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--encoder")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--no-boundary", action="store_true", help="skip Boundary IoU during training"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="__OUTPUT_LAST__",
        metavar="PATH",
        help="resume PATH; with no PATH, resume OUTPUT/last.pt",
    )
    parser.add_argument("--manifest", type=Path, help="event CSV manifest")
    parser.add_argument("--data-root", type=Path, help="override the dataset root")
    args = parser.parse_args()

    cfg = _resolved_config(load_config(args.config) if args.config.exists() else {})
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.limit is not None:
        cfg["data"]["limit"] = args.limit
    if args.output is not None:
        cfg["output"] = str(args.output)
    if args.encoder:
        cfg["model"]["encoder"] = args.encoder
    if args.no_pretrained:
        cfg["model"]["pretrained"] = False
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.manifest is not None:
        cfg["data"]["type"] = "manifest"
        cfg["data"]["manifest"] = str(args.manifest)
        cfg["data"]["data_root"] = str(args.data_root or args.manifest.parent)
    elif args.data_root is not None:
        if _data_kind(cfg["data"]) == "manifest":
            cfg["data"]["data_root"] = str(args.data_root)
        else:
            cfg["data"]["root"] = str(args.data_root)

    resume: str | Path | None
    if args.resume == "__OUTPUT_LAST__":
        resume = Path(cfg["output"]) / "last.pt"
    else:
        resume = args.resume
    run_training(
        cfg,
        device=args.device,
        resume=resume,
        val_limit=args.val_limit,
        compute_boundary=not args.no_boundary,
    )


if __name__ == "__main__":
    main()
