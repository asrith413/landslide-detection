#!/usr/bin/env python3
"""Train SA-LSNet or an ablation variant with a reproducible JSON config."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .data import build_dataloader
from .losses import BCEDiceLoss, MaskedObjectContrastiveLoss, TotalLoss
from .metrics import SegmentationMetrics
from .model import build_model
from .utils import count_parameters, load_config, merge_dicts, save_json, seed_everything


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
        "patience": 8,
        "contrastive_weight": 0.1,
        "pos_weight": 2.0,
        "boundary_weight": 0.0,
        "bce_weight": 0.5,
        "dice_weight": 0.5,
    },
    "output": "checkpoints/sa_lsnet",
}


def cosine_with_warmup(optimizer: torch.optim.Optimizer, warmup: int, total: int) -> LambdaLR:
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, fn)


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
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_seg = 0.0
    total_con = 0.0
    n = 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            out = model(x)
            losses = criterion(out, y)
        scaler.scale(losses["loss"]).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        bs = x.shape[0]
        total_loss += float(losses["loss"].detach()) * bs
        total_seg += float(losses["seg"]) * bs
        total_con += float(losses["contrastive"]) * bs
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
) -> tuple[dict[str, float], dict[str, float]]:
    model.eval()
    metrics = SegmentationMetrics()
    total_loss = 0.0
    n = 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            out = model(x)
            losses = criterion(out, y)
        metrics.update(out["logits"], y, compute_boundary=compute_boundary)
        total_loss += float(losses["loss"]) * x.shape[0]
        n += x.shape[0]
    result = metrics.compute()
    result["loss"] = total_loss / max(n, 1)
    return result, {"seg": 0.0, "contrastive": 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("ml/configs/sa_lsnet.json"))
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--val-limit", type=int, help="limit validation patches for smoke runs")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--encoder")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--no-boundary", action="store_true", help="skip Boundary IoU during training")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    cfg = merge_dicts(DEFAULTS, load_config(args.config) if args.config.exists() else {})
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

    seed_everything(cfg["seed"])
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device={device} torch={torch.__version__}")

    root = Path(cfg["data"]["root"])
    limit = cfg["data"]["limit"] or None
    train_loader = build_dataloader(
        root,
        "train",
        batch_size=cfg["data"]["batch_size"],
        augment=True,
        num_workers=cfg["data"]["num_workers"],
        limit=limit,
        seed=cfg["seed"],
    )
    val_loader = build_dataloader(
        root,
        "val",
        batch_size=cfg["data"]["batch_size"],
        augment=False,
        num_workers=cfg["data"]["num_workers"],
        limit=args.val_limit,
        seed=cfg["seed"],
    )
    if len(train_loader.dataset) == 0:
        raise SystemExit("empty training split; run prepare_landslide4sense.py")

    model = build_model(cfg).to(device)
    print(f"parameters={count_parameters(model):,}")

    seg = BCEDiceLoss(
        bce_weight=cfg["train"]["bce_weight"],
        dice_weight=cfg["train"]["dice_weight"],
        pos_weight=cfg["train"]["pos_weight"],
        boundary_weight=cfg["train"]["boundary_weight"],
    )
    contrastive = (
        MaskedObjectContrastiveLoss() if cfg["model"].get("use_contrastive", True) else None
    )
    criterion = TotalLoss(
        seg=seg,
        contrastive=contrastive,
        contrastive_weight=cfg["train"]["contrastive_weight"],
    )

    optimizer = AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = cfg["train"]["epochs"] * steps_per_epoch
    warmup_steps = cfg["train"]["warmup_epochs"] * steps_per_epoch
    scheduler = cosine_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"] and device.type == "cuda")

    out_dir = Path(cfg["output"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, out_dir / "config.json")
    history: list[dict[str, Any]] = []
    best_dice = -1.0
    best_epoch = -1
    bad = 0
    start = time.time()

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        t0 = time.time()
        tr = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            scaler,
            cfg["train"]["grad_clip"],
            cfg["train"]["amp"],
        )
        va, _ = validate(
            model,
            val_loader,
            criterion,
            device,
            cfg["train"]["amp"],
            compute_boundary=not args.no_boundary,
        )
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"val_{k}": v for k, v in va.items()},
            "seconds": round(time.time() - t0, 2),
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} train_loss={tr['loss']:.4f} "
            f"val_loss={va['loss']:.4f} iou={va['iou']:.4f} dice={va['dice']:.4f} "
            f"biou={va['boundary_iou']:.4f} ({row['seconds']:.1f}s)"
        )
        if va["dice"] > best_dice:
            best_dice = va["dice"]
            best_epoch = epoch
            bad = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "best_dice": best_dice,
                    "val": va,
                },
                out_dir / "best.pt",
            )
        else:
            bad += 1
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "best_dice": best_dice,
            },
            out_dir / "last.pt",
        )
        save_json(history, out_dir / "history.json")
        with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        if bad >= cfg["train"]["patience"]:
            print(f"early stopping at epoch {epoch} (best {best_epoch})")
            break

    save_json(
        {
            "experiment": cfg["experiment"],
            "best_epoch": best_epoch,
            "best_val_dice": best_dice,
            "epochs_run": len(history),
            "wall_seconds": round(time.time() - start, 1),
            "parameters": count_parameters(model),
            "device": str(device),
        },
        out_dir / "summary.json",
    )
    print(f"done: best val dice={best_dice:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
