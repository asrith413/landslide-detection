import copy
import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image
from torch import nn

from ml.landslide.evaluate import evaluate_loader
from ml.landslide.research_data import build_research_dataloader
from ml.landslide.train import run_training


class _TinySegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(3, 1, 1)

    def forward(self, image):
        return {"logits": self.head(image), "embedding": None}


class _AllPositiveModel(nn.Module):
    def forward(self, image):
        return {"logits": image[:, :1] * 0.0 + 10.0, "embedding": None}


def _write_training_manifest(root: Path) -> Path:
    rng = np.random.default_rng(7)
    rows = []
    for split, count in (("train", 3), ("val", 1), ("test", 1)):
        for index in range(count):
            image_name = f"{split}-{index}-image.png"
            mask_name = f"{split}-{index}-mask.png"
            image = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[2:6, 2:6] = 255
            Image.fromarray(image).save(root / image_name)
            Image.fromarray(mask).save(root / mask_name)
            rows.append(
                {
                    "image": image_name,
                    "mask": mask_name,
                    "split": split,
                    "event": f"{split.title()} Event {index}",
                }
            )
    manifest = root / "training-events.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image", "mask", "split", "event"))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _training_config(root: Path, manifest: Path, output: Path) -> dict:
    return {
        "experiment": "research-training-test",
        "seed": 13,
        "data": {
            "type": "event_manifest",
            "manifest": str(manifest),
            "data_root": str(root),
            "full_validation": False,
            "batch_size": 1,
            "num_workers": 0,
            "limit": 0,
            "pin_memory": False,
        },
        "model": {
            "encoder": "tiny",
            "pretrained": False,
            "use_structural": False,
            "fusion": "none",
            "use_contrastive": False,
            "width": 8,
            "decoder_channels": 8,
            "embed_dim": 4,
            "dropout": 0.0,
        },
        "train": {
            "epochs": 1,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "warmup_epochs": 0,
            "amp": False,
            "grad_clip": 1.0,
            "gradient_accumulation": 2,
            "patience": 10,
            "contrastive_weight": 0.0,
            "pos_weight": 1.0,
            "boundary_weight": 0.0,
            "bce_weight": 0.5,
            "dice_weight": 0.5,
        },
        "output": str(output),
    }


class ResearchTrainingTests(unittest.TestCase):
    def test_cpu_checkpoint_resume_and_incompatible_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            output = root / "output"
            data_root.mkdir()
            manifest = _write_training_manifest(data_root)
            config = _training_config(data_root, manifest, output)

            factory = lambda *args, **kwargs: _TinySegmentationModel()
            with (
                mock.patch("ml.landslide.train.build_model", side_effect=factory),
                mock.patch(
                    "ml.landslide.train.collect_runtime_report",
                    return_value={"cuda": {"runtime_available": False}},
                ),
            ):
                first = run_training(config, device="cpu", compute_boundary=False)

            self.assertEqual(first["epochs_run"], 1)
            self.assertTrue((output / "config.json").is_file())
            self.assertTrue((output / "runtime.json").is_file())
            self.assertTrue((output / "history.csv").is_file())
            checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
            required = {
                "schema_version",
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
            self.assertTrue(required <= checkpoint.keys())
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(len(checkpoint["history"]), 1)
            self.assertIsNotNone(checkpoint["manifest_sha256"])
            self.assertEqual(set(checkpoint["rng_state"]), {"python", "numpy", "torch", "cuda"})

            resumed_config = copy.deepcopy(config)
            resumed_config["train"]["epochs"] = 2
            with (
                mock.patch("ml.landslide.train.build_model", side_effect=factory),
                mock.patch(
                    "ml.landslide.train.collect_runtime_report",
                    return_value={"cuda": {"runtime_available": False}},
                ),
            ):
                resumed = run_training(
                    resumed_config,
                    device="cpu",
                    resume=output / "last.pt",
                    compute_boundary=False,
                )
            self.assertEqual(resumed["epochs_run"], 2)
            resumed_checkpoint = torch.load(
                output / "last.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(resumed_checkpoint["epoch"], 2)
            self.assertEqual(len(resumed_checkpoint["history"]), 2)

            incompatible = copy.deepcopy(resumed_config)
            incompatible["model"]["width"] = 9
            with (
                mock.patch("ml.landslide.train.build_model", side_effect=factory),
                mock.patch(
                    "ml.landslide.train.collect_runtime_report",
                    return_value={"cuda": {"runtime_available": False}},
                ),
                self.assertRaisesRegex(ValueError, "incompatible checkpoint: model"),
            ):
                run_training(
                    incompatible,
                    device="cpu",
                    resume=output / "last.pt",
                    compute_boundary=False,
                )

    def test_event_metrics_are_reported_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for index, event in enumerate(("Event Alpha", "Event Beta")):
                image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
                mask = np.full((8, 8), 255 if index == 0 else 0, dtype=np.uint8)
                image_path = root / f"image-{index}.png"
                mask_path = root / f"mask-{index}.png"
                Image.fromarray(image).save(image_path)
                Image.fromarray(mask).save(mask_path)
                rows.append(
                    {
                        "image": image_path.name,
                        "mask": mask_path.name,
                        "split": "test",
                        "event": event,
                    }
                )
            manifest = root / "events.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("image", "mask", "split", "event"))
                writer.writeheader()
                writer.writerows(rows)

            loader = build_research_dataloader(
                {
                    "type": "event_manifest",
                    "manifest": str(manifest),
                    "data_root": str(root),
                    "required_splits": ["test"],
                    "full_validation": False,
                    "pin_memory": False,
                },
                "test",
                batch_size=2,
                shuffle=False,
            )
            result = evaluate_loader(_AllPositiveModel(), loader, "cpu", compute_boundary=False)

            self.assertEqual(set(result["per_event"]), {"Event Alpha", "Event Beta"})
            self.assertAlmostEqual(result["per_event"]["Event Alpha"]["dice"], 1.0)
            self.assertAlmostEqual(result["per_event"]["Event Beta"]["dice"], 0.0)
            self.assertEqual(result["tp"], 64)
            self.assertEqual(result["fp"], 64)
            self.assertEqual(result["fn"], 0)
            self.assertEqual(result["tn"], 0)


if __name__ == "__main__":
    unittest.main()
