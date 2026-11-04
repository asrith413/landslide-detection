import hashlib
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from ml.landslide.deployment import (
    benchmark_onnx,
    benchmark_torch,
    collect_runtime_report,
    export_onnx_with_parity,
    load_checkpoint_model,
    quantize_onnx_dynamic,
    write_artifact_inventory,
)


def _has_optional_onnx_stack():
    try:
        importlib.import_module("onnx")
        importlib.import_module("onnxruntime")
    except (ImportError, OSError):
        return False
    return True


HAS_ONNX_STACK = _has_optional_onnx_stack()


class RuntimeReportTests(unittest.TestCase):
    def test_cpu_runtime_report_is_json_serializable(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            report = collect_runtime_report()

        self.assertFalse(report["cuda"]["runtime_available"])
        self.assertEqual(report["cuda"]["device_count"], 0)
        self.assertIn("version", report["python"])
        self.assertIn("version", report["torch"])
        json.dumps(report, allow_nan=False)

    def test_required_cuda_has_clear_error(self):
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            self.assertRaisesRegex(RuntimeError, "CUDA is required"),
        ):
            collect_runtime_report(require_cuda=True)


class BenchmarkTests(unittest.TestCase):
    def test_torch_cpu_benchmark_records_individual_timings(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        report = benchmark_torch(
            model,
            torch.randn(1, 4),
            device="cpu",
            warmup=1,
            repetitions=4,
        )

        self.assertEqual(report["backend"], "pytorch")
        self.assertEqual(len(report["measurements_ms"]), 4)
        self.assertGreater(report["latency_ms"]["median"], 0.0)
        self.assertGreaterEqual(report["latency_ms"]["p95"], report["latency_ms"]["median"])
        self.assertGreater(report["throughput_samples_per_second"], 0.0)
        self.assertIsNone(report["peak_allocated_cuda_memory_bytes"])
        self.assertGreater(report["model_size_bytes"], 0)
        json.dumps(report, allow_nan=False)


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_load_forces_pretrained_off_without_changing_reported_config(self):
        source_model = nn.Linear(3, 2)
        recorded_config = {"model": {"pretrained": True}}
        observed = {}

        def fake_build_model(config, pretrained_override=None):
            observed["config"] = config
            observed["pretrained_override"] = pretrained_override
            return nn.Linear(3, 2)

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "model.pt"
            torch.save(
                {"config": recorded_config, "model_state": source_model.state_dict()},
                checkpoint,
            )
            with mock.patch("ml.landslide.model.build_model", new=fake_build_model):
                model, config, _ = load_checkpoint_model(checkpoint)

        self.assertFalse(model.training)
        self.assertFalse(observed["config"]["model"]["pretrained"])
        self.assertIs(observed["pretrained_override"], False)
        self.assertTrue(config["model"]["pretrained"])


class ArtifactInventoryTests(unittest.TestCase):
    def test_inventory_hashes_recursively_and_excludes_itself(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            nested = root / "nested"
            nested.mkdir()
            (root / "a.bin").write_bytes(b"alpha")
            (nested / "b.bin").write_bytes(b"beta")
            (root / "manifest.json").write_text("stale", encoding="utf-8")

            report = write_artifact_inventory(root)
            paths = [item["path"] for item in report["files"]]

            self.assertEqual(paths, ["a.bin", "nested/b.bin"])
            self.assertEqual(report["file_count"], 2)
            self.assertEqual(report["files"][0]["sha256"], hashlib.sha256(b"alpha").hexdigest())
            saved = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, report)

            second = write_artifact_inventory(root)
            self.assertEqual(second["file_count"], 2)


@unittest.skipUnless(HAS_ONNX_STACK, "optional onnx and onnxruntime packages are required")
class OptionalOnnxTests(unittest.TestCase):
    def test_export_benchmark_and_quantization_report(self):
        from ml.landslide.model import build_model

        config = {
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
            }
        }
        model = build_model(config).eval()
        sample = torch.randn(1, 3, 32, 32)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "model.pt"
            onnx_path = root / "model.onnx"
            quantized_path = root / "model.int8.onnx"
            torch.save({"config": config, "model_state": model.state_dict()}, checkpoint)

            export_report = export_onnx_with_parity(
                checkpoint,
                onnx_path,
                image_size=32,
                sample=sample,
                atol=1e-4,
                rtol=1e-4,
            )
            self.assertTrue(export_report["parity_passed"])
            self.assertTrue((root / "model.json").is_file())

            benchmark_report = benchmark_onnx(onnx_path, sample, warmup=1, repetitions=2)
            self.assertEqual(len(benchmark_report["measurements_ms"]), 2)

            quantization_report = quantize_onnx_dynamic(
                onnx_path, quantized_path, sample, tolerance=1.0
            )
            self.assertIn(quantization_report["status"], {"success", "skipped"})
            self.assertTrue((root / "model.int8.json").is_file())


if __name__ == "__main__":
    unittest.main()
