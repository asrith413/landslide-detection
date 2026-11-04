import csv
import gc
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from ml.landslide.research_data import (
    EventManifestDataset,
    HRGLDDDataset,
    build_cas_event_manifest,
    validate_event_manifest,
)
from ml.landslide.storage import checksum_file, download_once, extract_zip_once


def _write_pair(root: Path, name: str, offset: int = 0) -> tuple[str, str]:
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    grid = np.arange(16, dtype=np.uint8).reshape(4, 4)
    image = np.stack(
        (
            (grid * 11 + offset).astype(np.uint8),
            (np.flipud(grid) * 9 + offset).astype(np.uint8),
            (np.fliplr(grid) * 7 + offset).astype(np.uint8),
        ),
        axis=-1,
    )
    mask = np.zeros((4, 4), dtype=np.uint16)
    mask[1:3, 1:3] = 65535
    image_path = image_dir / f"{name}.png"
    mask_path = mask_dir / f"{name}.tiff"
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)
    return image_path.relative_to(root).as_posix(), mask_path.relative_to(root).as_posix()


def _write_manifest(root: Path, rows: list[dict[str, str]]) -> Path:
    manifest = root / "events.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image", "mask", "split", "event"))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _three_split_manifest(root: Path) -> Path:
    rows = []
    for index, (split, event) in enumerate(
        (("train", "Train Event"), ("validation", "Val Event"), ("test", "Test Event"))
    ):
        image, mask = _write_pair(root, split, offset=index)
        rows.append({"image": image, "mask": mask, "split": split, "event": event})
    return _write_manifest(root, rows)


def _write_cas_archive(
    root: Path, directory: str, pairs: tuple[tuple[str, str], ...], offset: int
) -> None:
    sample_root = root / directory / "extracted" / "cas-sample"
    image_dir = sample_root / "img"
    mask_dir = sample_root / "mask"
    label_dir = sample_root / "label"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir()
    label_dir.mkdir()
    grid = np.arange(16, dtype=np.uint8).reshape(4, 4)
    for index, (image_name, mask_name) in enumerate(pairs):
        image = np.stack(
            (
                grid + offset + index,
                np.flipud(grid) + offset + index,
                np.fliplr(grid) + offset + index,
            ),
            axis=-1,
        ).astype(np.uint8)
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[index % 2 :: 2] = 255
        Image.fromarray(image).save(image_dir / image_name)
        Image.fromarray(mask).save(mask_dir / mask_name)
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(label_dir / "ignored.tif")


def _cas_archive_config(root: Path) -> dict[str, object]:
    _write_cas_archive(
        root,
        "archives/train",
        (("SlopeB.TIF", "slopeb.tiff"), ("alpha.tiff", "ALPHA.TIF")),
        1,
    )
    _write_cas_archive(root, "archives/validation", (("validation.tif", "VALIDATION.TIFF"),), 2)
    _write_cas_archive(root, "archives/test", (("test.TIFF", "TEST.tif"),), 3)
    return {
        "archives": [
            {
                "directory": "archives/test",
                "event": "Test Event",
                "split": "test",
                "archive": "test.zip",
                "sensor": "Sentinel-2",
                "gsd_m": 10,
            },
            {
                "directory": "archives/train",
                "event": "Train Event",
                "split": "train",
                "archive": "train.zip",
                "sensor": "PlanetScope",
                "gsd": 3,
            },
            {
                "directory": "archives/validation",
                "event": "Val Event",
                "split": "validation",
            },
        ],
        "events": {
            "Train Event": {"split": "train"},
            "Val Event": {"split": "val"},
            "Test Event": {"split": "test"},
        },
    }


class EventManifestTests(unittest.TestCase):
    def test_event_family_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for index, (split, event) in enumerate(
                (("train", "Storm Alpha"), ("val", "storm-alpha"), ("test", "Storm Beta"))
            ):
                image, mask = _write_pair(root, f"sample-{index}")
                rows.append({"image": image, "mask": mask, "split": split, "event": event})
            manifest = _write_manifest(root, rows)
            with self.assertRaisesRegex(ValueError, "crosses splits"):
                validate_event_manifest(manifest, full_decode=False)

    def test_decode_cache_is_reused_but_existence_is_rechecked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _three_split_manifest(root)
            cache = root / "decode-cache.json"
            with mock.patch("PIL.Image.open", wraps=Image.open) as image_open:
                first = validate_event_manifest(manifest, cache_path=cache)
                calls_after_first_pass = image_open.call_count
                second = validate_event_manifest(manifest, cache_path=cache)
                self.assertEqual(image_open.call_count, calls_after_first_pass)

            self.assertFalse(first["decode_cache_hit"])
            self.assertTrue(second["decode_cache_hit"])
            self.assertEqual(first["counts"], {"train": 1, "val": 1, "test": 1})
            self.assertEqual(first["decoded_pairs"], 3)
            json.dumps(second)

            (root / "masks" / "test.tiff").unlink()
            with self.assertRaises(FileNotFoundError):
                validate_event_manifest(manifest, cache_path=cache)

    def test_policy_aliases_and_dataset_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _three_split_manifest(root)
            policy = root / "policy.json"
            policy.write_text(
                json.dumps(
                    {
                        "events": {
                            "Canonical Train": {
                                "split": "train",
                                "aliases": ["Train Event"],
                            },
                            "Canonical Val": {
                                "split": "validation",
                                "aliases": ["Val Event"],
                            },
                            "Canonical Test": {
                                "split": "test",
                                "aliases": ["Test Event"],
                            },
                        },
                        "require_complete_coverage": True,
                    }
                ),
                encoding="utf-8",
            )
            report = validate_event_manifest(manifest, split_policy=policy)
            self.assertEqual(report["events"]["train"], ["Canonical Train"])

            first = EventManifestDataset(
                manifest,
                "train",
                split_policy=policy,
                augment=True,
                resize=6,
                seed=17,
            )
            second = EventManifestDataset(
                manifest,
                "train",
                split_policy=policy,
                augment=True,
                resize=6,
                seed=17,
            )
            sample = first[0]
            repeated = second[0]
            self.assertEqual(set(sample), {"image", "mask", "id", "path", "event"})
            self.assertEqual(tuple(sample["image"].shape), (3, 6, 6))
            self.assertEqual(tuple(sample["mask"].shape), (1, 6, 6))
            self.assertEqual(sample["event"], "Canonical Train")
            self.assertTrue(torch.equal(sample["image"], repeated["image"]))
            self.assertTrue(torch.equal(sample["mask"], repeated["mask"]))
            self.assertTrue(set(sample["mask"].unique().tolist()) <= {0.0, 1.0})


class CASEventManifestTests(unittest.TestCase):
    def test_build_is_deterministic_and_validates_tiffs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _cas_archive_config(root)
            config_path = root / "archives.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = root / "cas-events.csv"
            cache = root / "cas-validation-cache.json"

            first = build_cas_event_manifest(
                root,
                config_path,
                output,
                full_decode=True,
                validation_cache=cache,
            )
            first_bytes = output.read_bytes()
            second = build_cas_event_manifest(
                root,
                config_path,
                output,
                full_decode=True,
                validation_cache=cache,
            )

            self.assertEqual(output.read_bytes(), first_bytes)
            self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
            self.assertEqual(first["manifest_sha256"], hashlib.sha256(first_bytes).hexdigest())
            self.assertEqual(first["counts"], {"train": 2, "val": 1, "test": 1})
            self.assertTrue(second["validation"]["decode_cache_hit"])
            with output.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertEqual(
                    reader.fieldnames,
                    ["image", "mask", "split", "event", "archive", "sensor", "gsd_m"],
                )
            self.assertEqual(
                [row["image"] for row in rows],
                sorted((row["image"] for row in rows), key=str.casefold),
            )
            self.assertTrue(all(not Path(row["image"]).is_absolute() for row in rows))
            self.assertTrue(all("label" not in row["mask"] for row in rows))
            self.assertEqual({row["split"] for row in rows}, {"train", "val", "test"})

    def test_missing_policy_family_prevents_manifest_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _cas_archive_config(root)
            config["events"]["Unused Event"] = {"split": "train"}
            output = root / "cas-events.csv"

            with self.assertRaisesRegex(ValueError, "missing policy event families.*Unused Event"):
                build_cas_event_manifest(root, config, output)
            self.assertFalse(output.exists())


class HRGLDDTests(unittest.TestCase):
    def test_memory_map_float_normalization_and_nir_discard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            integer_root = root / "integer"
            float_root = root / "float"
            integer_root.mkdir()
            float_root.mkdir()
            rgb = np.array(
                [
                    [[0, 20, 40], [80, 60, 20]],
                    [[160, 100, 0], [240, 140, 80]],
                ],
                dtype=np.uint8,
            )
            integer_images = np.concatenate(
                (rgb[None], np.zeros((1, 2, 2, 1), dtype=np.uint8)), axis=-1
            )
            float_images = np.concatenate(
                (
                    rgb.astype(np.float32)[None] / 255.0,
                    np.ones((1, 2, 2, 1), dtype=np.float32),
                ),
                axis=-1,
            )
            mask = np.array([[[0, 255], [255, 0]]], dtype=np.uint8)
            np.save(integer_root / "trainX.npy", integer_images)
            np.save(integer_root / "trainY.npy", mask)
            np.save(float_root / "trainX.npy", float_images)
            np.save(float_root / "trainY.npy", mask)

            integer_dataset = HRGLDDDataset(integer_root, "train")
            float_dataset = HRGLDDDataset(float_root, "train")
            self.assertIsInstance(integer_dataset.images, np.memmap)
            integer_sample = integer_dataset[0]
            float_sample = float_dataset[0]
            self.assertTrue(torch.equal(integer_sample["image"], float_sample["image"]))
            self.assertEqual(tuple(integer_sample["mask"].shape), (1, 2, 2))
            self.assertEqual(integer_sample["event"], "HR-GLDD-external")

            del integer_sample, float_sample, integer_dataset, float_dataset
            gc.collect()


class StorageTests(unittest.TestCase):
    def test_download_reuses_verified_destination_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "archive.zip"
            destination.write_bytes(b"already-downloaded")
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            with mock.patch("urllib.request.urlopen") as urlopen:
                report = download_once(
                    "https://example.invalid/archive.zip",
                    destination,
                    checksum=digest,
                    expected_bytes=destination.stat().st_size,
                )
            urlopen.assert_not_called()
            self.assertEqual(report["action"], "reused")
            self.assertEqual(report["checksum"], digest)

    def test_checksum_and_zip_marker_preserve_existing_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "data.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("drive/data.txt", b"research-data")
            self.assertEqual(
                checksum_file(archive, "md5"), hashlib.md5(archive.read_bytes()).hexdigest()
            )
            destination = root / "extracted"
            first = extract_zip_once(archive, destination)
            preserved = destination / "drive" / "user-note.txt"
            preserved.write_text("keep", encoding="utf-8")
            second = extract_zip_once(archive, destination)

            self.assertEqual(first["action"], "extracted")
            self.assertEqual(second["action"], "reused")
            self.assertEqual((destination / "drive" / "data.txt").read_bytes(), b"research-data")
            self.assertEqual(preserved.read_text(encoding="utf-8"), "keep")
            json.dumps(second)

    def test_zip_slip_is_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../escaped.txt", b"bad")
                output.writestr("otherwise-safe.txt", b"not extracted")
            destination = root / "unsafe-output"
            with self.assertRaisesRegex(ValueError, "unsafe ZIP member"):
                extract_zip_once(archive, destination)
            self.assertFalse((root / "escaped.txt").exists())
            self.assertFalse((destination / "otherwise-safe.txt").exists())


if __name__ == "__main__":
    unittest.main()
