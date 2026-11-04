"""Validated real-data datasets for event manifests and HR-GLDD arrays.

The event manifest format is a UTF-8 CSV with the required columns
``image,mask,split,event``. Paths are confined to ``data_root`` and event
families are kept in exactly one split.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from .data import _augment, load_rgb, percentile_stretch, rgb_to_tensor
from .storage import checksum_file

_REQUIRED_COLUMNS = ("image", "mask", "split", "event")
_SPLITS = {"train": "train", "validation": "val", "val": "val", "test": "test"}
_CACHE_VERSION = 1


def canonicalize_split(split: str) -> str:
    """Return ``train``, ``val``, or ``test`` for a supported split name."""
    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a non-blank string")
    value = split.strip().casefold()
    try:
        return _SPLITS[value]
    except KeyError as exc:
        raise ValueError(
            f"unknown split {split!r}; expected train, validation/val, or test"
        ) from exc


def _event_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    key = " ".join("".join(c if c.isalnum() else " " for c in normalized).split())
    if not key:
        raise ValueError("event must contain at least one letter or number")
    return key


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _read_policy(
    split_policy: str | Path | Mapping[str, Any] | None,
    manifest_parent: Path,
    require_policy_coverage: bool | None,
) -> dict[str, Any]:
    if split_policy is None:
        if require_policy_coverage:
            raise ValueError("complete policy coverage requires a split policy")
        return {
            "aliases": {},
            "families": {},
            "sha256": None,
            "path": None,
            "require_coverage": False,
        }

    if isinstance(split_policy, Mapping):
        document = dict(split_policy)
        policy_hash = _json_hash(document)
        policy_path: str | None = None
    else:
        path = Path(split_policy).expanduser()
        if not path.is_absolute():
            path = manifest_parent / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"split policy does not exist: {path}")
        try:
            with path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read split policy {path}: {exc}") from exc
        policy_hash = checksum_file(path, "sha256")
        policy_path = str(path)

    if not isinstance(document, dict):
        raise TypeError("split policy JSON must contain an object")

    policy_requirement = document.get(
        "require_complete_coverage", document.get("require_complete_policy", False)
    )
    if not isinstance(policy_requirement, bool):
        raise TypeError("split policy require_complete_coverage must be a boolean")
    require_coverage = (
        policy_requirement if require_policy_coverage is None else require_policy_coverage
    )
    if not isinstance(require_coverage, bool):
        raise TypeError("require_policy_coverage must be a boolean")

    families: dict[str, dict[str, str]] = {}
    aliases: dict[str, str] = {}

    def add_family(name: Any, split: Any, family_aliases: Any = ()) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("policy event names must be non-blank strings")
        canonical_name = name.strip()
        canonical_key = _event_key(canonical_name)
        canonical_split = canonicalize_split(split)
        existing = families.get(canonical_key)
        if existing and (
            existing["split"] != canonical_split or existing["name"] != canonical_name
        ):
            raise ValueError(f"conflicting policy definitions for event {name!r}")
        families[canonical_key] = {"name": canonical_name, "split": canonical_split}

        if isinstance(family_aliases, str) or not isinstance(family_aliases, Sequence):
            raise TypeError(f"aliases for policy event {name!r} must be a list")
        for alias in (canonical_name, *family_aliases):
            if not isinstance(alias, str) or not alias.strip():
                raise ValueError(f"aliases for policy event {name!r} must be non-blank")
            alias_key = _event_key(alias)
            previous = aliases.get(alias_key)
            if previous is not None and previous != canonical_key:
                raise ValueError(f"policy alias {alias!r} belongs to multiple events")
            aliases[alias_key] = canonical_key

    def add_definition(name: Any, specification: Any) -> None:
        if isinstance(specification, str):
            add_family(name, specification)
            return
        if not isinstance(specification, dict) or "split" not in specification:
            raise ValueError(
                f"policy event {name!r} must map to a split or an object with split/aliases"
            )
        add_family(name, specification["split"], specification.get("aliases", ()))

    definitions = document.get("events")
    if definitions is not None:
        if isinstance(definitions, dict):
            for name, specification in definitions.items():
                add_definition(name, specification)
        elif isinstance(definitions, list):
            for item in definitions:
                if not isinstance(item, dict) or "name" not in item or "split" not in item:
                    raise ValueError("policy event lists require name and split for every item")
                add_family(item["name"], item["split"], item.get("aliases", ()))
        else:
            raise ValueError("split policy events must be an object or list")
    elif "splits" in document:
        split_groups = document["splits"]
        if not isinstance(split_groups, dict):
            raise ValueError("split policy splits must be an object")
        for split, names in split_groups.items():
            canonical_split = canonicalize_split(split)
            if isinstance(names, str) or not isinstance(names, list):
                raise TypeError(f"policy split {split!r} must contain an event list")
            for name in names:
                if isinstance(name, str):
                    add_family(name, canonical_split)
                elif isinstance(name, dict) and "name" in name:
                    add_family(name["name"], canonical_split, name.get("aliases", ()))
                else:
                    raise ValueError(f"invalid event entry in policy split {split!r}")
    else:
        reserved = {
            "aliases",
            "require_complete_coverage",
            "require_complete_policy",
        }
        definitions = {k: v for k, v in document.items() if k not in reserved}
        split_group_form = bool(definitions) and all(
            isinstance(k, str) and k.strip().casefold() in _SPLITS and isinstance(v, list)
            for k, v in definitions.items()
        )
        if split_group_form:
            for split, names in definitions.items():
                for name in names:
                    if isinstance(name, str):
                        add_family(name, split)
                    elif isinstance(name, dict) and "name" in name:
                        add_family(name["name"], split, name.get("aliases", ()))
                    else:
                        raise ValueError(f"invalid event entry in policy split {split!r}")
        else:
            for name, specification in definitions.items():
                add_definition(name, specification)

    top_level_aliases = document.get("aliases", {})
    if not isinstance(top_level_aliases, dict):
        raise TypeError("split policy aliases must be an object")
    for alias, target in top_level_aliases.items():
        if not isinstance(alias, str) or not isinstance(target, str):
            raise TypeError("split policy aliases must map strings to canonical event names")
        target_key = _event_key(target)
        if target_key not in families:
            raise ValueError(f"policy alias {alias!r} targets unknown event {target!r}")
        alias_key = _event_key(alias)
        previous = aliases.get(alias_key)
        if previous is not None and previous != target_key:
            raise ValueError(f"policy alias {alias!r} belongs to multiple events")
        aliases[alias_key] = target_key

    if not families:
        raise ValueError("split policy does not define any events")
    return {
        "aliases": aliases,
        "families": families,
        "sha256": policy_hash,
        "path": policy_path,
        "require_coverage": require_coverage,
    }


def _resolve_data_path(raw: str, root: Path, row_number: int, column: str) -> Path:
    if "\x00" in raw:
        raise ValueError(f"manifest row {row_number} {column} contains a null byte")
    portable = PurePosixPath(raw.replace("\\", "/"))
    if ".." in portable.parts:
        raise ValueError(f"manifest row {row_number} {column} contains path traversal: {raw!r}")

    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = root.joinpath(*portable.parts)
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"manifest row {row_number} {column} escapes data_root: {raw!r}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"manifest row {row_number} {column} does not exist: {resolved}")
    return resolved


def _mask_plane(mask: np.ndarray, context: str) -> np.ndarray:
    if mask.ndim == 2:
        return mask
    if mask.ndim == 3 and mask.shape[-1] == 1:
        return mask[..., 0]
    if mask.ndim == 3 and mask.shape[-1] in (3, 4):
        first = mask[..., 0]
        if all(np.array_equal(first, mask[..., channel]) for channel in range(1, mask.shape[-1])):
            return first
    raise ValueError(f"{context} must be a single-channel mask, got shape {mask.shape}")


def _validate_binary_values(values: np.ndarray | set[Any], context: str) -> set[int]:
    raw_values = values if isinstance(values, set) else set(np.unique(values).tolist())
    normalized: set[int] = set()
    for value in raw_values:
        if not isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
            raise TypeError(f"{context} has a non-numeric mask value")
        numeric = float(value)
        if not np.isfinite(numeric) or numeric != int(numeric):
            raise ValueError(f"{context} has a non-integral mask value {value!r}")
        normalized.add(int(numeric))
    valid_encodings = ({0, 1}, {0, 255}, {0, 65535})
    if not normalized or not any(normalized <= encoding for encoding in valid_encodings):
        preview = sorted(normalized)[:10]
        raise ValueError(f"{context} must use binary values 0/1, 0/255, or 0/65535; got {preview}")
    return normalized


def _load_mask(path: str | Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        image.load()
        mask = np.array(image)
    mask = _mask_plane(mask, f"mask {path}")
    _validate_binary_values(mask, f"mask {path}")
    return (mask > 0).astype(np.uint8)


def _decode_pair(record: Mapping[str, Any], expected_image_size: int | None) -> None:
    from PIL import Image

    try:
        with Image.open(record["image"]) as source:
            source.load()
            width, height = source.size
            converted = np.asarray(source.convert("RGB"))
        if converted.ndim != 3 or converted.shape[2] != 3:
            raise ValueError("RGB conversion did not produce three channels")
    except Exception as exc:
        raise ValueError(
            f"manifest row {record['row']} image cannot be decoded as RGB: {record['image']}: {exc}"
        ) from exc

    try:
        mask = _load_mask(record["mask"])
    except Exception as exc:
        raise ValueError(
            f"manifest row {record['row']} mask cannot be decoded: {record['mask']}: {exc}"
        ) from exc

    if mask.shape != (height, width):
        raise ValueError(
            f"manifest row {record['row']} image/mask dimensions differ: "
            f"image={(height, width)}, mask={mask.shape}"
        )
    if expected_image_size is not None and (height, width) != (
        expected_image_size,
        expected_image_size,
    ):
        raise ValueError(
            f"manifest row {record['row']} expected a square {expected_image_size}x"
            f"{expected_image_size} image, got {height}x{width}"
        )


def _read_cache(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {"version": _CACHE_VERSION, "entries": {}}
    if (
        not isinstance(value, dict)
        or value.get("version") != _CACHE_VERSION
        or not isinstance(value.get("entries"), dict)
    ):
        return {"version": _CACHE_VERSION, "entries": {}}
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary_name = handle.name
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _normalize_required_splits(required_splits: Sequence[str] | str) -> tuple[str, ...]:
    values = (required_splits,) if isinstance(required_splits, str) else required_splits
    normalized: list[str] = []
    for value in values:
        split = canonicalize_split(value)
        if split not in normalized:
            normalized.append(split)
    return tuple(normalized)


def _validated_manifest(
    manifest_path: str | Path,
    data_root: str | Path | None,
    *,
    split_policy: str | Path | Mapping[str, Any] | None,
    require_policy_coverage: bool | None,
    required_splits: Sequence[str] | str,
    full_decode: bool,
    expected_image_size: int | None,
    cache_path: str | Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"event manifest does not exist: {manifest}")
    root = Path(data_root).expanduser().resolve() if data_root is not None else manifest.parent
    if not root.is_dir():
        raise NotADirectoryError(f"data_root does not exist or is not a directory: {root}")
    if not isinstance(full_decode, bool):
        raise TypeError("full_decode must be a boolean")
    if expected_image_size is not None and (
        isinstance(expected_image_size, bool)
        or not isinstance(expected_image_size, int)
        or expected_image_size <= 0
    ):
        raise ValueError("expected_image_size must be a positive integer")

    manifest_hash = checksum_file(manifest, "sha256")
    policy = _read_policy(split_policy, manifest.parent, require_policy_coverage)
    required = _normalize_required_splits(required_splits)
    counts = {"train": 0, "val": 0, "test": 0}
    events: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    records: list[dict[str, Any]] = []
    seen_paths: dict[str, tuple[int, str]] = {}
    family_splits: dict[str, str] = {}
    family_names: dict[str, str] = {}

    try:
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError("event manifest is empty") from exc
            if not header or any(not name for name in header):
                raise ValueError("event manifest has a blank column name")
            if len(set(header)) != len(header):
                raise ValueError("event manifest has duplicate column names")
            missing = [name for name in _REQUIRED_COLUMNS if name not in header]
            if missing:
                raise ValueError(f"event manifest is missing required columns: {missing}")

            for row_number, values in enumerate(reader, start=2):
                if len(values) != len(header):
                    raise ValueError(
                        f"manifest row {row_number} has {len(values)} fields; "
                        f"expected {len(header)}"
                    )
                row = dict(zip(header, values))
                for column in _REQUIRED_COLUMNS:
                    if not row[column].strip():
                        raise ValueError(f"manifest row {row_number} has blank {column}")

                split = canonicalize_split(row["split"])
                source_event = row["event"].strip()
                source_key = _event_key(source_event)
                policy_family = policy["aliases"].get(source_key)
                if policy_family is None:
                    if policy["require_coverage"]:
                        raise ValueError(
                            f"manifest row {row_number} event {source_event!r} "
                            "is not in split policy"
                        )
                    family_key = source_key
                    family_name = family_names.setdefault(family_key, source_event)
                else:
                    family_key = policy_family
                    family = policy["families"][family_key]
                    family_name = family["name"]
                    if split != family["split"]:
                        raise ValueError(
                            f"manifest row {row_number} event {source_event!r} must be in "
                            f"{family['split']}, not {split}, according to split policy"
                        )

                previous_split = family_splits.setdefault(family_key, split)
                if previous_split != split:
                    raise ValueError(
                        f"event family {family_name!r} crosses splits {previous_split} and {split}"
                    )

                image_path = _resolve_data_path(row["image"].strip(), root, row_number, "image")
                mask_path = _resolve_data_path(row["mask"].strip(), root, row_number, "mask")
                for column, path in (("image", image_path), ("mask", mask_path)):
                    path_key = os.path.normcase(str(path))
                    previous = seen_paths.get(path_key)
                    if previous is not None:
                        raise ValueError(
                            f"manifest row {row_number} {column} duplicates row {previous[0]} "
                            f"{previous[1]} path: {path}"
                        )
                    seen_paths[path_key] = (row_number, column)

                records.append(
                    {
                        "image": str(image_path),
                        "mask": str(mask_path),
                        "split": split,
                        "event": family_name,
                        "source_event": source_event,
                        "id": image_path.stem,
                        "row": row_number,
                    }
                )
                counts[split] += 1
                events[split].add(family_name)
    except csv.Error as exc:
        raise ValueError(f"invalid CSV in event manifest {manifest}: {exc}") from exc

    if not records:
        raise ValueError("event manifest contains no data rows")
    if policy["require_coverage"]:
        missing_policy_families = sorted(
            (
                family["name"]
                for key, family in policy["families"].items()
                if key not in family_splits
            ),
            key=str.casefold,
        )
        if missing_policy_families:
            raise ValueError(
                f"event manifest is missing policy event families: {missing_policy_families}"
            )
    empty = [split for split in required if counts[split] == 0]
    if empty:
        raise ValueError(f"event manifest has empty required splits: {empty}")

    cache_hit = False
    decoded_pairs = 0
    resolved_cache: Path | None = None
    options = {
        "decoder_version": _CACHE_VERSION,
        "data_root": str(root),
        "expected_image_size": expected_image_size,
    }
    cache_key = _json_hash(
        {
            "manifest_sha256": manifest_hash,
            "policy_sha256": policy["sha256"],
            "validation_options": options,
        }
    )
    if full_decode:
        if cache_path is None:
            resolved_cache = manifest.with_suffix(manifest.suffix + ".validation-cache.json")
        else:
            resolved_cache = Path(cache_path).expanduser()
            if not resolved_cache.is_absolute():
                resolved_cache = manifest.parent / resolved_cache
            resolved_cache = resolved_cache.resolve()
        cache = _read_cache(resolved_cache)
        entry = cache["entries"].get(cache_key)
        if isinstance(entry, dict) and entry.get("decoded_pairs") == len(records):
            cache_hit = True
        else:
            for record in records:
                _decode_pair(record, expected_image_size)
                decoded_pairs += 1
            cache["entries"][cache_key] = {
                "manifest_sha256": manifest_hash,
                "policy_sha256": policy["sha256"],
                "validation_options": options,
                "decoded_pairs": len(records),
            }
            _write_json_atomic(resolved_cache, cache)

    report: dict[str, Any] = {
        "manifest": str(manifest),
        "data_root": str(root),
        "manifest_sha256": manifest_hash,
        "policy": policy["path"],
        "policy_sha256": policy["sha256"],
        "total": len(records),
        "counts": counts,
        "events": {split: sorted(names, key=str.casefold) for split, names in events.items()},
        "required_splits": list(required),
        "full_decode": full_decode,
        "decode_cache_hit": cache_hit,
        "decoded_pairs": decoded_pairs,
        "cache_path": str(resolved_cache) if resolved_cache is not None else None,
        "cache_key": cache_key if full_decode else None,
    }
    return records, report


def validate_event_manifest(
    manifest_path: str | Path,
    data_root: str | Path | None = None,
    *,
    split_policy: str | Path | Mapping[str, Any] | None = None,
    require_policy_coverage: bool | None = None,
    required_splits: Sequence[str] | str = ("train", "val", "test"),
    full_decode: bool = True,
    expected_image_size: int | None = None,
    cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate an event CSV and return a JSON-serializable report.

    CSV structure and file existence are checked on every call. Only successful
    image/mask decoding is cached. A split policy may define ``events`` as
    ``{"Canonical name": {"split": "train", "aliases": ["alias"]}}`` and set
    ``require_complete_coverage`` to true.
    """
    _, report = _validated_manifest(
        manifest_path,
        data_root,
        split_policy=split_policy,
        require_policy_coverage=require_policy_coverage,
        required_splits=required_splits,
        full_decode=full_decode,
        expected_image_size=expected_image_size,
        cache_path=cache_path,
    )
    return report


def load_event_manifest(
    manifest_path: str | Path,
    data_root: str | Path | None = None,
    *,
    split_policy: str | Path | Mapping[str, Any] | None = None,
    require_policy_coverage: bool | None = None,
    required_splits: Sequence[str] | str = ("train", "val", "test"),
    full_decode: bool = True,
    expected_image_size: int | None = None,
    cache_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load validated records and their JSON-serializable validation report."""
    return _validated_manifest(
        manifest_path,
        data_root,
        split_policy=split_policy,
        require_policy_coverage=require_policy_coverage,
        required_splits=required_splits,
        full_decode=full_decode,
        expected_image_size=expected_image_size,
        cache_path=cache_path,
    )


_CAS_MANIFEST_COLUMNS = ("image", "mask", "split", "event", "archive", "sensor", "gsd_m")


def _read_archive_config(
    archive_config: str | Path | Mapping[str, Any],
) -> tuple[dict[str, Any], str | Path | Mapping[str, Any]]:
    if isinstance(archive_config, Mapping):
        document = dict(archive_config)
        policy_source: str | Path | Mapping[str, Any] = document
    else:
        path = Path(archive_config).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CAS archive config does not exist: {path}")
        try:
            with path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read CAS archive config {path}: {exc}") from exc
        policy_source = path
    if not isinstance(document, dict):
        raise TypeError("CAS archive config must contain a JSON object")
    return document, policy_source


def _path_under_root(path: Path, root: Path, context: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{context} escapes dataset_root: {path}") from exc
    return resolved


def _resolve_archive_directory(raw: Any, root: Path, index: int) -> Path:
    if not isinstance(raw, (str, Path)):
        raise TypeError(f"archive config entry {index} directory must be a relative path")
    value = str(raw)
    if not value.strip() or "\x00" in value:
        raise ValueError(f"archive config entry {index} directory must be a relative path")
    portable = PurePosixPath(value.replace("\\", "/"))
    windows_path = PureWindowsPath(value)
    if portable.is_absolute() or windows_path.drive or ".." in portable.parts:
        raise ValueError(
            f"archive config entry {index} directory contains path traversal: {value!r}"
        )
    directory = _path_under_root(
        root.joinpath(*portable.parts), root, f"archive config entry {index} directory"
    )
    if not directory.is_dir():
        raise NotADirectoryError(
            f"archive config entry {index} directory does not exist: {directory}"
        )
    try:
        next(directory.iterdir())
    except StopIteration as exc:
        raise ValueError(f"archive config entry {index} directory is empty: {directory}") from exc
    return directory


def _find_cas_sample_root(directory: Path, dataset_root: Path) -> tuple[Path, Path, Path]:
    candidates: dict[str, tuple[Path, Path, Path]] = {}
    search = [directory]
    search.extend(path for path in directory.rglob("*") if path.is_dir())
    for candidate in search:
        image_dir = candidate / "img"
        mask_dir = candidate / "mask"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            continue
        sample_root = _path_under_root(candidate, dataset_root, "CAS sample root")
        image_dir = _path_under_root(image_dir, dataset_root, "CAS img directory")
        mask_dir = _path_under_root(mask_dir, dataset_root, "CAS mask directory")
        if image_dir == mask_dir:
            raise ValueError(f"CAS img and mask directories resolve to the same path: {image_dir}")
        candidates[os.path.normcase(str(sample_root))] = (sample_root, image_dir, mask_dir)

    ordered = sorted(candidates.values(), key=lambda item: (str(item[0]).casefold(), str(item[0])))
    if not ordered:
        raise ValueError(
            f"configured directory contains no CAS root with sibling img/mask directories: "
            f"{directory}"
        )
    if len(ordered) != 1:
        roots = [str(item[0]) for item in ordered]
        raise ValueError(
            f"configured directory contains multiple CAS roots with img/mask directories: {roots}"
        )
    return ordered[0]


def _index_cas_tiffs(directory: Path, dataset_root: Path, role: str) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    source_names: dict[str, str] = {}
    files = sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.casefold() in {".tif", ".tiff"}
        ),
        key=lambda path: (path.as_posix().casefold(), path.as_posix()),
    )
    if not files:
        raise ValueError(f"CAS {role} directory contains no .tif/.tiff files: {directory}")
    for path in files:
        resolved = _path_under_root(path, dataset_root, f"CAS {role} file")
        stem_key = path.stem.casefold()
        if stem_key in indexed:
            raise ValueError(
                f"CAS {role} directory has duplicate case-insensitive stem {path.stem!r}: "
                f"{source_names[stem_key]} and {path}"
            )
        indexed[stem_key] = resolved
        source_names[stem_key] = str(path)
    return indexed


def _cas_metadata(entry: Mapping[str, Any], key: str, *aliases: str) -> str:
    value: Any = None
    for name in (key, *aliases):
        if name in entry:
            value = entry[name]
            break
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        raise TypeError(f"archive metadata {key!r} must be a scalar value")
    return str(value).strip()


def build_cas_event_manifest(
    dataset_root: str | Path,
    archive_config: str | Path | Mapping[str, Any],
    output_path: str | Path,
    *,
    relative_paths: bool = True,
    validate: bool = True,
    full_decode: bool = False,
    validation_cache: str | Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic event manifest from extracted official CAS archives."""
    if not isinstance(relative_paths, bool) or not isinstance(validate, bool):
        raise TypeError("relative_paths and validate must be booleans")
    if not isinstance(full_decode, bool):
        raise TypeError("full_decode must be a boolean")
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"CAS dataset_root does not exist: {root}")
    config, policy_source = _read_archive_config(archive_config)
    archives = config.get("archives")
    if not isinstance(archives, list) or not archives:
        raise ValueError("CAS archive config requires a non-empty archives list")
    if validate and "events" not in config:
        raise ValueError("validated CAS archive config requires a split-policy events section")

    rows: list[dict[str, str]] = []
    declared_directories: dict[str, int] = {}
    selected_roots: dict[str, int] = {}
    used_files: dict[str, tuple[int, str]] = {}
    for index, entry in enumerate(archives, start=1):
        if not isinstance(entry, dict):
            raise TypeError(f"archive config entry {index} must be an object")
        missing = [name for name in ("directory", "event", "split") if name not in entry]
        if missing:
            raise ValueError(f"archive config entry {index} is missing required fields: {missing}")
        event = entry["event"]
        if not isinstance(event, str) or not event.strip():
            raise ValueError(f"archive config entry {index} event must be non-blank")
        event = event.strip()
        split = canonicalize_split(entry["split"])
        directory = _resolve_archive_directory(entry["directory"], root, index)
        directory_key = os.path.normcase(str(directory))
        previous = declared_directories.get(directory_key)
        if previous is not None:
            raise ValueError(
                f"archive config entries {previous} and {index} declare the same directory: "
                f"{directory}"
            )
        declared_directories[directory_key] = index

        sample_root, image_dir, mask_dir = _find_cas_sample_root(directory, root)
        sample_key = os.path.normcase(str(sample_root))
        previous = selected_roots.get(sample_key)
        if previous is not None:
            raise ValueError(
                f"archive config entries {previous} and {index} select the same CAS root: "
                f"{sample_root}"
            )
        selected_roots[sample_key] = index
        images = _index_cas_tiffs(image_dir, root, "img")
        masks = _index_cas_tiffs(mask_dir, root, "mask")
        if images.keys() != masks.keys():
            missing_masks = sorted(
                (images[key].stem for key in images.keys() - masks.keys()), key=str.casefold
            )
            missing_images = sorted(
                (masks[key].stem for key in masks.keys() - images.keys()), key=str.casefold
            )
            raise ValueError(
                f"CAS image/mask stems differ in {sample_root}: "
                f"missing masks={missing_masks}, missing images={missing_images}"
            )

        archive_name = _cas_metadata(entry, "archive")
        sensor = _cas_metadata(entry, "sensor")
        gsd_m = _cas_metadata(entry, "gsd_m", "gsd")
        for stem_key in sorted(images):
            image_path, mask_path = images[stem_key], masks[stem_key]
            for role, path in (("image", image_path), ("mask", mask_path)):
                file_key = os.path.normcase(str(path))
                prior = used_files.get(file_key)
                if prior is not None:
                    raise ValueError(
                        f"archive config entry {index} {role} reuses entry {prior[0]} "
                        f"{prior[1]} file: {path}"
                    )
                used_files[file_key] = (index, role)
            if relative_paths:
                image_value = image_path.relative_to(root).as_posix()
                mask_value = mask_path.relative_to(root).as_posix()
            else:
                image_value, mask_value = str(image_path), str(mask_path)
            rows.append(
                {
                    "image": image_value,
                    "mask": mask_value,
                    "split": split,
                    "event": event,
                    "archive": archive_name,
                    "sensor": sensor,
                    "gsd_m": gsd_m,
                }
            )

    rows.sort(
        key=lambda row: (
            row["image"].casefold(),
            row["image"],
            row["mask"].casefold(),
            row["mask"],
            row["event"].casefold(),
            row["event"],
            row["split"],
        )
    )
    counts = {"train": 0, "val": 0, "test": 0}
    events: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for row in rows:
        counts[row["split"]] += 1
        events[row["split"]].add(row["event"])

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not output.is_file():
        raise IsADirectoryError(f"CAS manifest output is not a file: {output}")
    cache_path = validation_cache
    if validate and full_decode and cache_path is None:
        cache_path = output.with_suffix(output.suffix + ".validation-cache.json")
    temporary_name: str | None = None
    validation_report: dict[str, Any] | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=output.parent,
            prefix=f".{output.name}.",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=_CAS_MANIFEST_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())

        temporary_path = Path(temporary_name)
        if validate:
            validation_report = validate_event_manifest(
                temporary_path,
                data_root=root,
                split_policy=policy_source,
                require_policy_coverage=True,
                required_splits=("train", "val", "test"),
                full_decode=full_decode,
                cache_path=cache_path,
            )
        os.replace(temporary_path, output)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass

    manifest_hash = checksum_file(output, "sha256")
    if validation_report is not None:
        validation_report["manifest"] = str(output)
    return {
        "manifest": str(output),
        "manifest_sha256": manifest_hash,
        "total": len(rows),
        "counts": counts,
        "events": {split: sorted(names, key=str.casefold) for split, names in events.items()},
        "archive_count": len(archives),
        "relative_paths": relative_paths,
        "validation": validation_report,
    }


def _resize_shape(resize: int | Sequence[int] | None) -> tuple[int, int] | None:
    if resize is None:
        return None
    if isinstance(resize, bool):
        raise TypeError("resize must be a positive integer or (height, width)")
    if isinstance(resize, int):
        shape = (resize, resize)
    elif isinstance(resize, Sequence) and not isinstance(resize, str) and len(resize) == 2:
        if any(isinstance(value, bool) or not isinstance(value, int) for value in resize):
            raise ValueError("resize dimensions must be positive integers")
        shape = (resize[0], resize[1])
    else:
        raise ValueError("resize must be a positive integer or (height, width)")
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("resize dimensions must be positive")
    return shape


def _resize_pair(
    image: np.ndarray, mask: np.ndarray, shape: tuple[int, int] | None
) -> tuple[np.ndarray, np.ndarray]:
    if shape is None or (image.shape[0], image.shape[1]) == shape:
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)
    from PIL import Image

    height, width = shape
    image = np.array(
        Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR),
        dtype=np.uint8,
        copy=True,
    )
    mask = np.array(
        Image.fromarray(mask.astype(np.uint8)).resize((width, height), Image.Resampling.NEAREST),
        dtype=np.uint8,
        copy=True,
    )
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class EventManifestDataset(Dataset):
    """RGB segmentation samples from a validated event CSV manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        *,
        data_root: str | Path | None = None,
        split_policy: str | Path | Mapping[str, Any] | None = None,
        require_policy_coverage: bool | None = None,
        required_splits: Sequence[str] | str = ("train", "val", "test"),
        full_validation: bool = True,
        expected_image_size: int | None = None,
        cache_path: str | Path | None = None,
        augment: bool = False,
        resize: int | Sequence[int] | None = None,
        limit: int | None = None,
        seed: int = 42,
    ) -> None:
        records, report = load_event_manifest(
            manifest_path,
            data_root,
            split_policy=split_policy,
            require_policy_coverage=require_policy_coverage,
            required_splits=required_splits,
            full_decode=full_validation,
            expected_image_size=expected_image_size,
            cache_path=cache_path,
        )
        self.split = canonicalize_split(split)
        self.records = [record for record in records if record["split"] == self.split]
        if not self.records:
            raise ValueError(f"manifest split {self.split!r} is empty")
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                raise ValueError("limit must be a non-negative integer")
            if limit > 0:
                self.records = self.records[:limit]
        self.validation_report = report
        self.augment = bool(augment)
        self.resize = _resize_shape(resize)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def set_worker_seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = percentile_stretch(load_rgb(record["image"]))
        mask = _load_mask(record["mask"])
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"image/mask dimensions changed after validation for {record['image']}"
            )
        if self.augment:
            image, mask = _augment(image, mask, self.rng)
        image, mask = _resize_pair(image, mask, self.resize)
        return {
            "image": rgb_to_tensor(image),
            "mask": torch.from_numpy(mask.astype(np.float32, copy=False)).unsqueeze(0),
            "id": record["id"],
            "path": record["image"],
            "event": record["event"],
        }


def _array_layouts(images: np.ndarray, masks: np.ndarray) -> tuple[str, str, tuple[int, int], int]:
    if images.ndim != 4:
        raise ValueError(f"HR-GLDD images must be four-dimensional, got {images.shape}")
    if masks.ndim not in (3, 4):
        raise ValueError(f"HR-GLDD masks must have 3 or 4 dimensions, got {masks.shape}")
    if images.shape[0] == 0:
        raise ValueError("HR-GLDD arrays are empty")
    if images.shape[0] != masks.shape[0]:
        raise ValueError(
            f"HR-GLDD image/mask sample counts differ: {images.shape[0]} != {masks.shape[0]}"
        )

    image_candidates: list[tuple[str, int, int, int]] = []
    if images.shape[-1] in (3, 4):
        image_candidates.append(("hwc", images.shape[1], images.shape[2], images.shape[-1]))
    if images.shape[1] in (3, 4):
        image_candidates.append(("chw", images.shape[2], images.shape[3], images.shape[1]))
    if not image_candidates:
        raise ValueError("HR-GLDD images must use NHWC or NCHW layout with 3 or 4 channels")

    mask_candidates: list[tuple[str, int, int]] = []
    if masks.ndim == 3:
        mask_candidates.append(("hw", masks.shape[1], masks.shape[2]))
    else:
        if masks.shape[-1] == 1:
            mask_candidates.append(("hwc", masks.shape[1], masks.shape[2]))
        if masks.shape[1] == 1:
            mask_candidates.append(("chw", masks.shape[2], masks.shape[3]))
    if not mask_candidates:
        raise ValueError("four-dimensional HR-GLDD masks must have one channel")

    matches = [
        (image_layout, mask_layout, (height, width), channels)
        for image_layout, height, width, channels in image_candidates
        for mask_layout, mask_height, mask_width in mask_candidates
        if (height, width) == (mask_height, mask_width)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"HR-GLDD image/mask layouts are incompatible or ambiguous: "
            f"images={images.shape}, masks={masks.shape}"
        )
    return matches[0]


def _image_to_uint8(image: np.ndarray, context: str) -> np.ndarray:
    if np.issubdtype(image.dtype, np.bool_) or not (
        np.issubdtype(image.dtype, np.integer) or np.issubdtype(image.dtype, np.floating)
    ):
        raise ValueError(f"{context} must contain integer or floating-point pixels")
    pixels = np.asarray(image)
    if pixels.size == 0:
        raise ValueError(f"{context} is empty")
    minimum = float(np.min(pixels))
    maximum = float(np.max(pixels))
    if not np.isfinite(minimum) or not np.isfinite(maximum) or minimum < 0:
        raise ValueError(f"{context} contains negative or non-finite pixels")
    pixels = pixels.astype(np.float32)
    if np.issubdtype(image.dtype, np.floating) and maximum <= 1.0 + 1e-6:
        pixels *= 255.0
    elif maximum > 255.0 + 1e-6:
        raise ValueError(f"{context} pixels must be in [0,1] or [0,255], got max {maximum}")
    return np.rint(np.clip(pixels, 0.0, 255.0)).astype(np.uint8)


class HRGLDDDataset(Dataset):
    """Memory-mapped official HR-GLDD train/val/test NumPy arrays."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        augment: bool = False,
        resize: int | Sequence[int] | None = None,
        limit: int | None = None,
        seed: int = 42,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"HR-GLDD root does not exist: {self.root}")
        self.split = canonicalize_split(split)
        prefix = "val" if self.split == "val" else self.split
        self.image_path = self.root / f"{prefix}X.npy"
        self.mask_path = self.root / f"{prefix}Y.npy"
        for path in (self.image_path, self.mask_path):
            if not path.is_file():
                raise FileNotFoundError(f"missing official HR-GLDD array: {path}")
        try:
            self.images = np.load(self.image_path, mmap_mode="r", allow_pickle=False)
            self.masks = np.load(self.mask_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"could not memory-map HR-GLDD arrays: {exc}") from exc

        self.image_layout, self.mask_layout, self.spatial_shape, self.channels = _array_layouts(
            self.images, self.masks
        )
        all_mask_values: set[Any] = set()
        for start in range(0, self.masks.shape[0], 64):
            all_mask_values.update(np.unique(self.masks[start : start + 64]).tolist())
        _validate_binary_values(all_mask_values, f"HR-GLDD masks {self.mask_path}")

        count = int(self.images.shape[0])
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                raise ValueError("limit must be a non-negative integer")
            if limit > 0:
                count = min(count, limit)
        self.count = count
        self.augment = bool(augment)
        self.resize = _resize_shape(resize)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def set_worker_seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("images", None)
        state.pop("masks", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.images = np.load(self.image_path, mmap_mode="r", allow_pickle=False)
        self.masks = np.load(self.mask_path, mmap_mode="r", allow_pickle=False)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += self.count
        if index < 0 or index >= self.count:
            raise IndexError(index)
        raw_image = np.asarray(self.images[index])
        if self.image_layout == "chw":
            # Channel four is NIR in the official arrays and is intentionally discarded.
            image = np.transpose(raw_image[:3], (1, 2, 0))
        else:
            image = raw_image[..., :3]
        image = percentile_stretch(
            _image_to_uint8(image, f"HR-GLDD image {self.image_path}[{index}]")
        )

        raw_mask = np.asarray(self.masks[index])
        if self.mask_layout == "chw":
            raw_mask = raw_mask[0]
        elif self.mask_layout == "hwc":
            raw_mask = raw_mask[..., 0]
        _validate_binary_values(raw_mask, f"HR-GLDD mask {self.mask_path}[{index}]")
        mask = (raw_mask > 0).astype(np.uint8)
        if self.augment:
            image, mask = _augment(image, mask, self.rng)
        image, mask = _resize_pair(image, mask, self.resize)
        return {
            "image": rgb_to_tensor(image),
            "mask": torch.from_numpy(mask.astype(np.float32, copy=False)).unsqueeze(0),
            "id": f"{self.split}-{index:06d}",
            "path": str(self.image_path),
            "event": "HR-GLDD-external",
        }


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    info = get_worker_info()
    if info is not None and hasattr(info.dataset, "set_worker_seed"):
        info.dataset.set_worker_seed(worker_seed)


def build_research_dataloader(
    data_cfg: Mapping[str, Any],
    split: str,
    *,
    batch_size: int | None = None,
    augment: bool | None = None,
    num_workers: int | None = None,
    limit: int | None = None,
    seed: int | None = None,
    shuffle: bool | None = None,
) -> DataLoader:
    """Build a deterministic event-manifest or HR-GLDD data loader from config."""
    if not isinstance(data_cfg, Mapping):
        raise TypeError("data_cfg must be a mapping")
    cfg = dict(data_cfg)
    canonical_split = canonicalize_split(split)
    seed = int(cfg.get("seed", 42) if seed is None else seed)
    batch_size = int(cfg.get("batch_size", 16) if batch_size is None else batch_size)
    num_workers = int(cfg.get("num_workers", 0) if num_workers is None else num_workers)
    augment = bool(cfg.get("augment", False) if augment is None else augment)
    limit = cfg.get("limit") if limit is None else limit
    if limit == 0:
        limit = None
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")

    kind = str(cfg.get("type", cfg.get("kind", cfg.get("dataset", "manifest")))).strip().casefold()
    if kind in {"hrgldd", "hr-gldd", "hr_gldd"}:
        root = cfg.get("root", cfg.get("data_root"))
        if root is None:
            raise KeyError("HR-GLDD data config requires root")
        dataset: Dataset = HRGLDDDataset(
            root,
            canonical_split,
            augment=augment,
            resize=cfg.get("resize", cfg.get("image_size")),
            limit=limit,
            seed=seed,
        )
    elif kind in {"manifest", "event_manifest", "events"}:
        manifest = cfg.get("manifest", cfg.get("manifest_path"))
        if manifest is None:
            raise KeyError("event data config requires manifest")
        dataset = EventManifestDataset(
            manifest,
            canonical_split,
            data_root=cfg.get("data_root", cfg.get("root")),
            split_policy=cfg.get("split_policy", cfg.get("policy")),
            require_policy_coverage=cfg.get("require_policy_coverage"),
            required_splits=cfg.get("required_splits", ("train", "val", "test")),
            full_validation=bool(cfg.get("full_validation", True)),
            expected_image_size=cfg.get("expected_image_size"),
            cache_path=cfg.get("validation_cache", cfg.get("cache_path")),
            augment=augment,
            resize=cfg.get("resize", cfg.get("image_size")),
            limit=limit,
            seed=seed,
        )
    else:
        raise ValueError(f"unknown research dataset type {kind!r}")

    if shuffle is None:
        shuffle = bool(cfg.get("shuffle", canonical_split == "train" and augment))
    generator = torch.Generator()
    generator.manual_seed(seed)
    pin_memory = bool(cfg.get("pin_memory", torch.cuda.is_available()))
    persistent_workers = bool(cfg.get("persistent_workers", num_workers > 0))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=bool(cfg.get("drop_last", False)),
        persistent_workers=persistent_workers if num_workers > 0 else False,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
