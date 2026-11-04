#!/usr/bin/env python3
"""Statically validate the committed Colab notebook and dataset ledgers."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REQUIRED_NOTEBOOK_TEXT = (
    "CAS Landslide Dataset",
    "event-disjoint",
    "SegFormer-B0",
    "MobileNetV3/LR-ASPP",
    "hard-negative region contrast",
    "validate_event_manifest",
    "run_training",
    "evaluate_external_hr_gldd",
    "export_onnx_with_parity",
    "quantize_onnx_dynamic",
    "write_artifact_inventory",
)

SECRET_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"ghp_[0-9A-Za-z]{20,}"),
    re.compile(r"github_pat_[0-9A-Za-z_]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object at {path}")
    return value


def _validate_cas_ledger(path: Path) -> None:
    ledger = _load_json(path)
    events = ledger.get("events")
    archives = ledger.get("archives")
    if not isinstance(events, dict) or len(events) != 9:
        raise ValueError("CAS ledger must define exactly nine canonical event families")
    if not isinstance(archives, list) or len(archives) != 16:
        raise ValueError("CAS ledger must define all 16 official data archives")
    if ledger.get("require_complete_coverage") is not True:
        raise ValueError("CAS ledger must require complete event coverage")

    roles = {"train": set(), "val": set(), "test": set()}
    directories: set[str] = set()
    for name, event in events.items():
        if not isinstance(event, dict) or event.get("split") not in roles:
            raise ValueError(f"invalid CAS event policy for {name!r}")
        roles[event["split"]].add(name)
    if not all(roles.values()):
        raise ValueError("CAS policy requires non-empty train, val, and test event roles")

    event_names = set(events)
    for item in archives:
        if not isinstance(item, dict):
            raise TypeError("CAS archive entries must be objects")
        required = {"archive", "directory", "event", "split", "url", "bytes", "md5"}
        missing = required.difference(item)
        if missing:
            raise ValueError(f"CAS archive entry is missing {sorted(missing)}")
        if item["event"] not in event_names:
            raise ValueError(f"CAS archive uses unknown event {item['event']!r}")
        if item["split"] != events[item["event"]]["split"]:
            raise ValueError(f"CAS archive split conflicts with event policy: {item['archive']}")
        if item["directory"] in directories:
            raise ValueError(f"duplicate CAS extraction directory: {item['directory']}")
        directories.add(item["directory"])
        if not str(item["url"]).startswith("https://zenodo.org/"):
            raise ValueError(f"CAS archive is not an official HTTPS Zenodo URL: {item['url']}")
        if not isinstance(item["bytes"], int) or item["bytes"] <= 0:
            raise ValueError(f"invalid CAS archive byte count: {item['archive']}")
        if re.fullmatch(r"[0-9a-f]{32}", str(item["md5"])) is None:
            raise ValueError(f"invalid CAS archive MD5: {item['archive']}")


def _validate_hr_ledger(path: Path) -> None:
    ledger = _load_json(path)
    files = ledger.get("files")
    expected = {
        "trainX.npy",
        "trainY.npy",
        "valX.npy",
        "valY.npy",
        "testX.npy",
        "testY.npy",
    }
    if not isinstance(files, list) or {item.get("name") for item in files} != expected:
        raise ValueError("HR-GLDD ledger must contain the six official NumPy arrays")
    for item in files:
        if not str(item.get("url", "")).startswith("https://zenodo.org/"):
            raise ValueError(f"HR-GLDD file is not an official HTTPS Zenodo URL: {item}")
        if not isinstance(item.get("bytes"), int) or item["bytes"] <= 0:
            raise ValueError(f"invalid HR-GLDD byte count: {item}")
        if re.fullmatch(r"[0-9a-f]{32}", str(item.get("md5", ""))) is None:
            raise ValueError(f"invalid HR-GLDD MD5: {item}")


def validate_notebook(path: str | Path) -> dict[str, int]:
    notebook_path = Path(path)
    notebook = _load_json(notebook_path)
    if notebook.get("nbformat") != 4:
        raise ValueError("notebook must use nbformat 4")
    if notebook.get("metadata", {}).get("accelerator") != "GPU":
        raise ValueError("notebook metadata must request a GPU accelerator")
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("notebook must contain cells")

    text = "\n".join("".join(cell.get("source", [])) for cell in cells if isinstance(cell, dict))
    missing = [token for token in REQUIRED_NOTEBOOK_TEXT if token not in text]
    if missing:
        raise ValueError(f"notebook is missing required workflow references: {missing}")
    if re.search(r"[A-Za-z]:\\\\", text):
        raise ValueError("notebook contains a hard-coded Windows path")
    code_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in cells
        if isinstance(cell, dict) and cell.get("cell_type") == "code"
    )
    if re.search(r"SyntheticLandslide|synthetic[_-]?(?:data|dataset)", code_text, re.IGNORECASE):
        raise ValueError("scientific Colab notebook code must not use synthetic data")
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            raise ValueError(f"notebook contains a credential-like value: {pattern.pattern}")

    code_cells = 0
    parameter_cells = 0
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict) or cell.get("cell_type") not in {"code", "markdown"}:
            raise ValueError(f"cell {index} has an unsupported type")
        if cell["cell_type"] != "code":
            continue
        code_cells += 1
        if cell.get("execution_count") is not None:
            raise ValueError(f"code cell {index} has a committed execution count")
        if cell.get("outputs") not in ([], None):
            raise ValueError(f"code cell {index} has committed outputs")
        if "parameters" in cell.get("metadata", {}).get("tags", []):
            parameter_cells += 1
        source = "".join(cell.get("source", []))
        try:
            compile(source, f"{notebook_path.name}:cell-{index}", "exec")
        except SyntaxError as exc:
            raise ValueError(f"code cell {index} does not compile: {exc}") from exc
    if parameter_cells != 1:
        raise ValueError(
            f"notebook must contain exactly one parameter cell, found {parameter_cells}"
        )

    repository = notebook_path.parent.parent
    _validate_cas_ledger(repository / "ml/configs/cas_colab.json")
    _validate_hr_ledger(repository / "ml/configs/hr_gldd_colab.json")
    return {"cells": len(cells), "code_cells": code_cells, "parameter_cells": parameter_cells}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "notebook",
        nargs="?",
        type=Path,
        default=Path("notebooks/Landslide_Training_Colab.ipynb"),
    )
    args = parser.parse_args()
    report = validate_notebook(args.notebook)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
