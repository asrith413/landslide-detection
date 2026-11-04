"""Reusable model deployment, validation, and benchmarking tools."""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import inspect
import io
import json
import math
import platform as platform_module
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


def _jsonable(value: Any) -> Any:
    """Convert common scientific Python values to strict JSON-compatible values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nvidia_smi_report() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {
            "available": False,
            "command": None,
            "returncode": None,
            "output": None,
            "error": None,
        }

    command = [
        executable,
        "--query-gpu=index,name,driver_version,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "command": command,
            "returncode": None,
            "output": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "available": completed.returncode == 0,
        "command": command,
        "returncode": int(completed.returncode),
        "output": completed.stdout.strip() or None,
        "error": completed.stderr.strip() or None,
    }


def collect_runtime_report(require_cuda: bool = False) -> dict[str, Any]:
    """Collect a JSON-serializable Python, PyTorch, and accelerator report.

    ``require_cuda`` fails explicitly when the installed PyTorch build or the
    current runtime cannot expose a CUDA device.
    """
    cuda_available = bool(torch.cuda.is_available())
    cuda_built = bool(
        torch.backends.cuda.is_built() if hasattr(torch.backends, "cuda") else torch.version.cuda
    )
    if require_cuda and not cuda_available:
        build = str(torch.version.cuda) if torch.version.cuda is not None else "CPU-only"
        raise RuntimeError(
            "CUDA is required but torch.cuda.is_available() is False "
            f"(PyTorch {torch.__version__}, CUDA build: {build}). Check the PyTorch "
            "installation, NVIDIA driver, and GPU visibility."
        )

    try:
        torchvision = importlib.import_module("torchvision")
        torchvision_report = {
            "available": True,
            "version": str(getattr(torchvision, "__version__", "unknown")),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - binary mismatches raise several exception types
        torchvision_report = {
            "available": False,
            "version": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    gpus: list[dict[str, Any]] = []
    cuda_error: str | None = None
    device_count = 0
    if cuda_available:
        try:
            device_count = int(torch.cuda.device_count())
            for index in range(device_count):
                properties = torch.cuda.get_device_properties(index)
                free_bytes: int | None = None
                total_bytes = int(properties.total_memory)
                memory_error: str | None = None
                try:
                    with torch.cuda.device(index):
                        free, total = torch.cuda.mem_get_info()
                    free_bytes = int(free)
                    total_bytes = int(total)
                except Exception as exc:  # noqa: BLE001 - driver failures are backend-specific
                    memory_error = f"{type(exc).__name__}: {exc}"
                gpus.append(
                    {
                        "index": index,
                        "name": str(properties.name),
                        "compute_capability": [int(properties.major), int(properties.minor)],
                        "total_vram_bytes": total_bytes,
                        "free_vram_bytes": free_bytes,
                        "memory_query_error": memory_error,
                    }
                )
        except Exception as exc:  # noqa: BLE001 - CUDA driver failures are backend-specific
            cuda_error = f"{type(exc).__name__}: {exc}"

    cudnn_available = bool(torch.backends.cudnn.is_available())
    cudnn_version = torch.backends.cudnn.version() if cudnn_available else None
    report = {
        "python": {
            "version": platform_module.python_version(),
            "implementation": platform_module.python_implementation(),
            "executable": sys.executable,
            "full_version": sys.version,
        },
        "platform": {
            "description": platform_module.platform(),
            "system": platform_module.system(),
            "release": platform_module.release(),
            "machine": platform_module.machine(),
            "processor": platform_module.processor(),
        },
        "torch": {
            "version": str(torch.__version__),
            "git_version": getattr(torch.version, "git_version", None),
        },
        "torchvision": torchvision_report,
        "cuda": {
            "built": cuda_built,
            "build_version": torch.version.cuda,
            "hip_build_version": getattr(torch.version, "hip", None),
            "runtime_available": cuda_available,
            "device_count": device_count,
            "gpu_names": [gpu["name"] for gpu in gpus],
            "devices": gpus,
            "error": cuda_error,
        },
        "cudnn": {
            "available": cudnn_available,
            "version": int(cudnn_version) if cudnn_version is not None else None,
        },
        "nvidia_smi": _nvidia_smi_report(),
    }
    return _jsonable(report)


def _resolve_torch_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {resolved} was requested but torch.cuda.is_available() is False"
        )
    return resolved


def _offline_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    safe_config = copy.deepcopy(dict(config))
    model_config = safe_config.get("model")
    if isinstance(model_config, Mapping):
        safe_config["model"] = dict(model_config)
        safe_config["model"]["pretrained"] = False
    else:
        safe_config["pretrained"] = False
    return safe_config


def load_checkpoint_model(
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    """Load the repository checkpoint contract without downloading encoder weights.

    Returns ``(model, original_config, checkpoint_payload)``. The original config is
    preserved for reporting while a copied config with ``pretrained=False`` is used
    to construct the model before loading the complete checkpoint state.
    """
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    load_kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    payload = torch.load(checkpoint_path, **load_kwargs)
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint {checkpoint_path} must contain a mapping")
    if not isinstance(payload.get("config"), Mapping):
        raise KeyError(f"checkpoint {checkpoint_path} does not contain a mapping 'config'")
    if "model_state" not in payload:
        raise KeyError(f"checkpoint {checkpoint_path} does not contain 'model_state'")

    original_config = copy.deepcopy(dict(payload["config"]))
    safe_config = _offline_model_config(original_config)

    # Import at call time so this remains compatible with concurrent build_model API work.
    model_module = importlib.import_module(".model", package=__package__)
    build_model = model_module.build_model
    signature = inspect.signature(build_model)
    positional_args: list[Any] = [safe_config]
    build_kwargs: dict[str, Any] = {}
    for name in (
        "pretrained_override",
        "pretrained",
        "load_pretrained",
        "download_pretrained",
    ):
        parameter = signature.parameters.get(name)
        if parameter is None:
            continue
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional_args.append(False)
        else:
            build_kwargs[name] = False
        break

    model = build_model(*positional_args, **build_kwargs)
    model.load_state_dict(payload["model_state"])
    model.to(_resolve_torch_device(device)).eval()
    return model, original_config, dict(payload)


def _require_module(name: str, install_name: str) -> Any:
    try:
        return importlib.import_module(name)
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"{name} is required for this deployment operation; install it with "
            f"`pip install {install_name}`"
        ) from exc


def _image_shape(image_size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(image_size, int) and not isinstance(image_size, bool):
        height = width = image_size
    else:
        values = list(image_size)
        if len(values) != 2:
            raise ValueError("image_size must be an integer or a (height, width) pair")
        height, width = int(values[0]), int(values[1])
    if height <= 0 or width <= 0:
        raise ValueError("image_size dimensions must be positive")
    return height, width


def _model_normalization(model: nn.Module) -> tuple[list[float], list[float]]:
    default_mean = [0.485, 0.456, 0.406]
    default_std = [0.229, 0.224, 0.225]
    mean = getattr(model, "rgb_mean", None)
    std = getattr(model, "rgb_std", None)
    if isinstance(mean, Tensor) and mean.numel() == 3:
        default_mean = [float(value) for value in mean.detach().cpu().flatten()]
    if isinstance(std, Tensor) and std.numel() == 3:
        default_std = [float(value) for value in std.detach().cpu().flatten()]
    return default_mean, default_std


def _prepare_rgb_sample(
    sample: Tensor | np.ndarray | None,
    height: int,
    width: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> Tensor:
    if sample is None:
        generator = torch.Generator(device="cpu").manual_seed(0)
        rgb = torch.rand((1, 3, height, width), generator=generator)
        mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        tensor = (rgb - mean_tensor) / std_tensor
    elif isinstance(sample, Tensor):
        tensor = sample.detach().to(device="cpu", dtype=torch.float32)
    else:
        tensor = torch.as_tensor(np.asarray(sample), dtype=torch.float32)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    expected = (1, 3, height, width)
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"sample must have static NCHW shape {expected}, got {tuple(tensor.shape)}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("sample contains non-finite values")
    return tensor.contiguous()


def _preprocessing_contract(
    height: int,
    width: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> dict[str, Any]:
    return {
        "source": {"color_space": "RGB", "layout": "HWC"},
        "percentile_stretch": {
            "scope": "per-channel",
            "low_percentile": 2.0,
            "high_percentile": 98.0,
            "output_dtype": "uint8",
            "output_range": [0, 255],
        },
        "tensor_conversion": {
            "layout": "NCHW",
            "dtype": "float32",
            "scale": 1.0 / 255.0,
        },
        "normalization": {"mean": list(mean), "std": list(std)},
        "model_input": {
            "name": "image",
            "shape": [1, 3, height, width],
            "layout": "NCHW",
            "channels": "RGB",
            "dynamic_axes": False,
        },
        "tiling": {"edge_padding": "replicate"},
    }


def _onnx_input_shape(model_proto: Any, input_name: str) -> list[int | None]:
    graph_input = next((item for item in model_proto.graph.input if item.name == input_name), None)
    if graph_input is None:
        raise RuntimeError(f"exported ONNX graph has no input named {input_name!r}")
    dimensions: list[int | None] = []
    for dimension in graph_input.type.tensor_type.shape.dim:
        dimensions.append(int(dimension.dim_value) if dimension.HasField("dim_value") else None)
    return dimensions


def _sidecar_path(path: Path) -> Path:
    candidate = path.with_suffix(".json")
    if candidate == path:
        candidate = path.with_name(path.name + ".report.json")
    return candidate


def export_onnx_with_parity(
    checkpoint: str | Path,
    out: str | Path,
    image_size: int | Sequence[int],
    opset: int = 18,
    sample: Tensor | np.ndarray | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict[str, Any]:
    """Export one static RGB tile and verify ONNX Runtime CPU parity.

    A JSON sidecar is always written after parity is measured. A parity mismatch
    writes a failed report and raises ``RuntimeError``.
    """
    if opset <= 0:
        raise ValueError("opset must be positive")
    if atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be non-negative")
    onnx = _require_module("onnx", "onnx")
    ort = _require_module("onnxruntime", "onnxruntime")
    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime CPUExecutionProvider is not available")

    height, width = _image_shape(image_size)
    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model, config, checkpoint_payload = load_checkpoint_model(checkpoint, device="cpu")
    model_module = importlib.import_module(".model", package=__package__)
    deployment_model = model_module.DeploymentModel(model).eval()
    mean, std = _model_normalization(model)
    input_tensor = _prepare_rgb_sample(sample, height, width, mean, std)

    with torch.inference_mode():
        torch_output = deployment_model(input_tensor).detach().cpu().numpy()

    export_signature = inspect.signature(torch.onnx.export)
    export_kwargs: dict[str, Any] = {
        "input_names": ["image"],
        "output_names": ["probability"],
        "opset_version": int(opset),
        "do_constant_folding": True,
        "dynamic_axes": None,
    }
    if "external_data" in export_signature.parameters:
        export_kwargs["external_data"] = False

    exporter = "legacy"
    if "dynamo" in export_signature.parameters:
        # The modern exporter requires onnxscript, which is not a core dependency.
        use_dynamo = importlib.util.find_spec("onnxscript") is not None
        export_kwargs["dynamo"] = use_dynamo
        exporter = "dynamo" if use_dynamo else "legacy"

    try:
        with torch.inference_mode():
            torch.onnx.export(
                deployment_model,
                (input_tensor,),
                str(output_path),
                **export_kwargs,
            )
    except Exception as first_error:
        if export_kwargs.get("dynamo") is not True:
            raise RuntimeError(
                f"ONNX export failed with the legacy exporter: {first_error}"
            ) from first_error
        export_kwargs["dynamo"] = False
        exporter = "legacy-fallback"
        try:
            with torch.inference_mode():
                torch.onnx.export(
                    deployment_model,
                    (input_tensor,),
                    str(output_path),
                    **export_kwargs,
                )
        except Exception as second_error:
            raise RuntimeError(
                "ONNX export failed with both dynamo and legacy exporters: "
                f"dynamo={first_error}; legacy={second_error}"
            ) from second_error

    if not output_path.is_file():
        raise RuntimeError(f"ONNX export did not create {output_path}")
    model_proto = onnx.load(str(output_path), load_external_data=False)
    onnx.checker.check_model(model_proto)
    expected_shape = [1, 3, height, width]
    actual_shape = _onnx_input_shape(model_proto, "image")
    if actual_shape != expected_shape:
        raise RuntimeError(
            f"ONNX input must be static {expected_shape}, exported graph has {actual_shape}"
        )
    external_initializers = [
        initializer.name
        for initializer in model_proto.graph.initializer
        if initializer.data_location == onnx.TensorProto.EXTERNAL or initializer.external_data
    ]
    if external_initializers:
        raise RuntimeError(
            "ONNX export references external tensor data despite single-file export: "
            + ", ".join(external_initializers[:5])
        )

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    ort_outputs = session.run(None, {session.get_inputs()[0].name: input_tensor.numpy()})
    if len(ort_outputs) != 1:
        raise RuntimeError(f"deployment ONNX model must return one output, got {len(ort_outputs)}")
    ort_output = np.asarray(ort_outputs[0])
    same_shape = ort_output.shape == torch_output.shape
    finite = bool(np.isfinite(torch_output).all() and np.isfinite(ort_output).all())
    if same_shape and finite:
        absolute_error = np.abs(torch_output.astype(np.float64) - ort_output.astype(np.float64))
        max_abs_error: float | None = float(absolute_error.max())
        mean_abs_error: float | None = float(absolute_error.mean())
        parity_passed = bool(np.allclose(torch_output, ort_output, atol=atol, rtol=rtol))
    else:
        max_abs_error = None
        mean_abs_error = None
        parity_passed = False

    report_path = _sidecar_path(output_path)
    report = {
        "status": "success" if parity_passed else "failed",
        "checkpoint": str(checkpoint),
        "onnx_model": str(output_path),
        "report": str(report_path),
        "sha256": _sha256_file(output_path),
        "size_bytes": int(output_path.stat().st_size),
        "opset": int(opset),
        "opset_imports": {
            item.domain or "ai.onnx": int(item.version) for item in model_proto.opset_import
        },
        "exporter": exporter,
        "single_file": True,
        "onnx_checker_passed": True,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "config": config,
        "checkpoint_metadata": {
            "epoch": checkpoint_payload.get("epoch"),
            "best_dice": checkpoint_payload.get("best_dice"),
        },
        "input": {"name": "image", "shape": expected_shape, "dtype": "float32"},
        "output": {
            "name": "probability",
            "shape": list(ort_output.shape),
            "dtype": str(ort_output.dtype),
            "range": [0.0, 1.0],
            "threshold": 0.5,
        },
        "preprocessing": _preprocessing_contract(height, width, mean, std),
        "providers": session.get_providers(),
        "torch_output_shape": list(torch_output.shape),
        "onnx_output_shape": list(ort_output.shape),
        "outputs_finite": finite,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "atol": float(atol),
        "rtol": float(rtol),
        "parity_passed": parity_passed,
    }
    report = _jsonable(report)
    _write_json(report, report_path)
    if not parity_passed:
        raise RuntimeError(
            "ONNX Runtime parity failed "
            f"(shapes {torch_output.shape} vs {ort_output.shape}, "
            f"max_abs_error={max_abs_error}, atol={atol}, rtol={rtol}); "
            f"see {report_path}"
        )
    return report


def _numpy_sample(sample: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(sample, Tensor):
        array = sample.detach().cpu().numpy()
    else:
        array = np.asarray(sample)
    array = np.ascontiguousarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("sample contains non-finite values")
    return array


def quantize_onnx_dynamic(
    float_model: str | Path,
    quantized_model: str | Path,
    sample: Tensor | np.ndarray,
    tolerance: float = 0.02,
) -> dict[str, Any]:
    """Dynamically quantize an ONNX model and validate its CPU output.

    Unsupported quantization operators or models produce a persisted ``skipped``
    report with the real exception. Numeric results outside ``tolerance`` produce
    a failed report and raise rather than accepting a low-quality artifact.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    ort = _require_module("onnxruntime", "onnxruntime")
    onnx = _require_module("onnx", "onnx")
    try:
        quantization = importlib.import_module("onnxruntime.quantization")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "onnxruntime.quantization is required; install it with `pip install onnxruntime`"
        ) from exc

    float_path = Path(float_model)
    quantized_path = Path(quantized_model)
    if not float_path.is_file():
        raise FileNotFoundError(f"float ONNX model not found: {float_path}")
    if float_path.resolve() == quantized_path.resolve():
        raise ValueError("float_model and quantized_model must be different paths")
    quantized_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = _sidecar_path(quantized_path)
    input_array = _numpy_sample(sample)

    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime CPUExecutionProvider is not available")
    float_session = ort.InferenceSession(str(float_path), providers=["CPUExecutionProvider"])
    if len(float_session.get_inputs()) != 1:
        raise ValueError("dynamic quantization parity requires a model with one input")
    float_outputs = float_session.run(None, {float_session.get_inputs()[0].name: input_array})
    if len(float_outputs) != 1:
        raise ValueError("dynamic quantization parity requires a model with one output")
    float_output = np.asarray(float_outputs[0])
    if not np.isfinite(float_output).all():
        raise RuntimeError("float ONNX model produced non-finite output")

    base_report: dict[str, Any] = {
        "float_model": str(float_path),
        "quantized_model": str(quantized_path),
        "report": str(report_path),
        "weight_type": "QInt8",
        "tolerance": float(tolerance),
        "float_size_bytes": int(float_path.stat().st_size),
        "float_sha256": _sha256_file(float_path),
        "sample_shape": list(input_array.shape),
        "provider": "CPUExecutionProvider",
    }

    def skipped(phase: str, reason: str, exc: Exception | None = None) -> dict[str, Any]:
        exists = quantized_path.is_file()
        report = {
            **base_report,
            "status": "skipped",
            "skipped": True,
            "phase": phase,
            "reason": reason,
            "error_type": type(exc).__name__ if exc is not None else None,
            "error": str(exc) if exc is not None else None,
            "artifact_usable": False,
            "quantized_size_bytes": int(quantized_path.stat().st_size) if exists else None,
            "quantized_sha256": _sha256_file(quantized_path) if exists else None,
            "max_abs_error": None,
            "mean_abs_error": None,
            "binary_mask_agreement": None,
        }
        report = _jsonable(report)
        _write_json(report, report_path)
        return report

    if quantized_path.exists():
        quantized_path.unlink()
    try:
        quantization.quantize_dynamic(
            str(float_path),
            str(quantized_path),
            weight_type=quantization.QuantType.QInt8,
        )
    except Exception as exc:  # noqa: BLE001 - ORT operator failures are version-specific
        return skipped("quantization", "ONNX Runtime could not quantize this model", exc)
    if not quantized_path.is_file():
        return skipped("quantization", "quantizer did not create an output model")

    try:
        quantized_proto = onnx.load(str(quantized_path), load_external_data=True)
        onnx.checker.check_model(quantized_proto)
        operator_counts = Counter(node.op_type for node in quantized_proto.graph.node)
    except Exception as exc:  # noqa: BLE001 - ONNX validation errors vary by release
        return skipped("validation", "quantized model is not a valid ONNX graph", exc)
    quantized_operator_count = sum(
        count
        for operator, count in operator_counts.items()
        if "Quantize" in operator or operator.startswith("QLinear") or operator.endswith("Integer")
    )
    if quantized_operator_count == 0:
        return skipped(
            "quantization",
            "model contains no operators supported by ONNX Runtime dynamic quantization",
        )

    try:
        quantized_session = ort.InferenceSession(
            str(quantized_path), providers=["CPUExecutionProvider"]
        )
        quantized_outputs = quantized_session.run(
            None, {quantized_session.get_inputs()[0].name: input_array}
        )
    except Exception as exc:  # noqa: BLE001 - ORT provider errors vary by release
        return skipped(
            "quantized_inference",
            "CPUExecutionProvider does not support the quantized graph",
            exc,
        )
    if len(quantized_outputs) != 1:
        return skipped(
            "quantized_inference",
            f"quantized model returned {len(quantized_outputs)} outputs instead of one",
        )

    quantized_output = np.asarray(quantized_outputs[0])
    same_shape = quantized_output.shape == float_output.shape
    finite = bool(np.isfinite(quantized_output).all())
    if same_shape and finite:
        error = np.abs(float_output.astype(np.float64) - quantized_output.astype(np.float64))
        max_abs_error: float | None = float(error.max())
        mean_abs_error: float | None = float(error.mean())
        mask_agreement: float | None = float(
            np.mean((float_output >= 0.5) == (quantized_output >= 0.5))
        )
        within_tolerance = bool(max_abs_error <= tolerance)
    else:
        max_abs_error = None
        mean_abs_error = None
        mask_agreement = None
        within_tolerance = False

    quantized_size = int(quantized_path.stat().st_size)
    report = {
        **base_report,
        "status": "success" if within_tolerance else "failed",
        "skipped": False,
        "artifact_usable": within_tolerance,
        "quantized_size_bytes": quantized_size,
        "size_reduction_bytes": int(base_report["float_size_bytes"] - quantized_size),
        "size_reduction_fraction": float(1.0 - quantized_size / base_report["float_size_bytes"]),
        "quantized_sha256": _sha256_file(quantized_path),
        "operator_counts": dict(sorted(operator_counts.items())),
        "quantized_operator_count": int(quantized_operator_count),
        "providers": quantized_session.get_providers(),
        "float_output_shape": list(float_output.shape),
        "quantized_output_shape": list(quantized_output.shape),
        "outputs_finite": finite,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "binary_mask_threshold": 0.5,
        "binary_mask_agreement": mask_agreement,
        "within_tolerance": within_tolerance,
    }
    report = _jsonable(report)
    _write_json(report, report_path)
    if not within_tolerance:
        raise RuntimeError(
            "quantized ONNX parity failed "
            f"(shapes {float_output.shape} vs {quantized_output.shape}, "
            f"max_abs_error={max_abs_error}, tolerance={tolerance}); see {report_path}"
        )
    return report


def _validate_benchmark_counts(warmup: int, repetitions: int) -> None:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")


def _latency_statistics(measurements_ms: Sequence[float]) -> dict[str, float]:
    values = np.asarray(measurements_ms, dtype=np.float64)
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
    }


def _batch_size(sample: Any) -> int:
    shape = getattr(sample, "shape", ())
    return int(shape[0]) if len(shape) > 0 else 1


def _serialized_model_size(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return int(buffer.tell())


def benchmark_torch(
    model: nn.Module,
    sample: Tensor | np.ndarray,
    device: str | torch.device = "cpu",
    warmup: int = 5,
    repetitions: int = 20,
) -> dict[str, Any]:
    """Benchmark individual PyTorch forwards with synchronization where required."""
    _validate_benchmark_counts(warmup, repetitions)
    resolved_device = _resolve_torch_device(device)
    if isinstance(sample, Tensor):
        input_tensor = sample.detach().to(resolved_device)
    else:
        input_tensor = torch.as_tensor(np.asarray(sample)).to(resolved_device)
    if input_tensor.numel() == 0:
        raise ValueError("sample must not be empty")

    was_training = model.training
    model.to(resolved_device).eval()

    def synchronize() -> None:
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)

    measurements: list[float] = []
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                model(input_tensor)
            synchronize()
            if resolved_device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(resolved_device)
            for _ in range(repetitions):
                synchronize()
                started = time.perf_counter()
                model(input_tensor)
                synchronize()
                measurements.append((time.perf_counter() - started) * 1000.0)
        peak_cuda_bytes = (
            int(torch.cuda.max_memory_allocated(resolved_device))
            if resolved_device.type == "cuda"
            else None
        )
    finally:
        model.train(was_training)

    latency = _latency_statistics(measurements)
    batch_size = _batch_size(input_tensor)
    elapsed_seconds = sum(measurements) / 1000.0
    if resolved_device.type == "cuda":
        device_name = torch.cuda.get_device_name(resolved_device)
    else:
        device_name = platform_module.processor() or platform_module.machine() or "CPU"
    report = {
        "backend": "pytorch",
        "provider": {
            "device": str(resolved_device),
            "device_name": device_name,
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cpu_threads": int(torch.get_num_threads()),
        },
        "sample_shape": list(input_tensor.shape),
        "batch_size": batch_size,
        "warmup": int(warmup),
        "repetitions": int(repetitions),
        "measurements_ms": measurements,
        "latency_ms": latency,
        "throughput_samples_per_second": float(batch_size * repetitions / elapsed_seconds),
        "peak_allocated_cuda_memory_bytes": peak_cuda_bytes,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "model_size_bytes": _serialized_model_size(model),
    }
    return _jsonable(report)


def benchmark_onnx(
    path: str | Path,
    sample: Tensor | np.ndarray,
    warmup: int = 5,
    repetitions: int = 20,
    providers: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Benchmark individual ONNX Runtime calls and report the active providers."""
    _validate_benchmark_counts(warmup, repetitions)
    ort = _require_module("onnxruntime", "onnxruntime")
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    input_array = _numpy_sample(sample)
    requested_providers = list(providers) if providers is not None else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=requested_providers)
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError(f"benchmark_onnx requires one model input, got {len(inputs)}")
    feeds = {inputs[0].name: input_array}

    for _ in range(warmup):
        session.run(None, feeds)
    measurements: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        session.run(None, feeds)
        measurements.append((time.perf_counter() - started) * 1000.0)

    latency = _latency_statistics(measurements)
    batch_size = _batch_size(input_array)
    elapsed_seconds = sum(measurements) / 1000.0
    active_providers = session.get_providers()
    report = {
        "backend": "onnxruntime",
        "model": str(model_path),
        "sha256": _sha256_file(model_path),
        "file_size_bytes": int(model_path.stat().st_size),
        "provider": {
            "onnxruntime_version": str(ort.__version__),
            "requested": requested_providers,
            "active": active_providers,
            "options": session.get_provider_options(),
            "available": ort.get_available_providers(),
        },
        "input": {
            "name": inputs[0].name,
            "shape": list(input_array.shape),
            "dtype": str(input_array.dtype),
        },
        "outputs": [output.name for output in session.get_outputs()],
        "batch_size": batch_size,
        "warmup": int(warmup),
        "repetitions": int(repetitions),
        "measurements_ms": measurements,
        "latency_ms": latency,
        "throughput_samples_per_second": float(batch_size * repetitions / elapsed_seconds),
        # ORT does not expose provider peak allocation through its Python inference API.
        "peak_allocated_cuda_memory_bytes": None,
    }
    return _jsonable(report)


def write_artifact_inventory(
    directory: str | Path,
    manifest_name: str | Path = "manifest.json",
) -> dict[str, Any]:
    """Hash every file below ``directory`` and write a self-excluding manifest."""
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"artifact directory not found: {root}")
    manifest = Path(manifest_name)
    if not manifest.is_absolute():
        manifest = root / manifest
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_resolved = manifest.resolve()

    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.resolve() == manifest_resolved:
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    report = {
        "algorithm": "sha256",
        "root": str(root.resolve()),
        "manifest": str(manifest),
        "file_count": len(files),
        "files": files,
    }
    report = _jsonable(report)
    _write_json(report, manifest)
    return report
