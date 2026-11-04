"""Idempotent download and ZIP extraction helpers for research datasets."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

_CHUNK_SIZE = 1024 * 1024
_SUPPORTED_HASHES = {"md5", "sha256"}


def checksum_file(
    path: str | Path, algorithm: str = "sha256", chunk_size: int = _CHUNK_SIZE
) -> str:
    """Return the lowercase MD5 or SHA-256 digest of a file."""
    source = Path(path)
    if not isinstance(algorithm, str):
        raise TypeError("checksum algorithm must be md5 or sha256")
    normalized_algorithm = algorithm.strip().casefold().replace("-", "")
    if normalized_algorithm not in _SUPPORTED_HASHES:
        raise ValueError("checksum algorithm must be md5 or sha256")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not source.is_file():
        raise FileNotFoundError(f"file does not exist: {source}")
    digest = hashlib.new(normalized_algorithm)
    with source.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _parse_checksum(checksum: str | None, algorithm: str) -> tuple[str, str | None]:
    if not isinstance(algorithm, str):
        raise TypeError("checksum algorithm must be md5 or sha256")
    if checksum is not None and not isinstance(checksum, str):
        raise ValueError("checksum must be a hexadecimal string")
    normalized_algorithm = algorithm.strip().casefold().replace("-", "")
    value = checksum
    if value is not None and ":" in value:
        prefix, value = value.split(":", 1)
        normalized_algorithm = prefix.strip().casefold().replace("-", "")
    if normalized_algorithm not in _SUPPORTED_HASHES:
        raise ValueError("checksum algorithm must be md5 or sha256")
    if value is None:
        return normalized_algorithm, None
    value = value.strip().casefold()
    expected_length = 32 if normalized_algorithm == "md5" else 64
    if len(value) != expected_length or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(
            f"invalid {normalized_algorithm} checksum; expected {expected_length} hex characters"
        )
    return normalized_algorithm, value


def _file_matches(
    path: Path,
    expected_bytes: int | None,
    checksum: str | None,
    checksum_algorithm: str,
) -> tuple[bool, str | None]:
    if not path.is_file():
        return False, None
    size = path.stat().st_size
    if expected_bytes is not None and size != expected_bytes:
        return False, None
    digest = checksum_file(path, checksum_algorithm) if checksum is not None else None
    if checksum is not None and digest != checksum:
        return False, digest
    return True, digest


def _space_guard(parent: Path, required_bytes: int, min_free_bytes: int) -> None:
    free = shutil.disk_usage(parent).free
    if free < required_bytes + min_free_bytes:
        raise OSError(
            f"insufficient disk space in {parent}: need at least "
            f"{required_bytes + min_free_bytes} bytes, have {free}"
        )


def _download_report(
    *,
    action: str,
    url: str,
    destination: Path,
    algorithm: str,
    digest: str | None,
    resumed: bool,
) -> dict[str, Any]:
    return {
        "action": action,
        "url": url,
        "destination": str(destination),
        "bytes": destination.stat().st_size,
        "checksum_algorithm": algorithm if digest is not None else None,
        "checksum": digest,
        "resumed": resumed,
    }


def download_once(
    url: str,
    destination: str | Path,
    *,
    checksum: str | None = None,
    checksum_algorithm: str = "sha256",
    expected_bytes: int | None = None,
    timeout: float = 60.0,
    chunk_size: int = _CHUNK_SIZE,
    min_free_bytes: int = 0,
) -> dict[str, Any]:
    """Download an HTTPS URL once, resuming a persistent ``.part`` when possible."""
    parsed = urlparse(url)
    if parsed.scheme.casefold() != "https" or not parsed.netloc:
        raise ValueError("download URL must be an absolute HTTPS URL")
    if expected_bytes is not None and (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
    ):
        raise ValueError("expected_bytes must be a non-negative integer")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or timeout <= 0
        or isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
        or isinstance(min_free_bytes, bool)
        or not isinstance(min_free_bytes, int)
        or min_free_bytes < 0
    ):
        raise ValueError("timeout/chunk_size must be positive and min_free_bytes non-negative")
    algorithm, expected_digest = _parse_checksum(checksum, checksum_algorithm)

    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not target.is_file():
        raise IsADirectoryError(f"download destination is not a file: {target}")
    matches, digest = _file_matches(target, expected_bytes, expected_digest, algorithm)
    if matches:
        return _download_report(
            action="reused",
            url=url,
            destination=target,
            algorithm=algorithm,
            digest=digest,
            resumed=False,
        )

    part = Path(f"{target}.part")
    if target.exists():
        target.unlink()
    if part.exists() and not part.is_file():
        raise IsADirectoryError(f"partial download path is not a file: {part}")
    if expected_bytes is not None and part.exists() and part.stat().st_size > expected_bytes:
        part.unlink()

    resume_offset = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "landslide-detection/0.1"}
    if resume_offset:
        headers["Range"] = f"bytes={resume_offset}-"
    request = urllib.request.Request(url, headers=headers, method="GET")

    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and resume_offset:
            matches, digest = _file_matches(part, expected_bytes, expected_digest, algorithm)
            if matches:
                os.replace(part, target)
                return _download_report(
                    action="downloaded",
                    url=url,
                    destination=target,
                    algorithm=algorithm,
                    digest=digest,
                    resumed=True,
                )
            part.unlink()
            return download_once(
                url,
                target,
                checksum=checksum,
                checksum_algorithm=checksum_algorithm,
                expected_bytes=expected_bytes,
                timeout=timeout,
                chunk_size=chunk_size,
                min_free_bytes=min_free_bytes,
            )
        raise RuntimeError(f"download failed for {url}: HTTP {exc.code} {exc.reason}") from exc
    except OSError as exc:
        raise RuntimeError(f"download failed for {url}: {exc}") from exc

    with response:
        final_url = response.geturl()
        if urlparse(final_url).scheme.casefold() != "https":
            raise RuntimeError(f"download redirected to a non-HTTPS URL: {final_url}")
        status = getattr(response, "status", response.getcode())
        content_range = response.headers.get("Content-Range", "")

        response_length: int | None = None
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                response_length = int(content_length)
            except ValueError as exc:
                raise RuntimeError(
                    f"server returned invalid Content-Length: {content_length}"
                ) from exc
            if response_length < 0:
                raise RuntimeError("server returned a negative Content-Length")

        range_start: int | None = None
        range_end: int | None = None
        server_total: int | None = None
        if content_range:
            match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", content_range, re.IGNORECASE)
            if not match:
                raise RuntimeError(f"server returned invalid Content-Range: {content_range}")
            range_start, range_end = int(match.group(1)), int(match.group(2))
            if range_end < range_start:
                raise RuntimeError(f"server returned invalid Content-Range: {content_range}")
            if match.group(3) != "*":
                server_total = int(match.group(3))
                if range_end >= server_total:
                    raise RuntimeError(f"server returned invalid Content-Range: {content_range}")
            if response_length is not None and response_length != range_end - range_start + 1:
                raise RuntimeError("Content-Length and Content-Range disagree")

        if status == 206:
            expected_start = resume_offset if resume_offset else 0
            if range_start != expected_start:
                if resume_offset:
                    part.unlink()
                raise RuntimeError(
                    f"server resumed at byte {range_start}, expected {expected_start}"
                )
        resumed = bool(resume_offset and status == 206)
        write_offset = resume_offset if resumed else 0
        mode = "ab" if resumed else "wb"
        if server_total is None and response_length is not None:
            server_total = write_offset + response_length

        if (
            expected_bytes is not None
            and server_total is not None
            and server_total != expected_bytes
        ):
            raise RuntimeError(
                f"download size disagrees with expected_bytes: server={server_total}, "
                f"expected={expected_bytes}"
            )
        total = expected_bytes if expected_bytes is not None else server_total
        required = max(0, total - write_offset) if total is not None else 0
        _space_guard(target.parent, required, min_free_bytes)

        received = 0
        try:
            with part.open(mode) as handle:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    handle.write(chunk)
                    received += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise RuntimeError(f"could not write partial download {part}: {exc}") from exc
        if response_length is not None and received != response_length:
            raise RuntimeError(
                f"download ended early: received {received} response bytes, "
                f"expected {response_length}; partial file kept at {part}"
            )

    final_size = part.stat().st_size
    if total is not None and final_size != total:
        raise RuntimeError(
            f"downloaded size mismatch: got {final_size}, server reported {total}; "
            f"partial file kept at {part}"
        )
    if expected_bytes is not None and final_size != expected_bytes:
        raise RuntimeError(
            f"downloaded size mismatch: got {final_size}, expected {expected_bytes}; "
            f"partial file kept at {part}"
        )
    digest = checksum_file(part, algorithm) if expected_digest is not None else None
    if expected_digest is not None and digest != expected_digest:
        part.unlink()
        raise RuntimeError(
            f"downloaded {algorithm} mismatch: got {digest}, expected {expected_digest}"
        )
    os.replace(part, target)
    return _download_report(
        action="downloaded",
        url=url,
        destination=target,
        algorithm=algorithm,
        digest=digest,
        resumed=resumed,
    )


def _safe_zip_target(info: zipfile.ZipInfo, destination: Path) -> tuple[Path, PurePosixPath]:
    if "\x00" in info.filename:
        raise ValueError(f"unsafe ZIP member contains a null byte: {info.filename!r}")
    portable = PurePosixPath(info.filename.replace("\\", "/"))
    if (
        not portable.parts
        or portable.is_absolute()
        or ".." in portable.parts
        or (portable.parts and re.match(r"^[A-Za-z]:$", portable.parts[0]))
    ):
        raise ValueError(f"unsafe ZIP member path: {info.filename!r}")
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ValueError(f"ZIP symbolic links are not allowed: {info.filename!r}")
    target = destination.joinpath(*portable.parts).resolve()
    try:
        target.relative_to(destination)
    except ValueError as exc:
        raise ValueError(f"ZIP member escapes destination: {info.filename!r}") from exc
    return target, portable


def _read_marker(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            marker = json.load(handle)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return marker if isinstance(marker, dict) else None


def _write_marker(path: Path, marker: dict[str, Any]) -> None:
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary_name = handle.name
            json.dump(marker, handle, indent=2, sort_keys=True)
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


def extract_zip_once(
    archive_path: str | Path,
    destination: str | Path,
    *,
    marker_name: str = ".landslide-extract.json",
    checksum: str | None = None,
    checksum_algorithm: str = "sha256",
    min_free_bytes: int = 0,
) -> dict[str, Any]:
    """Safely extract a ZIP once, preserving unrelated files in the destination."""
    archive = Path(archive_path).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"ZIP archive does not exist: {archive}")
    if Path(marker_name).name != marker_name or marker_name in {"", ".", ".."}:
        raise ValueError("marker_name must be a plain file name")
    if (
        isinstance(min_free_bytes, bool)
        or not isinstance(min_free_bytes, int)
        or min_free_bytes < 0
    ):
        raise ValueError("min_free_bytes must be non-negative")
    algorithm, expected_digest = _parse_checksum(checksum, checksum_algorithm)
    actual_digest = checksum_file(archive, algorithm)
    if expected_digest is not None and actual_digest != expected_digest:
        raise ValueError(
            f"archive {algorithm} mismatch: got {actual_digest}, expected {expected_digest}"
        )

    target_root = Path(destination).expanduser().resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    if not target_root.is_dir():
        raise NotADirectoryError(f"extraction destination is not a directory: {target_root}")
    marker_path = target_root / marker_name
    marker = _read_marker(marker_path)
    if (
        marker is not None
        and marker.get("archive_checksum_algorithm") == algorithm
        and marker.get("archive_checksum") == actual_digest
    ):
        return {
            "action": "reused",
            "archive": str(archive),
            "destination": str(target_root),
            "archive_checksum_algorithm": algorithm,
            "archive_checksum": actual_digest,
            "files": int(marker.get("files", 0)),
            "bytes": int(marker.get("bytes", 0)),
            "marker": str(marker_path),
        }

    try:
        archive_file = zipfile.ZipFile(archive, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"could not open ZIP archive {archive}: {exc}") from exc

    with archive_file:
        members: list[tuple[zipfile.ZipInfo, Path, PurePosixPath]] = []
        seen: set[str] = set()
        total_bytes = 0
        for info in archive_file.infolist():
            member_target, portable = _safe_zip_target(info, target_root)
            normalized_name = portable.as_posix().rstrip("/")
            if not normalized_name:
                continue
            if normalized_name == marker_name:
                raise ValueError(f"ZIP archive may not contain extraction marker {marker_name!r}")
            duplicate_key = os.path.normcase(str(member_target))
            if duplicate_key in seen:
                raise ValueError(f"ZIP archive contains duplicate path {info.filename!r}")
            seen.add(duplicate_key)
            members.append((info, member_target, portable))
            if not info.is_dir():
                total_bytes += info.file_size

        _space_guard(target_root, total_bytes, min_free_bytes)
        extracted_files = 0
        try:
            for info, member_target, _ in members:
                if info.is_dir():
                    member_target.mkdir(parents=True, exist_ok=True)
                    continue
                member_target.parent.mkdir(parents=True, exist_ok=True)
                resolved_parent = member_target.parent.resolve()
                try:
                    resolved_parent.relative_to(target_root)
                except ValueError as exc:
                    raise ValueError(
                        f"ZIP member parent escapes destination: {info.filename!r}"
                    ) from exc
                temporary_name: str | None = None
                try:
                    with (
                        archive_file.open(info, "r") as source,
                        tempfile.NamedTemporaryFile(
                            "wb",
                            dir=member_target.parent,
                            prefix=f".{member_target.name}.",
                            delete=False,
                        ) as output,
                    ):
                        temporary_name = output.name
                        shutil.copyfileobj(source, output, length=_CHUNK_SIZE)
                        output.flush()
                        os.fsync(output.fileno())
                    os.replace(temporary_name, member_target)
                    temporary_name = None
                finally:
                    if temporary_name is not None:
                        try:
                            Path(temporary_name).unlink()
                        except FileNotFoundError:
                            pass
                extracted_files += 1
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            raise RuntimeError(f"failed while extracting {archive}: {exc}") from exc

    marker = {
        "archive": str(archive),
        "archive_checksum_algorithm": algorithm,
        "archive_checksum": actual_digest,
        "files": extracted_files,
        "bytes": total_bytes,
    }
    _write_marker(marker_path, marker)
    return {
        "action": "extracted",
        "archive": str(archive),
        "destination": str(target_root),
        "archive_checksum_algorithm": algorithm,
        "archive_checksum": actual_digest,
        "files": extracted_files,
        "bytes": total_bytes,
        "marker": str(marker_path),
    }
