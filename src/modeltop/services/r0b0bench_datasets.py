"""Pinned, integrity-checked local assets for r0b0bench quality lanes."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import stat
import struct
import tempfile
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import httpx

DEFAULT_R0B0BENCH_DATASET_ROOT = Path(
    "~/.local/share/modeltop/datasets/r0b0bench"
).expanduser()
_MANIFEST_SCHEMA = 1
_MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024
_ARC_COMPRESSED_BYTES = 223_558


type R0b0benchAssetState = Literal["installed", "missing", "invalid"]
type R0b0benchAssetId = Literal[
    "qa", "ifeval", "humaneval", "gsm8k", "bfcl_run", "bfcl_ast"
]


@dataclass(frozen=True, slots=True)
class R0b0benchAssetSpec:
    """One immutable upstream asset and its normalized local output."""

    asset_id: R0b0benchAssetId
    label: str
    relative_path: Path
    source_url: str
    source_sha256: str
    source_revision: str
    output_sha256: str
    output_size: int
    row_count: int | None
    transform: Literal["direct", "gzip", "arc_range"]
    license_name: str
    license_url: str
    request_range: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class R0b0benchAssetStatus:
    """Integrity state for one installed asset."""

    asset_id: R0b0benchAssetId
    label: str
    state: R0b0benchAssetState
    path: Path


class R0b0benchDatasetError(Exception):
    """Bounded installation failure without response or dataset payloads."""


_ASSETS: tuple[R0b0benchAssetSpec, ...] = (
    R0b0benchAssetSpec(
        asset_id="qa",
        label="QA / ARC-Easy",
        relative_path=Path("qa/arc_easy_test.jsonl"),
        source_url=(
            "https://s3-us-west-2.amazonaws.com/ai2-website/data/ARC-V1-Feb2018.zip"
        ),
        source_sha256=(
            "0d3463518167984312e971ee3a08c8704a0db735959c164311b46eec367fcfd7"
        ),
        source_revision="ARC-V1-Feb2018",
        output_sha256=(
            "7c73345800fdb847ae5fca138cb26c65c718087a490170068631c50dee1d6fec"
        ),
        output_size=757_347,
        row_count=2_376,
        transform="arc_range",
        license_name="CC BY-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        request_range=(887_010, 1_110_674),
    ),
    R0b0benchAssetSpec(
        asset_id="ifeval",
        label="IFEval",
        relative_path=Path("ifeval/input_data.jsonl"),
        source_url=(
            "https://raw.githubusercontent.com/google-research/google-research/"
            "26d8ccdab6fec61b5c83ad6327ea8bda9e580288/"
            "instruction_following_eval/data/input_data.jsonl"
        ),
        source_sha256=(
            "67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49"
        ),
        source_revision="26d8ccdab6fec61b5c83ad6327ea8bda9e580288",
        output_sha256=(
            "67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49"
        ),
        output_size=207_111,
        row_count=541,
        transform="direct",
        license_name="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
    ),
    R0b0benchAssetSpec(
        asset_id="humaneval",
        label="HumanEval",
        relative_path=Path("humaneval/HumanEval.jsonl"),
        source_url=(
            "https://raw.githubusercontent.com/openai/human-eval/"
            "463c980b59e818ace59f6f9803cd92c749ceae61/"
            "data/HumanEval.jsonl.gz"
        ),
        source_sha256=(
            "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"
        ),
        source_revision="463c980b59e818ace59f6f9803cd92c749ceae61",
        output_sha256=(
            "1d49078ba3e2b196b9344535bef34a43021f038fad9561d6ee7c53450609a6a2"
        ),
        output_size=214_438,
        row_count=164,
        transform="gzip",
        license_name="MIT",
        license_url=(
            "https://github.com/openai/human-eval/blob/"
            "463c980b59e818ace59f6f9803cd92c749ceae61/LICENSE"
        ),
    ),
    R0b0benchAssetSpec(
        asset_id="gsm8k",
        label="GSM8K",
        relative_path=Path("gsm8k/test.jsonl"),
        source_url=(
            "https://raw.githubusercontent.com/openai/grade-school-math/"
            "b0bb162abedc65e1fdd8e93ed090fd7598ee68bc/"
            "grade_school_math/data/test.jsonl"
        ),
        source_sha256=(
            "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
        ),
        source_revision="b0bb162abedc65e1fdd8e93ed090fd7598ee68bc",
        output_sha256=(
            "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
        ),
        output_size=749_738,
        row_count=1_319,
        transform="direct",
        license_name="MIT",
        license_url=(
            "https://github.com/openai/grade-school-math/blob/"
            "b0bb162abedc65e1fdd8e93ed090fd7598ee68bc/LICENSE"
        ),
    ),
    R0b0benchAssetSpec(
        asset_id="bfcl_run",
        label="BFCL multi-turn adapter",
        relative_path=Path("bfcl/bfcl_run.py"),
        source_url=(
            "https://raw.githubusercontent.com/r0b0tlab/r0b0bench/"
            "d5ed83d8499a952546cf458e090be42ee4a48eef/"
            "scripts/bfcl/bfcl_run.py"
        ),
        source_sha256=(
            "3bd2d5575f75e1b89aa0d51a09967fbc26b156bc4529ba26dffcf2f292422a00"
        ),
        source_revision="d5ed83d8499a952546cf458e090be42ee4a48eef",
        output_sha256=(
            "3bd2d5575f75e1b89aa0d51a09967fbc26b156bc4529ba26dffcf2f292422a00"
        ),
        output_size=8_025,
        row_count=None,
        transform="direct",
        license_name="MIT",
        license_url=(
            "https://github.com/r0b0tlab/r0b0bench/blob/"
            "d5ed83d8499a952546cf458e090be42ee4a48eef/LICENSE"
        ),
    ),
    R0b0benchAssetSpec(
        asset_id="bfcl_ast",
        label="BFCL AST adapter",
        relative_path=Path("bfcl/bfcl_ast_run.py"),
        source_url=(
            "https://raw.githubusercontent.com/r0b0tlab/r0b0bench/"
            "d5ed83d8499a952546cf458e090be42ee4a48eef/"
            "scripts/bfcl/bfcl_ast_run.py"
        ),
        source_sha256=(
            "ac2aa535ce5b762cc16211c8b35d4c908d0fee2e48a13d5282d8a5dfc0242423"
        ),
        source_revision="d5ed83d8499a952546cf458e090be42ee4a48eef",
        output_sha256=(
            "ac2aa535ce5b762cc16211c8b35d4c908d0fee2e48a13d5282d8a5dfc0242423"
        ),
        output_size=8_260,
        row_count=None,
        transform="direct",
        license_name="MIT",
        license_url=(
            "https://github.com/r0b0tlab/r0b0bench/blob/"
            "d5ed83d8499a952546cf458e090be42ee4a48eef/LICENSE"
        ),
    ),
)


def _root(value: Path | None) -> Path:
    configured = os.environ.get("MODELTOP_R0B0BENCH_DATASET_ROOT")
    return Path(value or configured or DEFAULT_R0B0BENCH_DATASET_ROOT).expanduser()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_matches(path: Path, spec: R0b0benchAssetSpec) -> bool:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            return False
        if info.st_size != spec.output_size:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(64 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == spec.output_sha256
    except OSError:
        return False


def r0b0bench_asset_status(
    root: Path | None = None,
) -> tuple[R0b0benchAssetStatus, ...]:
    """Return missing/invalid/installed state without network access."""
    base = _root(root)
    rows: list[R0b0benchAssetStatus] = []
    for spec in _ASSETS:
        path = base / spec.relative_path
        if not path.exists():
            state = "missing"
        elif _file_matches(path, spec):
            state = "installed"
        else:
            state = "invalid"
        rows.append(R0b0benchAssetStatus(spec.asset_id, spec.label, state, path))
    return tuple(rows)


def r0b0bench_installed_paths(root: Path | None = None) -> dict[str, str]:
    """Return only validated paths accepted by runtime prerequisite fields."""
    base = _root(root)
    states = {row.asset_id: row.state for row in r0b0bench_asset_status(base)}
    resolved: dict[str, str] = {}
    fields: tuple[tuple[R0b0benchAssetId, str], ...] = (
        ("qa", "qa_data_path"),
        ("ifeval", "ifeval_data_path"),
        ("humaneval", "humaneval_data_path"),
        ("gsm8k", "gsm8k_data_path"),
    )
    specs = {spec.asset_id: spec for spec in _ASSETS}
    for asset_id, field_name in fields:
        if states[asset_id] == "installed":
            resolved[field_name] = str((base / specs[asset_id].relative_path).resolve())
    if states["bfcl_run"] == states["bfcl_ast"] == "installed":
        resolved["bfcl_scripts_directory"] = str((base / "bfcl").resolve())
    return resolved


def _download(client: httpx.Client, spec: R0b0benchAssetSpec) -> bytes:
    headers = {"Accept": "application/octet-stream"}
    if spec.request_range is not None:
        first, last = spec.request_range
        headers["Range"] = f"bytes={first}-{last}"
    try:
        with client.stream("GET", spec.source_url, headers=headers) as response:
            expected_status = 206 if spec.request_range is not None else 200
            if response.status_code != expected_status:
                raise R0b0benchDatasetError(
                    f"{spec.label}: download returned HTTP {response.status_code}"
                )
            if spec.request_range is not None:
                first, last = spec.request_range
                expected_range = f"bytes {first}-{last}/"
                if not (response.headers.get("content-range") or "").startswith(
                    expected_range
                ):
                    raise R0b0benchDatasetError(
                        f"{spec.label}: server ignored the pinned byte range"
                    )
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_bytes(64 * 1024):
                size += len(chunk)
                if size > _MAX_DOWNLOAD_BYTES:
                    raise R0b0benchDatasetError(
                        f"{spec.label}: download exceeded its size limit"
                    )
                chunks.append(chunk)
    except R0b0benchDatasetError:
        raise
    except httpx.HTTPError as error:
        raise R0b0benchDatasetError(
            f"{spec.label}: download failed ({type(error).__name__})"
        ) from error
    payload = b"".join(chunks)
    if _sha256(payload) != spec.source_sha256:
        raise R0b0benchDatasetError(f"{spec.label}: source checksum mismatch")
    return payload


def _arc_payload(fragment: bytes) -> bytes:
    if len(fragment) < 46 or fragment[:4] != b"PK\x03\x04":
        raise R0b0benchDatasetError("QA / ARC-Easy: malformed archive fragment")
    fields = struct.unpack_from("<IHHHHHIIIHH", fragment)
    flags, method, name_size, extra_size = fields[2], fields[3], fields[9], fields[10]
    if flags != 8 or method != 8:
        raise R0b0benchDatasetError("QA / ARC-Easy: unexpected archive encoding")
    start = 30 + name_size + extra_size
    end = start + _ARC_COMPRESSED_BYTES
    if end + 16 != len(fragment) or fragment[end : end + 4] != b"PK\x07\x08":
        raise R0b0benchDatasetError("QA / ARC-Easy: malformed archive bounds")
    try:
        raw = zlib.decompress(fragment[start:end], -zlib.MAX_WBITS)
    except zlib.error as error:
        raise R0b0benchDatasetError(
            "QA / ARC-Easy: archive decompression failed"
        ) from error
    crc, compressed_size, output_size = struct.unpack_from("<III", fragment, end + 4)
    if (
        crc != zlib.crc32(raw)
        or compressed_size != _ARC_COMPRESSED_BYTES
        or output_size != len(raw)
    ):
        raise R0b0benchDatasetError("QA / ARC-Easy: archive integrity mismatch")
    return raw


def _normalize_arc(payload: bytes) -> bytes:
    output: list[bytes] = []
    try:
        for line in payload.splitlines():
            if not line.strip():
                continue
            row = cast(dict[str, object], json.loads(line))
            question = cast(dict[str, object], row["question"])
            choices = cast(list[dict[str, object]], question["choices"])
            normalized = {
                "id": row["id"],
                "question": question["stem"],
                "choices": {
                    "text": [choice["text"] for choice in choices],
                    "label": [choice["label"] for choice in choices],
                },
                "answerKey": row["answerKey"],
            }
            output.append(
                (
                    json.dumps(
                        normalized,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
            )
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise R0b0benchDatasetError("QA / ARC-Easy: source schema mismatch") from error
    return b"".join(output)


def _transform(payload: bytes, spec: R0b0benchAssetSpec) -> bytes:
    if spec.transform == "direct":
        return payload
    if spec.transform == "gzip":
        try:
            return gzip.decompress(payload)
        except (OSError, EOFError) as error:
            raise R0b0benchDatasetError(
                f"{spec.label}: decompression failed"
            ) from error
    return _normalize_arc(_arc_payload(payload))


def _mapping(value: object) -> Mapping[str, object] | None:
    return cast(Mapping[str, object], value) if isinstance(value, dict) else None


def _valid_row(asset_id: R0b0benchAssetId, row: Mapping[str, object]) -> bool:
    if asset_id == "qa":
        choices = _mapping(row.get("choices"))
        return (
            isinstance(row.get("id"), str)
            and isinstance(row.get("question"), str)
            and choices is not None
            and isinstance(choices.get("text"), list)
            and isinstance(choices.get("label"), list)
            and len(cast(list[object], choices["text"]))
            == len(cast(list[object], choices["label"]))
            and isinstance(row.get("answerKey"), str)
        )
    if asset_id == "ifeval":
        return (
            isinstance(row.get("prompt"), str)
            and isinstance(row.get("instruction_id_list"), list)
            and isinstance(row.get("kwargs"), list)
        )
    if asset_id == "humaneval":
        return all(
            isinstance(row.get(key), str)
            for key in (
                "task_id",
                "prompt",
                "canonical_solution",
                "test",
                "entry_point",
            )
        )
    if asset_id == "gsm8k":
        return isinstance(row.get("question"), str) and isinstance(
            row.get("answer"), str
        )
    return True


def _validate_output(payload: bytes, spec: R0b0benchAssetSpec) -> None:
    if len(payload) != spec.output_size or _sha256(payload) != spec.output_sha256:
        raise R0b0benchDatasetError(f"{spec.label}: normalized checksum mismatch")
    if spec.row_count is None:
        return
    count = 0
    try:
        for line in payload.splitlines():
            if not line.strip():
                continue
            row = _mapping(json.loads(line))
            if row is None or not _valid_row(spec.asset_id, row):
                raise R0b0benchDatasetError(f"{spec.label}: normalized schema mismatch")
            count += 1
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise R0b0benchDatasetError(f"{spec.label}: invalid JSONL") from error
    if count != spec.row_count:
        raise R0b0benchDatasetError(f"{spec.label}: row count mismatch")


def _secure_directory(path: Path) -> Path:
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        resolved = path.resolve(strict=True)
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise R0b0benchDatasetError("dataset directory must be a real directory")
        path.chmod(0o700)
        return resolved
    except R0b0benchDatasetError:
        raise
    except OSError as error:
        raise R0b0benchDatasetError(
            f"unable to prepare dataset directory ({type(error).__name__})"
        ) from error


def _atomic_write(path: Path, payload: bytes) -> None:
    parent = _secure_directory(path.parent)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
        temporary = None
    except OSError as error:
        raise R0b0benchDatasetError(
            f"unable to write {path.name} ({type(error).__name__})"
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _manifest_payload(installed_at: datetime) -> bytes:
    payload = {
        "schema_version": _MANIFEST_SCHEMA,
        "installed_at": installed_at.isoformat(),
        "assets": [
            {
                "id": spec.asset_id,
                "path": str(spec.relative_path),
                "source_url": spec.source_url,
                "source_revision": spec.source_revision,
                "source_sha256": spec.source_sha256,
                "output_sha256": spec.output_sha256,
                "rows": spec.row_count,
                "license": spec.license_name,
                "license_url": spec.license_url,
            }
            for spec in _ASSETS
        ],
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def install_r0b0bench_assets(
    root: Path | None = None,
    *,
    client: httpx.Client | None = None,
) -> tuple[R0b0benchAssetStatus, ...]:
    """Install all pinned assets atomically per file and write provenance."""
    base = _secure_directory(_root(root))
    owned_client = client is None
    actual_client = client or httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(120.0, connect=30.0),
        headers={"User-Agent": "ModelTop-r0b0bench-dataset-installer/1"},
    )
    try:
        for spec in _ASSETS:
            destination = base / spec.relative_path
            if _file_matches(destination, spec):
                continue
            source = _download(actual_client, spec)
            output = _transform(source, spec)
            _validate_output(output, spec)
            _atomic_write(destination, output)
        _atomic_write(base / "manifest.json", _manifest_payload(datetime.now(UTC)))
    finally:
        if owned_client:
            actual_client.close()
    statuses = r0b0bench_asset_status(base)
    if any(row.state != "installed" for row in statuses):
        raise R0b0benchDatasetError("asset installation did not validate")
    return statuses
