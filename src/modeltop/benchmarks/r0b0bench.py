"""Hardened out-of-process boundary for the pinned r0b0bench CLI."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import signal
import stat
import sys
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn, Protocol, cast

from modeltop.benchmarks.models import (
    R0b0benchBenchmarkConfig,
    R0b0benchErrorCode,
    R0b0benchLaneResult,
    R0b0benchLaneStatus,
    R0b0benchMetric,
)
from modeltop.benchmarks.r0b0bench_contract import (
    R0B0BENCH_SYSTEMS_ORDER,
    R0b0benchLaneId,
    R0b0benchProfile,
    r0b0bench_ordered_selection,
    r0b0bench_profile_lanes,
)

logger = logging.getLogger(__name__)

R0B0BENCH_COMMIT = "d5ed83d8499a952546cf458e090be42ee4a48eef"
R0B0BENCH_VERSION = "1.0.0rc2"
R0B0BENCH_REPORT_SCHEMA = 2
DEFAULT_R0B0BENCH_OUTPUT_ROOT = Path("~/.local/share/modeltop/r0b0bench").expanduser()
BFCL_EVAL_VERSION = "2025.12.17"

_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_JSON_DEPTH = 16
_MAX_CONTAINER_ITEMS = 10_000
_MAX_STRING_CHARS = 64 * 1024
_RUN_ID = re.compile(r"r0b0bench-\d{8}T\d{6}Z-[0-9a-f]{8}\Z")
_LANE_MARKER = re.compile(
    r"=== lane (canary|bfcl_mt|bfcl_ast|latency|concurrency|throughput|"
    r"niah|qa|ifeval|humaneval|gsm8k|perf) ===\Z"
)
_ALLOWED_ENVIRONMENT = (
    "PATH",
    "LANG",
    "LC_ALL",
    "TZ",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "BFCL_NUM_THREADS",
    "BFCL_HTTP_TIMEOUT",
    "BFCL_MAX_RETRIES",
    "BFCL_MAX_TOKENS",
    "R0B0BENCH_BFCL_MODEL_REGISTRY",
    "R0B0BENCH_BFCL_MODEL_DISPLAY",
    "R0B0BENCH_BFCL_MODEL_URL",
    "R0B0BENCH_BFCL_MODEL_ORG",
    "R0B0BENCH_BFCL_MODEL_LICENSE",
)
_ERROR_MESSAGES: dict[R0b0benchErrorCode, str] = {
    "dependency_unavailable": "Pinned r0b0bench 1.0.0rc2 is unavailable.",
    "prerequisite_unavailable": (
        "Configure the required local prerequisite for the selected r0b0bench tests."
    ),
    "unsupported_authenticated_endpoint": (
        "r0b0bench 1.0.0rc2 does not support authenticated endpoints."
    ),
    "upstream_failure": "r0b0bench did not complete successfully.",
    "invalid_upstream_result": "r0b0bench returned an incompatible report.",
}

LaneStartedCallback = Callable[[R0b0benchLaneId, int, int], Awaitable[None]]
LaneFinishedCallback = Callable[[R0b0benchLaneResult, int, int], Awaitable[None]]


class R0b0benchRunnerError(Exception):
    """A bounded adapter failure safe to retain in application state."""

    def __init__(self, code: R0b0benchErrorCode) -> None:
        self.code: R0b0benchErrorCode = code
        super().__init__(_ERROR_MESSAGES[code])


@dataclass(frozen=True, slots=True)
class R0b0benchPrerequisiteIssue:
    lane_id: R0b0benchLaneId | None
    field_name: str


@dataclass(frozen=True, slots=True)
class R0b0benchPrerequisiteCheck:
    issues: tuple[R0b0benchPrerequisiteIssue, ...]
    child_environment: Mapping[str, str] = field(repr=False)


@dataclass(frozen=True, slots=True)
class R0b0benchRunnerRequest:
    benchmark_id: str
    config: R0b0benchBenchmarkConfig
    base_url: str = field(repr=False)
    model_id: str
    output_root: Path = field(repr=False)
    api_key: str | None = field(repr=False)


@dataclass(frozen=True, slots=True)
class R0b0benchPreparedRun:
    request: R0b0benchRunnerRequest
    child_environment: Mapping[str, str] = field(repr=False)
    run_directory: Path = field(repr=False)


@dataclass(frozen=True, slots=True)
class R0b0benchRunnerReport:
    upstream_run_id: str | None
    upstream_version: str
    schema_version: int | None
    profile: R0b0benchProfile
    model_id: str
    elapsed_seconds: float
    invalid_for_publish: bool
    infra_errors_total: int
    lanes: tuple[R0b0benchLaneResult, ...]
    unstarted_lanes: tuple[R0b0benchLaneId, ...]
    run_directory: Path = field(repr=False)
    cancelled: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        if self.infra_errors_total != sum(lane.infra_errors for lane in self.lanes):
            raise ValueError("infra_errors_total must match lane rows")
        if self.cancelled and not self.invalid_for_publish:
            raise ValueError("cancelled reports must be diagnostic")


class R0b0benchRunner(Protocol):
    async def prepare(
        self, request: R0b0benchRunnerRequest
    ) -> R0b0benchPreparedRun: ...

    async def run(
        self,
        prepared: R0b0benchPreparedRun,
        on_lane_started: LaneStartedCallback,
        on_lane_finished: LaneFinishedCallback,
    ) -> R0b0benchRunnerReport: ...


def _existing_path(value: str | None) -> Path | None:
    if not value:
        return None
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    return path


def _regular_file(value: str | None) -> Path | None:
    path = _existing_path(value)
    if path is None:
        return None
    try:
        if not path.is_file():
            return None
    except OSError:
        return None
    return path


def _executable_file(value: str | None) -> Path | None:
    path = _regular_file(value)
    if path is None or not os.access(path, os.X_OK):
        return None
    return path


def _environment_path(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def resolve_r0b0bench_prerequisites(
    config: R0b0benchBenchmarkConfig,
) -> R0b0benchPrerequisiteCheck:
    """Resolve selected local prerequisites without spawning or network I/O."""
    from modeltop.services.r0b0bench_datasets import r0b0bench_installed_paths

    installed = r0b0bench_installed_paths()
    selected = set(config.selected_lanes)
    issues: list[R0b0benchPrerequisiteIssue] = []
    overlay: dict[str, str] = {}

    if "niah" in selected:
        tokenizer = _existing_path(config.tokenizer_path)
        if tokenizer is None:
            issues.append(R0b0benchPrerequisiteIssue("niah", "tokenizer_path"))

    bfcl_lanes: tuple[R0b0benchLaneId, ...] = tuple(
        cast(R0b0benchLaneId, lane)
        for lane in ("bfcl_mt", "bfcl_ast")
        if lane in selected
    )
    if bfcl_lanes:
        python_value = config.bfcl_python or _environment_path(
            "R0B0BENCH_BFCL_PYTHON", "BFCL_PYTHON"
        )
        python_path = _executable_file(python_value or sys.executable)
        if python_path is None:
            for lane in bfcl_lanes:
                issues.append(R0b0benchPrerequisiteIssue(lane, "bfcl_python"))
        else:
            overlay["R0B0BENCH_BFCL_PYTHON"] = str(python_path)
        scripts_value = (
            config.bfcl_scripts_directory
            or _environment_path("R0B0BENCH_BFCL_SCRIPTS")
            or installed.get("bfcl_scripts_directory")
        )
        scripts = _existing_path(scripts_value)
        if scripts is None or not scripts.is_dir():
            for lane in bfcl_lanes:
                issues.append(
                    R0b0benchPrerequisiteIssue(lane, "bfcl_scripts_directory")
                )
        else:
            overlay["R0B0BENCH_BFCL_SCRIPTS"] = str(scripts)
            required_scripts = {
                "bfcl_mt": "bfcl_run.py",
                "bfcl_ast": "bfcl_ast_run.py",
            }
            for lane in bfcl_lanes:
                if _regular_file(str(scripts / required_scripts[lane])) is None:
                    issues.append(
                        R0b0benchPrerequisiteIssue(
                            lane,
                            "bfcl_scripts_directory",
                        )
                    )

    quality_specs: tuple[
        tuple[R0b0benchLaneId, str, str | None, tuple[str, ...]], ...
    ] = (
        ("qa", "qa_data_path", config.qa_data_path, ("R0B0BENCH_QA_DATA",)),
        (
            "ifeval",
            "ifeval_data_path",
            config.ifeval_data_path,
            ("R0B0BENCH_IFEVAL_DATA",),
        ),
        (
            "humaneval",
            "humaneval_data_path",
            config.humaneval_data_path,
            ("R0B0BENCH_HUMANEVAL_DATA",),
        ),
        (
            "gsm8k",
            "gsm8k_data_path",
            config.gsm8k_data_path,
            ("R0B0BENCH_GSM8K_DATA", "GSM8K_DATA"),
        ),
    )
    for lane, field_name, configured, fallbacks in quality_specs:
        if lane not in selected:
            continue
        resolved = _regular_file(
            configured or _environment_path(*fallbacks) or installed.get(field_name)
        )
        if resolved is None:
            issues.append(R0b0benchPrerequisiteIssue(lane, field_name))
        else:
            overlay[fallbacks[0]] = str(resolved)
    if "humaneval" in selected and not config.allow_unsafe_humaneval:
        issues.append(R0b0benchPrerequisiteIssue("humaneval", "allow_unsafe_humaneval"))

    return R0b0benchPrerequisiteCheck(tuple(issues), overlay)


def _base_child_environment(
    overlay: Mapping[str, str], *, model_id: str
) -> dict[str, str]:
    environment = {
        name: os.environ[name] for name in _ALLOWED_ENVIRONMENT if name in os.environ
    }
    environment.update(overlay)
    environment["R0B0BENCH_SERVED_MODEL"] = model_id
    environment["OPENAI_API_KEY"] = "EMPTY"
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def _raise(code: R0b0benchErrorCode) -> NoReturn:
    raise R0b0benchRunnerError(code)


async def _reap_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        await process.wait()
        return
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), timeout=5.0)
        return
    except TimeoutError:
        pass
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    await process.wait()


async def _probe(*arguments: str, environment: Mapping[str, str]) -> tuple[int, bytes]:
    try:
        process = await asyncio.create_subprocess_exec(
            *arguments,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=dict(environment),
            start_new_session=True,
        )
    except (OSError, ValueError):
        _raise("dependency_unavailable")
    try:
        output, _ = await process.communicate()
    except asyncio.CancelledError:
        await _reap_process(process)
        raise
    return process.returncode or 0, output[:4096]


def _secure_directory(path: Path) -> Path:
    try:
        info = path.lstat()
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        _raise("prerequisite_unavailable")
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        _raise("prerequisite_unavailable")
    if info.st_uid != os.getuid():
        _raise("prerequisite_unavailable")
    try:
        path.chmod(0o700)
        hardened = path.lstat()
    except OSError:
        _raise("prerequisite_unavailable")
    if stat.S_IMODE(hardened.st_mode) != 0o700:
        _raise("prerequisite_unavailable")
    return resolved


def _allocate_run_directory(request: R0b0benchRunnerRequest) -> tuple[Path, Path]:
    if _RUN_ID.fullmatch(request.benchmark_id) is None:
        _raise("prerequisite_unavailable")
    root = request.output_root.expanduser()
    try:
        if root.exists():
            root_resolved = _secure_directory(root)
        else:
            root.mkdir(parents=True, mode=0o700)
            root_resolved = _secure_directory(root)
        run_directory = root_resolved / request.benchmark_id
        os.mkdir(run_directory, 0o700)
    except R0b0benchRunnerError:
        raise
    except OSError:
        _raise("prerequisite_unavailable")
    run_resolved = _secure_directory(run_directory)
    if run_resolved.parent != root_resolved:
        _raise("prerequisite_unavailable")
    return root_resolved, run_resolved


def _verify_prepared_directory(prepared: R0b0benchPreparedRun) -> None:
    request = prepared.request
    expected = request.output_root.expanduser().resolve() / request.benchmark_id
    resolved = _secure_directory(prepared.run_directory)
    if resolved != expected or stat.S_IMODE(resolved.lstat().st_mode) != 0o700:
        _raise("prerequisite_unavailable")


def _reject_constant(_: str) -> NoReturn:
    raise ValueError("non-standard JSON constant")


def _validate_json_limits(value: object, depth: int = 0) -> None:
    if depth > _MAX_JSON_DEPTH:
        raise ValueError("JSON nesting limit exceeded")
    if isinstance(value, str):
        if len(value) > _MAX_STRING_CHARS:
            raise ValueError("JSON string limit exceeded")
        return
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        if len(mapping) > _MAX_CONTAINER_ITEMS:
            raise ValueError("JSON object limit exceeded")
        for key, item in mapping.items():
            if not isinstance(key, str) or len(key) > _MAX_STRING_CHARS:
                raise ValueError("invalid JSON key")
            _validate_json_limits(item, depth + 1)
        return
    if isinstance(value, list):
        sequence = cast(list[object], value)
        if len(sequence) > _MAX_CONTAINER_ITEMS:
            raise ValueError("JSON array limit exceeded")
        for item in sequence:
            _validate_json_limits(item, depth + 1)


def _read_json(path: Path, *, expected_parent: Path) -> dict[str, object]:
    try:
        if path.parent.resolve(strict=True) != expected_parent.resolve(strict=True):
            raise ValueError("unexpected JSON parent")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_JSON_BYTES:
                raise ValueError("invalid JSON file")
            raw = os.read(descriptor, _MAX_JSON_BYTES + 1)
        finally:
            os.close(descriptor)
        if len(raw) > _MAX_JSON_BYTES:
            raise ValueError("JSON file limit exceeded")
        value = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
        _validate_json_limits(value)
        if not isinstance(value, dict):
            raise ValueError("JSON root must be an object")
        return cast(dict[str, object], value)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, RuntimeError):
        _raise("invalid_upstream_result")


def _strict_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) for key in cast(dict[object, object], value)
    ):
        _raise("invalid_upstream_result")
    return cast(dict[str, object], value)


def _strict_list(value: object) -> list[object]:
    if not isinstance(value, list):
        _raise("invalid_upstream_result")
    return cast(list[object], value)


def _strict_string(value: object, *, maximum: int = 64 * 1024) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        _raise("invalid_upstream_result")
    return value


def _strict_bool(value: object) -> bool:
    if type(value) is not bool:
        _raise("invalid_upstream_result")
    return value


def _strict_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _raise("invalid_upstream_result")
    return value


def _strict_number(value: object, *, ratio: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _raise("invalid_upstream_result")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (ratio and result > 1):
        _raise("invalid_upstream_result")
    return result


def _strict_ratio(value: object) -> float:
    return _strict_number(value, ratio=True)


def _optional(
    summary: Mapping[str, object], key: str, parser: Callable[[object], Any]
) -> Any | None:
    if key not in summary:
        return None
    return parser(summary[key])


def _metric(name: str, value: object, unit: str | None) -> R0b0benchMetric:
    return R0b0benchMetric(
        name=name,
        value=cast(int | float | bool, value),
        unit=cast(Any, unit),
    )


def _flat_metrics(
    summary: Mapping[str, object],
    specs: tuple[tuple[str, str, str | None, Callable[[object], Any]], ...],
    *,
    prefix: str = "",
) -> list[R0b0benchMetric]:
    metrics: list[R0b0benchMetric] = []
    for source, name, unit, parser in specs:
        value = _optional(summary, source, parser)
        if value is not None:
            metrics.append(_metric(f"{prefix}{name}", value, unit))
    return metrics


_LATENCY_SPECS: tuple[tuple[str, str, str | None, Callable[[object], Any]], ...] = (
    ("ttft_ms_mean", "ttft_mean", "ms", _strict_number),
    ("itl_ms_mean", "itl_mean", "ms", _strict_number),
    ("itl_ms_p95_mean", "itl_p95_mean", "ms", _strict_number),
    ("e2el_ms_mean", "e2e_mean", "ms", _strict_number),
)
_THROUGHPUT_SPECS: tuple[tuple[str, str, str | None, Callable[[object], Any]], ...] = (
    (
        "median_client_output_tok_s",
        "decode_median_output_rate",
        "tokens/s",
        _strict_number,
    ),
    (
        "median_server_prompt_tok_s",
        "prefill_median_prompt_rate",
        "tokens/s",
        _strict_number,
    ),
)


def _latency_metrics(
    summary: Mapping[str, object], *, prefix: str = ""
) -> list[R0b0benchMetric]:
    metrics: list[R0b0benchMetric] = []
    if "stream" in summary:
        metrics.extend(
            _flat_metrics(
                _strict_mapping(summary["stream"]),
                _LATENCY_SPECS,
                prefix=prefix,
            )
        )
    failed = _optional(summary, "failed", _strict_count)
    if failed is not None:
        metrics.append(_metric(f"{prefix}failed_requests", failed, "count"))
    return metrics


def _concurrency_metrics(
    summary: Mapping[str, object], *, prefix: str = ""
) -> list[R0b0benchMetric]:
    if "rows" not in summary:
        return []
    rows = _strict_list(summary["rows"])
    parsed: list[tuple[int, float, int, int]] = []
    for row_value in rows:
        row = _strict_mapping(row_value)
        parsed.append(
            (
                _strict_count(row.get("concurrency")),
                _strict_number(row.get("aggregate_output_tok_s")),
                _strict_count(row.get("completed")),
                _strict_count(row.get("failed")),
            )
        )
    metrics = [
        _metric(
            f"{prefix}completed_requests",
            sum(row[2] for row in parsed),
            "count",
        ),
        _metric(f"{prefix}failed_requests", sum(row[3] for row in parsed), "count"),
    ]
    if parsed:
        _, chosen = max(enumerate(parsed), key=lambda pair: pair[1][1])
        metrics[:0] = [
            _metric(f"{prefix}peak_level", chosen[0], "count"),
            _metric(f"{prefix}peak_aggregate_output_rate", chosen[1], "tokens/s"),
        ]
    return metrics


def _throughput_metrics(
    summary: Mapping[str, object], *, prefix: str = ""
) -> list[R0b0benchMetric]:
    metrics: list[R0b0benchMetric] = []
    for section, spec in (
        ("decode", _THROUGHPUT_SPECS[:1]),
        ("prefill", _THROUGHPUT_SPECS[1:]),
    ):
        if section in summary:
            metrics.extend(
                _flat_metrics(_strict_mapping(summary[section]), spec, prefix=prefix)
            )
    failed = _optional(summary, "failed", _strict_count)
    if failed is not None:
        metrics.append(_metric(f"{prefix}failed_requests", failed, "count"))
    return metrics


def _normalize_metrics(
    lane_id: R0b0benchLaneId, summary: Mapping[str, object]
) -> tuple[R0b0benchMetric, ...]:
    if lane_id == "canary":
        return tuple(
            _flat_metrics(
                summary,
                (
                    ("passed", "passed", None, _strict_bool),
                    ("n", "cases", "count", _strict_count),
                ),
            )
        )
    if lane_id == "bfcl_mt":
        return tuple(
            _flat_metrics(
                summary,
                (
                    ("accuracy", "accuracy", "ratio", _strict_ratio),
                    ("expected_rows", "expected_rows", "count", _strict_count),
                ),
            )
        )
    if lane_id == "bfcl_ast":
        return tuple(
            _flat_metrics(
                summary,
                (
                    ("micro_accuracy", "micro_accuracy", "ratio", _strict_ratio),
                    ("micro_correct", "micro_correct", "count", _strict_count),
                    ("micro_total", "micro_total", "count", _strict_count),
                ),
            )
        )
    if lane_id == "latency":
        return tuple(_latency_metrics(summary))
    if lane_id == "concurrency":
        return tuple(_concurrency_metrics(summary))
    if lane_id == "throughput":
        return tuple(_throughput_metrics(summary))
    if lane_id == "niah":
        metrics = _flat_metrics(
            summary,
            (
                ("max_model_len", "max_model_len", "tokens", _strict_count),
                ("pass_count", "passed_depths", "count", _strict_count),
                ("total", "measured_depths", "count", _strict_count),
            ),
        )
        if "depths" in summary:
            depths = tuple(
                _strict_count(value) for value in _strict_list(summary["depths"])
            )
            if (
                len(depths) != 3
                or any(value <= 0 for value in depths)
                or not (depths[0] < depths[1] < depths[2])
            ):
                _raise("invalid_upstream_result")
            metrics.extend(
                _metric(name, value, "tokens")
                for name, value in zip(
                    ("depth_25", "depth_50", "depth_90"), depths, strict=True
                )
            )
        return tuple(metrics)
    if lane_id in {"qa", "ifeval", "gsm8k"}:
        return tuple(
            _flat_metrics(
                summary,
                (
                    ("accuracy", "accuracy", "ratio", _strict_ratio),
                    ("correct", "correct", "count", _strict_count),
                    ("n", "cases", "count", _strict_count),
                ),
            )
        )
    if lane_id == "humaneval":
        return tuple(
            _flat_metrics(
                summary,
                (
                    ("pass@1", "pass_at_1", "ratio", _strict_ratio),
                    ("n", "cases", "count", _strict_count),
                ),
            )
        )
    detail = _strict_mapping(summary.get("detail"))
    metrics: list[R0b0benchMetric] = []
    for component, normalizer in (
        ("latency", _latency_metrics),
        ("concurrency", _concurrency_metrics),
        ("throughput", _throughput_metrics),
    ):
        component_row = _strict_mapping(detail.get(component))
        component_summary = _strict_mapping(component_row.get("summary"))
        metrics.extend(normalizer(component_summary, prefix=f"{component}."))
    return tuple(metrics)


def _validate_artifacts(
    artifacts: Mapping[str, object], *, run_directory: Path
) -> None:
    for key, value in artifacts.items():
        _strict_string(key, maximum=256)
        raw = _strict_string(value)
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = run_directory / candidate
        try:
            resolved = candidate.resolve(strict=False)
            resolved.relative_to(run_directory)
        except (OSError, RuntimeError, ValueError):
            _raise("invalid_upstream_result")


def _parse_lane_object(
    value: object,
    *,
    expected_lane: R0b0benchLaneId,
    run_directory: Path,
) -> R0b0benchLaneResult:
    lane = _strict_mapping(value)
    if set(lane) != {
        "lane_id",
        "status",
        "summary",
        "artifacts",
        "infra_errors",
        "imported",
        "elapsed_s",
    }:
        _raise("invalid_upstream_result")
    if _strict_string(lane["lane_id"], maximum=32) != expected_lane:
        _raise("invalid_upstream_result")
    try:
        status_value = R0b0benchLaneStatus(_strict_string(lane["status"], maximum=32))
    except ValueError:
        _raise("invalid_upstream_result")
    summary = _strict_mapping(lane["summary"])
    artifacts = _strict_mapping(lane["artifacts"])
    _validate_artifacts(artifacts, run_directory=run_directory)
    _strict_bool(lane["imported"])
    elapsed = lane["elapsed_s"]
    elapsed_seconds = None if elapsed is None else _strict_number(elapsed)
    return R0b0benchLaneResult(
        lane_id=expected_lane,
        status=status_value,
        infra_errors=_strict_count(lane["infra_errors"]),
        elapsed_seconds=elapsed_seconds,
        metrics=_normalize_metrics(expected_lane, summary),
    )


def _lane_path(run_directory: Path, lane_id: R0b0benchLaneId) -> Path:
    lane_directory = run_directory / "lanes" / lane_id
    try:
        if lane_directory.resolve(strict=True) != lane_directory:
            _raise("invalid_upstream_result")
        info = lane_directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            _raise("invalid_upstream_result")
    except (OSError, RuntimeError):
        _raise("invalid_upstream_result")
    return lane_directory / "lane_result.json"


def _load_lane(
    run_directory: Path, lane_id: R0b0benchLaneId
) -> tuple[dict[str, object], R0b0benchLaneResult]:
    path = _lane_path(run_directory, lane_id)
    raw = _read_json(path, expected_parent=path.parent)
    return raw, _parse_lane_object(
        raw, expected_lane=lane_id, run_directory=run_directory
    )


def _parse_report(
    prepared: R0b0benchPreparedRun, reported_path: Path
) -> R0b0benchRunnerReport:
    request = prepared.request
    run_directory = prepared.run_directory
    expected_report = run_directory / "report.json"
    try:
        if reported_path.resolve(strict=True) != expected_report.resolve(strict=True):
            _raise("invalid_upstream_result")
    except (OSError, RuntimeError):
        _raise("invalid_upstream_result")
    report = _read_json(expected_report, expected_parent=run_directory)
    if set(report) != {
        "schema_version",
        "r0b0bench_version",
        "run_id",
        "profile",
        "base_url",
        "model",
        "systems_lanes",
        "invalid_for_publish",
        "started_utc",
        "elapsed_s",
        "lanes",
        "infra_errors_total",
    }:
        _raise("invalid_upstream_result")
    if (
        _strict_count(report["schema_version"]) != R0B0BENCH_REPORT_SCHEMA
        or _strict_string(report["r0b0bench_version"], maximum=32) != R0B0BENCH_VERSION
        or _strict_string(report["run_id"], maximum=128) != request.benchmark_id
        or _strict_string(report["profile"], maximum=32) != request.config.profile
        or _strict_string(report["base_url"]) != request.base_url
        or _strict_string(report["model"]) != request.model_id
    ):
        _raise("invalid_upstream_result")
    systems = tuple(
        _strict_string(value, maximum=32)
        for value in _strict_list(report["systems_lanes"])
    )
    if systems != R0B0BENCH_SYSTEMS_ORDER:
        _raise("invalid_upstream_result")
    try:
        started = datetime.fromisoformat(_strict_string(report["started_utc"]))
    except ValueError:
        _raise("invalid_upstream_result")
    if started.utcoffset() is None:
        _raise("invalid_upstream_result")
    elapsed = _strict_number(report["elapsed_s"])
    raw_lanes = _strict_list(report["lanes"])
    ordered = r0b0bench_ordered_selection(
        request.config.profile, request.config.selected_lanes
    )
    if len(raw_lanes) > len(ordered):
        _raise("invalid_upstream_result")
    lanes: list[R0b0benchLaneResult] = []
    for index, raw_lane in enumerate(raw_lanes):
        expected_lane = ordered[index]
        lane_copy, normalized = _load_lane(run_directory, expected_lane)
        if _strict_mapping(raw_lane) != lane_copy:
            _raise("invalid_upstream_result")
        lanes.append(normalized)
    lane_ids = tuple(lane.lane_id for lane in lanes)
    if lane_ids != ordered:
        canary_stop = (
            lane_ids == ("canary",)
            and lanes[0].status is R0b0benchLaneStatus.ERROR
            and lanes[0].infra_errors > 0
        )
        if not canary_stop:
            _raise("invalid_upstream_result")
    infra_errors = _strict_count(report["infra_errors_total"])
    if infra_errors != sum(lane.infra_errors for lane in lanes):
        _raise("invalid_upstream_result")
    canonical = r0b0bench_profile_lanes(request.config.profile)
    expected_invalid = (
        ordered != canonical
        or "perf" in request.config.selected_lanes
        or any(
            lane.lane_id != "perf" and lane.status is not R0b0benchLaneStatus.PASS
            for lane in lanes
        )
        or bool(infra_errors)
    )
    invalid = _strict_bool(report["invalid_for_publish"])
    if invalid != expected_invalid:
        _raise("invalid_upstream_result")
    return R0b0benchRunnerReport(
        upstream_run_id=request.benchmark_id,
        upstream_version=R0B0BENCH_VERSION,
        schema_version=R0B0BENCH_REPORT_SCHEMA,
        profile=request.config.profile,
        model_id=request.model_id,
        elapsed_seconds=elapsed,
        invalid_for_publish=invalid,
        infra_errors_total=infra_errors,
        lanes=tuple(lanes),
        unstarted_lanes=ordered[len(lanes) :],
        run_directory=run_directory,
        cancelled=False,
    )


def _partial_report(
    prepared: R0b0benchPreparedRun, *, elapsed_seconds: float
) -> R0b0benchRunnerReport:
    ordered = r0b0bench_ordered_selection(
        prepared.request.config.profile, prepared.request.config.selected_lanes
    )
    lanes: list[R0b0benchLaneResult] = []
    for lane_id in ordered:
        try:
            _, lane = _load_lane(prepared.run_directory, lane_id)
        except R0b0benchRunnerError:
            break
        lanes.append(lane)
    return R0b0benchRunnerReport(
        upstream_run_id=None,
        upstream_version=R0B0BENCH_VERSION,
        schema_version=None,
        profile=prepared.request.config.profile,
        model_id=prepared.request.model_id,
        elapsed_seconds=elapsed_seconds,
        invalid_for_publish=True,
        infra_errors_total=sum(lane.infra_errors for lane in lanes),
        lanes=tuple(lanes),
        unstarted_lanes=ordered[len(lanes) :],
        run_directory=prepared.run_directory,
        cancelled=True,
    )


async def _safe_callback(
    callback: Callable[..., Awaitable[None]], *args: object
) -> None:
    try:
        await callback(*args)
    except Exception:
        return


class SubprocessR0b0benchRunner:
    """Run and validate the pinned CLI in a private process group."""

    async def prepare(self, request: R0b0benchRunnerRequest) -> R0b0benchPreparedRun:
        try:
            if _RUN_ID.fullmatch(request.benchmark_id) is None:
                _raise("prerequisite_unavailable")
            if request.api_key not in {None, "", "EMPTY"}:
                _raise("unsupported_authenticated_endpoint")
            probe_environment = _base_child_environment({}, model_id=request.model_id)
            returncode, output = await _probe(
                sys.executable,
                "-m",
                "r0b0bench.cli",
                "--version",
                environment=probe_environment,
            )
            if returncode != 0 or output.decode("utf-8", "replace").strip() != (
                f"r0b0bench {R0B0BENCH_VERSION}"
            ):
                _raise("dependency_unavailable")
            prerequisites = resolve_r0b0bench_prerequisites(request.config)
            if prerequisites.issues:
                _raise("prerequisite_unavailable")
            environment = _base_child_environment(
                prerequisites.child_environment, model_id=request.model_id
            )
            if any(
                lane in request.config.selected_lanes
                for lane in ("bfcl_mt", "bfcl_ast")
            ):
                bfcl_python = environment["R0B0BENCH_BFCL_PYTHON"]
                code = (
                    "import importlib.metadata as m,sys;"
                    "sys.exit(0 if m.version('bfcl-eval')=="
                    f"'{BFCL_EVAL_VERSION}' else 1)"
                )
                bfcl_rc, _ = await _probe(
                    bfcl_python, "-c", code, environment=environment
                )
                if bfcl_rc != 0:
                    _raise("prerequisite_unavailable")
            root, run_directory = _allocate_run_directory(request)
            environment["HOME"] = str(run_directory)
            environment["TMPDIR"] = str(run_directory)
            normalized_request = R0b0benchRunnerRequest(
                benchmark_id=request.benchmark_id,
                config=request.config,
                base_url=request.base_url,
                model_id=request.model_id,
                output_root=root,
                api_key=request.api_key,
            )
            return R0b0benchPreparedRun(
                request=normalized_request,
                child_environment=environment,
                run_directory=run_directory,
            )
        except asyncio.CancelledError:
            raise
        except R0b0benchRunnerError as error:
            logger.error(
                "r0b0bench adapter failure benchmark=%s type=%s",
                request.benchmark_id,
                type(error).__name__,
            )
            raise
        except Exception as error:
            logger.error(
                "r0b0bench adapter failure benchmark=%s type=%s",
                request.benchmark_id,
                type(error).__name__,
            )
            _raise("prerequisite_unavailable")

    async def run(
        self,
        prepared: R0b0benchPreparedRun,
        on_lane_started: LaneStartedCallback,
        on_lane_finished: LaneFinishedCallback,
    ) -> R0b0benchRunnerReport:
        request = prepared.request
        started = time.monotonic()
        process: asyncio.subprocess.Process | None = None
        try:
            prerequisites = resolve_r0b0bench_prerequisites(request.config)
            if prerequisites.issues:
                _raise("prerequisite_unavailable")
            expected_environment = _base_child_environment(
                prerequisites.child_environment, model_id=request.model_id
            )
            expected_environment["HOME"] = str(prepared.run_directory)
            expected_environment["TMPDIR"] = str(prepared.run_directory)
            if dict(prepared.child_environment) != expected_environment:
                _raise("prerequisite_unavailable")
            _verify_prepared_directory(prepared)
            ordered = r0b0bench_ordered_selection(
                request.config.profile, request.config.selected_lanes
            )
            arguments = [
                sys.executable,
                "-m",
                "r0b0bench.cli",
                "run",
                "--profile",
                request.config.profile,
                "--base-url",
                request.base_url,
                "--model",
                request.model_id,
                "--output",
                str(request.output_root),
                "--run-id",
                request.benchmark_id,
                "--timeout",
                str(request.config.request_timeout_seconds),
            ]
            if request.config.tokenizer_path:
                arguments.extend(["--tokenizer", request.config.tokenizer_path])
            if ordered != r0b0bench_profile_lanes(request.config.profile):
                arguments.extend(["--only", ",".join(ordered)])
            try:
                process = await asyncio.create_subprocess_exec(
                    *arguments,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    env=dict(prepared.child_environment),
                    start_new_session=True,
                )
            except (OSError, ValueError):
                _raise("upstream_failure")
            assert process.stdout is not None
            reported_path: Path | None = None
            current_lane: R0b0benchLaneId | None = None
            buffer = bytearray()
            while True:
                chunk = await process.stdout.read(4096)
                if not chunk:
                    break
                buffer.extend(chunk)
                while b"\n" in buffer:
                    raw_line, _, remainder = buffer.partition(b"\n")
                    buffer = bytearray(remainder)
                    if len(raw_line) > 1024:
                        continue
                    line = raw_line.decode("utf-8", "replace").rstrip("\r")
                    marker = _LANE_MARKER.fullmatch(line)
                    if marker is not None:
                        lane_id = cast(R0b0benchLaneId, marker.group(1))
                        if lane_id not in ordered or (
                            current_lane is not None
                            and ordered.index(lane_id) <= ordered.index(current_lane)
                        ):
                            _raise("invalid_upstream_result")
                        current_lane = lane_id
                        await _safe_callback(
                            on_lane_started,
                            lane_id,
                            ordered.index(lane_id) + 1,
                            len(ordered),
                        )
                        continue
                    if line.startswith("report: "):
                        raw_path = line.removeprefix("report: ")
                        if not raw_path or len(raw_path) > 768:
                            _raise("invalid_upstream_result")
                        reported_path = Path(raw_path)
                        continue
                    if current_lane is not None and line.startswith("{"):
                        try:
                            hint = json.loads(line, parse_constant=_reject_constant)
                        except (json.JSONDecodeError, ValueError):
                            continue
                        if not isinstance(hint, dict):
                            continue
                        hint_mapping = cast(dict[object, object], hint)
                        if set(hint_mapping) != {
                            "lane",
                            "status",
                            "infra_errors",
                        }:
                            continue
                        if hint_mapping.get("lane") != current_lane:
                            continue
                        _, lane = _load_lane(prepared.run_directory, current_lane)
                        await _safe_callback(
                            on_lane_finished,
                            lane,
                            ordered.index(current_lane) + 1,
                            len(ordered),
                        )
            returncode = await process.wait()
            if reported_path is not None:
                report = _parse_report(prepared, reported_path)
                return report
            if returncode != 0:
                _raise("upstream_failure")
            _raise("invalid_upstream_result")
        except asyncio.CancelledError:
            if process is not None:
                await _reap_process(process)
            return _partial_report(
                prepared, elapsed_seconds=max(0.0, time.monotonic() - started)
            )
        except R0b0benchRunnerError as error:
            if process is not None:
                await _reap_process(process)
            logger.error(
                "r0b0bench adapter failure benchmark=%s type=%s",
                request.benchmark_id,
                type(error).__name__,
            )
            raise
        except Exception as error:
            if process is not None:
                await _reap_process(process)
            logger.error(
                "r0b0bench adapter failure benchmark=%s type=%s",
                request.benchmark_id,
                type(error).__name__,
            )
            _raise("invalid_upstream_result")
