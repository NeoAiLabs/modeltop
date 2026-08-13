"""Durable, redacted benchmark result archive."""
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportPrivateUsage=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Literal

from modeltop.benchmarks.models import (
    ConcurrencyBenchmarkResult,
    ContextBenchmarkResult,
    DrafterBenchmarkResult,
    MetricStatistics,
    PercentileStatistics,
    R0b0benchBenchmarkResult,
    SpeedTestResult,
    ToolCallingBenchmarkResult,
)
from modeltop.services.result_export import (
    ResultExportError,
    _hardware_payload,
    _safe_component,
    _safe_os_message,
    _statistics_payload,
)

logger = logging.getLogger(__name__)

ArchivedResultKind = Literal[
    "speed-test",
    "concurrency",
    "context",
    "tool-calling",
    "r0b0bench",
    "drafter",
]
_Result = (
    SpeedTestResult
    | ConcurrencyBenchmarkResult
    | ContextBenchmarkResult
    | ToolCallingBenchmarkResult
    | R0b0benchBenchmarkResult
    | DrafterBenchmarkResult
)
_SCHEMA_VERSION = "1.0"
_DEFAULT_HISTORY_DIRECTORY = Path("~/.local/share/modeltop/history")


@dataclass(frozen=True, slots=True)
class ArchiveEntry:
    """Small, indexable metadata for one immutable archived result."""

    result_id: str
    kind: ArchivedResultKind
    completed_at: datetime
    status: str
    server_id: str
    server_name: str
    model_id: str
    backend: str
    configuration_fingerprint: str
    summary: Mapping[str, int | float | str | bool | None]
    document_name: str


@dataclass(frozen=True, slots=True)
class ArchivedResultDocument:
    """One entry paired with its explicitly allowlisted details."""

    entry: ArchiveEntry
    details: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ResultArchiveSnapshot:
    """Newest-first valid archive entries and loaded documents."""

    entries: tuple[ArchiveEntry, ...] = ()
    documents: Mapping[str, ArchivedResultDocument] = field(default_factory=dict)
    archive_selection: tuple[str, ...] = ()
    load_error: str | None = None
    skipped_entries: tuple[ArchiveEntry, ...] = ()

    def __post_init__(self) -> None:
        if len(self.archive_selection) > 2:
            raise ValueError("archive selection must contain at most two IDs")


class ResultArchive:
    """Append-only JSON archive with atomic publication and redacted payloads."""

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = (directory or _DEFAULT_HISTORY_DIRECTORY).expanduser()

    def load_archive(
        self, archive_selection: tuple[str, ...] = ()
    ) -> ResultArchiveSnapshot:
        index_path = self.directory / "index.json"
        try:
            raw = json.loads(index_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return ResultArchiveSnapshot(archive_selection=archive_selection)
        except (OSError, json.JSONDecodeError) as error:
            logger.warning(
                "Result archive index unavailable error=%s", type(error).__name__
            )
            return ResultArchiveSnapshot(
                load_error=f"Result archive unavailable: {type(error).__name__}"
            )
        if (
            not isinstance(raw, dict)
            or raw.get("schema_version") != _SCHEMA_VERSION
            or not isinstance(raw.get("entries"), list)
        ):
            logger.warning("Result archive index invalid")
            return ResultArchiveSnapshot(
                load_error="Result archive unavailable: invalid index"
            )

        entries: list[ArchiveEntry] = []
        skipped: list[ArchiveEntry] = []
        documents: dict[str, ArchivedResultDocument] = {}
        invalid_index = False
        seen: set[str] = set()
        for item in raw["entries"]:
            try:
                entry = _entry_from_payload(item)
            except (ValueError, TypeError) as error:
                invalid_index = True
                logger.warning(
                    "Result archive index entry invalid error=%s", type(error).__name__
                )
                continue
            if entry.result_id in seen:
                invalid_index = True
                logger.warning("Result archive index contains a duplicate result ID")
                continue
            seen.add(entry.result_id)
            try:
                document = _document_from_payload(
                    json.loads(
                        (self.directory / entry.document_name).read_text(
                            encoding="utf-8"
                        )
                    )
                )
                if document.entry != entry:
                    raise ValueError("document metadata mismatch")
                entries.append(entry)
                documents[entry.result_id] = document
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                skipped.append(entry)
                logger.warning(
                    "Result archive entry skipped error=%s", type(error).__name__
                )
        entries.sort(key=lambda item: item.completed_at, reverse=True)
        skipped.sort(key=lambda item: item.completed_at, reverse=True)
        valid_selection = tuple(
            result_id for result_id in archive_selection if result_id in documents
        )[:2]
        if invalid_index:
            load_error = "Result archive unavailable: invalid index"
        elif skipped:
            load_error = "Result archive contains unreadable entries."
        else:
            load_error = None
        return ResultArchiveSnapshot(
            tuple(entries),
            documents,
            valid_selection,
            load_error,
            tuple(skipped),
        )

    def archive_result(
        self, result: _Result, archive_selection: tuple[str, ...] = ()
    ) -> ResultArchiveSnapshot:
        snapshot = self.load_archive(archive_selection)
        result_id, kind = _identity(result)
        known_ids = {entry.result_id for entry in snapshot.entries}
        known_ids.update(entry.result_id for entry in snapshot.skipped_entries)
        if result_id in known_ids or result_id in snapshot.documents:
            return snapshot
        if snapshot.load_error is not None and snapshot.load_error.startswith(
            "Result archive unavailable"
        ):
            return snapshot
        document = _result_document(result, kind)
        document = self._with_available_name(document)
        try:
            self._write_document(document)
            updated_entries = tuple(
                sorted(
                    (*snapshot.entries, document.entry),
                    key=lambda item: item.completed_at,
                    reverse=True,
                )
            )
            index_entries = tuple(
                sorted(
                    (*snapshot.entries, *snapshot.skipped_entries, document.entry),
                    key=lambda item: item.completed_at,
                    reverse=True,
                )
            )
            self._write_index(index_entries)
        except ResultExportError:
            raise
        documents = dict(snapshot.documents)
        documents[result_id] = document
        return ResultArchiveSnapshot(
            updated_entries,
            documents,
            snapshot.archive_selection,
            snapshot.load_error,
            snapshot.skipped_entries,
        )

    def _write_document(self, document: ArchivedResultDocument) -> None:
        path = self.directory / document.entry.document_name
        self._atomic_publish(
            path, _document_payload(document), "Result archive write failed"
        )

    def _with_available_name(
        self, document: ArchivedResultDocument
    ) -> ArchivedResultDocument:
        """Choose a new immutable filename without replacing a collision."""
        path = self.directory / document.entry.document_name
        suffix = 2
        while path.exists():
            path = path.with_name(
                f"{path.stem.rsplit('_', 1)[0]}_{suffix}{path.suffix}"
            )
            suffix += 1
        return replace(
            document,
            entry=replace(
                document.entry,
                document_name=str(path.relative_to(self.directory)),
            ),
        )

    def _write_index(self, entries: tuple[ArchiveEntry, ...]) -> None:
        self._atomic_replace(
            self.directory / "index.json",
            {
                "schema_version": _SCHEMA_VERSION,
                "entries": [_entry_payload(entry) for entry in entries],
            },
        )

    def _atomic_publish(
        self, path: Path, payload: Mapping[str, object], prefix: str
    ) -> None:
        temporary: Path | None = None
        failure: OSError | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = _write_temporary(path.parent, payload)
            try:
                os.link(temporary, path)
            except FileExistsError:
                # A concurrent equivalent writer won; keep its immutable document.
                return
        except OSError as error:
            failure = error
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as error:
                    failure = failure or error
        if failure is not None:
            logger.error("Result archive write failed error=%s", type(failure).__name__)
            raise ResultExportError(
                f"{prefix}: {_safe_os_message(failure)}"
            ) from failure

    def _atomic_replace(self, path: Path, payload: Mapping[str, object]) -> None:
        temporary: Path | None = None
        failure: OSError | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = _write_temporary(path.parent, payload)
            os.replace(temporary, path)
            temporary = None
        except OSError as error:
            failure = error
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as error:
                    failure = failure or error
        if failure is not None:
            logger.error(
                "Result archive index write failed error=%s", type(failure).__name__
            )
            raise ResultExportError(
                f"Result archive write failed: {_safe_os_message(failure)}"
            ) from failure


def _write_temporary(directory: Path, payload: Mapping[str, object]) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=directory, prefix=".modeltop-history-", suffix=".tmp"
    )
    temporary = Path(name)
    content = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())
    return temporary


def _identity(result: _Result) -> tuple[str, ArchivedResultKind]:
    if isinstance(result, SpeedTestResult):
        return result.run_id, "speed-test"
    if isinstance(result, ConcurrencyBenchmarkResult):
        return result.benchmark_id, "concurrency"
    if isinstance(result, ContextBenchmarkResult):
        return result.benchmark_id, "context"
    if isinstance(result, ToolCallingBenchmarkResult):
        return result.benchmark_id, "tool-calling"
    if isinstance(result, R0b0benchBenchmarkResult):
        return result.benchmark_id, "r0b0bench"
    return result.benchmark_id, "drafter"


def _config_fingerprint(result: _Result) -> str:
    config = result.config.model_dump(mode="json")
    content = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(content.encode()).hexdigest()


def _safe_config(result: _Result) -> dict[str, object]:
    unsafe = {
        "prompt",
        "system_prompt",
        "base_text",
        "retrieval_key",
        "tokenizer_path",
        "bfcl_python",
        "bfcl_scripts_directory",
        "qa_data_path",
        "ifeval_data_path",
        "humaneval_data_path",
        "gsm8k_data_path",
        "allow_unsafe_humaneval",
    }
    return {
        key: value
        for key, value in result.config.model_dump(mode="json").items()
        if key not in unsafe
    }


def _stat(
    value: MetricStatistics | PercentileStatistics,
) -> dict[str, int | float | None]:
    if isinstance(value, MetricStatistics):
        return _statistics_payload(value)
    return {
        name: getattr(value, name)
        for name in (
            "count",
            "mean",
            "median",
            "minimum",
            "maximum",
            "p50",
            "p90",
            "p95",
            "p99",
            "standard_deviation",
        )
    }


def _hardware_summary(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        name: getattr(value, name)
        for name in (
            "sample_count",
            "average_gpu_utilisation_percent",
            "maximum_gpu_utilisation_percent",
            "average_vram_used_bytes",
            "maximum_vram_used_bytes",
            "average_temperature_celsius",
            "maximum_temperature_celsius",
            "average_power_draw_watts",
            "maximum_power_draw_watts",
            "average_cpu_utilisation_percent",
            "maximum_cpu_utilisation_percent",
            "average_memory_used_bytes",
            "maximum_memory_used_bytes",
        )
    }


def _result_document(
    result: _Result, kind: ArchivedResultKind
) -> ArchivedResultDocument:
    result_id, _ = _identity(result)
    timestamp = result.completed_at.strftime("%Y%m%d-%H%M%S")
    component = _safe_component(result_id, fallback="run")
    name = f"runs/{kind}/{timestamp}_{component}.json"
    details, summary = _details_and_summary(result, kind)
    entry = ArchiveEntry(
        result_id,
        kind,
        result.completed_at,
        result.status.value,
        result.server_id,
        result.server_name,
        result.model_id,
        result.backend,
        _config_fingerprint(result),
        summary,
        name,
    )
    return ArchivedResultDocument(entry, details)


def _details_and_summary(
    result: _Result, kind: ArchivedResultKind
) -> tuple[dict[str, object], dict[str, int | float | str | bool | None]]:
    details: dict[str, object] = {
        "started_at": result.started_at.isoformat(),
        "configuration": _safe_config(result),
        "error": getattr(result, "error", None),
    }
    if kind in {"speed-test", "drafter"}:
        rows = []
        for run in result.run_results:  # type: ignore[union-attr]
            rows.append(
                {
                    name: getattr(run, name)
                    for name in run.__dataclass_fields__
                    if name not in {"response_character_count"}
                }
            )
        aggregates = {
            name: _stat(getattr(result, name))
            for name in (
                "ttft_ms",
                "output_tokens_per_second",
                "total_duration_s",
                "generation_duration_s",
                "prompt_tokens",
                "completion_tokens",
            )
        }
        if kind == "drafter":
            aggregates.update(
                {
                    name: _stat(getattr(result, name))
                    for name in ("draft_tokens", "accepted_tokens", "acceptance_rate")
                }
            )
            details["observations"] = [
                {"code": item.code, "message": item.message}
                for item in result.observations
            ]  # type: ignore[union-attr]
        details.update(
            {
                "counts": {
                    "configured_warmup_runs": result.warmup_runs,
                    "configured_measured_runs": result.measured_runs,
                    "attempted_warmup_runs": result.attempted_warmup_runs,
                    "attempted_measured_runs": result.attempted_measured_runs,
                    "successful_measured_runs": result.successful_runs,
                    "failed_measured_runs": result.failed_runs,
                    "cancelled_measured_runs": result.cancelled_runs,
                },
                "runs": rows,
                "aggregates": aggregates,
                "hardware": {
                    "before": _hardware_payload(result.hardware_before),
                    "after": _hardware_payload(result.hardware_after),
                },
            }
        )  # type: ignore[union-attr]
        return details, {
            "successful_runs": result.successful_runs,
            "mean_output_tokens_per_second": result.output_tokens_per_second.mean,
            "ttft_p95_ms": result.ttft_ms.p95,
        }
    if kind == "concurrency":
        levels = [
            {
                "concurrency": level.concurrency,
                "configured_requests": level.configured_requests,
                "attempted_requests": level.attempted_requests,
                "completed_requests": level.completed_requests,
                "successful_requests": level.successful_requests,
                "failed_requests": level.failed_requests,
                "cancelled_requests": level.cancelled_requests,
                "timed_out_requests": level.timed_out_requests,
                "wall_time_seconds": level.wall_time_seconds,
                "success_rate_percent": level.success_rate_percent,
                "total_prompt_tokens": level.total_prompt_tokens,
                "total_completion_tokens": level.total_completion_tokens,
                "token_count_mode": level.token_count_mode,
                "requests_per_second": level.requests_per_second,
                "aggregate_output_tokens_per_second": (
                    level.aggregate_output_tokens_per_second
                ),
                "ttft_ms": _stat(level.ttft_ms),
                "latency_seconds": _stat(level.latency_seconds),
                "generation_duration_seconds": _stat(level.generation_duration_seconds),
                "request_output_tokens_per_second": _stat(
                    level.request_output_tokens_per_second
                ),
                "completion_tokens": _stat(level.completion_tokens),
                "hardware_summary": _hardware_summary(level.hardware_summary),
                "output_length_warning": level.output_length_warning,
                "partial": level.partial,
                "early_stopped": level.early_stopped,
                "observations": [
                    {"code": x.code, "concurrency": x.concurrency, "message": x.message}
                    for x in level.observations
                ],
            }
            for level in result.levels
        ]  # type: ignore[union-attr]
        details.update(
            {
                "cancelled": result.cancelled,
                "warnings": list(result.warnings),
                "observations": [
                    {"code": x.code, "concurrency": x.concurrency, "message": x.message}
                    for x in result.observations
                ],
                "levels": levels,
            }
        )  # type: ignore[union-attr]
        peak = max(
            (level["aggregate_output_tokens_per_second"] for level in levels),
            default=None,
        )
        return details, {
            "peak_aggregate_output_tokens_per_second": peak,
            "level_count": len(levels),
            "cancelled": result.cancelled,
        }  # type: ignore[union-attr]
    if kind == "context":
        lengths = [
            {
                "target_length": row.target_length,
                "context_unit": row.context_unit,
                "effective_total_budget_tokens": row.effective_total_budget_tokens,
                "configured_requests": row.configured_requests,
                "attempted_requests": row.attempted_requests,
                "completed_requests": row.completed_requests,
                "accepted_requests": row.accepted_requests,
                "successful_requests": row.successful_requests,
                "failed_requests": row.failed_requests,
                "timed_out_requests": row.timed_out_requests,
                "cancelled_requests": row.cancelled_requests,
                "context_rejected_requests": row.context_rejected_requests,
                "success_rate_percent": row.success_rate_percent,
                "prompt_tokens": _stat(row.prompt_tokens),
                "ttft_ms": _stat(row.ttft_ms),
                "latency_seconds": _stat(row.latency_seconds),
                "output_tokens_per_second": _stat(row.output_tokens_per_second),
                "estimated_input_tokens_per_second": _stat(
                    row.estimated_input_tokens_per_second
                ),
                "completion_tokens": _stat(row.completion_tokens),
                "retrieval_attempts_by_position": list(
                    row.retrieval_attempts_by_position
                ),
                "retrieval_successes_by_position": list(
                    row.retrieval_successes_by_position
                ),
                "retrieval_rate_by_position": list(row.retrieval_rate_by_position),
                "hardware_summary": _hardware_summary(row.hardware_summary),
                "partial": row.partial,
                "early_stopped": row.early_stopped,
                "observations": [
                    {
                        "code": x.code,
                        "target_length": x.target_length,
                        "message": x.message,
                    }
                    for x in row.observations
                ],
            }
            for row in result.lengths
        ]  # type: ignore[union-attr]
        bounds = result.probe_bounds  # type: ignore[union-attr]
        details.update(
            {
                "highest_successful_prompt_tokens": (
                    result.highest_successful_prompt_tokens
                ),
                "first_fully_rejected_prompt_tokens": (
                    result.first_fully_rejected_prompt_tokens
                ),
                "possible_truncation": result.possible_truncation,
                "cancelled": result.cancelled,
                "warnings": list(result.warnings),
                "lengths": lengths,
                "probe_bounds": None
                if bounds is None
                else {
                    name: getattr(bounds, name) for name in bounds.__dataclass_fields__
                },
                "observations": [
                    {
                        "code": x.code,
                        "target_length": x.target_length,
                        "message": x.message,
                    }
                    for x in result.observations
                ],
            }
        )  # type: ignore[union-attr]
        return details, {
            "highest_successful_prompt_tokens": result.highest_successful_prompt_tokens,
            "length_count": len(lengths),
            "possible_truncation": result.possible_truncation,
        }  # type: ignore[union-attr]
    if kind == "r0b0bench":
        bench = result  # type: ignore[assignment]
        pass_count = sum(row.status.value == "PASS" for row in bench.lanes)
        fail_count = sum(row.status.value == "FAIL" for row in bench.lanes)
        error_count = sum(
            row.status.value in {"ERROR", "NOT_IMPLEMENTED"} for row in bench.lanes
        )
        details.update(
            {
                "profile": bench.config.profile,
                "selected_lanes": list(bench.config.selected_lanes),
                "upstream": {
                    "run_id": bench.upstream_run_id,
                    "version": bench.upstream_version,
                    "schema_version": bench.upstream_schema_version,
                    "commit": bench.upstream_commit,
                },
                "cancelled": bench.cancelled,
                "error_code": bench.error_code,
                "error_message": bench.error_message,
                "invalid_for_publish": bench.invalid_for_publish,
                "counts": {
                    "selected": bench.selected_count,
                    "completed": bench.completed_count,
                    "pass": pass_count,
                    "fail": fail_count,
                    "error": error_count,
                    "infra": bench.infra_errors_total,
                },
                "unstarted_lanes": list(bench.unstarted_lanes),
                "warning_codes": list(bench.warning_codes),
                "hardware_summary": _hardware_summary(bench.hardware_summary),
                "lanes": [
                    {
                        "lane_id": row.lane_id,
                        "status": row.status.value,
                        "infra_errors": row.infra_errors,
                        "elapsed_seconds": row.elapsed_seconds,
                        "metrics": [
                            {
                                "name": metric.name,
                                "value": metric.value,
                                "unit": metric.unit,
                            }
                            for metric in row.metrics
                        ],
                    }
                    for row in bench.lanes
                ],
            }
        )
        return details, {
            "selected_count": bench.selected_count,
            "completed_count": bench.completed_count,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "error_count": error_count,
            "infra_errors_total": bench.infra_errors_total,
            "invalid_for_publish": bench.invalid_for_publish,
        }

    tool = result  # type: ignore[assignment]
    details.update(
        {
            "upstream": {
                "run_id": tool.upstream_run_id,
                "config_fingerprint": tool.config_fingerprint,
                "integration_commit": tool.integration_commit,
                "version": tool.upstream_version,
                "schema_version": tool.schema_version,
            },
            "cancelled": tool.cancelled,
            "error_code": tool.error_code,
            "error_message": tool.error_message,
            "coverage": {
                "attempted_count": tool.attempted_count,
                "gradable_count": tool.gradable_count,
                "excluded_count": tool.excluded_count,
                "completion_rate_percent": tool.completion_rate_percent,
            },
            "scoring": {
                "final_score": tool.final_score,
                "total_points": tool.total_points,
                "max_points": tool.max_points,
                "rating": tool.rating,
                "category_k_gradable": tool.category_k_gradable,
                "safety_gate_passed": tool.safety_gate_passed,
                "deployability": tool.deployability,
                "responsiveness": tool.responsiveness,
                "median_turn_ms": tool.median_turn_ms,
            },
            "tokens": {
                "prompt_tokens": tool.prompt_tokens,
                "completion_tokens": tool.completion_tokens,
                "total_tokens": tool.total_tokens,
            },
            "categories": [
                {name: getattr(row, name) for name in row.__dataclass_fields__}
                for row in tool.categories
            ],
            "scenarios": [
                {name: getattr(row, name) for name in row.__dataclass_fields__}
                for row in tool.scenarios
            ],
            "warnings": list(tool.warnings),
            "hardware_summary": _hardware_summary(tool.hardware_summary),
        }
    )
    return details, {
        "final_score": tool.final_score,
        "completion_rate_percent": tool.completion_rate_percent,
        "total_points": tool.total_points,
        "max_points": tool.max_points,
    }


def _entry_payload(entry: ArchiveEntry) -> dict[str, object]:
    return {
        "result_id": entry.result_id,
        "kind": entry.kind,
        "completed_at": entry.completed_at.isoformat(),
        "status": entry.status,
        "server_id": entry.server_id,
        "server_name": entry.server_name,
        "model_id": entry.model_id,
        "backend": entry.backend,
        "configuration_fingerprint": entry.configuration_fingerprint,
        "summary": dict(entry.summary),
        "document_name": entry.document_name,
    }


def _document_payload(document: ArchivedResultDocument) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "entry": _entry_payload(document.entry),
        "details": dict(document.details),
    }


def _entry_from_payload(payload: object) -> ArchiveEntry:
    if not isinstance(payload, dict) or set(payload) != {
        "result_id",
        "kind",
        "completed_at",
        "status",
        "server_id",
        "server_name",
        "model_id",
        "backend",
        "configuration_fingerprint",
        "summary",
        "document_name",
    }:
        raise ValueError("invalid archive entry")
    kind = payload["kind"]
    if kind not in {
        "speed-test",
        "concurrency",
        "context",
        "tool-calling",
        "r0b0bench",
        "drafter",
    } or not isinstance(payload["summary"], dict):
        raise ValueError("invalid archive entry fields")
    completed_at = datetime.fromisoformat(str(payload["completed_at"]))
    if completed_at.utcoffset() is None:
        raise ValueError("archive timestamp must be timezone aware")
    return ArchiveEntry(
        str(payload["result_id"]),
        kind,
        completed_at,
        str(payload["status"]),
        str(payload["server_id"]),
        str(payload["server_name"]),
        str(payload["model_id"]),
        str(payload["backend"]),
        str(payload["configuration_fingerprint"]),
        payload["summary"],
        str(payload["document_name"]),
    )


def _document_from_payload(payload: object) -> ArchivedResultDocument:
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != _SCHEMA_VERSION
        or set(payload) != {"schema_version", "entry", "details"}
        or not isinstance(payload["details"], dict)
    ):
        raise ValueError("invalid archive document")
    return ArchivedResultDocument(
        _entry_from_payload(payload["entry"]), payload["details"]
    )


def load_archive(archive_selection: tuple[str, ...] = ()) -> ResultArchiveSnapshot:
    """Load the default archive for application startup."""
    return ResultArchive().load_archive(archive_selection)
