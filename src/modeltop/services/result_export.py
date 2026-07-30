"""Allowlisted JSON export and plain-text summaries for Speed Test results."""

import json
import logging
import os
import re
import tempfile
import unicodedata
from pathlib import Path

from modeltop.benchmarks.models import (
    MetricStatistics,
    SpeedTestResult,
    SpeedTestRunResult,
)
from modeltop.hardware.models import HardwareSnapshot, summarize_gpus

logger = logging.getLogger(__name__)

_DEFAULT_RESULTS_DIRECTORY = Path("~/.local/share/modeltop/results")
_COMPONENT_LIMIT = 64


class ResultExportError(Exception):
    """Sanitized filesystem failure exposed to the dashboard."""

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


def _statistics_payload(statistics: MetricStatistics) -> dict[str, int | float | None]:
    return {
        "count": statistics.count,
        "mean": statistics.mean,
        "median": statistics.median,
        "minimum": statistics.minimum,
        "maximum": statistics.maximum,
        "p95": statistics.p95,
        "standard_deviation": statistics.standard_deviation,
    }


def _run_payload(run: SpeedTestRunResult) -> dict[str, object]:
    return {
        "run_number": run.run_number,
        "phase": "warmup" if run.warmup else "measured",
        "warmup": run.warmup,
        "success": run.success,
        "cancelled": run.cancelled,
        "error": run.error,
        "prompt_tokens": run.prompt_tokens,
        "completion_tokens": run.completion_tokens,
        "total_tokens": run.total_tokens,
        "prompt_tokens_estimated": run.prompt_tokens_estimated,
        "completion_tokens_estimated": run.completion_tokens_estimated,
        "total_tokens_estimated": run.total_tokens_estimated,
        "ttft_ms": run.ttft_ms,
        "generation_duration_s": run.generation_duration_s,
        "total_duration_s": run.total_duration_s,
        "output_tokens_per_second": run.output_tokens_per_second,
        "finish_reason": run.finish_reason,
        "streamed": run.streamed,
        "response_character_count": run.response_character_count,
    }


def _hardware_payload(snapshot: HardwareSnapshot | None) -> dict[str, object] | None:
    if snapshot is None:
        return None
    return {
        "provider_name": snapshot.provider_name,
        "collected_at": snapshot.collected_at.isoformat(),
        "error": snapshot.error,
        "gpus": [
            {
                "index": gpu.index,
                "name": gpu.name,
                "utilisation_percent": gpu.utilisation_percent,
                "memory_used_bytes": gpu.memory_used_bytes,
                "memory_total_bytes": gpu.memory_total_bytes,
                "temperature_celsius": gpu.temperature_celsius,
                "power_draw_watts": gpu.power_draw_watts,
                "power_limit_watts": gpu.power_limit_watts,
                "fan_speed_percent": gpu.fan_speed_percent,
            }
            for gpu in snapshot.gpus
        ],
        "cpu": {
            "utilisation_percent": snapshot.cpu.utilisation_percent,
            "logical_core_count": snapshot.cpu.logical_core_count,
            "physical_core_count": snapshot.cpu.physical_core_count,
            "load_average_1m": snapshot.cpu.load_average_1m,
            "load_average_5m": snapshot.cpu.load_average_5m,
            "load_average_15m": snapshot.cpu.load_average_15m,
        },
        "memory": {
            "used_bytes": snapshot.memory.used_bytes,
            "total_bytes": snapshot.memory.total_bytes,
            "utilisation_percent": snapshot.memory.utilisation_percent,
        },
    }


def speed_test_result_payload(result: SpeedTestResult) -> dict[str, object]:
    """Return an explicit public schema with no request or response content."""
    return {
        "schema_version": "1.0",
        "benchmark": "speed-test",
        "run_id": result.run_id,
        "status": result.status.value,
        "started_at": result.started_at.isoformat(),
        "completed_at": result.completed_at.isoformat(),
        "server": {
            "id": result.server_id,
            "name": result.server_name,
            "endpoint": result.server_endpoint,
            "backend": result.backend,
        },
        "model": {"id": result.model_id},
        "configuration": result.config.model_dump(mode="json"),
        "counts": {
            "configured_warmup_runs": result.warmup_runs,
            "configured_measured_runs": result.measured_runs,
            "attempted_warmup_runs": result.attempted_warmup_runs,
            "attempted_measured_runs": result.attempted_measured_runs,
            "successful_measured_runs": result.successful_runs,
            "failed_measured_runs": result.failed_runs,
            "cancelled_measured_runs": result.cancelled_runs,
        },
        "runs": [_run_payload(run) for run in result.run_results],
        "aggregates": {
            "ttft_ms": _statistics_payload(result.ttft_ms),
            "output_tokens_per_second": _statistics_payload(
                result.output_tokens_per_second
            ),
            "total_duration_s": _statistics_payload(result.total_duration_s),
            "generation_duration_s": _statistics_payload(result.generation_duration_s),
            "prompt_tokens": _statistics_payload(result.prompt_tokens),
            "completion_tokens": _statistics_payload(result.completion_tokens),
            "estimated_token_metrics": result.estimated_measured_metrics,
        },
        "hardware": {
            "before": _hardware_payload(result.hardware_before),
            "after": _hardware_payload(result.hardware_after),
        },
        "error": result.error,
    }


def _safe_component(value: str, *, fallback: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    folded = normalized.encode("ascii", "ignore").decode("ascii").lower()
    component = re.sub(r"[^a-z0-9]+", "-", folded).strip("-")
    component = component[:_COMPONENT_LIMIT].rstrip("-")
    return component or fallback


def _safe_os_message(error: OSError) -> str:
    return error.strerror or type(error).__name__


def export_speed_test_result(
    result: SpeedTestResult,
    directory: Path | None = None,
) -> Path:
    """Durably write a new JSON file without ever replacing an existing export."""
    target_directory = (directory or _DEFAULT_RESULTS_DIRECTORY).expanduser().resolve()
    model = _safe_component(result.model_id, fallback="model")
    run_id = _safe_component(result.run_id, fallback="run")
    timestamp = result.completed_at.strftime("%Y%m%d-%H%M%S")
    stem = f"{timestamp}_speed-test_{model}_{run_id}"
    intended = target_directory / f"{stem}.json"
    temporary: Path | None = None
    published: Path | None = None
    failure: OSError | None = None

    try:
        target_directory.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target_directory,
            prefix=".modeltop-result-",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        content = (
            json.dumps(
                speed_test_result_payload(result),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            + "\n"
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())

        suffix = 1
        while True:
            candidate = (
                intended if suffix == 1 else target_directory / f"{stem}_{suffix}.json"
            )
            try:
                os.link(temporary, candidate)
            except FileExistsError:
                suffix += 1
                continue
            published = candidate
            break
    except OSError as error:
        failure = error
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as error:
                if failure is None:
                    failure = error

    if failure is not None:
        logger.error(
            "Result export failed error=%s path=%s",
            type(failure).__name__,
            intended,
        )
        raise ResultExportError(
            f"Result export failed: {_safe_os_message(failure)}"
        ) from failure
    if published is None:
        raise ResultExportError("Result export failed: output was not published")
    logger.info("Speed Test result exported run=%s path=%s", result.run_id, published)
    return published


def _number(value: float | None, *, estimated: bool = False) -> str:
    if value is None:
        return "--"
    return f"{'~' if estimated else ''}{value:.2f}"


def _summary_statistics(
    label: str,
    statistics: MetricStatistics,
    fields: tuple[str, ...],
    *,
    estimated: bool = False,
    unit: str = "",
) -> str:
    values: list[str] = []
    for field in fields:
        value = getattr(statistics, field)
        formatted = _number(value, estimated=estimated)
        if formatted != "--" and unit:
            formatted = f"{formatted} {unit}"
        values.append(f"{field} {formatted}")
    return f"{label}: " + " · ".join(values)


def _hardware_summary(snapshot: HardwareSnapshot | None) -> str:
    if snapshot is None:
        return "--"
    return summarize_gpus(snapshot.gpus).display_name


def format_speed_test_summary(result: SpeedTestResult) -> str:
    """Format a concise clipboard-safe summary from the same immutable result."""
    estimated = result.estimated_measured_metrics
    return "\n".join(
        (
            "ModelTop Speed Test",
            f"Status: {result.status.value.replace('_', ' ').upper()}",
            f"Model: {result.model_id}",
            f"Server: {result.server_name} ({result.server_id}) · "
            f"{result.server_endpoint} · Backend: {result.backend}",
            f"Runs: {result.successful_runs}/{result.measured_runs} successful · "
            f"{result.attempted_warmup_runs}/{result.warmup_runs} warm-ups attempted · "
            f"{result.failed_runs} failed · {result.cancelled_runs} cancelled",
            "Thinking: "
            + (
                "DISABLED (Qwen/vLLM request override)"
                if result.config.thinking_mode == "disabled"
                else "SERVER DEFAULT"
            ),
            _summary_statistics(
                "Output speed",
                result.output_tokens_per_second,
                ("mean", "median", "minimum", "maximum", "p95"),
                estimated=estimated,
                unit="tok/s",
            ),
            _summary_statistics(
                "TTFT",
                result.ttft_ms,
                ("mean", "median", "p95"),
                unit="ms",
            ),
            _summary_statistics(
                "Prompt tokens",
                result.prompt_tokens,
                ("mean", "median"),
                estimated=estimated,
            ),
            _summary_statistics(
                "Completion tokens",
                result.completion_tokens,
                ("mean", "median"),
                estimated=estimated,
            ),
            f"GPU before: {_hardware_summary(result.hardware_before)}",
            f"GPU after: {_hardware_summary(result.hardware_after)}",
        )
    )
