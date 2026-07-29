"""Allowlisted Speed Test JSON export and clipboard summary contracts."""

import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from modeltop.benchmarks.models import (
    MetricStatistics,
    SpeedTestConfig,
    SpeedTestResult,
    SpeedTestRunResult,
    SpeedTestStatus,
)
from modeltop.hardware.models import (
    CpuMetrics,
    GpuMetrics,
    HardwareSnapshot,
    MemoryMetrics,
)
from modeltop.services.result_export import (
    ResultExportError,
    export_speed_test_result,
    format_speed_test_summary,
    speed_test_result_payload,
)


def _stats(*values: float) -> MetricStatistics:
    if not values:
        return MetricStatistics(0, None, None, None, None, None, None)
    ordered = sorted(values)
    mean = sum(values) / len(values)
    return MetricStatistics(
        len(values),
        mean,
        ordered[len(values) // 2],
        ordered[0],
        ordered[-1],
        ordered[-1],
        None,
    )


def _snapshot(hour: int) -> HardwareSnapshot:
    return HardwareSnapshot(
        provider_name="fixture",
        gpus=(
            GpuMetrics(
                index=0,
                name="NVIDIA Fixture",
                uuid="GPU-secret-uuid",
                utilisation_percent=50,
                memory_used_bytes=1,
                memory_total_bytes=2,
                temperature_celsius=60,
                power_draw_watts=100,
                power_limit_watts=200,
                fan_speed_percent=40,
            ),
        ),
        cpu=CpuMetrics(10, 8, 4, 1, 2, 3),
        memory=MemoryMetrics(4, 8, 50),
        collected_at=datetime(2026, 7, 27, hour, tzinfo=UTC),
        error=None,
    )


def _result() -> SpeedTestResult:
    run = SpeedTestRunResult(
        run_number=1,
        warmup=False,
        success=True,
        cancelled=False,
        error=None,
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_tokens_estimated=True,
        completion_tokens_estimated=True,
        total_tokens_estimated=True,
        ttft_ms=100,
        generation_duration_s=2,
        total_duration_s=2.1,
        output_tokens_per_second=10,
        finish_reason="stop",
        streamed=True,
        response_character_count=len("generated-secret-text"),
    )
    return SpeedTestResult(
        run_id="speed-test-../../Ünique Run",
        status=SpeedTestStatus.COMPLETED,
        started_at=datetime(2026, 7, 27, 1, 2, 3, tzinfo=UTC),
        completed_at=datetime(2026, 7, 27, 1, 2, 4, tzinfo=UTC),
        server_id="server",
        server_name="Local",
        server_endpoint="localhost:8000",
        model_id="../Org/Mödél",
        backend="vLLM",
        config=SpeedTestConfig(
            prompt="exported benchmark prompt",
            warmup_runs=0,
            measured_runs=1,
        ),
        run_results=(run,),
        ttft_ms=_stats(100),
        output_tokens_per_second=_stats(10),
        total_duration_s=_stats(2.1),
        generation_duration_s=_stats(2),
        prompt_tokens=_stats(10),
        completion_tokens=_stats(20),
        hardware_before=_snapshot(1),
        hardware_after=_snapshot(2),
    )


def test_payload_is_complete_allowlisted_and_json_safe() -> None:
    result = _result()
    payload = speed_test_result_payload(result)
    assert payload["schema_version"] == "1.0"
    assert payload["status"] == "completed"
    assert payload["started_at"] == "2026-07-27T01:02:03+00:00"
    assert payload["configuration"] == result.config.model_dump(mode="json")
    assert payload["counts"] == {
        "configured_warmup_runs": 0,
        "configured_measured_runs": 1,
        "attempted_warmup_runs": 0,
        "attempted_measured_runs": 1,
        "successful_measured_runs": 1,
        "failed_measured_runs": 0,
        "cancelled_measured_runs": 0,
    }
    serialized = json.dumps(payload, allow_nan=False)
    assert "generated-secret-text" not in serialized
    assert "GPU-secret-uuid" not in serialized
    assert "api_key" not in serialized
    assert '"response_character_count": 21' in serialized
    hardware = payload["hardware"]
    assert isinstance(hardware, dict)
    assert hardware["before"] is not None

    for status in (SpeedTestStatus.FAILED, SpeedTestStatus.CANCELLED):
        changed = replace(result, status=status, error="Readable failure")
        assert speed_test_result_payload(changed)["status"] == status.value


def test_atomic_export_sanitizes_collisions_and_leaves_no_temporary_files(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "nested" / "results"
    result = _result()
    first = export_speed_test_result(result, directory)
    second = export_speed_test_result(result, directory)
    assert first.name == (
        "20260727-010204_speed-test_org-model_speed-test-unique-run.json"
    )
    assert second.name.endswith("_2.json")
    assert first.read_text(encoding="utf-8").endswith("\n")
    assert json.loads(first.read_text(encoding="utf-8"))["run_id"] == result.run_id
    assert not tuple(directory.glob(".modeltop-result-*.tmp"))


def test_link_failure_is_sanitized_and_cleans_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "results"

    def fail_link(source: os.PathLike[str], destination: os.PathLike[str]) -> None:
        raise OSError(95, "Operation not supported")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(ResultExportError, match="Operation not supported"):
        export_speed_test_result(_result(), directory)
    assert not tuple(directory.glob(".modeltop-result-*.tmp"))
    assert not tuple(directory.glob("*.json"))


def test_summary_formats_estimates_and_missing_values() -> None:
    summary = format_speed_test_summary(_result())
    assert summary == "\n".join(
        (
            "ModelTop Speed Test",
            "Status: COMPLETED",
            "Model: ../Org/Mödél",
            "Server: Local (server) · localhost:8000 · Backend: vLLM",
            "Runs: 1/1 successful · 0/0 warm-ups attempted · 0 failed · 0 cancelled",
            "Output speed: mean ~10.00 tok/s · median ~10.00 tok/s · "
            "minimum ~10.00 tok/s · maximum ~10.00 tok/s · p95 ~10.00 tok/s",
            "TTFT: mean 100.00 ms · median 100.00 ms · p95 100.00 ms",
            "Prompt tokens: mean ~10.00 · median ~10.00",
            "Completion tokens: mean ~20.00 · median ~20.00",
            "GPU before: NVIDIA Fixture",
            "GPU after: NVIDIA Fixture",
        )
    )
    missing = replace(
        _result(),
        output_tokens_per_second=_stats(),
        ttft_ms=_stats(),
        prompt_tokens=_stats(),
        completion_tokens=_stats(),
        hardware_before=None,
        hardware_after=None,
    )
    missing_summary = format_speed_test_summary(missing)
    assert "Output speed: mean --" in missing_summary
    assert "GPU before: --" in missing_summary
