"""Drafter domain model validation and terminal result contracts."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from modeltop.benchmarks.models import (
    DrafterBenchmarkConfig,
    DrafterBenchmarkResult,
    DrafterBenchmarkStatus,
    DrafterObservation,
    DrafterRunResult,
    MetricStatistics,
    drafter_benchmark_config_from_defaults,
)
from modeltop.benchmarks.prompts import DEFAULT_DRAFTER_PROMPT
from modeltop.models import DrafterBenchmarkDefaultsConfig


def _empty_stats() -> MetricStatistics:
    return MetricStatistics(0, None, None, None, None, None, None)


def _run(
    *,
    run_number: int = 1,
    warmup: bool = False,
    success: bool = True,
    cancelled: bool = False,
    draft_tokens: int | None = 6,
    accepted_tokens: int | None = 4,
    acceptance_rate: float | None = 4 / 6,
) -> DrafterRunResult:
    return DrafterRunResult(
        run_number=run_number,
        warmup=warmup,
        success=success,
        cancelled=cancelled,
        error=None if success else "failed",
        prompt_tokens=10 if success else None,
        completion_tokens=20 if success else None,
        total_tokens=30 if success else None,
        prompt_tokens_estimated=False,
        completion_tokens_estimated=False,
        total_tokens_estimated=False,
        draft_tokens=draft_tokens if success else None,
        accepted_tokens=accepted_tokens if success else None,
        acceptance_rate=acceptance_rate if success else None,
        ttft_ms=5.0 if success else None,
        generation_duration_s=1.0 if success else None,
        total_duration_s=1.2 if success else None,
        output_tokens_per_second=20.0 if success else None,
        finish_reason="stop" if success else None,
        streamed=True,
    )


def _result(
    *runs: DrafterRunResult,
    status: DrafterBenchmarkStatus = DrafterBenchmarkStatus.COMPLETED,
    observations: tuple[DrafterObservation, ...] = (),
) -> DrafterBenchmarkResult:
    started = datetime(2026, 7, 27, tzinfo=UTC)
    return DrafterBenchmarkResult(
        benchmark_id="drafter-1",
        status=status,
        started_at=started,
        completed_at=started + timedelta(seconds=1),
        server_id="server",
        server_name="Local",
        server_endpoint="http://localhost:8000/v1",
        model_id="model",
        backend="vLLM",
        config=DrafterBenchmarkConfig(),
        run_results=runs,
        ttft_ms=_empty_stats(),
        output_tokens_per_second=_empty_stats(),
        total_duration_s=_empty_stats(),
        generation_duration_s=_empty_stats(),
        prompt_tokens=_empty_stats(),
        completion_tokens=_empty_stats(),
        draft_tokens=_empty_stats(),
        accepted_tokens=_empty_stats(),
        acceptance_rate=_empty_stats(),
        observations=observations,
        hardware_before=None,
        hardware_after=None,
    )


def test_config_bounds_and_defaults() -> None:
    config = DrafterBenchmarkConfig()
    assert config.prompt == DEFAULT_DRAFTER_PROMPT
    assert config.warmup_runs == 1
    assert config.measured_runs == 5
    assert config.seed == 42
    assert config.continue_on_error is False

    with pytest.raises(ValidationError):
        DrafterBenchmarkConfig(prompt="   ")
    with pytest.raises(ValidationError):
        DrafterBenchmarkConfig(warmup_runs=-1)
    with pytest.raises(ValidationError):
        DrafterBenchmarkConfig(measured_runs=0)
    with pytest.raises(ValidationError):
        DrafterBenchmarkConfig(temperature=float("nan"))

    defaults = DrafterBenchmarkDefaultsConfig(measured_runs=7, max_tokens=128)
    from_defaults = drafter_benchmark_config_from_defaults(defaults)
    assert from_defaults.prompt == DEFAULT_DRAFTER_PROMPT
    assert from_defaults.measured_runs == 7
    assert from_defaults.max_tokens == 128


def test_status_active_and_terminal_membership() -> None:
    active = {
        DrafterBenchmarkStatus.PREPARING,
        DrafterBenchmarkStatus.WARMING_UP,
        DrafterBenchmarkStatus.RUNNING,
        DrafterBenchmarkStatus.CANCELLING,
    }
    terminal = {
        DrafterBenchmarkStatus.CANCELLED,
        DrafterBenchmarkStatus.COMPLETED,
        DrafterBenchmarkStatus.COMPLETED_WITH_ERRORS,
        DrafterBenchmarkStatus.FAILED,
    }
    for status in DrafterBenchmarkStatus:
        assert status.is_active is (status in active)
        assert status.is_terminal is (status in terminal)


def test_result_requires_terminal_status() -> None:
    with pytest.raises(ValueError, match="terminal"):
        _result(status=DrafterBenchmarkStatus.RUNNING)


def test_run_speculative_flags_and_result_availability() -> None:
    bare = _run(draft_tokens=None, accepted_tokens=None, acceptance_rate=None)
    assert not bare.speculative_telemetry_present

    present = _run()
    assert present.speculative_telemetry_present

    unavailable = _result(
        _run(warmup=True),
        _run(draft_tokens=None, accepted_tokens=None, acceptance_rate=None),
    )
    assert not unavailable.speculative_telemetry_available
    assert unavailable.successful_runs == 1
    assert unavailable.attempted_measured_runs == 1

    available = _result(_run(warmup=True), _run(), _run(run_number=2))
    assert available.speculative_telemetry_available
    assert available.successful_runs == 2


def test_observation_codes_are_closed_set() -> None:
    observation = DrafterObservation(
        code="speculative_telemetry_unavailable",
        message=(
            "Server did not report draft/accept usage fields; throughput metrics only."
        ),
    )
    assert observation.code == "speculative_telemetry_unavailable"
