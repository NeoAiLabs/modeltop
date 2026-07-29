"""Immutable Speed Test configuration and lifecycle contracts."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from modeltop.benchmarks.models import (
    DEFAULT_SPEED_TEST_PROMPT,
    MetricStatistics,
    SpeedTestConfig,
    SpeedTestResult,
    SpeedTestRunResult,
    SpeedTestStatus,
    initial_speed_test_state,
    speed_test_config_for_preset,
)


def _empty_statistics() -> MetricStatistics:
    return MetricStatistics(0, None, None, None, None, None, None)


def _run(*, warmup: bool = False, success: bool = True) -> SpeedTestRunResult:
    return SpeedTestRunResult(
        run_number=1,
        warmup=warmup,
        success=success,
        cancelled=False,
        error=None if success else "failed",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_tokens_estimated=False,
        completion_tokens_estimated=False,
        total_tokens_estimated=False,
        ttft_ms=5.0,
        generation_duration_s=1.0,
        total_duration_s=1.005,
        output_tokens_per_second=20.0,
        finish_reason="stop",
        streamed=True,
        response_character_count=80,
    )


def _result() -> SpeedTestResult:
    empty = _empty_statistics()
    now = datetime.now(UTC)
    return SpeedTestResult(
        run_id="speed-test-id",
        status=SpeedTestStatus.COMPLETED,
        started_at=now,
        completed_at=now,
        server_id="server",
        server_name="Server",
        server_endpoint="localhost:8000",
        model_id="model",
        backend="test",
        config=SpeedTestConfig(),
        run_results=(_run(warmup=True), _run()),
        ttft_ms=empty,
        output_tokens_per_second=empty,
        total_duration_s=empty,
        generation_duration_s=empty,
        prompt_tokens=empty,
        completion_tokens=empty,
        hardware_before=None,
        hardware_after=None,
    )


def test_exact_presets_and_custom_preservation() -> None:
    expected = {
        "quick": (1, 3, 128),
        "standard": (1, 5, 256),
        "long": (1, 3, 1024),
    }
    for preset, values in expected.items():
        config = speed_test_config_for_preset(preset)  # type: ignore[arg-type]
        assert (config.warmup_runs, config.measured_runs, config.max_tokens) == values
        assert config.prompt == DEFAULT_SPEED_TEST_PROMPT
        assert (config.temperature, config.top_p, config.seed) == (0.0, 1.0, 42)
        assert config.request_timeout_seconds == 300.0
        assert not config.continue_on_error

    edited = SpeedTestConfig(
        prompt="edited",
        warmup_runs=2,
        measured_runs=7,
        max_tokens=99,
        temperature=1.2,
        top_p=0.8,
        seed=None,
        request_timeout_seconds=12,
        continue_on_error=True,
    )
    custom = speed_test_config_for_preset("custom", edited)
    assert custom == edited.model_copy(update={"preset": "custom"})
    assert SpeedTestConfig() == speed_test_config_for_preset("standard")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt", " \n"),
        ("warmup_runs", -1),
        ("warmup_runs", 21),
        ("measured_runs", 0),
        ("measured_runs", 101),
        ("max_tokens", 0),
        ("max_tokens", 32769),
        ("temperature", -0.01),
        ("temperature", 2.01),
        ("temperature", float("nan")),
        ("top_p", 0),
        ("top_p", 1.01),
        ("request_timeout_seconds", 0),
        ("request_timeout_seconds", float("inf")),
        ("seed", 1.5),
    ],
)
def test_invalid_config_boundaries(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        SpeedTestConfig.model_validate({field: value})


def test_valid_config_boundaries_and_frozen_values() -> None:
    config = SpeedTestConfig(
        warmup_runs=0,
        measured_runs=100,
        max_tokens=32768,
        temperature=2,
        top_p=0.0001,
        request_timeout_seconds=0.0001,
    )
    with pytest.raises(ValidationError):
        config.measured_runs = 3  # pyright: ignore[reportAttributeAccessIssue]

    result = _result()
    with pytest.raises(FrozenInstanceError):
        result.status = SpeedTestStatus.FAILED  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(FrozenInstanceError):
        result.run_results[0].success = False  # pyright: ignore[reportAttributeAccessIssue]


def test_status_helpers_counts_and_state_latest_result() -> None:
    assert SpeedTestStatus.PREPARING.is_active
    assert SpeedTestStatus.CANCELLING.is_active
    assert not SpeedTestStatus.COMPLETED.is_active
    assert SpeedTestStatus.CANCELLED.is_terminal
    assert not SpeedTestStatus.IDLE.is_terminal

    result = replace(
        _result(),
        run_results=(
            _run(warmup=True),
            _run(),
            replace(_run(), run_number=2, success=False, error="failed"),
            replace(_run(), run_number=3, success=False, cancelled=True),
        ),
    )
    assert result.attempted_warmup_runs == 1
    assert result.attempted_measured_runs == 3
    assert result.successful_runs == 1
    assert result.failed_runs == 1
    assert result.cancelled_runs == 1

    state = replace(initial_speed_test_state(), results=(result,))
    assert state.latest_result is result
    assert state.result_by_id(result.run_id) is result
    with pytest.raises(ValueError, match="2,000"):
        replace(state, live_output_preview="x" * 2001)
