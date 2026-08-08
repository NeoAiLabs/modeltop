"""Concurrency configuration, immutable result, and state-lane contracts."""

from dataclasses import FrozenInstanceError, fields, replace
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from modeltop.benchmarks.models import (
    ConcurrencyBenchmarkConfig,
    ConcurrencyBenchmarkProgress,
    ConcurrencyBenchmarkResult,
    ConcurrencyBenchmarkState,
    ConcurrencyBenchmarkStatus,
    ConcurrencyLevelResult,
    ConcurrencyRequestProgress,
    ConcurrencyRequestResult,
    HardwareBenchmarkSummary,
    PercentileStatistics,
    SaturationObservation,
    SpeedTestStatus,
    concurrency_benchmark_config_from_defaults,
    initial_concurrency_benchmark_state,
)
from modeltop.benchmarks.prompts import DEFAULT_CONCURRENCY_PROMPT
from modeltop.models import ConcurrencyBenchmarkDefaultsConfig
from modeltop.state import initial_application_state


def _empty_statistics() -> PercentileStatistics:
    return PercentileStatistics(
        count=0,
        mean=None,
        median=None,
        minimum=None,
        maximum=None,
        p50=None,
        p90=None,
        p95=None,
        p99=None,
        standard_deviation=None,
    )


def _request(
    *,
    concurrency: int = 1,
    sequence: int = 1,
    success: bool = True,
    cancelled: bool = False,
    timed_out: bool = False,
) -> ConcurrencyRequestResult:
    return ConcurrencyRequestResult(
        request_id=f"benchmark-c{concurrency}-r{sequence:04d}",
        concurrency_level=concurrency,
        sequence_number=sequence,
        started_at=10.0,
        first_token_at=10.1 if success else None,
        completed_at=11.0,
        queue_wait_seconds=0.25,
        prompt_tokens=8 if success else None,
        completion_tokens=16 if success else None,
        total_tokens=24 if success else None,
        prompt_tokens_estimated=False,
        completion_tokens_estimated=False,
        total_tokens_estimated=False,
        ttft_ms=100.0 if success else None,
        total_latency_seconds=1.0,
        generation_duration_seconds=0.9 if success else None,
        output_tokens_per_second=16 / 0.9 if success else None,
        finish_reason="stop" if success else None,
        status_code=200,
        success=success,
        cancelled=cancelled,
        timed_out=timed_out,
        streamed=True,
        response_character_count=64 if success else 0,
        error_type=None if success else "RequestError",
        error_message=None if success else "Request failed",
    )


def _level(concurrency: int = 1) -> ConcurrencyLevelResult:
    request = _request(concurrency=concurrency)
    empty = _empty_statistics()
    return ConcurrencyLevelResult(
        concurrency=concurrency,
        configured_requests=1,
        attempted_requests=1,
        completed_requests=1,
        successful_requests=1,
        failed_requests=0,
        cancelled_requests=0,
        timed_out_requests=0,
        wall_time_seconds=1.25,
        success_rate_percent=100.0,
        total_prompt_tokens=8,
        total_completion_tokens=16,
        token_count_mode="exact",
        requests_per_second=0.8,
        aggregate_output_tokens_per_second=12.8,
        ttft_ms=empty,
        latency_seconds=empty,
        generation_duration_seconds=empty,
        request_output_tokens_per_second=empty,
        completion_tokens=empty,
        hardware_samples=(),
        hardware_summary=None,
        output_length_warning=None,
        partial=False,
        early_stopped=False,
        observations=(),
        requests=(request,),
    )


def _result(
    *,
    status: ConcurrencyBenchmarkStatus = ConcurrencyBenchmarkStatus.COMPLETED,
    cancelled: bool = False,
    levels: tuple[ConcurrencyLevelResult, ...] | None = None,
) -> ConcurrencyBenchmarkResult:
    started_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    return ConcurrencyBenchmarkResult(
        benchmark_id="benchmark-id",
        status=status,
        server_id="server",
        server_name="Server",
        server_endpoint="localhost:8000",
        model_id="model",
        backend="test",
        started_at=started_at,
        completed_at=started_at + timedelta(seconds=7.5),
        config=ConcurrencyBenchmarkConfig(),
        levels=(_level(),) if levels is None else levels,
        cancelled=cancelled,
        error="failed safely" if status is ConcurrencyBenchmarkStatus.ERROR else None,
        warnings=(),
        observations=(),
    )


def test_exact_default_config_and_fixed_prompt() -> None:
    config = ConcurrencyBenchmarkConfig()
    assert DEFAULT_CONCURRENCY_PROMPT == (
        "Explain speculative decoding in one concise paragraph."
    )
    assert config.mode == "sweep"
    assert config.prompt == DEFAULT_CONCURRENCY_PROMPT
    assert config.system_prompt is None
    assert config.concurrency_levels == (1, 2, 4, 8)
    assert config.requests_per_level == 16
    assert config.warmup_requests == 2
    assert config.max_tokens == 256
    assert config.temperature == 0.0
    assert config.top_p == 1.0
    assert config.seed == 42
    assert config.request_timeout_seconds == 120.0
    assert config.delay_between_levels_seconds == 3.0
    assert config.stream is True
    assert config.maximum_concurrency == 128
    assert config.unique_prompt_suffix_per_request is True


def test_defaults_conversion_copies_every_effective_yaml_value() -> None:
    defaults = ConcurrencyBenchmarkDefaultsConfig(
        default_levels=(2, 6, 10),
        requests_per_level=27,
        warmup_requests=3,
        max_tokens=513,
        temperature=0.75,
        top_p=0.9,
        request_timeout_seconds=45.5,
        delay_between_levels_seconds=0.25,
        maximum_concurrency=64,
        unique_prompt_suffix_per_request=False,
    )
    config = concurrency_benchmark_config_from_defaults(defaults)
    assert config == ConcurrencyBenchmarkConfig(
        mode="sweep",
        prompt=DEFAULT_CONCURRENCY_PROMPT,
        system_prompt=None,
        concurrency_levels=(2, 6, 10),
        requests_per_level=27,
        warmup_requests=3,
        max_tokens=513,
        temperature=0.75,
        top_p=0.9,
        seed=42,
        request_timeout_seconds=45.5,
        delay_between_levels_seconds=0.25,
        stream=True,
        maximum_concurrency=64,
        unique_prompt_suffix_per_request=False,
    )


def test_fixed_and_sweep_cardinality_and_canonical_level_order() -> None:
    fixed = ConcurrencyBenchmarkConfig(mode="fixed", concurrency_levels=(4,))
    assert fixed.concurrency_levels == (4,)

    sweep = ConcurrencyBenchmarkConfig(mode="sweep", concurrency_levels=(8, 1, 4, 2))
    assert sweep.concurrency_levels == (1, 2, 4, 8)

    with pytest.raises(ValidationError, match="exactly one"):
        ConcurrencyBenchmarkConfig(mode="fixed", concurrency_levels=(2, 4))
    with pytest.raises(ValidationError, match="at least two"):
        ConcurrencyBenchmarkConfig(mode="sweep", concurrency_levels=(2,))
    with pytest.raises(ValidationError, match="unique"):
        ConcurrencyBenchmarkConfig(mode="sweep", concurrency_levels=(1, 2, 2))


def test_prompt_validation_and_system_prompt_normalization() -> None:
    with pytest.raises(ValidationError, match="blank"):
        ConcurrencyBenchmarkConfig(prompt=" \n\t")

    blank = ConcurrencyBenchmarkConfig(system_prompt="  \n ")
    assert blank.system_prompt is None
    trimmed = ConcurrencyBenchmarkConfig(system_prompt="  Keep answers concise.  ")
    assert trimmed.system_prompt == "Keep answers concise."
    preserved = ConcurrencyBenchmarkConfig(prompt="  deliberate user whitespace  ")
    assert preserved.prompt == "  deliberate user whitespace  "


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("concurrency_levels", []),
        ("concurrency_levels", [0, 1]),
        ("concurrency_levels", [-1, 1]),
        ("concurrency_levels", [1, "2"]),
        ("concurrency_levels", [True, 2]),
        ("requests_per_level", 0),
        ("requests_per_level", 1001),
        ("requests_per_level", 1.0),
        ("warmup_requests", -1),
        ("warmup_requests", 1001),
        ("max_tokens", 0),
        ("max_tokens", 1.0),
        ("temperature", True),
        ("temperature", "0.5"),
        ("temperature", -0.0001),
        ("temperature", 2.0001),
        ("temperature", float("nan")),
        ("temperature", float("inf")),
        ("temperature", float("-inf")),
        ("top_p", 0.0),
        ("top_p", 1.0001),
        ("top_p", float("nan")),
        ("top_p", float("inf")),
        ("top_p", float("-inf")),
        ("request_timeout_seconds", 0.0),
        ("request_timeout_seconds", -0.1),
        ("request_timeout_seconds", float("nan")),
        ("request_timeout_seconds", float("inf")),
        ("request_timeout_seconds", float("-inf")),
        ("delay_between_levels_seconds", -0.0001),
        ("delay_between_levels_seconds", float("nan")),
        ("delay_between_levels_seconds", float("inf")),
        ("delay_between_levels_seconds", float("-inf")),
        ("maximum_concurrency", 0),
        ("maximum_concurrency", 1.0),
        ("seed", 1.5),
        ("seed", "42"),
        ("seed", True),
        ("stream", False),
        ("stream", 1),
        ("stream", "true"),
    ],
)
def test_runtime_config_rejects_every_invalid_numeric_boundary(
    field: str, value: object
) -> None:
    with pytest.raises(ValidationError):
        ConcurrencyBenchmarkConfig.model_validate({field: value})


def test_valid_runtime_boundaries_and_safety_maximum_override() -> None:
    config = ConcurrencyBenchmarkConfig(
        mode="fixed",
        concurrency_levels=(256,),
        requests_per_level=1000,
        warmup_requests=1000,
        max_tokens=1,
        temperature=2.0,
        top_p=0.000001,
        seed=None,
        request_timeout_seconds=0.000001,
        delay_between_levels_seconds=0.0,
        maximum_concurrency=256,
    )
    assert config.concurrency_levels == (256,)
    assert config.maximum_concurrency == 256
    minimums = ConcurrencyBenchmarkConfig(
        mode="fixed",
        concurrency_levels=(1,),
        requests_per_level=1,
        warmup_requests=0,
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        request_timeout_seconds=0.000001,
        delay_between_levels_seconds=0.0,
        maximum_concurrency=1,
    )
    assert minimums.requests_per_level == 1
    assert minimums.warmup_requests == 0

    with pytest.raises(ValidationError, match="maximum_concurrency"):
        ConcurrencyBenchmarkConfig(
            mode="fixed", concurrency_levels=(129,), maximum_concurrency=128
        )


def test_runtime_config_and_yaml_defaults_are_frozen() -> None:
    config = ConcurrencyBenchmarkConfig()
    with pytest.raises(ValidationError):
        config.requests_per_level = 2  # pyright: ignore[reportAttributeAccessIssue]

    defaults = ConcurrencyBenchmarkDefaultsConfig()
    with pytest.raises(ValidationError):
        defaults.max_tokens = 32  # pyright: ignore[reportAttributeAccessIssue]


def test_request_timing_outcomes_and_token_estimation_invariants() -> None:
    exact = _request()
    assert not exact.token_count_is_estimated
    estimated = replace(exact, completion_tokens_estimated=True)
    assert estimated.token_count_is_estimated

    for changes in (
        {"started_at": float("nan")},
        {"queue_wait_seconds": -0.1},
        {"first_token_at": 9.9},
        {"completed_at": 10.05},
        {"total_latency_seconds": float("inf")},
        {"completion_tokens": -1},
        {"response_character_count": -1},
        {"cancelled": True},
        {"timed_out": True},
    ):
        with pytest.raises(ValueError):
            replace(exact, **changes)  # type: ignore[arg-type]

    timed_out = _request(success=False, timed_out=True)
    cancelled = _request(success=False, cancelled=True)
    failed = _request(success=False)
    assert timed_out.timed_out and not timed_out.cancelled
    assert cancelled.cancelled and not cancelled.timed_out
    assert not failed.success and not failed.cancelled and not failed.timed_out


def test_level_count_invariants_and_immutable_rows() -> None:
    level = _level()
    assert level.successful_requests == 1
    with pytest.raises(FrozenInstanceError):
        level.partial = True  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(FrozenInstanceError):
        level.requests[0].success = False  # pyright: ignore[reportAttributeAccessIssue]

    with pytest.raises(ValueError, match="sum"):
        replace(level, successful_requests=0)
    with pytest.raises(ValueError, match="configured"):
        replace(level, attempted_requests=2, completed_requests=1)
    with pytest.raises(ValueError, match="every terminal"):
        replace(level, requests=())


def test_terminal_result_requires_utc_order_and_matching_cancellation() -> None:
    result = _result()
    assert result.wall_time_seconds == 7.5
    with pytest.raises(FrozenInstanceError):
        result.error = "changed"  # pyright: ignore[reportAttributeAccessIssue]

    with pytest.raises(ValueError, match="terminal"):
        replace(result, status=ConcurrencyBenchmarkStatus.RUNNING)
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        replace(result, started_at=datetime(2026, 1, 2, 3, 4, 5))
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        replace(
            result,
            completed_at=datetime(
                2026, 1, 2, 3, 4, 6, tzinfo=timezone(timedelta(hours=1))
            ),
        )
    with pytest.raises(ValueError, match="precede"):
        replace(result, completed_at=result.started_at - timedelta(seconds=1))
    with pytest.raises(ValueError, match="cancelled"):
        replace(result, cancelled=True)

    cancelled = _result(
        status=ConcurrencyBenchmarkStatus.CANCELLED,
        cancelled=True,
        levels=(),
    )
    assert cancelled.cancelled
    assert cancelled.status.is_terminal


def test_result_levels_are_unique_and_ordered() -> None:
    with pytest.raises(ValueError, match="ordered"):
        _result(levels=(_level(2), _level(1)))
    with pytest.raises(ValueError, match="unique"):
        _result(levels=(_level(1), _level(1)))


def test_status_helpers_and_initial_concurrency_state() -> None:
    assert ConcurrencyBenchmarkStatus.VALIDATING.is_active
    assert ConcurrencyBenchmarkStatus.WARMING_UP.is_active
    assert ConcurrencyBenchmarkStatus.RUNNING.is_active
    assert ConcurrencyBenchmarkStatus.BETWEEN_LEVELS.is_active
    assert ConcurrencyBenchmarkStatus.CANCELLING.is_active
    assert not ConcurrencyBenchmarkStatus.IDLE.is_active
    assert ConcurrencyBenchmarkStatus.COMPLETED.is_terminal
    assert ConcurrencyBenchmarkStatus.CANCELLED.is_terminal
    assert ConcurrencyBenchmarkStatus.ERROR.is_terminal

    config = ConcurrencyBenchmarkConfig(mode="fixed", concurrency_levels=(7,))
    state = initial_concurrency_benchmark_state(config)
    assert state.config is config
    assert state.status is ConcurrencyBenchmarkStatus.IDLE
    assert state.active_benchmark_id is None
    assert state.progress is None
    assert state.benchmark_started_at is None
    assert state.latest_result is None
    assert state.benchmark_error is None
    assert not state.is_active
    assert not state.is_terminal


def test_initial_application_state_has_an_independent_concurrency_lane() -> None:
    custom = ConcurrencyBenchmarkConfig(mode="fixed", concurrency_levels=(7,))
    state = initial_application_state(
        "server", hardware_enabled=False, concurrency_config=custom
    )
    assert state.concurrency_benchmark.config is custom
    assert state.concurrency_benchmark.status is ConcurrencyBenchmarkStatus.IDLE
    assert state.speed_test.status.value == "idle"
    assert state.generation_status.value == "idle"
    assert not state.benchmark_is_active

    active_concurrency = replace(
        state,
        concurrency_benchmark=replace(
            state.concurrency_benchmark,
            status=ConcurrencyBenchmarkStatus.RUNNING,
            active_benchmark_id="benchmark-id",
        ),
        active_view="concurrency",
    )
    assert active_concurrency.active_view == "concurrency"
    assert active_concurrency.benchmark_is_active
    assert not active_concurrency.speed_test.is_active

    active_speed_test = replace(
        state,
        speed_test=replace(
            state.speed_test,
            status=SpeedTestStatus.RUNNING,
        ),
    )
    assert active_speed_test.benchmark_is_active
    assert not active_speed_test.concurrency_benchmark.is_active

    default_state = initial_application_state("server", hardware_enabled=True)
    assert default_state.concurrency_benchmark.config == ConcurrencyBenchmarkConfig()
    assert default_state.concurrency_benchmark is not state.concurrency_benchmark


def test_exact_concurrency_pydantic_field_contracts() -> None:
    assert tuple(ConcurrencyBenchmarkDefaultsConfig.model_fields) == (
        "default_levels",
        "requests_per_level",
        "warmup_requests",
        "max_tokens",
        "temperature",
        "top_p",
        "request_timeout_seconds",
        "delay_between_levels_seconds",
        "maximum_concurrency",
        "unique_prompt_suffix_per_request",
    )
    assert tuple(ConcurrencyBenchmarkConfig.model_fields) == (
        "mode",
        "prompt",
        "system_prompt",
        "concurrency_levels",
        "requests_per_level",
        "warmup_requests",
        "max_tokens",
        "temperature",
        "top_p",
        "seed",
        "request_timeout_seconds",
        "delay_between_levels_seconds",
        "stream",
        "maximum_concurrency",
        "unique_prompt_suffix_per_request",
        "thinking_mode",
    )


def test_exact_concurrency_dataclass_field_contracts() -> None:
    expected = {
        PercentileStatistics: (
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
        ),
        ConcurrencyRequestResult: (
            "request_id",
            "concurrency_level",
            "sequence_number",
            "started_at",
            "first_token_at",
            "completed_at",
            "queue_wait_seconds",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_tokens_estimated",
            "completion_tokens_estimated",
            "total_tokens_estimated",
            "ttft_ms",
            "total_latency_seconds",
            "generation_duration_seconds",
            "output_tokens_per_second",
            "finish_reason",
            "status_code",
            "success",
            "cancelled",
            "timed_out",
            "streamed",
            "response_character_count",
            "error_type",
            "error_message",
        ),
        ConcurrencyLevelResult: (
            "concurrency",
            "configured_requests",
            "attempted_requests",
            "completed_requests",
            "successful_requests",
            "failed_requests",
            "cancelled_requests",
            "timed_out_requests",
            "wall_time_seconds",
            "success_rate_percent",
            "total_prompt_tokens",
            "total_completion_tokens",
            "token_count_mode",
            "requests_per_second",
            "aggregate_output_tokens_per_second",
            "ttft_ms",
            "latency_seconds",
            "generation_duration_seconds",
            "request_output_tokens_per_second",
            "completion_tokens",
            "hardware_samples",
            "hardware_summary",
            "output_length_warning",
            "partial",
            "early_stopped",
            "observations",
            "requests",
        ),
        HardwareBenchmarkSummary: (
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
        ),
        SaturationObservation: ("code", "concurrency", "message"),
        ConcurrencyBenchmarkResult: (
            "benchmark_id",
            "status",
            "server_id",
            "server_name",
            "server_endpoint",
            "model_id",
            "backend",
            "started_at",
            "completed_at",
            "config",
            "levels",
            "cancelled",
            "error",
            "warnings",
            "observations",
        ),
        ConcurrencyRequestProgress: (
            "request_id",
            "concurrency_level",
            "sequence_number",
            "state",
            "queued_at",
            "started_at",
            "latest_metrics",
            "error",
        ),
        ConcurrencyBenchmarkProgress: (
            "phase",
            "active_concurrency_level",
            "next_concurrency_level",
            "delay_remaining_seconds",
            "configured_requests",
            "active_request_count",
            "queued_request_count",
            "completed_request_count",
            "successful_request_count",
            "failed_request_count",
            "timed_out_request_count",
            "cancelled_request_count",
            "elapsed_seconds",
            "aggregate_output_tokens_per_second",
            "requests_per_second",
            "median_ttft_ms",
            "request_rows",
            "completed_levels",
            "warnings",
        ),
        ConcurrencyBenchmarkState: (
            "config",
            "status",
            "active_benchmark_id",
            "progress",
            "benchmark_started_at",
            "latest_result",
            "benchmark_error",
        ),
    }
    for model, names in expected.items():
        assert tuple(field.name for field in fields(model)) == names
