"""Pure Concurrency aggregation and scaling coverage."""

from datetime import UTC, datetime
from math import inf, nan

import pytest

from modeltop.benchmarks.concurrency import (
    GPU_SATURATION_PERCENT,
    analyze_scaling,
    build_concurrency_level_result,
    classify_token_count_mode,
    summarize_hardware_samples,
)
from modeltop.benchmarks.models import ConcurrencyRequestResult
from modeltop.benchmarks.statistics import calculate_percentile_statistics
from modeltop.hardware.models import (
    CpuMetrics,
    GpuMetrics,
    HardwareSnapshot,
    MemoryMetrics,
)


def _request(
    sequence: int,
    *,
    concurrency: int = 2,
    success: bool = True,
    cancelled: bool = False,
    timed_out: bool = False,
    tokens: int | None = 10,
    estimated: bool = False,
    ttft: float | None = 100.0,
    latency: float | None = 1.0,
    speed: float | None = 10.0,
) -> ConcurrencyRequestResult:
    return ConcurrencyRequestResult(
        request_id=f"bench-c{concurrency}-r{sequence:04d}",
        concurrency_level=concurrency,
        sequence_number=sequence,
        started_at=float(sequence),
        first_token_at=float(sequence) + 0.1 if ttft is not None else None,
        completed_at=float(sequence) + 1.0,
        queue_wait_seconds=0.25,
        prompt_tokens=5 if success else None,
        completion_tokens=tokens if success else None,
        total_tokens=(tokens + 5 if success and tokens is not None else None),
        prompt_tokens_estimated=estimated,
        completion_tokens_estimated=estimated,
        total_tokens_estimated=estimated,
        ttft_ms=ttft if success else None,
        total_latency_seconds=latency if success else None,
        generation_duration_seconds=0.9 if success else None,
        output_tokens_per_second=speed if success else None,
        finish_reason="stop" if success else None,
        status_code=200 if success else 500,
        success=success,
        cancelled=cancelled,
        timed_out=timed_out,
        streamed=True,
        response_character_count=20 if success else 0,
        error_type=None if success else "Failure",
        error_message=None if success else "failed",
    )


def test_nearest_rank_statistics_and_sample_deviation() -> None:
    empty = calculate_percentile_statistics([None])
    assert empty.count == 0 and empty.p99 is None
    one = calculate_percentile_statistics([7])
    assert one.p50 == one.p99 == 7
    assert one.standard_deviation is None
    even = calculate_percentile_statistics([4, 1, 3, 2])
    assert even.median == 2.5
    assert (even.p50, even.p90, even.p95, even.p99) == (2, 4, 4, 4)
    assert even.standard_deviation == pytest.approx(1.2909944487)
    odd = calculate_percentile_statistics([5, None, 1, 3])
    assert odd.median == 3 and odd.p50 == 3
    for value in (nan, inf, -inf):
        with pytest.raises(ValueError, match="finite"):
            calculate_percentile_statistics([value])


def test_level_denominators_counts_and_token_modes() -> None:
    requests = (
        _request(1, speed=999.0),
        _request(2, estimated=True, speed=888.0),
        _request(3, success=False),
        _request(4, success=False, timed_out=True),
        _request(5, success=False, cancelled=True),
    )
    level = build_concurrency_level_result(
        concurrency=2,
        configured_requests=8,
        wall_time_seconds=4.0,
        requests=requests,
    )
    assert (
        level.successful_requests,
        level.failed_requests,
        level.timed_out_requests,
        level.cancelled_requests,
    ) == (2, 1, 1, 1)
    assert level.success_rate_percent == 40.0
    assert level.total_completion_tokens == 20
    assert level.requests_per_second == 0.5
    assert level.aggregate_output_tokens_per_second == 5.0
    assert level.request_output_tokens_per_second.mean == 943.5
    assert level.token_count_mode == "mixed"
    assert level.partial
    assert classify_token_count_mode([_request(1)]) == "exact"
    assert classify_token_count_mode([_request(1, estimated=True)]) == "estimated"
    assert classify_token_count_mode([_request(1, tokens=None)]) == "unavailable"


def _snapshot(
    second: int,
    *,
    gpu_util: float | None,
    cpu_util: float | None,
) -> HardwareSnapshot:
    return HardwareSnapshot(
        provider_name="fixture",
        gpus=(
            GpuMetrics(
                index=0,
                name="GPU",
                uuid="0",
                utilisation_percent=gpu_util,
                memory_used_bytes=100,
                memory_total_bytes=1000,
                temperature_celsius=50.0,
                power_draw_watts=20.0,
                power_limit_watts=100.0,
                fan_speed_percent=None,
            ),
            GpuMetrics(
                index=1,
                name="GPU",
                uuid="1",
                utilisation_percent=gpu_util,
                memory_used_bytes=200,
                memory_total_bytes=1000,
                temperature_celsius=60.0,
                power_draw_watts=30.0,
                power_limit_watts=100.0,
                fan_speed_percent=None,
            ),
        ),
        cpu=CpuMetrics(cpu_util, 8, 4, None, None, None),
        memory=MemoryMetrics(400, 1000, 40.0),
        collected_at=datetime(2026, 1, 1, 0, 0, second, tzinfo=UTC),
        error=None,
    )


def test_hardware_summary_deduplicates_and_ignores_partial_metrics() -> None:
    first = _snapshot(1, gpu_util=90.0, cpu_util=20.0)
    second = _snapshot(2, gpu_util=100.0, cpu_util=None)
    summary = summarize_hardware_samples((first, first, second))
    assert summary is not None
    assert summary.sample_count == 2
    assert summary.average_gpu_utilisation_percent == 95.0
    assert summary.maximum_gpu_utilisation_percent == 100.0
    assert summary.average_vram_used_bytes == 300.0
    assert summary.maximum_temperature_celsius == 60.0
    assert summary.average_power_draw_watts == 50.0
    assert summary.average_cpu_utilisation_percent == 20.0
    assert summarize_hardware_samples(()) is None


def test_scaling_thresholds_require_two_levels_and_are_adjacent() -> None:
    first = build_concurrency_level_result(
        concurrency=1,
        configured_requests=2,
        wall_time_seconds=1.0,
        requests=(
            _request(1, concurrency=1),
            _request(2, concurrency=1),
        ),
    )
    assert not any(
        observation.code in {"peak_throughput", "throughput_plateau"}
        for observation in analyze_scaling((first,))
    )
    second = build_concurrency_level_result(
        concurrency=2,
        configured_requests=2,
        wall_time_seconds=2.0,
        requests=(
            _request(
                1,
                concurrency=2,
                ttft=151.0,
                latency=1.51,
                speed=7.4,
            ),
            _request(
                2,
                concurrency=2,
                ttft=151.0,
                latency=1.51,
                speed=7.4,
            ),
        ),
        hardware_samples=(
            _snapshot(
                1,
                gpu_util=GPU_SATURATION_PERCENT,
                cpu_util=1.0,
            ),
        ),
    )
    observations = analyze_scaling((first, second))
    codes = {observation.code for observation in observations}
    assert "peak_throughput" in codes
    assert "gpu_saturation" in codes
    assert "ttft_degradation" in codes
    assert "latency_degradation" in codes
    assert "request_speed_degradation" in codes
