"""Available-value and measured-run-only Speed Test statistics."""

import math
from statistics import stdev

import pytest

from modeltop.benchmarks.models import SpeedTestRunResult
from modeltop.benchmarks.statistics import (
    build_speed_test_aggregates,
    calculate_metric_statistics,
)


def _run(
    run_number: int,
    *,
    warmup: bool = False,
    success: bool = True,
    ttft_ms: float | None = None,
    speed: float | None = None,
    prompt_tokens: int | None = None,
) -> SpeedTestRunResult:
    return SpeedTestRunResult(
        run_number=run_number,
        warmup=warmup,
        success=success,
        cancelled=False,
        error=None if success else "failed",
        prompt_tokens=prompt_tokens,
        completion_tokens=20 if success else None,
        total_tokens=None,
        prompt_tokens_estimated=False,
        completion_tokens_estimated=False,
        total_tokens_estimated=False,
        ttft_ms=ttft_ms,
        generation_duration_s=2.0 if success else None,
        total_duration_s=2.5 if success else None,
        output_tokens_per_second=speed,
        finish_reason="stop" if success else None,
        streamed=True,
        response_character_count=10,
    )


def test_empty_single_and_many_statistics() -> None:
    empty = calculate_metric_statistics([None, None])
    assert empty.count == 0
    assert empty.mean is None
    assert empty.standard_deviation is None

    single = calculate_metric_statistics([None, 4])
    assert single.count == 1
    assert (single.mean, single.median, single.p95) == (4.0, 4.0, 4.0)
    assert single.standard_deviation is None

    many = calculate_metric_statistics([1, 2, None, 3, 100])
    assert many.count == 4
    assert many.mean == 26.5
    assert many.median == 2.5
    assert (many.minimum, many.maximum, many.p95) == (1.0, 100.0, 100.0)
    assert many.standard_deviation == stdev([1.0, 2.0, 3.0, 100.0])


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_values_are_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        calculate_metric_statistics([1, None, value])


def test_aggregates_filter_runs_and_metrics_independently() -> None:
    aggregates = build_speed_test_aggregates(
        (
            _run(1, warmup=True, ttft_ms=1, speed=100, prompt_tokens=1),
            _run(1, ttft_ms=10, speed=None, prompt_tokens=11),
            _run(2, ttft_ms=None, speed=20, prompt_tokens=None),
            _run(3, success=False, ttft_ms=999, speed=999, prompt_tokens=999),
        )
    )
    assert aggregates.ttft_ms.count == 1
    assert aggregates.ttft_ms.mean == 10
    assert aggregates.output_tokens_per_second.count == 1
    assert aggregates.output_tokens_per_second.mean == 20
    assert aggregates.prompt_tokens.count == 1
    assert aggregates.prompt_tokens.mean == 11
    assert aggregates.completion_tokens.count == 2
    assert aggregates.completion_tokens.mean == 20
    assert aggregates.total_duration_s.count == 2
    assert aggregates.generation_duration_s.count == 2
