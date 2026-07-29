"""Pure metric aggregation for sequential and concurrency benchmarks."""

import math
from collections.abc import Iterable
from statistics import fmean, median, stdev

from modeltop.benchmarks.models import (
    DrafterAggregates,
    DrafterRunResult,
    HardwareBenchmarkSummary,
    MetricStatistics,
    PercentileStatistics,
    SpeedTestAggregates,
    SpeedTestRunResult,
)
from modeltop.hardware.models import HardwareSnapshot, summarize_gpus


def calculate_metric_statistics(
    values: Iterable[float | int | None],
) -> MetricStatistics:
    """Summarize finite available values with nearest-rank p95."""
    available: list[float] = []
    for value in values:
        if value is None:
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("Metric values must be finite")
        available.append(numeric)

    if not available:
        return MetricStatistics(0, None, None, None, None, None, None)

    ordered = sorted(available)
    count = len(ordered)
    return MetricStatistics(
        count=count,
        mean=fmean(ordered),
        median=float(median(ordered)),
        minimum=ordered[0],
        maximum=ordered[-1],
        p95=ordered[math.ceil(0.95 * count) - 1],
        standard_deviation=stdev(ordered) if count >= 2 else None,
    )


def calculate_percentile_statistics(
    values: Iterable[float | int | None],
) -> PercentileStatistics:
    """Summarize finite values using the nearest-rank percentile rule."""
    available: list[float] = []
    for value in values:
        if value is None:
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("Metric values must be finite")
        available.append(numeric)

    if not available:
        return PercentileStatistics(
            0, None, None, None, None, None, None, None, None, None
        )

    ordered = sorted(available)
    count = len(ordered)

    def nearest_rank(percentile: float) -> float:
        return ordered[math.ceil(percentile * count) - 1]

    return PercentileStatistics(
        count=count,
        mean=fmean(ordered),
        median=float(median(ordered)),
        minimum=ordered[0],
        maximum=ordered[-1],
        p50=nearest_rank(0.50),
        p90=nearest_rank(0.90),
        p95=nearest_rank(0.95),
        p99=nearest_rank(0.99),
        standard_deviation=stdev(ordered) if count >= 2 else None,
    )


def build_speed_test_aggregates(
    run_results: Iterable[SpeedTestRunResult],
) -> SpeedTestAggregates:
    """Aggregate each metric independently from successful measured runs."""
    successful = tuple(
        result for result in run_results if not result.warmup and result.success
    )
    return SpeedTestAggregates(
        ttft_ms=calculate_metric_statistics(result.ttft_ms for result in successful),
        output_tokens_per_second=calculate_metric_statistics(
            result.output_tokens_per_second for result in successful
        ),
        total_duration_s=calculate_metric_statistics(
            result.total_duration_s for result in successful
        ),
        generation_duration_s=calculate_metric_statistics(
            result.generation_duration_s for result in successful
        ),
        prompt_tokens=calculate_metric_statistics(
            result.prompt_tokens for result in successful
        ),
        completion_tokens=calculate_metric_statistics(
            result.completion_tokens for result in successful
        ),
    )


def build_drafter_aggregates(
    run_results: Iterable[DrafterRunResult],
) -> DrafterAggregates:
    """Aggregate each metric independently from successful measured runs."""
    successful = tuple(
        result for result in run_results if not result.warmup and result.success
    )
    return DrafterAggregates(
        ttft_ms=calculate_metric_statistics(result.ttft_ms for result in successful),
        output_tokens_per_second=calculate_metric_statistics(
            result.output_tokens_per_second for result in successful
        ),
        total_duration_s=calculate_metric_statistics(
            result.total_duration_s for result in successful
        ),
        generation_duration_s=calculate_metric_statistics(
            result.generation_duration_s for result in successful
        ),
        prompt_tokens=calculate_metric_statistics(
            result.prompt_tokens for result in successful
        ),
        completion_tokens=calculate_metric_statistics(
            result.completion_tokens for result in successful
        ),
        draft_tokens=calculate_metric_statistics(
            result.draft_tokens for result in successful
        ),
        accepted_tokens=calculate_metric_statistics(
            result.accepted_tokens for result in successful
        ),
        acceptance_rate=calculate_metric_statistics(
            result.acceptance_rate for result in successful
        ),
    )


def deduplicate_hardware_samples(
    samples: Iterable[HardwareSnapshot],
) -> tuple[HardwareSnapshot, ...]:
    """Return snapshots ordered and uniquely keyed by collection timestamp."""
    by_time = {sample.collected_at: sample for sample in samples}
    return tuple(by_time[timestamp] for timestamp in sorted(by_time))


def _mean(values: Iterable[float | int | None]) -> float | None:
    available = [float(value) for value in values if value is not None]
    return fmean(available) if available else None


def _maximum(values: Iterable[float | int | None]) -> float | int | None:
    available = [value for value in values if value is not None]
    return max(available) if available else None


def summarize_hardware_samples(
    samples: Iterable[HardwareSnapshot],
) -> HardwareBenchmarkSummary | None:
    """Summarize independently available local metrics from unique snapshots."""
    unique = deduplicate_hardware_samples(samples)
    if not unique:
        return None
    gpu_summaries = tuple(summarize_gpus(sample.gpus) for sample in unique)
    fields = (
        tuple(summary.utilisation_percent for summary in gpu_summaries),
        tuple(summary.memory_used_bytes for summary in gpu_summaries),
        tuple(summary.temperature_celsius for summary in gpu_summaries),
        tuple(summary.power_draw_watts for summary in gpu_summaries),
        tuple(sample.cpu.utilisation_percent for sample in unique),
        tuple(sample.memory.used_bytes for sample in unique),
    )
    if not any(value is not None for values in fields for value in values):
        return None
    return HardwareBenchmarkSummary(
        sample_count=len(unique),
        average_gpu_utilisation_percent=_mean(fields[0]),
        maximum_gpu_utilisation_percent=_maximum(fields[0]),
        average_vram_used_bytes=_mean(fields[1]),
        maximum_vram_used_bytes=_maximum(fields[1]),
        average_temperature_celsius=_mean(fields[2]),
        maximum_temperature_celsius=_maximum(fields[2]),
        average_power_draw_watts=_mean(fields[3]),
        maximum_power_draw_watts=_maximum(fields[3]),
        average_cpu_utilisation_percent=_mean(fields[4]),
        maximum_cpu_utilisation_percent=_maximum(fields[4]),
        average_memory_used_bytes=_mean(fields[5]),
        maximum_memory_used_bytes=_maximum(fields[5]),
    )
