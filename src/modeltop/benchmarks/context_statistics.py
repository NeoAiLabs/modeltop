"""Pure Context benchmark aggregation and cautious typed observations."""

from collections.abc import Iterable
from dataclasses import replace
from itertools import pairwise

from modeltop.benchmarks.context_retrieval import detect_possible_truncation
from modeltop.benchmarks.models import (
    ContextBenchmarkConfig,
    ContextLengthResult,
    ContextObservation,
    ContextRequestResult,
    RetrievalPosition,
)
from modeltop.benchmarks.statistics import (
    calculate_percentile_statistics,
    deduplicate_hardware_samples,
    summarize_hardware_samples,
)
from modeltop.hardware.models import HardwareSnapshot


def build_context_length_result(
    *,
    target_length: int,
    config: ContextBenchmarkConfig,
    configured_requests: int,
    requests: Iterable[ContextRequestResult],
    hardware_samples: Iterable[HardwareSnapshot],
    early_stopped: bool = False,
) -> ContextLengthResult:
    """Aggregate measured attempts only; callers never pass warm-up requests."""
    terminal = tuple(requests)
    successful = tuple(request for request in terminal if request.success)
    attempted = len(terminal)
    accepted = sum(request.accepted is True for request in terminal)
    timeouts = sum(request.timed_out for request in terminal)
    cancelled = sum(request.cancelled for request in terminal)
    rejected = sum(request.context_rejected for request in terminal)
    failures = sum(
        not request.success
        and not request.timed_out
        and not request.cancelled
        and not request.context_rejected
        for request in terminal
    )
    positions: tuple[RetrievalPosition, ...] = config.retrieval_positions
    retrieval_attempts: list[tuple[RetrievalPosition, int]] = []
    retrieval_successes: list[tuple[RetrievalPosition, int]] = []
    retrieval_rates: list[tuple[RetrievalPosition, float]] = []
    for position in positions:
        scores = tuple(
            score
            for request in terminal
            for score in request.retrieval_results
            if score.marker == "MODELTOP_RETRIEVAL_KEY" and score.position == position
        )
        success_count = sum(score.status == "pass" for score in scores)
        retrieval_attempts.append((position, len(scores)))
        retrieval_successes.append((position, success_count))
        retrieval_rates.append(
            (position, success_count / len(scores) * 100 if scores else 0.0)
        )
    unique_hardware = deduplicate_hardware_samples(hardware_samples)
    effective_budget = max(
        (
            request.measurement.effective_prompt_tokens
            + (
                config.retrieval_maximum_output_tokens
                if config.mode == "retrieval"
                else config.maximum_output_tokens
            )
            for request in terminal
        ),
        default=target_length
        + (
            config.retrieval_maximum_output_tokens
            if config.mode == "retrieval"
            else config.maximum_output_tokens
        ),
    )
    result = ContextLengthResult(
        target_length=target_length,
        context_unit=config.context_unit,
        effective_total_budget_tokens=effective_budget,
        configured_requests=configured_requests,
        attempted_requests=attempted,
        completed_requests=attempted,
        accepted_requests=accepted,
        successful_requests=len(successful),
        failed_requests=failures,
        timed_out_requests=timeouts,
        cancelled_requests=cancelled,
        context_rejected_requests=rejected,
        success_rate_percent=(len(successful) / attempted * 100 if attempted else 0.0),
        prompt_tokens=calculate_percentile_statistics(
            request.measurement.effective_prompt_tokens for request in successful
        ),
        ttft_ms=calculate_percentile_statistics(
            request.ttft_ms for request in successful
        ),
        latency_seconds=calculate_percentile_statistics(
            request.total_latency_seconds for request in successful
        ),
        output_tokens_per_second=calculate_percentile_statistics(
            request.output_tokens_per_second for request in successful
        ),
        estimated_input_tokens_per_second=calculate_percentile_statistics(
            request.estimated_input_tokens_per_second for request in successful
        ),
        completion_tokens=calculate_percentile_statistics(
            request.completion_tokens for request in successful
        ),
        retrieval_attempts_by_position=tuple(retrieval_attempts),
        retrieval_successes_by_position=tuple(retrieval_successes),
        retrieval_rate_by_position=tuple(retrieval_rates),
        hardware_samples=unique_hardware,
        hardware_summary=summarize_hardware_samples(unique_hardware),
        partial=attempted < configured_requests or cancelled > 0,
        early_stopped=early_stopped,
        observations=(),
        requests=terminal,
    )
    return replace(result, observations=build_context_observations((result,), config))


def build_context_observations(
    lengths: Iterable[ContextLengthResult],
    config: ContextBenchmarkConfig,
) -> tuple[ContextObservation, ...]:
    """Build closed observations without causal or definite-truncation claims."""
    ordered = tuple(lengths)
    observations: list[ContextObservation] = []
    for length in ordered:
        if (
            length.attempted_requests == length.configured_requests
            and length.configured_requests > 0
            and length.context_rejected_requests == length.configured_requests
        ):
            observations.append(
                ContextObservation(
                    "first_context_rejection",
                    length.target_length,
                    f"First confirmed context rejection at {length.target_length}.",
                )
            )
            break

    successful_lengths = tuple(
        length
        for length in ordered
        if length.successful_requests > 0
        and length.ttft_ms.median is not None
        and length.prompt_tokens.median is not None
    )
    for previous, current in pairwise(successful_lengths):
        assert previous.ttft_ms.median is not None
        assert current.ttft_ms.median is not None
        assert previous.prompt_tokens.median is not None
        assert current.prompt_tokens.median is not None
        if previous.ttft_ms.median <= 0 or previous.prompt_tokens.median <= 0:
            continue
        prompt_ratio = current.prompt_tokens.median / previous.prompt_tokens.median
        if prompt_ratio <= 0:
            continue
        scaled_ttft = (current.ttft_ms.median / previous.ttft_ms.median) / prompt_ratio
        if scaled_ttft > 1.5:
            observations.append(
                ContextObservation(
                    "sharp_ttft_increase",
                    current.target_length,
                    "TTFT increased faster than prompt size across adjacent lengths.",
                )
            )

    output_baseline = next(
        (
            length
            for length in ordered
            if length.output_tokens_per_second.median is not None
            and length.output_tokens_per_second.median > 0
        ),
        None,
    )
    if output_baseline is not None:
        assert output_baseline.output_tokens_per_second.median is not None
        for length in ordered:
            median_speed = length.output_tokens_per_second.median
            if (
                median_speed is not None
                and median_speed < output_baseline.output_tokens_per_second.median * 0.8
            ):
                observations.append(
                    ContextObservation(
                        "output_speed_degradation",
                        length.target_length,
                        "Median output speed fell more than 20% from the "
                        "shortest successful length.",
                    )
                )
                break

    comparable_vram = tuple(
        length
        for length in ordered
        if length.hardware_summary is not None
        and length.hardware_summary.maximum_vram_used_bytes is not None
    )
    if len(comparable_vram) >= 2:
        first = comparable_vram[0]
        last = comparable_vram[-1]
        assert first.hardware_summary is not None
        assert last.hardware_summary is not None
        first_vram = first.hardware_summary.maximum_vram_used_bytes
        last_vram = last.hardware_summary.maximum_vram_used_bytes
        assert first_vram is not None and last_vram is not None
        if last_vram > first_vram:
            observations.append(
                ContextObservation(
                    "vram_growth",
                    last.target_length,
                    "Local maximum VRAM increased by "
                    f"{int(last_vram - first_vram)} bytes.",
                )
            )

    for length in ordered:
        successful_requests = tuple(
            request
            for request in length.requests
            if request.success and request.ttft_ms is not None
        )
        if (
            config.reuse_prompt
            and not config.unique_prompt_suffix_per_run
            and len(successful_requests) >= 2
        ):
            first_ttft = successful_requests[0].ttft_ms
            later = [request.ttft_ms for request in successful_requests[1:]]
            assert first_ttft is not None
            if first_ttft > 0 and any(
                value is not None and value < first_ttft * 0.8 for value in later
            ):
                observations.append(
                    ContextObservation(
                        "possible_prompt_caching",
                        length.target_length,
                        "A byte-identical reused prompt had a substantially "
                        "lower later TTFT.",
                    )
                )
        for request in length.requests:
            direction = detect_possible_truncation(request.retrieval_results)
            if direction is not None:
                detail = ", ".join(
                    f"{score.marker}={score.status}"
                    for score in request.retrieval_results
                )
                observations.append(
                    ContextObservation(
                        direction,  # type: ignore[arg-type]
                        length.target_length,
                        f"Possible prompt truncation detected: {detail}.",
                    )
                )

    if config.mode == "retrieval" and len(ordered) >= 2:

        def retrieval_rate(length: ContextLengthResult) -> float | None:
            attempts = sum(value for _, value in length.retrieval_attempts_by_position)
            successes = sum(
                value for _, value in length.retrieval_successes_by_position
            )
            return successes / attempts * 100 if attempts else None

        baseline = retrieval_rate(ordered[0])
        if baseline is not None:
            for length in ordered[1:]:
                current = retrieval_rate(length)
                if current is not None and baseline - current > 20:
                    observations.append(
                        ContextObservation(
                            "retrieval_degradation",
                            length.target_length,
                            "Retrieval success fell more than 20 percentage "
                            "points from the shortest length.",
                        )
                    )
                    break
    return tuple(observations)
