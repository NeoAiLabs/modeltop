"""Bounded concurrent benchmark execution and pure scaling analysis."""

import asyncio
import hashlib
import logging
import math
from collections.abc import Callable, Iterable, Sequence
from contextlib import suppress
from dataclasses import replace
from datetime import datetime
from itertools import pairwise
from statistics import fmean

from modeltop.benchmarks.base import Benchmark, BenchmarkContext
from modeltop.benchmarks.hardware_sampling import sample_hardware_snapshots
from modeltop.benchmarks.models import (
    ConcurrencyBenchmarkConfig,
    ConcurrencyBenchmarkProgress,
    ConcurrencyBenchmarkResult,
    ConcurrencyBenchmarkStatus,
    ConcurrencyLevelResult,
    ConcurrencyRequestProgress,
    ConcurrencyRequestResult,
    SaturationObservation,
    TokenCountMode,
)
from modeltop.benchmarks.runner import run_bounded_workers
from modeltop.benchmarks.statistics import (
    calculate_percentile_statistics,
    deduplicate_hardware_samples,
    summarize_hardware_samples,
)
from modeltop.chat.models import ChatMessage, GenerationMetrics, GenerationSettings
from modeltop.hardware.models import HardwareSnapshot
from modeltop.services.generation import (
    GenerationCancelled,
    GenerationFailed,
    GenerationOutcome,
    GenerationProgress,
    GenerationRequest,
    GenerationService,
)

logger = logging.getLogger(__name__)

GPU_SATURATION_PERCENT = 95.0
THROUGHPUT_PLATEAU_IMPROVEMENT_PERCENT = 5.0
LATENCY_DEGRADATION_PERCENT = 50.0
TTFT_DEGRADATION_PERCENT = 50.0
REQUEST_SPEED_DEGRADATION_PERCENT = 25.0
RELIABILITY_DEGRADATION_PERCENT = 99.0
OUTPUT_LENGTH_CV_PERCENT = 25.0
EARLY_STOP_FAILURE_COUNT = 8

ProgressCallback = Callable[
    [ConcurrencyBenchmarkStatus, ConcurrencyBenchmarkProgress], None
]


def classify_token_count_mode(
    requests: Iterable[ConcurrencyRequestResult],
) -> TokenCountMode:
    """Classify available successful completion counts by their source."""
    counted = tuple(
        request
        for request in requests
        if request.success and request.completion_tokens is not None
    )
    if not counted:
        return "unavailable"
    estimated = {request.completion_tokens_estimated for request in counted}
    if estimated == {False}:
        return "exact"
    if estimated == {True}:
        return "estimated"
    return "mixed"


def _output_length_warning(
    successful: Sequence[ConcurrencyRequestResult],
) -> str | None:
    values = [
        float(request.completion_tokens)
        for request in successful
        if request.completion_tokens is not None
    ]
    if len(values) < 2:
        return None
    mean = fmean(values)
    if mean <= 0:
        return None
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    coefficient = math.sqrt(variance) / mean * 100.0
    if coefficient <= OUTPUT_LENGTH_CV_PERCENT:
        return None
    return (
        f"Output-length variation is {coefficient:.1f}% (sample coefficient of "
        "variation); throughput comparisons may not be like-for-like."
    )


def build_concurrency_level_result(
    *,
    concurrency: int,
    configured_requests: int,
    wall_time_seconds: float,
    requests: Iterable[ConcurrencyRequestResult],
    hardware_samples: Iterable[HardwareSnapshot] = (),
    early_stopped: bool = False,
) -> ConcurrencyLevelResult:
    """Aggregate one measured level from terminal attempted requests."""
    terminal = tuple(sorted(requests, key=lambda request: request.sequence_number))
    successful = tuple(request for request in terminal if request.success)
    cancelled = sum(request.cancelled for request in terminal)
    timed_out = sum(request.timed_out for request in terminal)
    failed = sum(
        not request.success and not request.cancelled and not request.timed_out
        for request in terminal
    )
    prompt_tokens = sum(request.prompt_tokens or 0 for request in successful)
    completion_tokens = sum(request.completion_tokens or 0 for request in successful)
    elapsed = max(0.0, wall_time_seconds)
    attempted = len(terminal)
    unique_hardware = deduplicate_hardware_samples(hardware_samples)
    warning = _output_length_warning(successful)
    observations: tuple[SaturationObservation, ...] = ()
    if warning is not None:
        observations = (
            SaturationObservation("output_length_variance", concurrency, warning),
        )
    return ConcurrencyLevelResult(
        concurrency=concurrency,
        configured_requests=configured_requests,
        attempted_requests=attempted,
        completed_requests=attempted,
        successful_requests=len(successful),
        failed_requests=failed,
        cancelled_requests=cancelled,
        timed_out_requests=timed_out,
        wall_time_seconds=elapsed,
        success_rate_percent=(
            len(successful) / attempted * 100.0 if attempted else 0.0
        ),
        total_prompt_tokens=prompt_tokens,
        total_completion_tokens=completion_tokens,
        token_count_mode=classify_token_count_mode(successful),
        requests_per_second=(len(successful) / elapsed if elapsed > 0 else 0.0),
        aggregate_output_tokens_per_second=(
            completion_tokens / elapsed if elapsed > 0 else 0.0
        ),
        ttft_ms=calculate_percentile_statistics(
            request.ttft_ms for request in successful
        ),
        latency_seconds=calculate_percentile_statistics(
            request.total_latency_seconds for request in successful
        ),
        generation_duration_seconds=calculate_percentile_statistics(
            request.generation_duration_seconds for request in successful
        ),
        request_output_tokens_per_second=calculate_percentile_statistics(
            request.output_tokens_per_second for request in successful
        ),
        completion_tokens=calculate_percentile_statistics(
            request.completion_tokens for request in successful
        ),
        hardware_samples=unique_hardware,
        hardware_summary=summarize_hardware_samples(unique_hardware),
        output_length_warning=warning,
        partial=attempted < configured_requests or cancelled > 0,
        early_stopped=early_stopped,
        observations=observations,
        requests=terminal,
    )


def analyze_scaling(
    levels: Iterable[ConcurrencyLevelResult],
) -> tuple[SaturationObservation, ...]:
    """Return workload-specific observations across adjacent completed levels."""
    ordered = tuple(sorted(levels, key=lambda level: level.concurrency))
    observations: list[SaturationObservation] = []
    first_failure_level: ConcurrencyLevelResult | None = None
    for level in ordered:
        observations.extend(level.observations)
        hardware = level.hardware_summary
        if (
            hardware is not None
            and hardware.average_gpu_utilisation_percent is not None
            and hardware.average_gpu_utilisation_percent >= GPU_SATURATION_PERCENT
        ):
            observations.append(
                SaturationObservation(
                    "gpu_saturation",
                    level.concurrency,
                    f"LOCAL HARDWARE average GPU utilisation reached "
                    f"{hardware.average_gpu_utilisation_percent:.1f}% at concurrency "
                    f"{level.concurrency}.",
                )
            )
        if (
            first_failure_level is None
            and level.successful_requests < level.attempted_requests
        ):
            first_failure_level = level
        if level.success_rate_percent < RELIABILITY_DEGRADATION_PERCENT:
            observations.append(
                SaturationObservation(
                    "reliability_degradation",
                    level.concurrency,
                    f"Success rate fell to {level.success_rate_percent:.1f}% at "
                    f"concurrency {level.concurrency}.",
                )
            )
    if first_failure_level is not None:
        observations.append(
            SaturationObservation(
                "first_failures",
                first_failure_level.concurrency,
                f"The first request failures occurred at concurrency "
                f"{first_failure_level.concurrency}.",
            )
        )
    if len(ordered) < 2:
        return tuple(observations)

    peak = max(ordered, key=lambda level: level.aggregate_output_tokens_per_second)
    observations.append(
        SaturationObservation(
            "peak_throughput",
            peak.concurrency,
            f"Peak measured aggregate throughput was "
            f"{peak.aggregate_output_tokens_per_second:.1f} tok/s at concurrency "
            f"{peak.concurrency}.",
        )
    )
    ttft_levels = tuple(level for level in ordered if level.ttft_ms.median is not None)
    if ttft_levels:
        lowest = min(ttft_levels, key=lambda level: float(level.ttft_ms.median or 0.0))
        observations.append(
            SaturationObservation(
                "lowest_median_ttft",
                lowest.concurrency,
                f"Lowest measured median TTFT was {lowest.ttft_ms.median:.1f} ms "
                f"at concurrency {lowest.concurrency}.",
            )
        )

    for previous, current in pairwise(ordered):
        previous_throughput = previous.aggregate_output_tokens_per_second
        if previous_throughput > 0:
            improvement = (
                (current.aggregate_output_tokens_per_second - previous_throughput)
                / previous_throughput
                * 100.0
            )
            if improvement < THROUGHPUT_PLATEAU_IMPROVEMENT_PERCENT:
                observations.append(
                    SaturationObservation(
                        "throughput_plateau",
                        current.concurrency,
                        f"Aggregate throughput improved {improvement:.1f}% from "
                        f"concurrency {previous.concurrency} to {current.concurrency}.",
                    )
                )
        pairs = (
            (
                previous.latency_seconds.p95,
                current.latency_seconds.p95,
                LATENCY_DEGRADATION_PERCENT,
                "latency_degradation",
                "p95 latency",
            ),
            (
                previous.ttft_ms.p95,
                current.ttft_ms.p95,
                TTFT_DEGRADATION_PERCENT,
                "ttft_degradation",
                "p95 TTFT",
            ),
        )
        for old, new, threshold, code, label in pairs:
            if old is not None and old > 0 and new is not None:
                increase = (new - old) / old * 100.0
                if increase > threshold:
                    observations.append(
                        SaturationObservation(
                            code,  # type: ignore[arg-type]
                            current.concurrency,
                            f"{label} increased {increase:.1f}% versus concurrency "
                            f"{previous.concurrency}.",
                        )
                    )
        old_speed = previous.request_output_tokens_per_second.median
        new_speed = current.request_output_tokens_per_second.median
        if old_speed is not None and old_speed > 0 and new_speed is not None:
            drop = (old_speed - new_speed) / old_speed * 100.0
            if drop > REQUEST_SPEED_DEGRADATION_PERCENT:
                observations.append(
                    SaturationObservation(
                        "request_speed_degradation",
                        current.concurrency,
                        f"Median per-request speed dropped {drop:.1f}% versus "
                        f"concurrency {previous.concurrency}.",
                    )
                )
    return tuple(observations)


class ConcurrencyBenchmark(Benchmark[ConcurrencyBenchmarkResult]):
    """Run warm-up and measured requests through a bounded worker pool."""

    def __init__(
        self,
        generation_service: GenerationService,
        config: ConcurrencyBenchmarkConfig,
        *,
        server_id: str,
        server_name: str,
        server_endpoint: str,
        model_id: str,
        backend: str,
        hardware_refresh_interval_seconds: float,
        on_progress: ProgressCallback,
    ) -> None:
        self._generation_service = generation_service
        self._config = config
        self._server_id = server_id
        self._server_name = server_name
        self._server_endpoint = server_endpoint
        self._model_id = model_id
        self._backend = backend
        self._sample_interval = min(hardware_refresh_interval_seconds, 0.5)
        self._on_progress = on_progress
        self._cancel_event = asyncio.Event()
        self._levels: list[ConcurrencyLevelResult] = []
        self._warnings: list[str] = []
        self._rows: dict[str, ConcurrencyRequestProgress] = {}
        self._terminal: dict[str, ConcurrencyRequestResult] = {}
        self._phase: str | None = None
        self._level: int | None = None
        self._level_started_at: float | None = None
        self._next_level: int | None = None
        self._delay_remaining: float | None = None
        self._context: BenchmarkContext | None = None

    def request_cancellation(self) -> None:
        """Stop future dequeues and interrupt cancellible waits."""
        self._cancel_event.set()

    def _progress(self, status: ConcurrencyBenchmarkStatus) -> None:
        now = self._context.monotonic_clock() if self._context is not None else 0.0
        elapsed = (
            max(0.0, now - self._level_started_at)
            if self._level_started_at is not None
            else 0.0
        )
        terminal = tuple(self._terminal.values())
        successful = tuple(request for request in terminal if request.success)
        active_metrics = tuple(
            row.latest_metrics
            for row in self._rows.values()
            if row.state == "running" and row.latest_metrics is not None
        )
        completion_tokens = sum(
            request.completion_tokens or 0 for request in successful
        ) + sum(metrics.completion_tokens or 0 for metrics in active_metrics)
        ttft_values = [
            request.ttft_ms for request in terminal if request.ttft_ms is not None
        ] + [
            metrics.ttft_ms for metrics in active_metrics if metrics.ttft_ms is not None
        ]
        median_ttft = calculate_percentile_statistics(ttft_values).median
        rows = tuple(sorted(self._rows.values(), key=lambda row: row.sequence_number))
        self._on_progress(
            status,
            ConcurrencyBenchmarkProgress(
                phase=self._phase,  # type: ignore[arg-type]
                active_concurrency_level=self._level,
                next_concurrency_level=self._next_level,
                delay_remaining_seconds=self._delay_remaining,
                configured_requests=len(rows),
                active_request_count=sum(row.state == "running" for row in rows),
                queued_request_count=sum(row.state == "queued" for row in rows),
                completed_request_count=len(terminal),
                successful_request_count=len(successful),
                failed_request_count=sum(
                    not request.success
                    and not request.cancelled
                    and not request.timed_out
                    for request in terminal
                ),
                timed_out_request_count=sum(request.timed_out for request in terminal),
                cancelled_request_count=sum(request.cancelled for request in terminal),
                elapsed_seconds=elapsed,
                aggregate_output_tokens_per_second=(
                    completion_tokens / elapsed if elapsed > 0 else 0.0
                ),
                requests_per_second=(len(successful) / elapsed if elapsed > 0 else 0.0),
                median_ttft_ms=median_ttft,
                request_rows=rows,
                completed_levels=tuple(self._levels),
                warnings=tuple(self._warnings),
            ),
        )

    def _make_rows(self, context: BenchmarkContext, level: int, count: int) -> None:
        queued_at = context.monotonic_clock()
        self._rows = {
            f"{context.benchmark_id}-c{level}-r{sequence:04d}": (
                ConcurrencyRequestProgress(
                    request_id=f"{context.benchmark_id}-c{level}-r{sequence:04d}",
                    concurrency_level=level,
                    sequence_number=sequence,
                    state="queued",
                    queued_at=queued_at,
                    started_at=None,
                    latest_metrics=None,
                    error=None,
                )
            )
            for sequence in range(1, count + 1)
        }
        self._terminal = {}

    async def _execute_request(
        self,
        context: BenchmarkContext,
        level: int,
        sequence: int,
        *,
        publish: bool,
    ) -> ConcurrencyRequestResult:
        request_id = f"{context.benchmark_id}-c{level}-r{sequence:04d}"
        started_at = context.monotonic_clock()
        row = self._rows.get(request_id)
        queue_wait = max(0.0, started_at - row.queued_at) if row is not None else 0.0
        if publish and row is not None:
            self._rows[request_id] = replace(
                row, state="running", started_at=started_at
            )
            self._progress(ConcurrencyBenchmarkStatus.RUNNING)

        messages: list[ChatMessage] = []
        if self._config.system_prompt is not None:
            messages.append(ChatMessage("system", self._config.system_prompt))
        messages.append(ChatMessage("user", self._config.prompt))
        request = GenerationRequest(
            server_id=self._server_id,
            model_id=self._model_id,
            messages=tuple(messages),
            settings=GenerationSettings(
                temperature=self._config.temperature,
                top_p=self._config.top_p,
                max_tokens=self._config.max_tokens,
                seed=self._config.seed,
                stream=True,
            ),
            request_timeout_seconds=self._config.request_timeout_seconds,
        )
        latest_outcome: GenerationOutcome | None = None

        def update(progress: GenerationProgress) -> None:
            if publish and request_id in self._rows:
                self._rows[request_id] = replace(
                    self._rows[request_id], latest_metrics=progress.metrics
                )
                self._progress(ConcurrencyBenchmarkStatus.RUNNING)

        success = False
        cancelled = False
        timed_out = False
        error_type: str | None = None
        error_message: str | None = None
        timeout = asyncio.timeout(self._config.request_timeout_seconds)
        try:
            async with timeout:
                try:
                    latest_outcome = await self._generation_service.run(request, update)
                except GenerationCancelled as error:
                    latest_outcome = error.outcome
                    if timeout.expired():
                        timed_out = True
                        error_type = "TimeoutError"
                        error_message = "Request timed out"
                    else:
                        cancelled = True
                        error_type = "CancelledError"
                        error_message = "Request cancelled"
            if not timed_out and not cancelled:
                if latest_outcome.content:
                    success = True
                else:
                    error_type = "EmptyResponse"
                    error_message = "Empty model response"
        except TimeoutError:
            timed_out = True
            error_type = "TimeoutError"
            error_message = "Request timed out"
        except GenerationFailed as error:
            latest_outcome = error.outcome
            error_type = type(error.error).__name__
            error_message = error.error.user_message
        except GenerationCancelled as error:
            latest_outcome = error.outcome
            cancelled = True
            error_type = "CancelledError"
            error_message = "Request cancelled"

        completed_at = context.monotonic_clock()
        metrics = (
            latest_outcome.metrics
            if latest_outcome is not None
            else GenerationMetrics(
                request_started_at=started_at, completed_at=completed_at
            )
        )
        result = ConcurrencyRequestResult(
            request_id=request_id,
            concurrency_level=level,
            sequence_number=sequence,
            started_at=metrics.request_started_at,
            first_token_at=metrics.first_token_at,
            completed_at=metrics.completed_at or completed_at,
            queue_wait_seconds=queue_wait,
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.completion_tokens,
            total_tokens=metrics.total_tokens,
            prompt_tokens_estimated=metrics.prompt_tokens_estimated,
            completion_tokens_estimated=metrics.completion_tokens_estimated,
            total_tokens_estimated=metrics.total_tokens_estimated,
            ttft_ms=metrics.ttft_ms,
            total_latency_seconds=metrics.total_duration_s,
            generation_duration_seconds=metrics.active_generation_duration_s,
            output_tokens_per_second=metrics.output_tokens_per_second,
            finish_reason=metrics.finish_reason,
            status_code=latest_outcome.status_code if latest_outcome else None,
            success=success,
            cancelled=cancelled,
            timed_out=timed_out,
            streamed=metrics.streamed,
            response_character_count=(
                len(latest_outcome.content) if latest_outcome else 0
            ),
            error_type=error_type,
            error_message=error_message,
        )
        if publish and request_id in self._rows:
            state = (
                "done"
                if success
                else "timeout"
                if timed_out
                else "cancelled"
                if cancelled
                else "error"
            )
            self._rows[request_id] = replace(
                self._rows[request_id],
                state=state,
                latest_metrics=metrics,
                error=error_message,
            )
        if success:
            logger.debug("Concurrency request completed request=%s", request_id)
        else:
            logger.warning(
                "Concurrency request failed request=%s error=%s status=%s",
                request_id,
                error_type,
                result.status_code,
            )
        return result

    async def _run_phase(
        self,
        context: BenchmarkContext,
        level: int,
        count: int,
        *,
        publish: bool,
    ) -> tuple[ConcurrencyRequestResult, ...]:
        if publish:
            self._make_rows(context, level, count)
        results: list[ConcurrencyRequestResult] = []
        stop_early = False

        async def worker(sequence: int) -> ConcurrencyRequestResult:
            logger.debug(
                "Concurrency request starting benchmark=%s level=%d sequence=%d",
                context.benchmark_id,
                level,
                sequence,
            )
            return await self._execute_request(
                context, level, sequence, publish=publish
            )

        def on_result(result: ConcurrencyRequestResult) -> None:
            nonlocal stop_early
            results.append(result)
            if result.success and not result.streamed:
                warning = (
                    "Server used non-stream fallback; TTFT, generation duration, "
                    "and per-request speed are unavailable."
                )
                if warning not in self._warnings:
                    self._warnings.append(warning)
            if publish:
                self._terminal[result.request_id] = result
                first = {
                    item.sequence_number: item
                    for item in results
                    if item.sequence_number <= EARLY_STOP_FAILURE_COUNT
                }
                stop_early = len(first) == EARLY_STOP_FAILURE_COUNT and all(
                    not item.success for item in first.values()
                )
                self._progress(ConcurrencyBenchmarkStatus.RUNNING)

        await run_bounded_workers(
            range(1, count + 1),
            level,
            worker,
            on_result,
            lambda: self._cancel_event.is_set() or stop_early,
        )
        return tuple(results)

    async def _sample_hardware(
        self,
        context: BenchmarkContext,
        level_started_at: datetime,
        stop: asyncio.Event,
        samples: list[HardwareSnapshot],
    ) -> None:
        await sample_hardware_snapshots(
            read_snapshot=context.read_hardware_snapshot,
            measured_phase_started_at=level_started_at,
            stop_event=stop,
            interval_seconds=self._sample_interval,
            append_sample=samples.append,
        )

    async def _between_levels(self, context: BenchmarkContext, next_level: int) -> None:
        delay = self._config.delay_between_levels_seconds
        if delay <= 0 or self._cancel_event.is_set():
            return
        self._phase = None
        self._level = None
        self._next_level = next_level
        deadline = context.monotonic_clock() + delay
        while not self._cancel_event.is_set():
            remaining = max(0.0, deadline - context.monotonic_clock())
            self._delay_remaining = remaining
            self._progress(ConcurrencyBenchmarkStatus.BETWEEN_LEVELS)
            if remaining <= 0:
                break
            with suppress(TimeoutError):
                await asyncio.wait_for(
                    self._cancel_event.wait(), timeout=min(remaining, 0.1)
                )
        self._delay_remaining = None
        self._next_level = None

    def _result(
        self,
        context: BenchmarkContext,
        status: ConcurrencyBenchmarkStatus,
        *,
        error: str | None = None,
    ) -> ConcurrencyBenchmarkResult:
        observations = analyze_scaling(self._levels)
        return ConcurrencyBenchmarkResult(
            benchmark_id=context.benchmark_id,
            status=status,
            server_id=self._server_id,
            server_name=self._server_name,
            server_endpoint=self._server_endpoint,
            model_id=self._model_id,
            backend=self._backend,
            started_at=context.started_at,
            completed_at=context.utc_now(),
            config=self._config,
            levels=tuple(self._levels),
            cancelled=status is ConcurrencyBenchmarkStatus.CANCELLED,
            error=error,
            warnings=tuple(self._warnings),
            observations=observations,
        )

    def cancelled_result(self, context: BenchmarkContext) -> ConcurrencyBenchmarkResult:
        """Build the partial terminal result after structured task cancellation."""
        return self._result(
            context,
            ConcurrencyBenchmarkStatus.CANCELLED,
            error="Benchmark cancelled — partial results retained",
        )

    async def run(self, context: BenchmarkContext) -> ConcurrencyBenchmarkResult:
        """Run sorted levels sequentially and preserve all terminal measured work."""
        self._context = context
        prompt_hash = hashlib.sha256(self._config.prompt.encode()).hexdigest()[:12]
        logger.info(
            "Concurrency benchmark starting benchmark=%s server=%s model=%s mode=%s "
            "levels=%s requests=%d warmups=%d max_tokens=%d temperature=%s top_p=%s "
            "seed=%s timeout=%s prompt_chars=%d prompt_sha256=%s",
            context.benchmark_id,
            self._server_id,
            self._model_id,
            self._config.mode,
            self._config.concurrency_levels,
            self._config.requests_per_level,
            self._config.warmup_requests,
            self._config.max_tokens,
            self._config.temperature,
            self._config.top_p,
            self._config.seed,
            self._config.request_timeout_seconds,
            len(self._config.prompt),
            prompt_hash,
        )
        try:
            for index, level in enumerate(self._config.concurrency_levels):
                if self._cancel_event.is_set():
                    break
                if index:
                    await self._between_levels(context, level)
                    if self._cancel_event.is_set():
                        break
                if self._config.warmup_requests:
                    self._phase = "warmup"
                    self._level = level
                    self._rows = {}
                    self._terminal = {}
                    self._progress(ConcurrencyBenchmarkStatus.WARMING_UP)
                    warmups = await self._run_phase(
                        context,
                        level,
                        self._config.warmup_requests,
                        publish=False,
                    )
                    failures = sum(not request.success for request in warmups)
                    logger.info(
                        "Concurrency warm-up completed benchmark=%s level=%d "
                        "attempted=%d failed=%d",
                        context.benchmark_id,
                        level,
                        len(warmups),
                        failures,
                    )
                    if warmups and failures == len(warmups):
                        return self._result(
                            context,
                            ConcurrencyBenchmarkStatus.ERROR,
                            error=(
                                f"Every warm-up request failed at concurrency {level}"
                            ),
                        )
                    if failures:
                        self._warnings.append(
                            f"{failures} of {len(warmups)} warm-up requests failed at "
                            f"concurrency {level}."
                        )
                if self._cancel_event.is_set():
                    break

                self._phase = "measured"
                self._level = level
                self._level_started_at = context.monotonic_clock()
                level_started_utc = context.utc_now()
                self._make_rows(context, level, self._config.requests_per_level)
                self._progress(ConcurrencyBenchmarkStatus.RUNNING)
                hardware_samples: list[HardwareSnapshot] = []
                sampler_stop = asyncio.Event()
                sampler = asyncio.create_task(
                    self._sample_hardware(
                        context, level_started_utc, sampler_stop, hardware_samples
                    ),
                    name=f"concurrency-hardware-{level}",
                )
                early_stopped = False
                try:
                    requests = await self._run_phase(
                        context,
                        level,
                        self._config.requests_per_level,
                        publish=True,
                    )
                    first = {
                        request.sequence_number: request
                        for request in requests
                        if request.sequence_number <= EARLY_STOP_FAILURE_COUNT
                    }
                    early_stopped = len(first) == EARLY_STOP_FAILURE_COUNT and all(
                        not request.success for request in first.values()
                    )
                finally:
                    sampler_stop.set()
                    await sampler
                wall_time = max(0.0, context.monotonic_clock() - self._level_started_at)
                level_result = build_concurrency_level_result(
                    concurrency=level,
                    configured_requests=self._config.requests_per_level,
                    wall_time_seconds=wall_time,
                    requests=requests,
                    hardware_samples=hardware_samples,
                    early_stopped=early_stopped,
                )
                self._levels.append(level_result)
                logger.info(
                    "Concurrency level completed benchmark=%s level=%d attempted=%d "
                    "success=%d tok_s=%.2f req_s=%.2f token_mode=%s hardware=%s",
                    context.benchmark_id,
                    level,
                    level_result.attempted_requests,
                    level_result.successful_requests,
                    level_result.aggregate_output_tokens_per_second,
                    level_result.requests_per_second,
                    level_result.token_count_mode,
                    "available" if level_result.hardware_summary else "unavailable",
                )
                self._progress(ConcurrencyBenchmarkStatus.RUNNING)
                if early_stopped:
                    logger.warning(
                        "Concurrency early stop benchmark=%s level=%d",
                        context.benchmark_id,
                        level,
                    )
                    break
        except asyncio.CancelledError:
            self._cancel_event.set()
            if self._level_started_at is not None and self._terminal:
                level = self._level
                if level is not None and not any(
                    item.concurrency == level for item in self._levels
                ):
                    self._levels.append(
                        build_concurrency_level_result(
                            concurrency=level,
                            configured_requests=self._config.requests_per_level,
                            wall_time_seconds=max(
                                0.0,
                                context.monotonic_clock() - self._level_started_at,
                            ),
                            requests=self._terminal.values(),
                        )
                    )
            raise

        if self._cancel_event.is_set():
            return self._result(
                context,
                ConcurrencyBenchmarkStatus.CANCELLED,
                error="Benchmark cancelled — partial results retained",
            )
        result = self._result(context, ConcurrencyBenchmarkStatus.COMPLETED)
        logger.info(
            "Concurrency benchmark completed benchmark=%s observations=%s",
            context.benchmark_id,
            tuple(observation.code for observation in result.observations),
        )
        return result
