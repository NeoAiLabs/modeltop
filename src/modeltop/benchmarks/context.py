"""Sequential Context Length benchmark execution through GenerationService."""

import asyncio
import logging
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, replace

from modeltop.api.errors import ContextLimitError, RequestTimeoutError
from modeltop.benchmarks.base import Benchmark, BenchmarkContext
from modeltop.benchmarks.context_builder import BuiltContextPrompt, build_context_prompt
from modeltop.benchmarks.context_probe import ContextProbePlanner
from modeltop.benchmarks.context_retrieval import (
    RETRIEVAL_INSTRUCTION,
    TRI_MARKER_INSTRUCTION,
    RetrievalMarkerSpec,
    RetrievalPromptSpec,
    generate_retrieval_key,
    measure_retrieval_placements,
    score_single_retrieval,
    score_tri_marker_retrieval,
)
from modeltop.benchmarks.context_statistics import (
    build_context_length_result,
    build_context_observations,
)
from modeltop.benchmarks.hardware_sampling import sample_hardware_snapshots
from modeltop.benchmarks.models import (
    ContextBenchmarkConfig,
    ContextBenchmarkProgress,
    ContextBenchmarkResult,
    ContextBenchmarkStatus,
    ContextLengthResult,
    ContextProbeBounds,
    ContextPromptBuildProgress,
    ContextPromptMeasurement,
    ContextRequestProgress,
    ContextRequestResult,
    RetrievalPosition,
)
from modeltop.benchmarks.runner import run_bounded_workers
from modeltop.benchmarks.token_budget import effective_context_output_budget
from modeltop.chat.models import GenerationSettings
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

type ProgressCallback = Callable[
    [ContextBenchmarkStatus, ContextBenchmarkProgress], None
]


@dataclass(frozen=True, slots=True)
class _AttemptPlan:
    run_number: int
    sequence_number: int
    absolute_attempt: int
    position: RetrievalPosition | None = None
    tri_marker: bool = False
    warmup: bool = False


class _FatalContextError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ContextBenchmark(Benchmark[ContextBenchmarkResult]):
    """Run fixed, sweep, probe, and retrieval Context workloads sequentially."""

    def __init__(
        self,
        generation_service: GenerationService,
        config: ContextBenchmarkConfig,
        *,
        server_id: str,
        server_name: str,
        server_endpoint: str,
        model_id: str,
        backend: str,
        on_progress: ProgressCallback,
    ) -> None:
        self._generation_service = generation_service
        self._config = config
        self._server_id = server_id
        self._server_name = server_name
        self._server_endpoint = server_endpoint
        self._model_id = model_id
        self._backend = backend
        self._on_progress = on_progress
        self._cancel_event = asyncio.Event()
        self._context: BenchmarkContext | None = None
        self._lengths: list[ContextLengthResult] = []
        self._warnings: list[str] = []
        self._active_target: int | None = None
        self._next_target: int | None = None
        self._target_index = 0
        self._target_count = 0
        self._run_number = 0
        self._configured_runs = 0
        self._delay_remaining: float | None = None
        self._build_progress: ContextPromptBuildProgress | None = None
        self._active_request: ContextRequestProgress | None = None
        self._probe_bounds: ContextProbeBounds | None = None
        self._current_requests: list[ContextRequestResult] = []
        self._current_hardware: list[HardwareSnapshot] = []
        self._current_configured = 0

    def request_cancellation(self) -> None:
        """Stop future prompt builds and request dequeues."""
        self._cancel_event.set()

    def _publish(self, status: ContextBenchmarkStatus) -> None:
        cached_hardware = (
            self._context.read_hardware_snapshot()
            if self._context is not None
            else None
        )
        self._on_progress(
            status,
            ContextBenchmarkProgress(
                build=self._build_progress,
                active_target_length=self._active_target,
                next_target_length=self._next_target,
                target_index=self._target_index,
                target_count=self._target_count,
                run_number=self._run_number,
                configured_runs=self._configured_runs,
                delay_remaining_seconds=self._delay_remaining,
                active_request=self._active_request,
                completed_lengths=tuple(self._lengths),
                probe_bounds=self._probe_bounds,
                probe_stage=(
                    self._probe_bounds.stage if self._probe_bounds is not None else None
                ),
                warnings=tuple(self._warnings),
                cached_hardware=cached_hardware,
            ),
        )

    def _on_build_progress(self, progress: ContextPromptBuildProgress) -> None:
        self._build_progress = progress
        self._publish(ContextBenchmarkStatus.BUILDING_PROMPT)

    def _retrieval_prompt(
        self,
        target: int,
        plan: _AttemptPlan,
    ) -> RetrievalPromptSpec | None:
        if plan.position is None:
            return None
        if plan.tri_marker:
            positions: tuple[RetrievalPosition, ...] = ("beginning", "middle", "end")
            labels = ("BEGIN_MARKER", "MIDDLE_MARKER", "END_MARKER")
            markers = tuple(
                RetrievalMarkerSpec(
                    label,
                    generate_retrieval_key(
                        random_seed=self._config.random_seed,
                        target=target,
                        position=position,
                        absolute_attempt=plan.absolute_attempt,
                        marker_index=index,
                        filler="",
                        manual_key=self._config.retrieval_key,
                        regenerate_per_run=self._config.retrieval_regenerate_per_run,
                    ),
                    position,
                )
                for index, (label, position) in enumerate(
                    zip(labels, positions, strict=True)
                )
            )
            return RetrievalPromptSpec(markers, TRI_MARKER_INSTRUCTION)
        key = generate_retrieval_key(
            random_seed=self._config.random_seed,
            target=target,
            position=plan.position,
            absolute_attempt=plan.absolute_attempt,
            marker_index=0,
            filler="",
            manual_key=self._config.retrieval_key,
            regenerate_per_run=self._config.retrieval_regenerate_per_run,
        )
        return RetrievalPromptSpec(
            (RetrievalMarkerSpec("MODELTOP_RETRIEVAL_KEY", key, plan.position),),
            RETRIEVAL_INSTRUCTION,
        )

    async def _build_prompt(
        self,
        target: int,
        plan: _AttemptPlan,
    ) -> tuple[BuiltContextPrompt, RetrievalPromptSpec | None]:
        retrieval_prompt = self._retrieval_prompt(target, plan)
        self._build_progress = None
        self._publish(ContextBenchmarkStatus.BUILDING_PROMPT)
        prompt = await build_context_prompt(
            self._config,
            target,
            self._generation_service.token_counter,
            retrieval_prompt=retrieval_prompt,
            absolute_attempt=plan.absolute_attempt,
            on_progress=self._on_build_progress,
        )
        self._warnings.extend(
            warning for warning in prompt.warnings if warning not in self._warnings
        )
        return prompt, retrieval_prompt

    @staticmethod
    def _measurement_with_server_usage(
        measurement: ContextPromptMeasurement,
        outcome: GenerationOutcome,
    ) -> ContextPromptMeasurement:
        metrics = outcome.metrics
        server_tokens = (
            metrics.prompt_tokens
            if not metrics.prompt_tokens_estimated and measurement.estimated
            else None
        )
        if server_tokens is None:
            return measurement
        difference = server_tokens - measurement.local_prompt_tokens
        percent = (
            difference / measurement.local_prompt_tokens * 100
            if measurement.local_prompt_tokens > 0
            else None
        )
        return replace(
            measurement,
            server_prompt_tokens=server_tokens,
            server_token_difference=difference,
            server_token_difference_percent=percent,
        )

    def _result_from_outcome(
        self,
        target: int,
        plan: _AttemptPlan,
        prompt: BuiltContextPrompt,
        retrieval_prompt: RetrievalPromptSpec | None,
        outcome: GenerationOutcome,
        *,
        state: str,
        accepted: bool | None,
        error_type: str | None = None,
        error_message: str | None = None,
        timed_out: bool = False,
        cancelled: bool = False,
    ) -> ContextRequestResult:
        measurement = self._measurement_with_server_usage(prompt.measurement, outcome)
        metrics = outcome.metrics
        retrieval_results = ()
        if retrieval_prompt is not None and state == "done":
            placements = measure_retrieval_placements(
                prompt.messages[-1].content, retrieval_prompt
            )
            expected = tuple(marker.key for marker in retrieval_prompt.markers)
            if plan.tri_marker:
                assert len(expected) == 3 and len(placements) == 3
                retrieval_results = score_tri_marker_retrieval(
                    outcome.content,
                    expected=(expected[0], expected[1], expected[2]),
                    realised_placements=(
                        placements[0],
                        placements[1],
                        placements[2],
                    ),
                )
            else:
                assert plan.position is not None
                retrieval_results = (
                    score_single_retrieval(
                        outcome.content,
                        expected=expected[0],
                        position=plan.position,
                        realised_placement_percent=placements[0],
                        case_insensitive=(
                            self._config.retrieval_case_insensitive_match
                        ),
                        containment=self._config.retrieval_containment_match,
                    ),
                )
        input_rate = None
        if (
            self._config.estimated_input_rate_enabled
            and metrics.ttft_ms is not None
            and metrics.ttft_ms > 0
        ):
            input_rate = measurement.effective_prompt_tokens / (metrics.ttft_ms / 1000)
        success = state == "done"
        context_rejected = state == "rejected"
        return ContextRequestResult(
            request_id=(
                f"{self._context.benchmark_id if self._context else 'context'}-"
                f"t{target}-r{plan.sequence_number:04d}"
            ),
            target_length=target,
            run_number=plan.run_number,
            sequence_number=plan.sequence_number,
            measurement=measurement,
            requested_at=metrics.request_started_at,
            first_token_at=metrics.first_token_at,
            completed_at=metrics.completed_at,
            ttft_ms=metrics.ttft_ms,
            total_latency_seconds=metrics.total_duration_s,
            generation_duration_seconds=metrics.active_generation_duration_s,
            completion_tokens=metrics.completion_tokens,
            completion_tokens_estimated=metrics.completion_tokens_estimated,
            output_tokens_per_second=metrics.output_tokens_per_second,
            estimated_input_tokens_per_second=input_rate,
            finish_reason=metrics.finish_reason,
            status_code=outcome.status_code,
            streamed=metrics.streamed,
            success=success,
            state=state,  # type: ignore[arg-type]
            accepted=accepted,
            context_rejected=context_rejected,
            timed_out=timed_out,
            cancelled=cancelled,
            retrieval_results=retrieval_results,
            response_character_count=len(outcome.content),
            error_type=error_type,
            error_message=error_message,
        )

    async def _attempt(
        self,
        target: int,
        plan: _AttemptPlan,
        prompt: BuiltContextPrompt,
        retrieval_prompt: RetrievalPromptSpec | None,
    ) -> ContextRequestResult:
        settings = GenerationSettings(
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            max_tokens=effective_context_output_budget(self._config),
            seed=self._config.seed,
            stream=True,
            enable_thinking=(
                False if self._config.thinking_mode == "disabled" else None
            ),
        )
        status = (
            ContextBenchmarkStatus.WARMING_UP
            if plan.warmup
            else (
                ContextBenchmarkStatus.PROBING
                if self._config.mode == "probe"
                else ContextBenchmarkStatus.RUNNING
            )
        )
        request_id = (
            f"{self._context.benchmark_id if self._context else 'context'}-"
            f"t{target}-r{plan.sequence_number:04d}"
        )

        def on_progress(progress: GenerationProgress) -> None:
            self._active_request = ContextRequestProgress(
                request_id=request_id,
                target_length=target,
                run_number=plan.run_number,
                sequence_number=plan.sequence_number,
                state="running",
                latest_metrics=progress.metrics,
                response_character_count=len(progress.content),
                error=None,
            )
            self._publish(status)

        request = GenerationRequest(
            server_id=self._server_id,
            model_id=self._model_id,
            messages=prompt.messages,
            settings=settings,
            request_timeout_seconds=self._config.request_timeout_seconds,
        )
        timeout = asyncio.timeout(self._config.request_timeout_seconds)
        try:
            async with timeout:
                outcome = await self._generation_service.run(request, on_progress)
        except GenerationCancelled as error:
            if not timeout.expired():
                raise
            result = self._result_from_outcome(
                target,
                plan,
                prompt,
                retrieval_prompt,
                error.outcome,
                state="timeout",
                accepted=None,
                error_type="RequestTimeoutError",
                error_message=(
                    "Request timed out; context acceptance could not be confirmed."
                ),
                timed_out=True,
            )
        except TimeoutError:
            now = self._context.monotonic_clock() if self._context else 0.0
            empty_metrics = (
                replace(
                    self._active_request.latest_metrics,
                    completed_at=now,
                    cancelled=True,
                )
                if self._active_request and self._active_request.latest_metrics
                else None
            )
            if empty_metrics is None:
                raise _FatalContextError(
                    "Request timed out; context acceptance could not be confirmed."
                ) from None
            outcome = GenerationOutcome("", empty_metrics)
            result = self._result_from_outcome(
                target,
                plan,
                prompt,
                retrieval_prompt,
                outcome,
                state="timeout",
                accepted=None,
                error_type="RequestTimeoutError",
                error_message=(
                    "Request timed out; context acceptance could not be confirmed."
                ),
                timed_out=True,
            )
        except GenerationFailed as failure:
            if isinstance(failure.error, RequestTimeoutError):
                result = self._result_from_outcome(
                    target,
                    plan,
                    prompt,
                    retrieval_prompt,
                    failure.outcome,
                    state="timeout",
                    accepted=None,
                    error_type=type(failure.error).__name__,
                    error_message=(
                        "Request timed out; context acceptance could not be confirmed."
                    ),
                    timed_out=True,
                )
            elif isinstance(failure.error, ContextLimitError):
                result = self._result_from_outcome(
                    target,
                    plan,
                    prompt,
                    retrieval_prompt,
                    failure.outcome,
                    state="rejected",
                    accepted=False,
                    error_type=type(failure.error).__name__,
                    error_message=failure.error.user_message,
                )
            else:
                result = self._result_from_outcome(
                    target,
                    plan,
                    prompt,
                    retrieval_prompt,
                    failure.outcome,
                    state="error",
                    accepted=None,
                    error_type=type(failure.error).__name__,
                    error_message=failure.error.user_message,
                )
        else:
            accepted = bool(outcome.content or outcome.metrics.finish_reason)
            result = self._result_from_outcome(
                target,
                plan,
                prompt,
                retrieval_prompt,
                outcome,
                state="done" if accepted else "error",
                accepted=True if accepted else None,
                error_type=None if accepted else "EmptyResponseError",
                error_message=None
                if accepted
                else "Server returned no valid response.",
            )
        self._active_request = ContextRequestProgress(
            request_id=result.request_id,
            target_length=target,
            run_number=plan.run_number,
            sequence_number=plan.sequence_number,
            state=result.state,
            latest_metrics=None,
            response_character_count=result.response_character_count,
            error=result.error_message,
        )
        self._publish(status)
        return result

    def _plans_for_target(
        self,
    ) -> tuple[tuple[_AttemptPlan, ...], tuple[_AttemptPlan, ...]]:
        absolute = 0
        warmups: list[_AttemptPlan] = []
        measured: list[_AttemptPlan] = []
        if self._config.mode == "retrieval":
            first_position = self._config.retrieval_positions[0]
            for run in range(1, self._config.warmup_requests + 1):
                absolute += 1
                warmups.append(
                    _AttemptPlan(run, run, absolute, first_position, warmup=True)
                )
            sequence = 0
            for position in self._config.retrieval_positions:
                for run in range(1, self._config.repetitions_per_length + 1):
                    absolute += 1
                    sequence += 1
                    measured.append(_AttemptPlan(run, sequence, absolute, position))
            if self._config.retrieval_truncation_detection:
                for run in range(1, self._config.repetitions_per_length + 1):
                    absolute += 1
                    sequence += 1
                    measured.append(
                        _AttemptPlan(
                            run,
                            sequence,
                            absolute,
                            "middle",
                            tri_marker=True,
                        )
                    )
        else:
            for run in range(1, self._config.warmup_requests + 1):
                absolute += 1
                warmups.append(_AttemptPlan(run, run, absolute, warmup=True))
            for run in range(1, self._config.repetitions_per_length + 1):
                absolute += 1
                measured.append(_AttemptPlan(run, run, absolute))
        return tuple(warmups), tuple(measured)

    async def _run_target(self, target: int) -> ContextLengthResult:
        assert self._context is not None
        warmups, measured = self._plans_for_target()
        self._current_requests = []
        self._current_hardware = []
        self._current_configured = len(measured)
        self._configured_runs = len(measured)
        reusable = (
            self._config.mode != "retrieval"
            and self._config.reuse_prompt
            and not self._config.unique_prompt_suffix_per_run
        )
        shared_prompt: BuiltContextPrompt | None = None
        shared_spec: RetrievalPromptSpec | None = None
        if reusable:
            seed_plan = warmups[0] if warmups else measured[0]
            shared_prompt, shared_spec = await self._build_prompt(target, seed_plan)

        try:
            for plan in warmups:
                if self._cancel_event.is_set():
                    raise asyncio.CancelledError
                prompt, spec = (
                    (shared_prompt, shared_spec)
                    if shared_prompt is not None
                    else await self._build_prompt(target, plan)
                )
                assert prompt is not None
                result = await self._attempt(target, plan, prompt, spec)
                if result.context_rejected:
                    warning = f"Warm-up request was context-rejected at {target}."
                    if warning not in self._warnings:
                        self._warnings.append(warning)
                elif result.timed_out and not self._config.continue_after_timeout:
                    raise _FatalContextError(
                        "Warm-up timed out before measured requests could start."
                    )
                elif result.state == "error":
                    raise _FatalContextError(
                        result.error_message or "Context warm-up failed."
                    )

            measured_started_at = self._context.utc_now()
            sampler_stop = asyncio.Event()
            sampler = asyncio.create_task(
                sample_hardware_snapshots(
                    read_snapshot=self._context.read_hardware_snapshot,
                    measured_phase_started_at=measured_started_at,
                    stop_event=sampler_stop,
                    interval_seconds=self._config.hardware_sample_interval_seconds,
                    append_sample=self._current_hardware.append,
                ),
                name=f"context-hardware-{target}",
            )
            stop_after_result = False

            async def worker(plan: _AttemptPlan) -> ContextRequestResult:
                prompt, spec = (
                    (shared_prompt, shared_spec)
                    if shared_prompt is not None
                    else await self._build_prompt(target, plan)
                )
                assert prompt is not None
                self._run_number = plan.run_number
                return await self._attempt(target, plan, prompt, spec)

            def on_result(result: ContextRequestResult) -> None:
                nonlocal stop_after_result
                self._current_requests.append(result)
                if result.timed_out and not self._config.continue_after_timeout:
                    stop_after_result = True
                if result.state == "error":
                    stop_after_result = True

            try:
                await run_bounded_workers(
                    measured,
                    1,
                    worker,
                    on_result,
                    lambda: self._cancel_event.is_set() or stop_after_result,
                )
            finally:
                sampler_stop.set()
                await sampler
            result = build_context_length_result(
                target_length=target,
                config=self._config,
                configured_requests=len(measured),
                requests=self._current_requests,
                hardware_samples=self._current_hardware,
            )
            if any(request.state == "error" for request in self._current_requests):
                message = next(
                    request.error_message
                    for request in self._current_requests
                    if request.state == "error"
                )
                raise _FatalContextError(message or "Context request failed.")
            return result
        finally:
            shared_prompt = None
            shared_spec = None

    async def _between_targets(self, next_target: int) -> None:
        assert self._context is not None
        delay = self._config.delay_between_lengths_seconds
        if delay <= 0 or self._cancel_event.is_set():
            return
        self._next_target = next_target
        deadline = self._context.monotonic_clock() + delay
        while not self._cancel_event.is_set():
            remaining = max(0.0, deadline - self._context.monotonic_clock())
            self._delay_remaining = remaining
            self._publish(ContextBenchmarkStatus.BETWEEN_LENGTHS)
            if remaining <= 0:
                break
            with suppress(TimeoutError):
                await asyncio.wait_for(
                    self._cancel_event.wait(), timeout=min(remaining, 0.1)
                )
        self._delay_remaining = None
        self._next_target = None

    def _commit_partial_current(self) -> None:
        if self._active_target is None or not self._current_requests:
            return
        if any(length.target_length == self._active_target for length in self._lengths):
            return
        self._lengths.append(
            build_context_length_result(
                target_length=self._active_target,
                config=self._config,
                configured_requests=self._current_configured,
                requests=self._current_requests,
                hardware_samples=self._current_hardware,
            )
        )

    def _result(
        self,
        context: BenchmarkContext,
        status: ContextBenchmarkStatus,
        *,
        error: str | None = None,
    ) -> ContextBenchmarkResult:
        observations = build_context_observations(self._lengths, self._config)
        lengths = tuple(
            replace(
                length,
                observations=tuple(
                    observation
                    for observation in observations
                    if observation.target_length == length.target_length
                ),
            )
            for length in self._lengths
        )
        successful_tokens = [
            request.measurement.effective_prompt_tokens
            for length in lengths
            for request in length.requests
            if request.success
        ]
        fully_rejected = [
            length
            for length in lengths
            if length.attempted_requests == length.configured_requests
            and length.configured_requests > 0
            and length.context_rejected_requests == length.configured_requests
        ]
        return ContextBenchmarkResult(
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
            lengths=lengths,
            highest_successful_prompt_tokens=(
                max(successful_tokens) if successful_tokens else None
            ),
            first_fully_rejected_prompt_tokens=(
                min(
                    request.measurement.effective_prompt_tokens
                    for length in fully_rejected
                    for request in length.requests
                )
                if fully_rejected
                else None
            ),
            probe_bounds=self._probe_bounds,
            possible_truncation=any(
                observation.code
                in {"possible_left_truncation", "possible_right_truncation"}
                for observation in observations
            ),
            cancelled=status is ContextBenchmarkStatus.CANCELLED,
            error=error,
            warnings=tuple(self._warnings),
            observations=observations,
        )

    def cancelled_result(self, context: BenchmarkContext) -> ContextBenchmarkResult:
        """Retain completed/current attempts and hardware after cancellation."""
        self._commit_partial_current()
        return self._result(
            context,
            ContextBenchmarkStatus.CANCELLED,
            error="Benchmark cancelled — partial results retained",
        )

    def error_result(
        self, context: BenchmarkContext, safe_message: str
    ) -> ContextBenchmarkResult:
        """Retain completed/current attempts and hardware after an unexpected error."""
        self._commit_partial_current()
        return self._result(context, ContextBenchmarkStatus.ERROR, error=safe_message)

    async def run(self, context: BenchmarkContext) -> ContextBenchmarkResult:
        """Run the immutable Context plan and preserve the latest terminal result."""
        self._context = context
        logger.info(
            "Context benchmark starting benchmark=%s server=%s model=%s mode=%s "
            "targets=%s unit=%s source=%s repetitions=%d warmups=%d",
            context.benchmark_id,
            self._server_id,
            self._model_id,
            self._config.mode,
            self._config.target_lengths,
            self._config.context_unit,
            self._config.content_source,
            self._config.repetitions_per_length,
            self._config.warmup_requests,
        )
        partial_rejection_streak = 0
        try:
            if self._config.mode == "probe":
                safety = (
                    self._config.maximum_context_test_tokens
                    - self._config.maximum_output_tokens
                )
                planner = ContextProbePlanner(
                    start_tokens=self._config.probe_start_tokens,
                    maximum_tokens=self._config.probe_maximum_tokens,
                    resolution_tokens=self._config.probe_resolution_tokens,
                    safety_maximum_tokens=(
                        safety
                        // self._config.probe_resolution_tokens
                        * self._config.probe_resolution_tokens
                    ),
                )
                self._target_count = 0
                while (target := planner.next_candidate()) is not None:
                    if self._cancel_event.is_set():
                        raise asyncio.CancelledError
                    self._active_target = target
                    self._target_index += 1
                    length = await self._run_target(target)
                    self._lengths.append(length)
                    outcomes = tuple(request.accepted for request in length.requests)
                    planner.record(target, outcomes)
                    self._probe_bounds = planner.bounds
                    self._publish(ContextBenchmarkStatus.PROBING)
                self._probe_bounds = planner.bounds
            else:
                targets = self._config.target_lengths
                self._target_count = len(targets)
                for index, target in enumerate(targets, start=1):
                    if self._cancel_event.is_set():
                        raise asyncio.CancelledError
                    self._active_target = target
                    self._target_index = index
                    length = await self._run_target(target)
                    self._lengths.append(length)
                    fully_attempted = (
                        length.attempted_requests == length.configured_requests
                        and length.configured_requests > 0
                    )
                    all_rejected = (
                        fully_attempted
                        and length.context_rejected_requests
                        == length.configured_requests
                    )
                    partial_rejected = (
                        fully_attempted
                        and length.accepted_requests > 0
                        and length.context_rejected_requests > 0
                        and length.accepted_requests + length.context_rejected_requests
                        == length.configured_requests
                    )
                    partial_rejection_streak = (
                        partial_rejection_streak + 1 if partial_rejected else 0
                    )
                    if self._config.early_stop_enabled and (
                        all_rejected or partial_rejection_streak >= 2
                    ):
                        self._lengths[-1] = replace(length, early_stopped=True)
                        break
                    if length.partial and not self._config.continue_after_timeout:
                        break
                    if index < len(targets):
                        await self._between_targets(targets[index])
            self._active_target = None
            result = self._result(context, ContextBenchmarkStatus.COMPLETED)
        except asyncio.CancelledError:
            result = self.cancelled_result(context)
        except _FatalContextError as error:
            result = self.error_result(context, error.message)
        finally:
            self._active_request = None
            self._build_progress = None
            self._current_requests = []
            self._current_hardware = []
        logger.info(
            "Context benchmark terminal benchmark=%s status=%s lengths=%d",
            context.benchmark_id,
            result.status,
            len(result.lengths),
        )
        return result
