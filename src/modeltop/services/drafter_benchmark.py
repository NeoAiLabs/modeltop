"""Sequential Drafter benchmark orchestration with latest-only retention."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from uuid import uuid4

from modeltop.benchmarks.models import (
    DrafterBenchmarkConfig,
    DrafterBenchmarkProgress,
    DrafterBenchmarkResult,
    DrafterBenchmarkStatus,
    DrafterObservation,
    DrafterPhase,
    DrafterRunResult,
)
from modeltop.benchmarks.statistics import build_drafter_aggregates
from modeltop.chat.models import ChatMessage, GenerationMetrics, GenerationSettings
from modeltop.hardware.models import HardwareSnapshot
from modeltop.models import DiscoveredModel, ServerConfig, format_backend_label
from modeltop.services.generation import (
    GenerationCancelled,
    GenerationFailed,
    GenerationOutcome,
    GenerationProgress,
    GenerationRequest,
    GenerationService,
)
from modeltop.state import ApplicationState, ApplicationStateStore, ServerStatus

logger = logging.getLogger(__name__)

type StateCallback = Callable[[ApplicationState], None]
type UtcNow = Callable[[], datetime]
type BenchmarkIdFactory = Callable[[datetime], str]


class DrafterBenchmarkOperationError(Exception):
    """Readable synchronous rejection before benchmark network I/O."""

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


@dataclass(frozen=True, slots=True)
class PendingDrafterBenchmark:
    """Immutable metadata captured by the atomic Drafter reservation."""

    benchmark_id: str
    config: DrafterBenchmarkConfig
    server_id: str
    server_name: str
    server_endpoint: str
    model_id: str
    backend: str
    started_at: datetime
    hardware_before: HardwareSnapshot | None


def _default_benchmark_id_factory(started_at: datetime) -> str:
    timestamp = started_at.strftime("%Y%m%dT%H%M%S%fZ")
    return f"drafter-{timestamp}-{uuid4().hex[:8]}"


def build_drafter_observations(
    run_results: Sequence[DrafterRunResult],
    acceptance_rate_mean: float | None,
) -> tuple[DrafterObservation, ...]:
    """Derive the closed observation set for a terminal Drafter result."""
    successful = tuple(
        result for result in run_results if not result.warmup and result.success
    )
    if not successful:
        return ()

    with_telemetry = tuple(
        result for result in successful if result.speculative_telemetry_present
    )
    observations: list[DrafterObservation] = []
    if not with_telemetry:
        observations.append(
            DrafterObservation(
                code="speculative_telemetry_unavailable",
                message=(
                    "Server did not report draft/accept usage fields; "
                    "throughput metrics only."
                ),
            )
        )
    elif len(with_telemetry) < len(successful):
        observations.append(
            DrafterObservation(
                code="partial_speculative_telemetry",
                message=(
                    "Some measured runs lacked draft/accept usage fields; "
                    "speculative aggregates use available runs only."
                ),
            )
        )
    elif acceptance_rate_mean is not None and acceptance_rate_mean < 0.30:
        observations.append(
            DrafterObservation(
                code="low_mean_acceptance_rate",
                message=(
                    f"Mean acceptance rate is {acceptance_rate_mean:.2f}; "
                    "draft quality may be limiting speedup."
                ),
            )
        )
    return tuple(observations)


class DrafterBenchmarkService:
    """Reserve, execute, and retain one sequential Drafter benchmark at a time."""

    def __init__(
        self,
        generation_service: GenerationService,
        state_store: ApplicationStateStore,
        server: ServerConfig,
        on_state_change: StateCallback,
        *,
        utc_now: UtcNow = lambda: datetime.now(UTC),
        benchmark_id_factory: BenchmarkIdFactory = _default_benchmark_id_factory,
    ) -> None:
        self._generation_service = generation_service
        self._state_store = state_store
        self._server = server
        self._on_state_change = on_state_change
        self._utc_now = utc_now
        self._benchmark_id_factory = benchmark_id_factory

    @property
    def state(self) -> ApplicationState:
        return self._state_store.state

    def _publish(
        self, transform: Callable[[ApplicationState], ApplicationState]
    ) -> ApplicationState:
        state = self._state_store.update(transform)
        self._on_state_change(state)
        return state

    def begin_benchmark(
        self, config: DrafterBenchmarkConfig
    ) -> PendingDrafterBenchmark:
        """Atomically validate and reserve a Drafter run before network I/O."""

        pending: PendingDrafterBenchmark | None = None

        def reserve(state: ApplicationState) -> ApplicationState:
            nonlocal pending
            if state.tool_calling_benchmark.is_active:
                raise DrafterBenchmarkOperationError(
                    "A Tool Calling benchmark is already running."
                )
            if state.context_benchmark.is_active:
                raise DrafterBenchmarkOperationError(
                    "A Context benchmark is already running."
                )
            if state.concurrency_benchmark.is_active:
                raise DrafterBenchmarkOperationError(
                    "A Concurrency benchmark is already running."
                )
            if state.drafter_benchmark.is_active:
                raise DrafterBenchmarkOperationError(
                    "A Drafter benchmark is already running."
                )
            if state.speed_test.is_active:
                raise DrafterBenchmarkOperationError("A Speed Test is already running.")
            if state.active_generation_id is not None:
                raise DrafterBenchmarkOperationError(
                    "A Chat generation is already running."
                )
            if state.server_status is not ServerStatus.ONLINE:
                raise DrafterBenchmarkOperationError("The selected server is offline.")
            server_id = state.selected_server_id
            model_id = state.selected_model_id
            if server_id is None or model_id is None:
                raise DrafterBenchmarkOperationError(
                    "Select an available model before starting Drafter."
                )
            selected_model = next(
                (model for model in state.available_models if model.id == model_id),
                DiscoveredModel(id=model_id),
            )
            backend = self._server.backend_label
            if backend == "--":
                backend = format_backend_label(selected_model.owned_by)
            started_at = self._utc_now()
            benchmark_id = self._benchmark_id_factory(started_at)
            pending = PendingDrafterBenchmark(
                benchmark_id=benchmark_id,
                config=config,
                server_id=server_id,
                server_name=self._server.name,
                server_endpoint=self._server.endpoint_label,
                model_id=model_id,
                backend=backend,
                started_at=started_at,
                hardware_before=state.hardware_snapshot,
            )
            lane = state.drafter_benchmark
            return replace(
                state,
                drafter_benchmark=replace(
                    lane,
                    config=config,
                    status=DrafterBenchmarkStatus.PREPARING,
                    active_benchmark_id=benchmark_id,
                    progress=None,
                    started_at=started_at,
                    benchmark_error=None,
                ),
            )

        try:
            self._publish(reserve)
        except DrafterBenchmarkOperationError:
            raise
        except Exception:
            if pending is not None:
                self._finalize(
                    pending,
                    (),
                    DrafterBenchmarkStatus.FAILED,
                    "Drafter benchmark failed",
                )
            logger.exception("Drafter reservation callback failed")
            raise
        if pending is None:
            raise RuntimeError("Drafter reservation did not produce a request")
        logger.info(
            "Drafter reserved id=%s server=%s model=%s",
            pending.benchmark_id,
            pending.server_id,
            pending.model_id,
        )
        return pending

    async def run_benchmark(
        self, pending: PendingDrafterBenchmark
    ) -> DrafterBenchmarkResult:
        """Run warm-ups then measured requests with at most one request in flight."""
        lane = self.state.drafter_benchmark
        if lane.active_benchmark_id != pending.benchmark_id or not lane.is_active:
            raise DrafterBenchmarkOperationError(
                "This Drafter benchmark is no longer active."
            )

        had_errors = False
        run_results: list[DrafterRunResult] = []
        current_phase: DrafterPhase | None = None
        current_run = 0
        current_metrics: GenerationMetrics | None = None
        current_recorded = False
        completed_measured = 0

        try:
            phases = (
                (True, pending.config.warmup_runs),
                (False, pending.config.measured_runs),
            )
            for warmup, total in phases:
                for run_number in range(1, total + 1):
                    current_phase = "warmup" if warmup else "measured"
                    current_run = run_number
                    current_metrics = None
                    current_recorded = False
                    self._begin_request(
                        pending,
                        warmup=warmup,
                        run_number=run_number,
                        total=total,
                        completed_measured=completed_measured,
                        last_error=None,
                    )
                    request = GenerationRequest(
                        server_id=pending.server_id,
                        model_id=pending.model_id,
                        messages=(ChatMessage("user", pending.config.prompt),),
                        settings=GenerationSettings(
                            temperature=pending.config.temperature,
                            top_p=pending.config.top_p,
                            max_tokens=pending.config.max_tokens,
                            seed=pending.config.seed,
                            stream=True,
                        ),
                        request_timeout_seconds=(
                            pending.config.request_timeout_seconds
                        ),
                    )

                    def on_progress(
                        progress: GenerationProgress,
                        *,
                        _completed: int = completed_measured,
                    ) -> None:
                        nonlocal current_metrics
                        current_metrics = progress.metrics
                        self._handle_progress(
                            pending,
                            progress,
                            completed_measured=_completed,
                        )

                    try:
                        outcome = await self._generation_service.run(
                            request, on_progress
                        )
                    except GenerationCancelled as error:
                        run_result = self._run_result(
                            run_number,
                            warmup,
                            error.outcome,
                            success=False,
                            cancelled=True,
                            error_message=None,
                        )
                        run_results.append(run_result)
                        current_recorded = True
                        self._publish_last_error(pending, run_result.error)
                        self._finalize(
                            pending,
                            tuple(run_results),
                            DrafterBenchmarkStatus.CANCELLED,
                            error_message=None,
                        )
                        raise asyncio.CancelledError from error
                    except GenerationFailed as error:
                        run_result = self._run_result(
                            run_number,
                            warmup,
                            error.outcome,
                            success=False,
                            cancelled=False,
                            error_message=error.error.user_message,
                        )
                        run_results.append(run_result)
                        current_recorded = True
                        had_errors = True
                        self._publish_last_error(pending, run_result.error)
                        logger.warning(
                            "Drafter request failed id=%s phase=%s index=%d error=%s",
                            pending.benchmark_id,
                            current_phase,
                            run_number,
                            type(error.error).__name__,
                        )
                        if not pending.config.continue_on_error:
                            return self._finalize(
                                pending,
                                tuple(run_results),
                                DrafterBenchmarkStatus.FAILED,
                                error.error.user_message,
                            )
                    else:
                        run_result = self._run_result(
                            run_number,
                            warmup,
                            outcome,
                            success=True,
                            cancelled=False,
                            error_message=None,
                        )
                        run_results.append(run_result)
                        current_recorded = True
                        if not warmup:
                            completed_measured += 1
                        self._publish_last_error(pending, None)

            status = (
                DrafterBenchmarkStatus.COMPLETED_WITH_ERRORS
                if had_errors
                else DrafterBenchmarkStatus.COMPLETED
            )
            return self._finalize(
                pending, tuple(run_results), status, error_message=None
            )
        except asyncio.CancelledError:
            if self._is_active(pending.benchmark_id):
                if current_phase is not None and not current_recorded:
                    partial = self._partial_run_result(
                        current_run,
                        current_phase == "warmup",
                        current_metrics,
                        cancelled=True,
                        error_message=None,
                    )
                    run_results.append(partial)
                self._finalize(
                    pending,
                    tuple(run_results),
                    DrafterBenchmarkStatus.CANCELLED,
                    error_message=None,
                )
            raise
        except Exception as error:
            if self._is_active(pending.benchmark_id):
                if current_phase is not None and not current_recorded:
                    partial = self._partial_run_result(
                        current_run,
                        current_phase == "warmup",
                        current_metrics,
                        cancelled=False,
                        error_message="Drafter benchmark failed",
                    )
                    run_results.append(partial)
                self._finalize(
                    pending,
                    tuple(run_results),
                    DrafterBenchmarkStatus.FAILED,
                    error_message="Drafter benchmark failed",
                )
            logger.exception(
                "Unexpected Drafter failure id=%s error=%s",
                pending.benchmark_id,
                type(error).__name__,
            )
            raise
        finally:
            if self._is_active(pending.benchmark_id):
                self._finalize(
                    pending,
                    tuple(run_results),
                    DrafterBenchmarkStatus.FAILED,
                    error_message="Drafter benchmark failed",
                )

    def request_cancellation(self, benchmark_id: str | None = None) -> bool:
        """Publish cancelling for the matching active reservation."""
        changed = False

        def transform(state: ApplicationState) -> ApplicationState:
            nonlocal changed
            lane = state.drafter_benchmark
            if not lane.is_active or (
                benchmark_id is not None and lane.active_benchmark_id != benchmark_id
            ):
                return state
            changed = True
            return replace(
                state,
                drafter_benchmark=replace(
                    lane, status=DrafterBenchmarkStatus.CANCELLING
                ),
            )

        self._publish(transform)
        return changed

    def cancel_reservation(
        self, pending: PendingDrafterBenchmark
    ) -> DrafterBenchmarkResult | None:
        """Finalize cancellation if a reserved worker never entered its coroutine."""
        if not self._is_active(pending.benchmark_id):
            return None
        self.request_cancellation(pending.benchmark_id)
        return self._finalize(
            pending, (), DrafterBenchmarkStatus.CANCELLED, error_message=None
        )

    def _is_active(self, benchmark_id: str) -> bool:
        lane = self.state.drafter_benchmark
        return lane.active_benchmark_id == benchmark_id and lane.is_active

    def _begin_request(
        self,
        pending: PendingDrafterBenchmark,
        *,
        warmup: bool,
        run_number: int,
        total: int,
        completed_measured: int,
        last_error: str | None,
    ) -> None:
        status = (
            DrafterBenchmarkStatus.WARMING_UP
            if warmup
            else DrafterBenchmarkStatus.RUNNING
        )
        phase: DrafterPhase = "warmup" if warmup else "measured"

        def transform(state: ApplicationState) -> ApplicationState:
            lane = state.drafter_benchmark
            if lane.active_benchmark_id != pending.benchmark_id or not lane.is_active:
                return state
            return replace(
                state,
                drafter_benchmark=replace(
                    lane,
                    status=status,
                    progress=DrafterBenchmarkProgress(
                        current_phase=phase,
                        current_run=run_number,
                        phase_total=total,
                        completed_measured_runs=completed_measured,
                        configured_measured_runs=pending.config.measured_runs,
                        latest_metrics=None,
                        last_error=last_error,
                        warnings=(),
                    ),
                ),
            )

        self._publish(transform)
        logger.info(
            "Drafter request starting id=%s phase=%s index=%d total=%d",
            pending.benchmark_id,
            phase,
            run_number,
            total,
        )

    def _handle_progress(
        self,
        pending: PendingDrafterBenchmark,
        progress: GenerationProgress,
        *,
        completed_measured: int,
    ) -> None:
        def transform(state: ApplicationState) -> ApplicationState:
            lane = state.drafter_benchmark
            if (
                lane.active_benchmark_id != pending.benchmark_id
                or not lane.is_active
                or lane.progress is None
            ):
                return state
            return replace(
                state,
                drafter_benchmark=replace(
                    lane,
                    progress=replace(
                        lane.progress,
                        latest_metrics=progress.metrics,
                        completed_measured_runs=completed_measured,
                    ),
                ),
            )

        self._publish(transform)

    def _publish_last_error(
        self, pending: PendingDrafterBenchmark, last_error: str | None
    ) -> None:
        def transform(state: ApplicationState) -> ApplicationState:
            lane = state.drafter_benchmark
            if (
                lane.active_benchmark_id != pending.benchmark_id
                or not lane.is_active
                or lane.progress is None
            ):
                return state
            return replace(
                state,
                drafter_benchmark=replace(
                    lane,
                    progress=replace(lane.progress, last_error=last_error),
                ),
            )

        self._publish(transform)

    @staticmethod
    def _run_result(
        run_number: int,
        warmup: bool,
        outcome: GenerationOutcome,
        *,
        success: bool,
        cancelled: bool,
        error_message: str | None,
    ) -> DrafterRunResult:
        return DrafterBenchmarkService._partial_run_result(
            run_number,
            warmup,
            outcome.metrics,
            success=success,
            cancelled=cancelled,
            error_message=error_message,
        )

    @staticmethod
    def _partial_run_result(
        run_number: int,
        warmup: bool,
        metrics: GenerationMetrics | None,
        *,
        success: bool = False,
        cancelled: bool,
        error_message: str | None,
    ) -> DrafterRunResult:
        return DrafterRunResult(
            run_number=run_number,
            warmup=warmup,
            success=success,
            cancelled=cancelled,
            error=error_message,
            prompt_tokens=metrics.prompt_tokens if metrics is not None else None,
            completion_tokens=(
                metrics.completion_tokens if metrics is not None else None
            ),
            total_tokens=metrics.total_tokens if metrics is not None else None,
            prompt_tokens_estimated=(
                metrics.prompt_tokens_estimated if metrics is not None else False
            ),
            completion_tokens_estimated=(
                metrics.completion_tokens_estimated if metrics is not None else False
            ),
            total_tokens_estimated=(
                metrics.total_tokens_estimated if metrics is not None else False
            ),
            draft_tokens=metrics.draft_tokens if metrics is not None else None,
            accepted_tokens=metrics.accepted_tokens if metrics is not None else None,
            acceptance_rate=metrics.acceptance_rate if metrics is not None else None,
            ttft_ms=metrics.ttft_ms if metrics is not None else None,
            generation_duration_s=(
                metrics.active_generation_duration_s if metrics is not None else None
            ),
            total_duration_s=(
                metrics.total_duration_s if metrics is not None else None
            ),
            output_tokens_per_second=(
                metrics.output_tokens_per_second if metrics is not None else None
            ),
            finish_reason=metrics.finish_reason if metrics is not None else None,
            streamed=metrics.streamed if metrics is not None else True,
        )

    def _finalize(
        self,
        pending: PendingDrafterBenchmark,
        run_results: tuple[DrafterRunResult, ...],
        status: DrafterBenchmarkStatus,
        error_message: str | None,
    ) -> DrafterBenchmarkResult:
        completed_at = self._utc_now()
        terminal: DrafterBenchmarkResult | None = None

        def transform(state: ApplicationState) -> ApplicationState:
            nonlocal terminal
            lane = state.drafter_benchmark
            if (
                lane.latest_result is not None
                and lane.latest_result.benchmark_id == pending.benchmark_id
                and lane.active_benchmark_id is None
            ):
                terminal = lane.latest_result
                return state
            if lane.active_benchmark_id != pending.benchmark_id:
                if (
                    lane.latest_result is not None
                    and lane.latest_result.benchmark_id == pending.benchmark_id
                ):
                    terminal = lane.latest_result
                    return state
                raise DrafterBenchmarkOperationError(
                    "This Drafter benchmark is no longer active."
                )
            aggregates = build_drafter_aggregates(run_results)
            observations = build_drafter_observations(
                run_results, aggregates.acceptance_rate.mean
            )
            terminal = DrafterBenchmarkResult(
                benchmark_id=pending.benchmark_id,
                status=status,
                started_at=pending.started_at,
                completed_at=completed_at,
                server_id=pending.server_id,
                server_name=pending.server_name,
                server_endpoint=pending.server_endpoint,
                model_id=pending.model_id,
                backend=pending.backend,
                config=pending.config,
                run_results=run_results,
                ttft_ms=aggregates.ttft_ms,
                output_tokens_per_second=aggregates.output_tokens_per_second,
                total_duration_s=aggregates.total_duration_s,
                generation_duration_s=aggregates.generation_duration_s,
                prompt_tokens=aggregates.prompt_tokens,
                completion_tokens=aggregates.completion_tokens,
                draft_tokens=aggregates.draft_tokens,
                accepted_tokens=aggregates.accepted_tokens,
                acceptance_rate=aggregates.acceptance_rate,
                observations=observations,
                hardware_before=pending.hardware_before,
                hardware_after=state.hardware_snapshot,
                error=error_message,
            )
            return replace(
                state,
                drafter_benchmark=replace(
                    lane,
                    config=pending.config,
                    status=status,
                    active_benchmark_id=None,
                    progress=None,
                    started_at=None,
                    latest_result=terminal,
                    benchmark_error=error_message,
                ),
            )

        state = self._state_store.update(transform)
        try:
            self._on_state_change(state)
        except Exception as error:
            logger.error(
                "Drafter terminal callback failed id=%s error=%s",
                pending.benchmark_id,
                type(error).__name__,
            )
        if terminal is None:
            raise RuntimeError("Drafter finalization did not produce a result")
        logger.info(
            "Drafter finalized id=%s status=%s successful=%d failed=%d cancelled=%d",
            terminal.benchmark_id,
            terminal.status,
            terminal.successful_runs,
            terminal.failed_runs,
            terminal.cancelled_runs,
        )
        return terminal
