"""Strictly sequential single-request Speed Test orchestration."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from uuid import uuid4

from modeltop.benchmarks.models import (
    SpeedTestConfig,
    SpeedTestResult,
    SpeedTestRunResult,
    SpeedTestStatus,
)
from modeltop.benchmarks.statistics import build_speed_test_aggregates
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
type RunIdFactory = Callable[[datetime], str]


class SpeedTestOperationError(Exception):
    """Readable synchronous rejection before benchmark network I/O."""

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


@dataclass(frozen=True, slots=True)
class PendingSpeedTest:
    """Immutable metadata captured by the atomic benchmark reservation."""

    run_id: str
    config: SpeedTestConfig
    server_id: str
    server_name: str
    server_endpoint: str
    model_id: str
    backend: str
    started_at: datetime
    hardware_before: HardwareSnapshot | None


def _default_run_id_factory(started_at: datetime) -> str:
    timestamp = started_at.strftime("%Y%m%dT%H%M%S%fZ")
    return f"speed-test-{timestamp}-{uuid4().hex[:8]}"


class SpeedTestService:
    """Reserve, execute, and retain one sequential Speed Test at a time."""

    def __init__(
        self,
        generation_service: GenerationService,
        state_store: ApplicationStateStore,
        server: ServerConfig,
        on_state_change: StateCallback,
        *,
        utc_now: UtcNow = lambda: datetime.now(UTC),
        run_id_factory: RunIdFactory = _default_run_id_factory,
    ) -> None:
        self._generation_service = generation_service
        self._state_store = state_store
        self._server = server
        self._on_state_change = on_state_change
        self._utc_now = utc_now
        self._run_id_factory = run_id_factory

    @property
    def state(self) -> ApplicationState:
        return self._state_store.state

    def _publish(
        self, transform: Callable[[ApplicationState], ApplicationState]
    ) -> ApplicationState:
        state = self._state_store.update(transform)
        self._on_state_change(state)
        return state

    def begin_test(self, config: SpeedTestConfig) -> PendingSpeedTest:
        """Atomically validate and reserve a benchmark before network I/O."""

        pending: PendingSpeedTest | None = None

        def reserve(state: ApplicationState) -> ApplicationState:
            nonlocal pending
            if state.tool_calling_benchmark.is_active:
                raise SpeedTestOperationError(
                    "A Tool Calling benchmark is already running."
                )
            if state.r0b0bench_benchmark.is_active:
                raise SpeedTestOperationError(
                    "An r0b0bench benchmark is already running."
                )
            if state.context_benchmark.is_active:
                raise SpeedTestOperationError("A Context benchmark is already running.")
            if state.concurrency_benchmark.is_active:
                raise SpeedTestOperationError(
                    "A Concurrency benchmark is already running."
                )
            if state.drafter_benchmark.is_active:
                raise SpeedTestOperationError("A Drafter benchmark is already running.")
            if state.speed_test.is_active:
                raise SpeedTestOperationError("A Speed Test is already running.")
            if state.active_generation_id is not None:
                raise SpeedTestOperationError("A Chat generation is already running.")
            if state.server_status is not ServerStatus.ONLINE:
                raise SpeedTestOperationError("The selected server is offline.")
            server_id = state.selected_server_id
            model_id = state.selected_model_id
            if server_id is None or model_id is None:
                raise SpeedTestOperationError(
                    "Select an available model before starting Speed Test."
                )
            selected_model = next(
                (model for model in state.available_models if model.id == model_id),
                DiscoveredModel(id=model_id),
            )
            backend = self._server.backend_label
            if backend == "--":
                backend = format_backend_label(selected_model.owned_by)
            started_at = self._utc_now()
            run_id = self._run_id_factory(started_at)
            pending = PendingSpeedTest(
                run_id=run_id,
                config=config,
                server_id=server_id,
                server_name=self._server.name,
                server_endpoint=self._server.endpoint_label,
                model_id=model_id,
                backend=backend,
                started_at=started_at,
                hardware_before=state.hardware_snapshot,
            )
            return replace(
                state,
                speed_test=replace(
                    state.speed_test,
                    config=config,
                    status=SpeedTestStatus.PREPARING,
                    run_id=run_id,
                    current_phase=None,
                    current_run=0,
                    phase_total=0,
                    latest_metrics=None,
                    live_output_preview="",
                    run_results=(),
                    last_error=None,
                ),
            )

        try:
            self._publish(reserve)
        except SpeedTestOperationError:
            raise
        except Exception:
            if pending is not None:
                self._finalize(pending, SpeedTestStatus.FAILED, "Speed Test failed")
            logger.exception("Speed Test reservation callback failed")
            raise
        if pending is None:
            raise RuntimeError("Speed Test reservation did not produce a request")
        logger.info(
            "Speed Test reserved run=%s server=%s model=%s preset=%s",
            pending.run_id,
            pending.server_id,
            pending.model_id,
            pending.config.preset,
        )
        return pending

    async def run_test(self, pending: PendingSpeedTest) -> SpeedTestResult:
        """Run warm-ups then measured requests with at most one request in flight."""
        state = self.state.speed_test
        if state.run_id != pending.run_id or not state.is_active:
            raise SpeedTestOperationError("This Speed Test is no longer active.")

        had_errors = False
        current_phase: str | None = None
        current_run = 0
        current_metrics: GenerationMetrics | None = None
        current_character_count = 0
        current_recorded = False

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
                    current_character_count = 0
                    current_recorded = False
                    self._begin_request(pending.run_id, warmup, run_number, total)
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
                            enable_thinking=(
                                False
                                if pending.config.thinking_mode == "disabled"
                                else None
                            ),
                        ),
                        request_timeout_seconds=(
                            pending.config.request_timeout_seconds
                        ),
                    )

                    def on_progress(progress: GenerationProgress) -> None:
                        nonlocal current_metrics, current_character_count
                        current_metrics = progress.metrics
                        current_character_count = len(progress.content)
                        self._handle_progress(pending.run_id, progress)

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
                        self._append_run(pending.run_id, run_result)
                        current_recorded = True
                        self._finalize(
                            pending, SpeedTestStatus.CANCELLED, error_message=None
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
                        self._append_run(pending.run_id, run_result)
                        current_recorded = True
                        had_errors = True
                        logger.warning(
                            "Speed Test request failed run=%s phase=%s index=%d "
                            "error=%s",
                            pending.run_id,
                            current_phase,
                            run_number,
                            type(error.error).__name__,
                        )
                        if not pending.config.continue_on_error:
                            return self._finalize(
                                pending,
                                SpeedTestStatus.FAILED,
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
                        self._append_run(pending.run_id, run_result)
                        current_recorded = True

            status = (
                SpeedTestStatus.COMPLETED_WITH_ERRORS
                if had_errors
                else SpeedTestStatus.COMPLETED
            )
            return self._finalize(pending, status, error_message=None)
        except asyncio.CancelledError:
            if self._is_active(pending.run_id):
                if current_phase is not None and not current_recorded:
                    partial = self._partial_run_result(
                        current_run,
                        current_phase == "warmup",
                        current_metrics,
                        current_character_count,
                        cancelled=True,
                        error_message=None,
                    )
                    self._append_run(pending.run_id, partial)
                self._finalize(pending, SpeedTestStatus.CANCELLED, error_message=None)
            raise
        except Exception as error:
            if self._is_active(pending.run_id):
                if current_phase is not None and not current_recorded:
                    partial = self._partial_run_result(
                        current_run,
                        current_phase == "warmup",
                        current_metrics,
                        current_character_count,
                        cancelled=False,
                        error_message="Speed Test failed",
                    )
                    self._append_run(pending.run_id, partial)
                self._finalize(
                    pending, SpeedTestStatus.FAILED, error_message="Speed Test failed"
                )
            logger.exception(
                "Unexpected Speed Test failure run=%s error=%s",
                pending.run_id,
                type(error).__name__,
            )
            raise
        finally:
            if self._is_active(pending.run_id):
                self._finalize(
                    pending, SpeedTestStatus.FAILED, error_message="Speed Test failed"
                )

    def request_cancellation(self, run_id: str | None = None) -> bool:
        """Publish cancelling for the matching active reservation."""
        changed = False

        def transform(state: ApplicationState) -> ApplicationState:
            nonlocal changed
            speed = state.speed_test
            if not speed.is_active or (run_id is not None and speed.run_id != run_id):
                return state
            changed = True
            return replace(
                state,
                speed_test=replace(speed, status=SpeedTestStatus.CANCELLING),
            )

        self._publish(transform)
        return changed

    def cancel_reservation(self, pending: PendingSpeedTest) -> SpeedTestResult | None:
        """Finalize cancellation if a reserved worker never entered its coroutine."""
        if not self._is_active(pending.run_id):
            return None
        self.request_cancellation(pending.run_id)
        return self._finalize(pending, SpeedTestStatus.CANCELLED, error_message=None)

    def _is_active(self, run_id: str) -> bool:
        speed = self.state.speed_test
        return speed.run_id == run_id and speed.is_active

    def _begin_request(
        self, run_id: str, warmup: bool, run_number: int, total: int
    ) -> None:
        status = SpeedTestStatus.WARMING_UP if warmup else SpeedTestStatus.RUNNING

        def transform(state: ApplicationState) -> ApplicationState:
            speed = state.speed_test
            if speed.run_id != run_id or not speed.is_active:
                return state
            return replace(
                state,
                speed_test=replace(
                    speed,
                    status=status,
                    current_phase="warmup" if warmup else "measured",
                    current_run=run_number,
                    phase_total=total,
                    latest_metrics=None,
                    live_output_preview="",
                ),
            )

        self._publish(transform)
        logger.info(
            "Speed Test request starting run=%s phase=%s index=%d total=%d",
            run_id,
            "warmup" if warmup else "measured",
            run_number,
            total,
        )

    def _handle_progress(self, run_id: str, progress: GenerationProgress) -> None:
        def transform(state: ApplicationState) -> ApplicationState:
            speed = state.speed_test
            if speed.run_id != run_id or not speed.is_active:
                return state
            return replace(
                state,
                speed_test=replace(
                    speed,
                    latest_metrics=progress.metrics,
                    live_output_preview=progress.content[-2000:],
                ),
            )

        self._publish(transform)

    def _append_run(self, run_id: str, result: SpeedTestRunResult) -> None:
        def transform(state: ApplicationState) -> ApplicationState:
            speed = state.speed_test
            if speed.run_id != run_id or not speed.is_active:
                return state
            return replace(
                state,
                speed_test=replace(
                    speed,
                    run_results=(*speed.run_results, result),
                    live_output_preview="",
                    last_error=result.error,
                ),
            )

        self._publish(transform)
        logger.info(
            "Speed Test request finished run=%s phase=%s index=%d success=%s "
            "cancelled=%s prompt_tokens=%s completion_tokens=%s duration=%s "
            "finish=%s",
            run_id,
            "warmup" if result.warmup else "measured",
            result.run_number,
            result.success,
            result.cancelled,
            result.prompt_tokens,
            result.completion_tokens,
            result.total_duration_s,
            result.finish_reason,
        )

    @staticmethod
    def _run_result(
        run_number: int,
        warmup: bool,
        outcome: GenerationOutcome,
        *,
        success: bool,
        cancelled: bool,
        error_message: str | None,
    ) -> SpeedTestRunResult:
        return SpeedTestService._partial_run_result(
            run_number,
            warmup,
            outcome.metrics,
            len(outcome.content),
            success=success,
            cancelled=cancelled,
            error_message=error_message,
        )

    @staticmethod
    def _partial_run_result(
        run_number: int,
        warmup: bool,
        metrics: GenerationMetrics | None,
        character_count: int,
        *,
        success: bool = False,
        cancelled: bool,
        error_message: str | None,
    ) -> SpeedTestRunResult:
        return SpeedTestRunResult(
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
            response_character_count=character_count,
        )

    def _finalize(
        self,
        pending: PendingSpeedTest,
        status: SpeedTestStatus,
        error_message: str | None,
    ) -> SpeedTestResult:
        completed_at = self._utc_now()
        terminal: SpeedTestResult | None = None

        def transform(state: ApplicationState) -> ApplicationState:
            nonlocal terminal
            speed = state.speed_test
            existing = speed.result_by_id(pending.run_id)
            if existing is not None:
                terminal = existing
                return state
            if speed.run_id != pending.run_id:
                raise SpeedTestOperationError("This Speed Test is no longer active.")
            aggregates = build_speed_test_aggregates(speed.run_results)
            terminal = SpeedTestResult(
                run_id=pending.run_id,
                status=status,
                started_at=pending.started_at,
                completed_at=completed_at,
                server_id=pending.server_id,
                server_name=pending.server_name,
                server_endpoint=pending.server_endpoint,
                model_id=pending.model_id,
                backend=pending.backend,
                config=pending.config,
                run_results=speed.run_results,
                ttft_ms=aggregates.ttft_ms,
                output_tokens_per_second=aggregates.output_tokens_per_second,
                total_duration_s=aggregates.total_duration_s,
                generation_duration_s=aggregates.generation_duration_s,
                prompt_tokens=aggregates.prompt_tokens,
                completion_tokens=aggregates.completion_tokens,
                hardware_before=pending.hardware_before,
                hardware_after=state.hardware_snapshot,
                error=error_message,
            )
            return replace(
                state,
                speed_test=replace(
                    speed,
                    status=status,
                    run_id=None,
                    current_phase=None,
                    current_run=0,
                    phase_total=0,
                    latest_metrics=None,
                    live_output_preview="",
                    last_error=error_message,
                    results=(*speed.results, terminal),
                ),
            )

        state = self._state_store.update(transform)
        try:
            self._on_state_change(state)
        except Exception as error:
            logger.error(
                "Speed Test terminal callback failed run=%s error=%s",
                pending.run_id,
                type(error).__name__,
            )
        if terminal is None:
            raise RuntimeError("Speed Test finalization did not produce a result")
        logger.info(
            "Speed Test finalized run=%s status=%s successful=%d failed=%d "
            "cancelled=%d",
            terminal.run_id,
            terminal.status,
            terminal.successful_runs,
            terminal.failed_runs,
            terminal.cancelled_runs,
        )
        return terminal
