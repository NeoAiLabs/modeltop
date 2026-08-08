"""Service-owned reservation and lifecycle for Context Length benchmarks."""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from uuid import uuid4

from modeltop.benchmarks.base import BenchmarkContext
from modeltop.benchmarks.context import ContextBenchmark
from modeltop.benchmarks.models import (
    ContextBenchmarkConfig,
    ContextBenchmarkProgress,
    ContextBenchmarkResult,
    ContextBenchmarkStatus,
)
from modeltop.models import DiscoveredModel, ServerConfig, format_backend_label
from modeltop.services.generation import GenerationService
from modeltop.state import ApplicationState, ApplicationStateStore, ServerStatus

logger = logging.getLogger(__name__)

type StateCallback = Callable[[ApplicationState], None]
type UtcNow = Callable[[], datetime]
type MonotonicClock = Callable[[], float]
type BenchmarkIdFactory = Callable[[datetime], str]


class ContextBenchmarkOperationError(Exception):
    """Readable synchronous rejection before Context network I/O."""

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


@dataclass(frozen=True, slots=True)
class PendingContextBenchmark:
    """Immutable metadata captured by the atomic Context reservation."""

    benchmark_id: str
    config: ContextBenchmarkConfig
    server_id: str
    server_name: str
    server_endpoint: str
    model_id: str
    backend: str
    started_at: datetime


def _default_benchmark_id_factory(started_at: datetime) -> str:
    timestamp = started_at.strftime("%Y%m%dT%H%M%S%fZ")
    return f"context-{timestamp}-{uuid4().hex[:8]}"


class ContextBenchmarkService:
    """Atomically reserve, execute, cancel, and retain one Context result."""

    def __init__(
        self,
        generation_service: GenerationService,
        state_store: ApplicationStateStore,
        server: ServerConfig,
        on_state_change: StateCallback,
        *,
        utc_now: UtcNow = lambda: datetime.now(UTC),
        monotonic_clock: MonotonicClock = time.perf_counter,
        benchmark_id_factory: BenchmarkIdFactory = _default_benchmark_id_factory,
    ) -> None:
        self._generation_service = generation_service
        self._state_store = state_store
        self._server = server
        self._on_state_change = on_state_change
        self._utc_now = utc_now
        self._monotonic_clock = monotonic_clock
        self._benchmark_id_factory = benchmark_id_factory
        self._benchmark: ContextBenchmark | None = None
        self._active_id: str | None = None
        self._last_progress_publish_at: float | None = None
        self._last_progress_key: tuple[object, ...] | None = None
        self._buffered_progress: ContextBenchmarkProgress | None = None
        self._buffered_status: ContextBenchmarkStatus | None = None
        self._finalized_ids: set[str] = set()

    @property
    def state(self) -> ApplicationState:
        return self._state_store.state

    def _publish(
        self, transform: Callable[[ApplicationState], ApplicationState]
    ) -> ApplicationState:
        state = self._state_store.update(transform)
        try:
            self._on_state_change(state)
        except Exception:
            logger.exception("Context state callback failed")
        return state

    def begin_benchmark(
        self, config: ContextBenchmarkConfig
    ) -> PendingContextBenchmark:
        """Atomically reserve Context traffic before prompt construction or I/O."""
        pending: PendingContextBenchmark | None = None

        def reserve(state: ApplicationState) -> ApplicationState:
            nonlocal pending
            if state.tool_calling_benchmark.is_active:
                raise ContextBenchmarkOperationError(
                    "A Tool Calling benchmark is already running."
                )
            if state.r0b0bench_benchmark.is_active:
                raise ContextBenchmarkOperationError(
                    "An r0b0bench benchmark is already running."
                )
            if state.context_benchmark.is_active:
                raise ContextBenchmarkOperationError(
                    "A Context benchmark is already running."
                )
            if state.concurrency_benchmark.is_active:
                raise ContextBenchmarkOperationError(
                    "A Concurrency benchmark is already running."
                )
            if state.drafter_benchmark.is_active:
                raise ContextBenchmarkOperationError(
                    "A Drafter benchmark is already running."
                )
            if state.speed_test.is_active:
                raise ContextBenchmarkOperationError("A Speed Test is already running.")
            if state.active_generation_id is not None:
                raise ContextBenchmarkOperationError(
                    "A Chat generation is already running."
                )
            if state.is_refreshing:
                raise ContextBenchmarkOperationError(
                    "Refresh in progress; retry the benchmark when it completes"
                )
            if state.server_status is not ServerStatus.ONLINE:
                raise ContextBenchmarkOperationError("The selected server is offline.")
            server_id = state.selected_server_id
            model_id = state.selected_model_id
            if server_id is None or model_id is None:
                raise ContextBenchmarkOperationError(
                    "Select an available model before starting Context Length."
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
            pending = PendingContextBenchmark(
                benchmark_id,
                config,
                server_id,
                self._server.name,
                self._server.endpoint_label,
                model_id,
                backend,
                started_at,
            )
            lane = state.context_benchmark
            return replace(
                state,
                context_benchmark=replace(
                    lane,
                    config=config,
                    status=ContextBenchmarkStatus.VALIDATING,
                    active_benchmark_id=benchmark_id,
                    progress=None,
                    benchmark_started_at=started_at,
                    benchmark_error=None,
                ),
            )

        self._publish(reserve)
        if pending is None:
            raise RuntimeError("Context reservation did not produce metadata")
        self._active_id = pending.benchmark_id
        self._last_progress_publish_at = None
        self._last_progress_key = None
        logger.info(
            "Context benchmark reserved benchmark=%s server=%s model=%s",
            pending.benchmark_id,
            pending.server_id,
            pending.model_id,
        )
        return pending

    def _accept_progress(
        self,
        pending: PendingContextBenchmark,
        status: ContextBenchmarkStatus,
        progress: ContextBenchmarkProgress,
    ) -> None:
        if self._active_id != pending.benchmark_id:
            return
        now = self._monotonic_clock()
        key = (
            status,
            progress.active_target_length,
            progress.next_target_length,
            progress.probe_stage,
            progress.active_request.state if progress.active_request else None,
        )
        force = key != self._last_progress_key
        self._buffered_status = status
        self._buffered_progress = progress
        if (
            not force
            and self._last_progress_publish_at is not None
            and now - self._last_progress_publish_at < 0.1
        ):
            return
        self._flush_progress(pending, now=now)

    def _flush_progress(
        self, pending: PendingContextBenchmark, *, now: float | None = None
    ) -> None:
        status = self._buffered_status
        progress = self._buffered_progress
        if (
            status is None
            or progress is None
            or self._active_id != pending.benchmark_id
        ):
            return

        def update(state: ApplicationState) -> ApplicationState:
            lane = state.context_benchmark
            if lane.active_benchmark_id != pending.benchmark_id:
                return state
            return replace(
                state,
                context_benchmark=replace(
                    lane,
                    status=status,
                    progress=progress,
                    benchmark_error=None,
                ),
            )

        self._publish(update)
        self._last_progress_publish_at = self._monotonic_clock() if now is None else now
        self._last_progress_key = (
            status,
            progress.active_target_length,
            progress.next_target_length,
            progress.probe_stage,
            progress.active_request.state if progress.active_request else None,
        )

    def _context(self, pending: PendingContextBenchmark) -> BenchmarkContext:
        return BenchmarkContext(
            benchmark_id=pending.benchmark_id,
            started_at=pending.started_at,
            monotonic_clock=self._monotonic_clock,
            utc_now=self._utc_now,
            read_hardware_snapshot=lambda: self._state_store.state.hardware_snapshot,
        )

    async def run_benchmark(
        self, pending: PendingContextBenchmark
    ) -> ContextBenchmarkResult:
        """Own all Context tasks and commit one terminal immutable result."""
        lane = self.state.context_benchmark
        if (
            lane.active_benchmark_id != pending.benchmark_id
            or lane.status is not ContextBenchmarkStatus.VALIDATING
        ):
            raise ContextBenchmarkOperationError(
                "This Context benchmark is no longer active."
            )
        context = self._context(pending)
        benchmark = ContextBenchmark(
            self._generation_service,
            pending.config,
            server_id=pending.server_id,
            server_name=pending.server_name,
            server_endpoint=pending.server_endpoint,
            model_id=pending.model_id,
            backend=pending.backend,
            on_progress=lambda status, progress: self._accept_progress(
                pending, status, progress
            ),
        )
        self._benchmark = benchmark
        try:
            try:
                result = await benchmark.run(context)
            except asyncio.CancelledError:
                result = benchmark.cancelled_result(context)
                self._finalize(pending, result)
                raise
            except Exception:
                logger.exception(
                    "Context benchmark failed benchmark=%s", pending.benchmark_id
                )
                result = benchmark.error_result(context, "Context benchmark failed")
            self._flush_progress(pending)
            self._finalize(pending, result)
            return result
        finally:
            if self._active_id == pending.benchmark_id:
                self._active_id = None
            if self._benchmark is benchmark:
                self._benchmark = None
            self._buffered_progress = None
            self._buffered_status = None

    def request_cancellation(self, benchmark_id: str | None = None) -> None:
        """Publish cancellation first, then stop prompt/request work."""
        active_id = self._active_id
        if active_id is None or (
            benchmark_id is not None and benchmark_id != active_id
        ):
            return

        def mark(state: ApplicationState) -> ApplicationState:
            lane = state.context_benchmark
            if lane.active_benchmark_id != active_id or not lane.is_active:
                return state
            return replace(
                state,
                context_benchmark=replace(
                    lane, status=ContextBenchmarkStatus.CANCELLING
                ),
            )

        self._publish(mark)
        if self._benchmark is not None:
            self._benchmark.request_cancellation()

    def cancel_reservation(self, pending: PendingContextBenchmark) -> None:
        """Finalize a reservation whose worker never entered its coroutine."""
        lane = self.state.context_benchmark
        if lane.active_benchmark_id != pending.benchmark_id:
            return
        context = self._context(pending)
        benchmark = ContextBenchmark(
            self._generation_service,
            pending.config,
            server_id=pending.server_id,
            server_name=pending.server_name,
            server_endpoint=pending.server_endpoint,
            model_id=pending.model_id,
            backend=pending.backend,
            on_progress=lambda _status, _progress: None,
        )
        self._finalize(pending, benchmark.cancelled_result(context))
        if self._active_id == pending.benchmark_id:
            self._active_id = None

    def _finalize(
        self,
        pending: PendingContextBenchmark,
        result: ContextBenchmarkResult,
    ) -> None:
        if pending.benchmark_id in self._finalized_ids:
            return
        self._finalized_ids.add(pending.benchmark_id)

        def finish(state: ApplicationState) -> ApplicationState:
            lane = state.context_benchmark
            if lane.active_benchmark_id != pending.benchmark_id:
                return state
            return replace(
                state,
                context_benchmark=replace(
                    lane,
                    config=pending.config,
                    status=result.status,
                    active_benchmark_id=None,
                    progress=None,
                    benchmark_started_at=None,
                    latest_result=result,
                    benchmark_error=result.error,
                ),
            )

        self._publish(finish)
