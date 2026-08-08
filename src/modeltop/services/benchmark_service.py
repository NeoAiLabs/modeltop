"""Service-owned reservation and lifecycle for Concurrency benchmarks."""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from uuid import uuid4

from modeltop.benchmarks.base import BenchmarkContext
from modeltop.benchmarks.concurrency import ConcurrencyBenchmark
from modeltop.benchmarks.models import (
    ConcurrencyBenchmarkConfig,
    ConcurrencyBenchmarkProgress,
    ConcurrencyBenchmarkResult,
    ConcurrencyBenchmarkStatus,
)
from modeltop.models import DiscoveredModel, ServerConfig, format_backend_label
from modeltop.services.generation import GenerationService
from modeltop.state import ApplicationState, ApplicationStateStore, ServerStatus

logger = logging.getLogger(__name__)

type StateCallback = Callable[[ApplicationState], None]
type UtcNow = Callable[[], datetime]
type MonotonicClock = Callable[[], float]
type BenchmarkIdFactory = Callable[[datetime], str]


class BenchmarkOperationError(Exception):
    """Readable synchronous rejection before benchmark network I/O."""

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


@dataclass(frozen=True, slots=True)
class PendingConcurrencyBenchmark:
    """Immutable metadata captured by the atomic reservation."""

    benchmark_id: str
    config: ConcurrencyBenchmarkConfig
    server_id: str
    server_name: str
    server_endpoint: str
    model_id: str
    backend: str
    started_at: datetime


def _default_benchmark_id_factory(started_at: datetime) -> str:
    timestamp = started_at.strftime("%Y%m%dT%H%M%S%fZ")
    return f"concurrency-{timestamp}-{uuid4().hex[:8]}"


class BenchmarkService:
    """Atomically reserve, execute, cancel, and retain one benchmark."""

    def __init__(
        self,
        generation_service: GenerationService,
        state_store: ApplicationStateStore,
        server: ServerConfig,
        hardware_refresh_interval_seconds: float,
        on_state_change: StateCallback,
        *,
        utc_now: UtcNow = lambda: datetime.now(UTC),
        monotonic_clock: MonotonicClock = time.perf_counter,
        benchmark_id_factory: BenchmarkIdFactory = _default_benchmark_id_factory,
    ) -> None:
        self._generation_service = generation_service
        self._state_store = state_store
        self._server = server
        self._hardware_refresh_interval_seconds = hardware_refresh_interval_seconds
        self._on_state_change = on_state_change
        self._utc_now = utc_now
        self._monotonic_clock = monotonic_clock
        self._benchmark_id_factory = benchmark_id_factory
        self._benchmark: ConcurrencyBenchmark | None = None
        self._active_id: str | None = None
        self._last_progress_publish_at: float | None = None
        self._last_progress_key: tuple[object, ...] | None = None
        self._buffered_progress: ConcurrencyBenchmarkProgress | None = None
        self._buffered_status: ConcurrencyBenchmarkStatus | None = None

    @property
    def state(self) -> ApplicationState:
        """Return the latest coherent application state."""
        return self._state_store.state

    def _publish(
        self, transform: Callable[[ApplicationState], ApplicationState]
    ) -> ApplicationState:
        state = self._state_store.update(transform)
        self._on_state_change(state)
        return state

    def begin_benchmark(
        self, config: ConcurrencyBenchmarkConfig
    ) -> PendingConcurrencyBenchmark:
        """Atomically validate and reserve benchmark traffic before network I/O."""
        pending: PendingConcurrencyBenchmark | None = None

        def reserve(state: ApplicationState) -> ApplicationState:
            nonlocal pending
            if state.tool_calling_benchmark.is_active:
                raise BenchmarkOperationError(
                    "A Tool Calling benchmark is already running."
                )
            if state.r0b0bench_benchmark.is_active:
                raise BenchmarkOperationError(
                    "An r0b0bench benchmark is already running."
                )
            if state.context_benchmark.is_active:
                raise BenchmarkOperationError("A Context benchmark is already running.")
            if state.concurrency_benchmark.is_active:
                raise BenchmarkOperationError(
                    "A Concurrency benchmark is already running."
                )
            if state.drafter_benchmark.is_active:
                raise BenchmarkOperationError("A Drafter benchmark is already running.")
            if state.speed_test.is_active:
                raise BenchmarkOperationError("A Speed Test is already running.")
            if state.active_generation_id is not None:
                raise BenchmarkOperationError("A Chat generation is already running.")
            if state.is_refreshing:
                raise BenchmarkOperationError(
                    "Refresh in progress; retry the benchmark when it completes"
                )
            if state.server_status is not ServerStatus.ONLINE:
                raise BenchmarkOperationError("The selected server is offline.")
            server_id = state.selected_server_id
            model_id = state.selected_model_id
            if server_id is None or model_id is None:
                raise BenchmarkOperationError(
                    "Select an available model before starting Concurrency."
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
            pending = PendingConcurrencyBenchmark(
                benchmark_id=benchmark_id,
                config=config,
                server_id=server_id,
                server_name=self._server.name,
                server_endpoint=self._server.endpoint_label,
                model_id=model_id,
                backend=backend,
                started_at=started_at,
            )
            lane = state.concurrency_benchmark
            return replace(
                state,
                concurrency_benchmark=replace(
                    lane,
                    config=config,
                    status=ConcurrencyBenchmarkStatus.VALIDATING,
                    active_benchmark_id=benchmark_id,
                    progress=None,
                    benchmark_started_at=started_at,
                    benchmark_error=None,
                ),
            )

        self._publish(reserve)
        if pending is None:
            raise RuntimeError("Benchmark reservation did not produce metadata")
        self._active_id = pending.benchmark_id
        self._last_progress_publish_at = None
        self._last_progress_key = None
        logger.info(
            "Concurrency benchmark reserved benchmark=%s server=%s model=%s",
            pending.benchmark_id,
            pending.server_id,
            pending.model_id,
        )
        return pending

    def _accept_progress(
        self,
        pending: PendingConcurrencyBenchmark,
        status: ConcurrencyBenchmarkStatus,
        progress: ConcurrencyBenchmarkProgress,
    ) -> None:
        if self._active_id != pending.benchmark_id:
            return
        now = self._monotonic_clock()
        key = (
            status,
            progress.phase,
            progress.active_concurrency_level,
            progress.next_concurrency_level,
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
        self, pending: PendingConcurrencyBenchmark, *, now: float | None = None
    ) -> None:
        status = self._buffered_status
        progress = self._buffered_progress
        if status is None or progress is None:
            return
        if self._active_id != pending.benchmark_id:
            return

        def update(state: ApplicationState) -> ApplicationState:
            lane = state.concurrency_benchmark
            if lane.active_benchmark_id != pending.benchmark_id:
                return state
            return replace(
                state,
                concurrency_benchmark=replace(
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
            progress.phase,
            progress.active_concurrency_level,
            progress.next_concurrency_level,
        )

    def _context(self, pending: PendingConcurrencyBenchmark) -> BenchmarkContext:
        return BenchmarkContext(
            benchmark_id=pending.benchmark_id,
            started_at=pending.started_at,
            monotonic_clock=self._monotonic_clock,
            utc_now=self._utc_now,
            read_hardware_snapshot=lambda: self._state_store.state.hardware_snapshot,
        )

    async def run_benchmark(
        self, pending: PendingConcurrencyBenchmark
    ) -> ConcurrencyBenchmarkResult:
        """Own benchmark tasks and publish one terminal immutable result."""
        lane = self.state.concurrency_benchmark
        if (
            lane.active_benchmark_id != pending.benchmark_id
            or lane.status is not ConcurrencyBenchmarkStatus.VALIDATING
        ):
            raise BenchmarkOperationError("This benchmark is no longer active.")
        context = self._context(pending)
        benchmark = ConcurrencyBenchmark(
            self._generation_service,
            pending.config,
            server_id=pending.server_id,
            server_name=pending.server_name,
            server_endpoint=pending.server_endpoint,
            model_id=pending.model_id,
            backend=pending.backend,
            hardware_refresh_interval_seconds=self._hardware_refresh_interval_seconds,
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
                logger.info(
                    "Concurrency benchmark cancelled benchmark=%s",
                    pending.benchmark_id,
                )
                raise
            except Exception:
                logger.exception(
                    "Concurrency benchmark failed benchmark=%s",
                    pending.benchmark_id,
                )
                result = ConcurrencyBenchmarkResult(
                    benchmark_id=pending.benchmark_id,
                    status=ConcurrencyBenchmarkStatus.ERROR,
                    server_id=pending.server_id,
                    server_name=pending.server_name,
                    server_endpoint=pending.server_endpoint,
                    model_id=pending.model_id,
                    backend=pending.backend,
                    started_at=pending.started_at,
                    completed_at=self._utc_now(),
                    config=pending.config,
                    levels=(),
                    cancelled=False,
                    error="Concurrency benchmark failed",
                    warnings=(),
                    observations=(),
                )
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
        """Publish cancellation first, then stop dequeues and cancellible waits."""
        active_id = self._active_id
        if active_id is None or (
            benchmark_id is not None and benchmark_id != active_id
        ):
            return

        def mark(state: ApplicationState) -> ApplicationState:
            lane = state.concurrency_benchmark
            if lane.active_benchmark_id != active_id or not lane.is_active:
                return state
            return replace(
                state,
                concurrency_benchmark=replace(
                    lane, status=ConcurrencyBenchmarkStatus.CANCELLING
                ),
            )

        self._publish(mark)
        if self._benchmark is not None:
            self._benchmark.request_cancellation()

    def cancel_reservation(self, pending: PendingConcurrencyBenchmark) -> None:
        """Finalize a reservation whose worker never entered its coroutine."""
        lane = self.state.concurrency_benchmark
        if lane.active_benchmark_id != pending.benchmark_id:
            return
        result = ConcurrencyBenchmarkResult(
            benchmark_id=pending.benchmark_id,
            status=ConcurrencyBenchmarkStatus.CANCELLED,
            server_id=pending.server_id,
            server_name=pending.server_name,
            server_endpoint=pending.server_endpoint,
            model_id=pending.model_id,
            backend=pending.backend,
            started_at=pending.started_at,
            completed_at=self._utc_now(),
            config=pending.config,
            levels=(),
            cancelled=True,
            error="Benchmark cancelled — partial results retained",
            warnings=(),
            observations=(),
        )
        self._finalize(pending, result)
        if self._active_id == pending.benchmark_id:
            self._active_id = None

    def _finalize(
        self,
        pending: PendingConcurrencyBenchmark,
        result: ConcurrencyBenchmarkResult,
    ) -> None:
        def finish(state: ApplicationState) -> ApplicationState:
            lane = state.concurrency_benchmark
            if lane.active_benchmark_id != pending.benchmark_id:
                return state
            return replace(
                state,
                concurrency_benchmark=replace(
                    lane,
                    config=pending.config,
                    status=result.status,
                    active_benchmark_id=None,
                    benchmark_started_at=None,
                    latest_result=result,
                    benchmark_error=result.error,
                ),
            )

        self._publish(finish)
