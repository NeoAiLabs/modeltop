"""Atomic reservation and lifecycle for native Tool Calling benchmarks."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from uuid import uuid4

from modeltop.benchmarks.base import BenchmarkContext
from modeltop.benchmarks.models import (
    ToolCallingBenchmarkConfig,
    ToolCallingBenchmarkProgress,
    ToolCallingBenchmarkResult,
    ToolCallingBenchmarkStatus,
)
from modeltop.benchmarks.tool_calling import (
    TOOL_EVAL_BENCH_COMMIT,
    ToolCallingBenchmark,
    UpstreamBenchmarkRunner,
    map_upstream_backend,
    run_upstream_benchmark,
)
from modeltop.models import DiscoveredModel, ServerConfig, format_backend_label
from modeltop.state import ApplicationState, ApplicationStateStore, ServerStatus

logger = logging.getLogger(__name__)

type StateCallback = Callable[[ApplicationState], None]
type UtcNow = Callable[[], datetime]
type MonotonicClock = Callable[[], float]
type BenchmarkIdFactory = Callable[[datetime], str]


class ToolCallingBenchmarkOperationError(Exception):
    """Readable synchronous rejection before upstream import or network I/O."""

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


@dataclass(frozen=True, slots=True)
class PendingToolCallingBenchmark:
    """Immutable metadata captured by the atomic traffic reservation."""

    benchmark_id: str
    config: ToolCallingBenchmarkConfig
    server_id: str
    server_name: str
    server_base_url: str
    server_endpoint: str
    model_id: str
    backend: str
    backend_hint: str | None
    started_at: datetime
    api_key: str | None = field(repr=False)


def _default_benchmark_id_factory(started_at: datetime) -> str:
    timestamp = started_at.strftime("%Y%m%dT%H%M%S%fZ")
    return f"tool-calling-{timestamp}-{uuid4().hex[:8]}"


class ToolCallingBenchmarkService:
    """Reserve, execute, cancel, and retain one normalized latest result."""

    def __init__(
        self,
        state_store: ApplicationStateStore,
        server: ServerConfig,
        on_state_change: StateCallback,
        *,
        upstream_runner: UpstreamBenchmarkRunner = run_upstream_benchmark,
        utc_now: UtcNow = lambda: datetime.now(UTC),
        monotonic_clock: MonotonicClock = time.perf_counter,
        benchmark_id_factory: BenchmarkIdFactory = _default_benchmark_id_factory,
    ) -> None:
        self._state_store = state_store
        self._server = server
        self._on_state_change = on_state_change
        self._upstream_runner = upstream_runner
        self._utc_now = utc_now
        self._monotonic_clock = monotonic_clock
        self._benchmark_id_factory = benchmark_id_factory
        self._benchmark: ToolCallingBenchmark | None = None
        self._active_id: str | None = None
        self._last_progress_publish_at: float | None = None
        self._buffered_progress: ToolCallingBenchmarkProgress | None = None
        self._finalized_ids: set[str] = set()

    @property
    def state(self) -> ApplicationState:
        return self._state_store.state

    def _publish(
        self,
        transform: Callable[[ApplicationState], ApplicationState],
    ) -> ApplicationState:
        state = self._state_store.update(transform)
        try:
            self._on_state_change(state)
        except Exception as error:
            logger.info(
                "Tool Calling state callback failed exception_class=%s",
                type(error).__name__,
            )
        return state

    def begin_benchmark(
        self,
        config: ToolCallingBenchmarkConfig,
    ) -> PendingToolCallingBenchmark:
        """Atomically validate and reserve all benchmark/discovery traffic."""
        pending: PendingToolCallingBenchmark | None = None

        def reserve(state: ApplicationState) -> ApplicationState:
            nonlocal pending
            if state.tool_calling_benchmark.is_active:
                raise ToolCallingBenchmarkOperationError(
                    "A Tool Calling benchmark is already running."
                )
            if state.r0b0bench_benchmark.is_active:
                raise ToolCallingBenchmarkOperationError(
                    "An r0b0bench benchmark is already running."
                )
            if state.context_benchmark.is_active:
                raise ToolCallingBenchmarkOperationError(
                    "A Context benchmark is already running."
                )
            if state.concurrency_benchmark.is_active:
                raise ToolCallingBenchmarkOperationError(
                    "A Concurrency benchmark is already running."
                )
            if state.drafter_benchmark.is_active:
                raise ToolCallingBenchmarkOperationError(
                    "A Drafter benchmark is already running."
                )
            if state.speed_test.is_active:
                raise ToolCallingBenchmarkOperationError(
                    "A Speed Test is already running."
                )
            if state.active_generation_id is not None:
                raise ToolCallingBenchmarkOperationError(
                    "A Chat generation is already running."
                )
            if state.is_refreshing:
                raise ToolCallingBenchmarkOperationError(
                    "Refresh in progress; retry Tool Calling when it completes."
                )
            if state.server_status is not ServerStatus.ONLINE:
                raise ToolCallingBenchmarkOperationError(
                    "The selected server is offline."
                )
            server_id = state.selected_server_id
            model_id = state.selected_model_id
            if server_id is None or model_id is None:
                raise ToolCallingBenchmarkOperationError(
                    "Select an available model before starting Tool Calling."
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
            pending = PendingToolCallingBenchmark(
                benchmark_id=benchmark_id,
                config=config,
                server_id=server_id,
                server_name=self._server.name,
                server_base_url=self._server.base_url,
                server_endpoint=self._server.endpoint_label,
                model_id=model_id,
                backend=backend,
                backend_hint=self._server.backend_hint or selected_model.owned_by,
                started_at=started_at,
                api_key=self._server.api_key,
            )
            lane = state.tool_calling_benchmark
            return replace(
                state,
                tool_calling_benchmark=replace(
                    lane,
                    config=config,
                    status=ToolCallingBenchmarkStatus.VALIDATING,
                    active_benchmark_id=benchmark_id,
                    progress=None,
                    benchmark_started_at=started_at,
                    benchmark_error=None,
                ),
            )

        self._publish(reserve)
        if pending is None:
            raise RuntimeError("Tool Calling reservation did not produce metadata")
        self._active_id = pending.benchmark_id
        self._last_progress_publish_at = None
        self._buffered_progress = None
        logger.info(
            "Tool Calling reserved benchmark=%s model=%s suite=%s",
            pending.benchmark_id,
            pending.model_id,
            pending.config.suite,
        )
        return pending

    async def run_benchmark(
        self,
        pending: PendingToolCallingBenchmark,
    ) -> ToolCallingBenchmarkResult:
        """Own the upstream client and commit exactly one terminal result."""
        lane = self.state.tool_calling_benchmark
        if (
            lane.active_benchmark_id != pending.benchmark_id
            or lane.status is not ToolCallingBenchmarkStatus.VALIDATING
        ):
            raise ToolCallingBenchmarkOperationError(
                "This Tool Calling benchmark is no longer active."
            )
        benchmark = self._make_benchmark(pending)
        self._benchmark = benchmark
        self._mark_running(pending)
        try:
            result = await benchmark.run(self._context(pending))
            self._flush_progress(pending)
            self._finalize(pending, result)
            return result
        finally:
            if self._active_id == pending.benchmark_id:
                self._active_id = None
            if self._benchmark is benchmark:
                self._benchmark = None
            self._buffered_progress = None

    def request_cancellation(self, benchmark_id: str | None = None) -> None:
        """Publish CANCELLING before interrupting the active upstream request."""
        active_id = self._active_id
        if active_id is None or (
            benchmark_id is not None and benchmark_id != active_id
        ):
            return

        def mark(state: ApplicationState) -> ApplicationState:
            lane = state.tool_calling_benchmark
            if lane.active_benchmark_id != active_id or not lane.is_active:
                return state
            return replace(
                state,
                tool_calling_benchmark=replace(
                    lane,
                    status=ToolCallingBenchmarkStatus.CANCELLING,
                ),
            )

        self._publish(mark)
        if self._benchmark is not None:
            self._benchmark.request_cancellation()

    def cancel_reservation(
        self,
        pending: PendingToolCallingBenchmark,
    ) -> ToolCallingBenchmarkResult | None:
        """Finalize cancellation when a worker never entered its coroutine."""
        lane = self.state.tool_calling_benchmark
        if lane.active_benchmark_id != pending.benchmark_id:
            return None
        _, warnings = map_upstream_backend(pending.backend_hint)
        result = ToolCallingBenchmarkResult(
            benchmark_id=pending.benchmark_id,
            upstream_run_id=None,
            config_fingerprint=None,
            server_id=pending.server_id,
            server_name=pending.server_name,
            server_endpoint=pending.server_endpoint,
            model_id=pending.model_id,
            backend=pending.backend,
            integration_commit=TOOL_EVAL_BENCH_COMMIT,
            upstream_version=None,
            schema_version=None,
            config=pending.config,
            started_at=pending.started_at,
            completed_at=self._utc_now(),
            status=ToolCallingBenchmarkStatus.CANCELLED,
            cancelled=True,
            error_code=None,
            error_message=None,
            attempted_count=pending.config.scenario_count,
            gradable_count=0,
            excluded_count=0,
            completion_rate_percent=None,
            final_score=None,
            total_points=None,
            max_points=None,
            rating=None,
            category_k_gradable=False,
            safety_gate_passed=None,
            deployability=None,
            responsiveness=None,
            median_turn_ms=None,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            categories=(),
            scenarios=(),
            warnings=warnings,
            hardware_summary=None,
        )
        self._finalize(pending, result)
        if self._active_id == pending.benchmark_id:
            self._active_id = None
        return result

    def _make_benchmark(
        self,
        pending: PendingToolCallingBenchmark,
    ) -> ToolCallingBenchmark:
        return ToolCallingBenchmark(
            config=pending.config,
            server_id=pending.server_id,
            server_name=pending.server_name,
            server_base_url=pending.server_base_url,
            server_endpoint=pending.server_endpoint,
            model_id=pending.model_id,
            backend=pending.backend,
            backend_hint=pending.backend_hint,
            api_key=pending.api_key,
            progress_callback=lambda progress: self._accept_progress(pending, progress),
            upstream_runner=self._upstream_runner,
        )

    def _context(self, pending: PendingToolCallingBenchmark) -> BenchmarkContext:
        return BenchmarkContext(
            benchmark_id=pending.benchmark_id,
            started_at=pending.started_at,
            monotonic_clock=self._monotonic_clock,
            utc_now=self._utc_now,
            read_hardware_snapshot=lambda: self.state.hardware_snapshot,
        )

    def _mark_running(self, pending: PendingToolCallingBenchmark) -> None:
        def mark(state: ApplicationState) -> ApplicationState:
            lane = state.tool_calling_benchmark
            if lane.active_benchmark_id != pending.benchmark_id:
                return state
            return replace(
                state,
                tool_calling_benchmark=replace(
                    lane,
                    status=ToolCallingBenchmarkStatus.RUNNING,
                ),
            )

        self._publish(mark)

    async def _accept_progress(
        self,
        pending: PendingToolCallingBenchmark,
        progress: ToolCallingBenchmarkProgress,
    ) -> None:
        if self._active_id != pending.benchmark_id:
            return
        self._buffered_progress = progress
        now = self._monotonic_clock()
        terminal = progress.completed_count == progress.configured_count
        if (
            not terminal
            and self._last_progress_publish_at is not None
            and now - self._last_progress_publish_at < 0.1
        ):
            return
        self._flush_progress(pending, now=now)

    def _flush_progress(
        self,
        pending: PendingToolCallingBenchmark,
        *,
        now: float | None = None,
    ) -> None:
        progress = self._buffered_progress
        if progress is None or self._active_id != pending.benchmark_id:
            return

        def update(state: ApplicationState) -> ApplicationState:
            lane = state.tool_calling_benchmark
            if lane.active_benchmark_id != pending.benchmark_id:
                return state
            status = lane.status
            if status is ToolCallingBenchmarkStatus.VALIDATING:
                status = ToolCallingBenchmarkStatus.RUNNING
            return replace(
                state,
                tool_calling_benchmark=replace(
                    lane,
                    status=status,
                    progress=progress,
                    benchmark_error=None,
                ),
            )

        self._publish(update)
        self._last_progress_publish_at = self._monotonic_clock() if now is None else now

    def _finalize(
        self,
        pending: PendingToolCallingBenchmark,
        result: ToolCallingBenchmarkResult,
    ) -> None:
        if pending.benchmark_id in self._finalized_ids:
            return
        self._finalized_ids.add(pending.benchmark_id)

        def finish(state: ApplicationState) -> ApplicationState:
            lane = state.tool_calling_benchmark
            if lane.active_benchmark_id != pending.benchmark_id:
                return state
            return replace(
                state,
                tool_calling_benchmark=replace(
                    lane,
                    config=pending.config,
                    status=result.status,
                    active_benchmark_id=None,
                    progress=None,
                    benchmark_started_at=None,
                    latest_result=result,
                    benchmark_error=result.error_message,
                ),
            )

        self._publish(finish)
