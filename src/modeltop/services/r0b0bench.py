"""Atomic reservation and lifecycle for out-of-process r0b0bench runs."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from modeltop.benchmarks.models import (
    R0b0benchBenchmarkConfig,
    R0b0benchBenchmarkProgress,
    R0b0benchBenchmarkResult,
    R0b0benchBenchmarkStatus,
    R0b0benchErrorCode,
    R0b0benchLaneId,
    R0b0benchLaneResult,
    R0b0benchLaneStatus,
    R0b0benchWarningCode,
)
from modeltop.benchmarks.r0b0bench import (
    DEFAULT_R0B0BENCH_OUTPUT_ROOT,
    R0B0BENCH_COMMIT,
    R0b0benchPreparedRun,
    R0b0benchRunner,
    R0b0benchRunnerError,
    R0b0benchRunnerReport,
    R0b0benchRunnerRequest,
)
from modeltop.benchmarks.r0b0bench_contract import (
    r0b0bench_ordered_selection,
    r0b0bench_profile_lanes,
)
from modeltop.benchmarks.statistics import summarize_hardware_samples
from modeltop.hardware.models import HardwareSnapshot
from modeltop.models import DiscoveredModel, ServerConfig, format_backend_label
from modeltop.state import ApplicationState, ApplicationStateStore, ServerStatus

logger = logging.getLogger(__name__)

type StateCallback = Callable[[ApplicationState], None]
type UtcNow = Callable[[], datetime]
type MonotonicClock = Callable[[], float]
type BenchmarkIdFactory = Callable[[datetime], str]


class R0b0benchBenchmarkOperationError(Exception):
    """Readable synchronous rejection before child creation or network I/O."""

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


@dataclass(frozen=True, slots=True)
class PendingR0b0benchBenchmark:
    """Immutable server, model, and config captured by the reservation."""

    benchmark_id: str
    config: R0b0benchBenchmarkConfig
    server_id: str
    server_name: str
    server_base_url: str = field(repr=False)
    server_endpoint: str
    model_id: str
    backend: str
    started_at: datetime
    started_monotonic: float = field(repr=False)
    api_key: str | None = field(repr=False)


def _default_benchmark_id_factory(started_at: datetime) -> str:
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    return f"r0b0bench-{timestamp}-{uuid4().hex[:8]}"


class R0b0benchBenchmarkService:
    """Reserve, validate, execute, cancel, and finalize one r0b0bench run."""

    def __init__(
        self,
        state_store: ApplicationStateStore,
        server: ServerConfig,
        on_state_change: StateCallback,
        *,
        runner: R0b0benchRunner,
        output_root: Path = DEFAULT_R0B0BENCH_OUTPUT_ROOT,
        utc_now: UtcNow = lambda: datetime.now(UTC),
        monotonic_clock: MonotonicClock = time.perf_counter,
        benchmark_id_factory: BenchmarkIdFactory = _default_benchmark_id_factory,
    ) -> None:
        self._state_store = state_store
        self._server = server
        self._on_state_change = on_state_change
        self._runner = runner
        self._output_root = output_root
        self._utc_now = utc_now
        self._monotonic_clock = monotonic_clock
        self._benchmark_id_factory = benchmark_id_factory
        self._active_id: str | None = None
        self._task: asyncio.Task[object] | None = None
        self._prepared: R0b0benchPreparedRun | None = None
        self._completed: list[R0b0benchLaneResult] = []
        self._current_lane: R0b0benchLaneId | None = None
        self._hardware_samples: list[HardwareSnapshot] = []
        self._last_progress_publish_at: float | None = None
        self._buffered_progress: R0b0benchBenchmarkProgress | None = None
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
        except Exception as error:
            logger.info(
                "r0b0bench state callback failed exception_class=%s",
                type(error).__name__,
            )
        return state

    def begin_benchmark(
        self, config: R0b0benchBenchmarkConfig
    ) -> PendingR0b0benchBenchmark:
        """Atomically reserve all benchmark, chat, refresh, and model traffic."""
        pending: PendingR0b0benchBenchmark | None = None

        def reserve(state: ApplicationState) -> ApplicationState:
            nonlocal pending
            if state.r0b0bench_benchmark.is_active:
                raise R0b0benchBenchmarkOperationError(
                    "An r0b0bench benchmark is already running."
                )
            if state.tool_calling_benchmark.is_active:
                raise R0b0benchBenchmarkOperationError(
                    "A Tool Calling benchmark is already running."
                )
            if state.context_benchmark.is_active:
                raise R0b0benchBenchmarkOperationError(
                    "A Context benchmark is already running."
                )
            if state.concurrency_benchmark.is_active:
                raise R0b0benchBenchmarkOperationError(
                    "A Concurrency benchmark is already running."
                )
            if state.drafter_benchmark.is_active:
                raise R0b0benchBenchmarkOperationError(
                    "A Drafter benchmark is already running."
                )
            if state.speed_test.is_active:
                raise R0b0benchBenchmarkOperationError(
                    "A Speed Test is already running."
                )
            if state.active_generation_id is not None:
                raise R0b0benchBenchmarkOperationError(
                    "A Chat generation is already running."
                )
            if state.is_refreshing:
                raise R0b0benchBenchmarkOperationError(
                    "Refresh in progress; retry r0b0bench when it completes."
                )
            if state.server_status is not ServerStatus.ONLINE:
                raise R0b0benchBenchmarkOperationError(
                    "The selected server is offline."
                )
            server_id = state.selected_server_id
            model_id = state.selected_model_id
            if server_id is None or model_id is None:
                raise R0b0benchBenchmarkOperationError(
                    "Select an available model before starting r0b0bench."
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
            pending = PendingR0b0benchBenchmark(
                benchmark_id=benchmark_id,
                config=config,
                server_id=server_id,
                server_name=self._server.name,
                server_base_url=self._server.base_url,
                server_endpoint=self._server.endpoint_label,
                model_id=model_id,
                backend=backend,
                started_at=started_at,
                started_monotonic=self._monotonic_clock(),
                api_key=self._server.api_key,
            )
            lane = state.r0b0bench_benchmark
            return replace(
                state,
                r0b0bench_benchmark=replace(
                    lane,
                    config=config,
                    status=R0b0benchBenchmarkStatus.VALIDATING,
                    active_benchmark_id=benchmark_id,
                    progress=None,
                    benchmark_started_at=started_at,
                    benchmark_error=None,
                ),
            )

        self._publish(reserve)
        if pending is None:
            raise RuntimeError("r0b0bench reservation did not produce metadata")
        self._active_id = pending.benchmark_id
        self._task = None
        self._prepared = None
        self._completed = []
        self._current_lane = None
        self._hardware_samples = []
        self._last_progress_publish_at = None
        self._buffered_progress = None
        return pending

    async def run_benchmark(
        self, pending: PendingR0b0benchBenchmark
    ) -> R0b0benchBenchmarkResult:
        """Prepare and run the child, committing exactly one terminal result."""
        lane = self.state.r0b0bench_benchmark
        if (
            lane.active_benchmark_id != pending.benchmark_id
            or lane.status is not R0b0benchBenchmarkStatus.VALIDATING
        ):
            raise R0b0benchBenchmarkOperationError(
                "This r0b0bench benchmark is no longer active."
            )
        current_task = asyncio.current_task()
        if current_task is None:
            raise RuntimeError("r0b0bench requires an active asyncio task")
        self._task = cast_task = current_task
        request = R0b0benchRunnerRequest(
            benchmark_id=pending.benchmark_id,
            config=pending.config,
            base_url=pending.server_base_url,
            model_id=pending.model_id,
            output_root=self._output_root,
            api_key=pending.api_key,
        )
        try:
            try:
                prepared = await self._runner.prepare(request)
            except asyncio.CancelledError:
                result = self._cancelled_result(pending, prepared=self._prepared)
                self._finalize(pending, result)
                return result
            self._prepared = prepared
            self._mark_running(pending)
            self._publish_progress(pending, force=True)
            try:
                report = await self._runner.run(
                    prepared,
                    lambda lane_id, index, total: self._lane_started(
                        pending, lane_id, index, total
                    ),
                    lambda row, index, total: self._lane_finished(
                        pending, row, index, total
                    ),
                )
            except asyncio.CancelledError:
                result = self._cancelled_result(pending, prepared=prepared)
            except R0b0benchRunnerError as error:
                result = self._error_result(
                    pending,
                    error.code,
                    prepared=prepared,
                )
            except Exception:
                result = self._error_result(
                    pending, "upstream_failure", prepared=prepared
                )
            else:
                self._completed = list(report.lanes)
                result = self._result_from_report(pending, report)
            self._flush_progress(pending)
            self._finalize(pending, result)
            return result
        except R0b0benchRunnerError as error:
            result = self._error_result(
                pending,
                error.code,
                prepared=self._prepared,
            )
            self._finalize(pending, result)
            return result
        except Exception:
            result = self._error_result(
                pending, "upstream_failure", prepared=self._prepared
            )
            self._finalize(pending, result)
            return result
        finally:
            if self._active_id == pending.benchmark_id:
                self._active_id = None
            if self._task is cast_task:
                self._task = None
            self._prepared = None
            self._buffered_progress = None

    def request_cancellation(self, benchmark_id: str | None = None) -> None:
        """Publish CANCELLING before interrupting runner preparation or execution."""
        active_id = self._active_id
        if active_id is None or (
            benchmark_id is not None and benchmark_id != active_id
        ):
            return

        def mark(state: ApplicationState) -> ApplicationState:
            lane = state.r0b0bench_benchmark
            if lane.active_benchmark_id != active_id or not lane.is_active:
                return state
            return replace(
                state,
                r0b0bench_benchmark=replace(
                    lane, status=R0b0benchBenchmarkStatus.CANCELLING
                ),
            )

        self._publish(mark)
        if self._task is not None:
            self._task.cancel()

    def cancel_reservation(
        self, pending: PendingR0b0benchBenchmark
    ) -> R0b0benchBenchmarkResult | None:
        """Finalize an empty cancellation when a Textual worker never entered."""
        lane = self.state.r0b0bench_benchmark
        if lane.active_benchmark_id != pending.benchmark_id:
            return None
        result = self._cancelled_result(pending, prepared=None)
        self._finalize(pending, result)
        if self._active_id == pending.benchmark_id:
            self._active_id = None
        return result

    def _mark_running(self, pending: PendingR0b0benchBenchmark) -> None:
        def mark(state: ApplicationState) -> ApplicationState:
            lane = state.r0b0bench_benchmark
            if lane.active_benchmark_id != pending.benchmark_id:
                return state
            return replace(
                state,
                r0b0bench_benchmark=replace(
                    lane, status=R0b0benchBenchmarkStatus.RUNNING
                ),
            )

        self._publish(mark)

    async def _lane_started(
        self,
        pending: PendingR0b0benchBenchmark,
        lane_id: R0b0benchLaneId,
        _index: int,
        _total: int,
    ) -> None:
        if self._active_id != pending.benchmark_id:
            return
        self._current_lane = lane_id
        self._publish_progress(pending, force=True)

    async def _lane_finished(
        self,
        pending: PendingR0b0benchBenchmark,
        row: R0b0benchLaneResult,
        _index: int,
        _total: int,
    ) -> None:
        if self._active_id != pending.benchmark_id:
            return
        if row.lane_id not in {existing.lane_id for existing in self._completed}:
            self._completed.append(row)
        self._current_lane = None
        self._publish_progress(pending, force=True)

    def _publish_progress(
        self, pending: PendingR0b0benchBenchmark, *, force: bool = False
    ) -> None:
        snapshot = self.state.hardware_snapshot
        if snapshot is not None:
            self._hardware_samples.append(snapshot)
        statuses = tuple(row.status for row in self._completed)
        progress = R0b0benchBenchmarkProgress(
            configured_count=len(pending.config.selected_lanes),
            completed_count=len(self._completed),
            pass_count=statuses.count(R0b0benchLaneStatus.PASS),
            fail_count=statuses.count(R0b0benchLaneStatus.FAIL),
            skip_count=statuses.count(R0b0benchLaneStatus.SKIP),
            error_count=statuses.count(R0b0benchLaneStatus.ERROR),
            not_implemented_count=statuses.count(R0b0benchLaneStatus.NOT_IMPLEMENTED),
            current_lane=self._current_lane,
            elapsed_seconds=max(
                0.0, self._monotonic_clock() - pending.started_monotonic
            ),
            cached_hardware=snapshot,
        )
        self._buffered_progress = progress
        now = self._monotonic_clock()
        if (
            not force
            and self._last_progress_publish_at is not None
            and now - self._last_progress_publish_at < 0.1
        ):
            return
        self._flush_progress(pending, now=now)

    def _flush_progress(
        self,
        pending: PendingR0b0benchBenchmark,
        *,
        now: float | None = None,
    ) -> None:
        progress = self._buffered_progress
        if progress is None or self._active_id != pending.benchmark_id:
            return

        def update(state: ApplicationState) -> ApplicationState:
            lane = state.r0b0bench_benchmark
            if lane.active_benchmark_id != pending.benchmark_id:
                return state
            return replace(
                state,
                r0b0bench_benchmark=replace(
                    lane, progress=progress, benchmark_error=None
                ),
            )

        self._publish(update)
        self._last_progress_publish_at = self._monotonic_clock() if now is None else now

    def _warnings(
        self,
        config: R0b0benchBenchmarkConfig,
        *,
        lanes: tuple[R0b0benchLaneResult, ...],
        unstarted: tuple[R0b0benchLaneId, ...],
        cancelled: bool,
    ) -> tuple[R0b0benchWarningCode, ...]:
        warnings: list[R0b0benchWarningCode] = []
        ordered = r0b0bench_ordered_selection(config.profile, config.selected_lanes)
        if ordered != r0b0bench_profile_lanes(config.profile):
            warnings.append("filtered_selection")
        if "perf" in config.selected_lanes:
            warnings.append("perf_composite")
        if (
            unstarted
            and lanes
            and lanes[0].lane_id == "canary"
            and lanes[0].status is R0b0benchLaneStatus.ERROR
            and lanes[0].infra_errors > 0
        ):
            warnings.append("canary_infrastructure_stop")
        if cancelled:
            warnings.append("cancelled_partial")
        return tuple(warnings)

    def _result_from_report(
        self,
        pending: PendingR0b0benchBenchmark,
        report: R0b0benchRunnerReport,
    ) -> R0b0benchBenchmarkResult:
        has_problems = any(
            lane.infra_errors
            or lane.status
            in {R0b0benchLaneStatus.ERROR, R0b0benchLaneStatus.NOT_IMPLEMENTED}
            for lane in report.lanes
        )
        if report.cancelled:
            status = R0b0benchBenchmarkStatus.CANCELLED
        elif has_problems:
            status = R0b0benchBenchmarkStatus.COMPLETED_WITH_ERRORS
        else:
            status = R0b0benchBenchmarkStatus.COMPLETED
        return R0b0benchBenchmarkResult(
            benchmark_id=pending.benchmark_id,
            upstream_run_id=report.upstream_run_id,
            upstream_version=report.upstream_version,
            upstream_schema_version=report.schema_version,
            upstream_commit=R0B0BENCH_COMMIT,
            config=pending.config,
            server_id=pending.server_id,
            server_name=pending.server_name,
            server_endpoint=pending.server_endpoint,
            model_id=pending.model_id,
            backend=pending.backend,
            started_at=pending.started_at,
            completed_at=self._utc_now(),
            status=status,
            cancelled=report.cancelled,
            error_code=None,
            error_message=None,
            selected_count=len(pending.config.selected_lanes),
            completed_count=len(report.lanes),
            unstarted_lanes=report.unstarted_lanes,
            lanes=report.lanes,
            infra_errors_total=report.infra_errors_total,
            invalid_for_publish=report.invalid_for_publish,
            warning_codes=self._warnings(
                pending.config,
                lanes=report.lanes,
                unstarted=report.unstarted_lanes,
                cancelled=report.cancelled,
            ),
            hardware_summary=summarize_hardware_samples(tuple(self._hardware_samples)),
            run_directory=report.run_directory,
        )

    def _cancelled_result(
        self,
        pending: PendingR0b0benchBenchmark,
        *,
        prepared: R0b0benchPreparedRun | None,
    ) -> R0b0benchBenchmarkResult:
        ordered = r0b0bench_ordered_selection(
            pending.config.profile, pending.config.selected_lanes
        )
        lanes = tuple(self._completed)
        return R0b0benchBenchmarkResult(
            benchmark_id=pending.benchmark_id,
            upstream_run_id=None,
            upstream_version=None,
            upstream_schema_version=None,
            upstream_commit=R0B0BENCH_COMMIT,
            config=pending.config,
            server_id=pending.server_id,
            server_name=pending.server_name,
            server_endpoint=pending.server_endpoint,
            model_id=pending.model_id,
            backend=pending.backend,
            started_at=pending.started_at,
            completed_at=self._utc_now(),
            status=R0b0benchBenchmarkStatus.CANCELLED,
            cancelled=True,
            error_code=None,
            error_message=None,
            selected_count=len(ordered),
            completed_count=len(lanes),
            unstarted_lanes=ordered[len(lanes) :],
            lanes=lanes,
            infra_errors_total=sum(row.infra_errors for row in lanes),
            invalid_for_publish=True,
            warning_codes=self._warnings(
                pending.config,
                lanes=lanes,
                unstarted=ordered[len(lanes) :],
                cancelled=True,
            ),
            hardware_summary=summarize_hardware_samples(tuple(self._hardware_samples)),
            run_directory=None if prepared is None else prepared.run_directory,
        )

    def _error_result(
        self,
        pending: PendingR0b0benchBenchmark,
        code: R0b0benchErrorCode,
        *,
        prepared: R0b0benchPreparedRun | None,
    ) -> R0b0benchBenchmarkResult:
        ordered = r0b0bench_ordered_selection(
            pending.config.profile, pending.config.selected_lanes
        )
        lanes = tuple(self._completed)
        unstarted = ordered[len(lanes) :]
        message = str(R0b0benchRunnerError(code))
        return R0b0benchBenchmarkResult(
            benchmark_id=pending.benchmark_id,
            upstream_run_id=None,
            upstream_version=None,
            upstream_schema_version=None,
            upstream_commit=R0B0BENCH_COMMIT,
            config=pending.config,
            server_id=pending.server_id,
            server_name=pending.server_name,
            server_endpoint=pending.server_endpoint,
            model_id=pending.model_id,
            backend=pending.backend,
            started_at=pending.started_at,
            completed_at=self._utc_now(),
            status=R0b0benchBenchmarkStatus.ERROR,
            cancelled=False,
            error_code=code,
            error_message=message,
            selected_count=len(ordered),
            completed_count=len(lanes),
            unstarted_lanes=unstarted,
            lanes=lanes,
            infra_errors_total=sum(row.infra_errors for row in lanes),
            invalid_for_publish=True,
            warning_codes=self._warnings(
                pending.config,
                lanes=lanes,
                unstarted=unstarted,
                cancelled=False,
            ),
            hardware_summary=summarize_hardware_samples(tuple(self._hardware_samples)),
            run_directory=None if prepared is None else prepared.run_directory,
        )

    def _finalize(
        self,
        pending: PendingR0b0benchBenchmark,
        result: R0b0benchBenchmarkResult,
    ) -> None:
        if pending.benchmark_id in self._finalized_ids:
            return
        self._finalized_ids.add(pending.benchmark_id)

        def finish(state: ApplicationState) -> ApplicationState:
            lane = state.r0b0bench_benchmark
            if lane.active_benchmark_id != pending.benchmark_id:
                return state
            return replace(
                state,
                r0b0bench_benchmark=replace(
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
