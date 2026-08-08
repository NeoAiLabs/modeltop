"""Immutable runtime state snapshots."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Literal

from modeltop.benchmarks.models import (
    ConcurrencyBenchmarkConfig,
    ConcurrencyBenchmarkState,
    ContextBenchmarkConfig,
    ContextBenchmarkState,
    DrafterBenchmarkConfig,
    DrafterBenchmarkState,
    R0b0benchBenchmarkConfig,
    R0b0benchBenchmarkState,
    SpeedTestState,
    ToolCallingBenchmarkConfig,
    ToolCallingBenchmarkState,
    initial_concurrency_benchmark_state,
    initial_context_benchmark_state,
    initial_drafter_benchmark_state,
    initial_r0b0bench_benchmark_state,
    initial_speed_test_state,
    initial_tool_calling_benchmark_state,
)
from modeltop.chat.models import GenerationMetrics, GenerationStatus
from modeltop.chat.session import ChatSession
from modeltop.hardware.models import HardwareSnapshot
from modeltop.models import DiscoveredModel
from modeltop.services.result_archive import ResultArchiveSnapshot, load_archive

type ActiveView = Literal[
    "overview",
    "chat",
    "speed-test",
    "concurrency",
    "context",
    "tool-calling",
    "r0b0bench",
    "drafter",
    "results",
    "settings",
]


class ServerStatus(StrEnum):
    """Current server connectivity state."""

    CONNECTING = "connecting"
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"


class HardwareStatus(StrEnum):
    """Current local hardware monitoring state."""

    INITIALISING = "initialising"
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ApplicationState:
    """A coherent dashboard state snapshot."""

    selected_server_id: str | None
    selected_model_id: str | None
    server_status: ServerStatus
    available_models: tuple[DiscoveredModel, ...]
    connection_latency_ms: float | None
    last_refresh_time: datetime | None
    last_error: str | None
    is_refreshing: bool
    hardware_status: HardwareStatus
    hardware_snapshot: HardwareSnapshot | None
    hardware_last_refresh_time: datetime | None
    hardware_last_error: str | None
    hardware_is_refreshing: bool
    active_view: ActiveView
    chat_session: ChatSession
    generation_status: GenerationStatus
    generation_metrics: GenerationMetrics | None
    generation_error: str | None
    generation_notice: str | None
    current_response: str
    generation_id: int
    active_generation_id: int | None
    speed_test: SpeedTestState
    concurrency_benchmark: ConcurrencyBenchmarkState
    context_benchmark: ContextBenchmarkState
    tool_calling_benchmark: ToolCallingBenchmarkState
    r0b0bench_benchmark: R0b0benchBenchmarkState
    drafter_benchmark: DrafterBenchmarkState
    result_archive: ResultArchiveSnapshot

    @property
    def benchmark_is_active(self) -> bool:
        """Return whether any benchmark lane owns generation traffic."""
        return (
            self.speed_test.is_active
            or self.concurrency_benchmark.is_active
            or self.context_benchmark.is_active
            or self.tool_calling_benchmark.is_active
            or self.r0b0bench_benchmark.is_active
            or self.drafter_benchmark.is_active
        )


def initial_application_state(
    server_id: str,
    *,
    hardware_enabled: bool,
    concurrency_config: ConcurrencyBenchmarkConfig | None = None,
    result_archive: ResultArchiveSnapshot | None = None,
    context_config: ContextBenchmarkConfig | None = None,
    tool_calling_config: ToolCallingBenchmarkConfig | None = None,
    r0b0bench_config: R0b0benchBenchmarkConfig | None = None,
    drafter_config: DrafterBenchmarkConfig | None = None,
) -> ApplicationState:
    """Build the initial state for independent server and hardware lanes."""
    return ApplicationState(
        selected_server_id=server_id,
        selected_model_id=None,
        server_status=ServerStatus.CONNECTING,
        available_models=(),
        connection_latency_ms=None,
        last_refresh_time=None,
        last_error=None,
        is_refreshing=False,
        hardware_status=(
            HardwareStatus.INITIALISING
            if hardware_enabled
            else HardwareStatus.UNAVAILABLE
        ),
        hardware_snapshot=None,
        hardware_last_refresh_time=None,
        hardware_last_error=(
            None if hardware_enabled else "Hardware monitoring disabled"
        ),
        hardware_is_refreshing=False,
        active_view="overview",
        chat_session=ChatSession(),
        generation_status=GenerationStatus.IDLE,
        generation_metrics=None,
        generation_error=None,
        generation_notice=None,
        current_response="",
        generation_id=0,
        active_generation_id=None,
        speed_test=initial_speed_test_state(),
        concurrency_benchmark=initial_concurrency_benchmark_state(
            concurrency_config
            if concurrency_config is not None
            else ConcurrencyBenchmarkConfig()
        ),
        context_benchmark=initial_context_benchmark_state(
            context_config if context_config is not None else ContextBenchmarkConfig()
        ),
        tool_calling_benchmark=initial_tool_calling_benchmark_state(
            tool_calling_config
            if tool_calling_config is not None
            else ToolCallingBenchmarkConfig()
        ),
        r0b0bench_benchmark=initial_r0b0bench_benchmark_state(
            r0b0bench_config
            if r0b0bench_config is not None
            else R0b0benchBenchmarkConfig()
        ),
        drafter_benchmark=initial_drafter_benchmark_state(
            drafter_config if drafter_config is not None else DrafterBenchmarkConfig()
        ),
        result_archive=result_archive if result_archive is not None else load_archive(),
    )


class ApplicationStateStore:
    """Event-loop-owned store for atomic immutable state replacements."""

    def __init__(self, initial_state: ApplicationState) -> None:
        self._state = initial_state

    @property
    def state(self) -> ApplicationState:
        """Return the latest immutable state snapshot."""
        return self._state

    def update(
        self, transform: Callable[[ApplicationState], ApplicationState]
    ) -> ApplicationState:
        """Apply one synchronous transform to the latest snapshot."""
        self._state = transform(self._state)
        return self._state
