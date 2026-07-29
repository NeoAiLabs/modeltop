"""Atomic ownership and cancellation tests for Tool Calling service."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

from modeltop.benchmarks.models import (
    ContextBenchmarkStatus,
    ToolCallingBenchmarkConfig,
    ToolCallingBenchmarkStatus,
)
from modeltop.benchmarks.tool_calling import (
    ScenarioStartCallback,
    UpstreamBenchmarkRunner,
    suite_registry,
)
from modeltop.models import DiscoveredModel, ServerConfig
from modeltop.services.model_discovery import ModelDiscoveryService
from modeltop.services.server_monitor import ServerMonitor
from modeltop.services.tool_calling import (
    ToolCallingBenchmarkOperationError,
    ToolCallingBenchmarkService,
)
from modeltop.state import (
    ApplicationStateStore,
    ServerStatus,
    initial_application_state,
)


def _store() -> ApplicationStateStore:
    return ApplicationStateStore(
        replace(
            initial_application_state("server", hardware_enabled=False),
            server_status=ServerStatus.ONLINE,
            selected_model_id="model",
            available_models=(DiscoveredModel(id="model", owned_by="vllm"),),
        )
    )


def _service(
    runner: UpstreamBenchmarkRunner,
    store: ApplicationStateStore | None = None,
) -> tuple[ToolCallingBenchmarkService, ApplicationStateStore]:
    actual_store = store or _store()
    service = ToolCallingBenchmarkService(
        actual_store,
        ServerConfig(
            id="server",
            name="Server",
            base_url="http://127.0.0.1:8000/v1",
            api_key="SENSITIVE_API_KEY",
            backend_hint="vllm",
        ),
        lambda _state: None,
        upstream_runner=runner,
        benchmark_id_factory=lambda _started: "tool-calling-test",
    )
    return service, actual_store


def _broken_runner() -> UpstreamBenchmarkRunner:
    async def run(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("SENSITIVE_UPSTREAM_MESSAGE")

    return cast(UpstreamBenchmarkRunner, run)


def test_reservation_is_atomic_and_error_finalization_releases_lane() -> None:
    service, store = _service(_broken_runner())
    pending = service.begin_benchmark(ToolCallingBenchmarkConfig(suite="core"))
    lane = store.state.tool_calling_benchmark
    assert lane.status is ToolCallingBenchmarkStatus.VALIDATING
    assert lane.active_benchmark_id == pending.benchmark_id
    assert store.state.benchmark_is_active

    with pytest.raises(ToolCallingBenchmarkOperationError, match="already running"):
        service.begin_benchmark(ToolCallingBenchmarkConfig(suite="core"))

    result = asyncio.run(service.run_benchmark(pending))
    assert result.status is ToolCallingBenchmarkStatus.ERROR
    assert result.error_code == "upstream_failure"
    assert "SENSITIVE_UPSTREAM_MESSAGE" not in repr(result)
    lane = store.state.tool_calling_benchmark
    assert lane.latest_result is result
    assert lane.active_benchmark_id is None
    assert lane.status is ToolCallingBenchmarkStatus.ERROR
    assert not store.state.benchmark_is_active


def test_reservation_rejects_other_lanes_refresh_offline_and_missing_model() -> None:
    states = (
        replace(
            _store().state,
            context_benchmark=replace(
                _store().state.context_benchmark,
                status=ContextBenchmarkStatus.RUNNING,
                active_benchmark_id="context",
            ),
        ),
        replace(_store().state, is_refreshing=True),
        replace(_store().state, server_status=ServerStatus.OFFLINE),
        replace(_store().state, selected_model_id=None),
    )
    expected = ("Context", "Refresh", "offline", "Select")
    for state, message in zip(states, expected, strict=True):
        service, _ = _service(_broken_runner(), ApplicationStateStore(state))
        with pytest.raises(ToolCallingBenchmarkOperationError, match=message):
            service.begin_benchmark(ToolCallingBenchmarkConfig(suite="core"))


def test_cancel_reservation_creates_empty_terminal_result_once() -> None:
    service, store = _service(_broken_runner())
    pending = service.begin_benchmark(ToolCallingBenchmarkConfig(suite="core"))
    result = service.cancel_reservation(pending)
    assert result is not None
    assert result.status is ToolCallingBenchmarkStatus.CANCELLED
    assert result.scenarios == ()
    assert store.state.tool_calling_benchmark.latest_result is result
    assert service.cancel_reservation(pending) is None


def test_worker_cancellation_waits_for_executor_cleanup_and_flushes_state() -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        cleaned = asyncio.Event()

        async def gated(**kwargs: Any) -> dict[str, Any]:
            identity = suite_registry("core")[0]
            definition = SimpleNamespace(
                id=identity[0],
                category=SimpleNamespace(value=identity[1]),
                title=identity[2],
            )
            on_start = cast(ScenarioStartCallback, kwargs["on_scenario_start"])
            await on_start(definition, 0, 15)
            started.set()
            try:
                await asyncio.Event().wait()
                raise AssertionError("gate unexpectedly released")
            finally:
                cleaned.set()

        service, store = _service(cast(UpstreamBenchmarkRunner, gated))
        pending = service.begin_benchmark(ToolCallingBenchmarkConfig(suite="core"))
        task = asyncio.create_task(service.run_benchmark(pending))
        await started.wait()
        assert store.state.tool_calling_benchmark.progress is not None
        service.request_cancellation(pending.benchmark_id)
        assert (
            store.state.tool_calling_benchmark.status
            is ToolCallingBenchmarkStatus.CANCELLING
        )
        task.cancel()
        result = await task
        assert cleaned.is_set()
        assert result.status is ToolCallingBenchmarkStatus.CANCELLED
        assert store.state.tool_calling_benchmark.latest_result is result
        assert store.state.tool_calling_benchmark.active_benchmark_id is None

    asyncio.run(scenario())


def test_server_monitor_blocks_refresh_and_selection_at_state_owner() -> None:
    store = _store()
    active_lane = replace(
        store.state.tool_calling_benchmark,
        status=ToolCallingBenchmarkStatus.RUNNING,
        active_benchmark_id="tool-calling-test",
    )
    store.update(lambda state: replace(state, tool_calling_benchmark=active_lane))
    monitor = ServerMonitor(
        ServerConfig(id="server", name="Server", base_url="http://127.0.0.1:8000/v1"),
        cast(ModelDiscoveryService, object()),
        store,
        lambda _state: None,
    )
    assert not monitor.begin_refresh()
    assert not monitor.select_model("model")
    assert not store.state.is_refreshing

    store.update(
        lambda state: replace(
            state,
            tool_calling_benchmark=replace(
                state.tool_calling_benchmark,
                status=ToolCallingBenchmarkStatus.COMPLETED,
                active_benchmark_id=None,
            ),
        )
    )
    assert monitor.select_model("model")
    assert monitor.begin_refresh()
