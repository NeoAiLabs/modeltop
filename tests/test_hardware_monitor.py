"""Independent hardware monitor selection and state-transition tests."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import pytest

import modeltop.services.hardware_monitor as monitor_module
from modeltop.hardware.base import (
    HardwareCollectionError,
    HardwareProvider,
    HardwareProviderUnavailable,
    SystemMetricsResult,
)
from modeltop.hardware.models import (
    CpuMetrics,
    GpuMetrics,
    HardwareSnapshot,
    MemoryMetrics,
)
from modeltop.models import HardwareConfig
from modeltop.services.hardware_monitor import HardwareMonitor
from modeltop.state import (
    ApplicationState,
    ApplicationStateStore,
    HardwareStatus,
    ServerStatus,
    initial_application_state,
)


class _FakeCollector:
    def collect(self) -> SystemMetricsResult:
        return SystemMetricsResult(
            CpuMetrics(20.0, 8, 4, 1.0, 2.0, 3.0),
            MemoryMetrics(4, 8, 50.0),
            False,
        )


class _FakeProvider(HardwareProvider):
    def __init__(
        self,
        name: str,
        outcomes: list[HardwareSnapshot | BaseException],
        *,
        gate: bool = False,
    ) -> None:
        self.name = name
        self.outcomes = outcomes
        self.collect_count = 0
        self.close_count = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.gate = gate

    async def collect(self) -> HardwareSnapshot:
        self.collect_count += 1
        self.started.set()
        if self.gate:
            await self.release.wait()
        outcome = self.outcomes[min(self.collect_count - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def close(self) -> None:
        self.close_count += 1


def _snapshot(
    *,
    gpu_count: int = 1,
    error: str | None = None,
    second: int = 0,
) -> HardwareSnapshot:
    gpus = tuple(
        GpuMetrics(
            index,
            f"GPU {index}",
            f"gpu-{index}",
            10.0 + index,
            2,
            8,
            50.0,
            100.0,
            200.0,
            30.0,
        )
        for index in range(gpu_count)
    )
    return HardwareSnapshot(
        "fixture",
        gpus,
        CpuMetrics(20.0, 8, 4, 1.0, 2.0, 3.0),
        MemoryMetrics(4, 8, 50.0),
        datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=second),
        error,
    )


def _store() -> ApplicationStateStore:
    return ApplicationStateStore(
        initial_application_state("server", hardware_enabled=True)
    )


def _factory(
    operation: Callable[[], HardwareProvider | BaseException],
    calls: list[str],
    name: str,
) -> Callable[[object], Awaitable[HardwareProvider]]:
    async def create(collector: object) -> HardwareProvider:
        calls.append(name)
        outcome = operation()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return create


def test_auto_falls_back_once_then_reuses_selected_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(monitor_module, "SystemMetricsCollector", _FakeCollector)
        calls: list[str] = []
        provider = _FakeProvider("nvidia-smi", [_snapshot(), _snapshot(second=1)])
        monitor = HardwareMonitor(
            HardwareConfig(),
            _store(),
            lambda state: None,
            nvml_factory=_factory(
                lambda: HardwareProviderUnavailable("NVML unavailable", "detail"),
                calls,
                "nvml",
            ),
            nvidia_smi_factory=_factory(lambda: provider, calls, "smi"),
        )
        assert monitor.begin_refresh()
        first = await monitor.refresh()
        assert first.success
        assert monitor.provider is provider
        assert monitor.state.hardware_status is HardwareStatus.AVAILABLE
        assert monitor.begin_refresh()
        await monitor.refresh()
        assert calls == ["nvml", "smi"]
        assert provider.collect_count == 2
        await monitor.aclose()
        await monitor.aclose()
        assert provider.close_count == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["nvml", "nvidia-smi"])
def test_explicit_modes_do_not_fallback(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(monitor_module, "SystemMetricsCollector", _FakeCollector)
        calls: list[str] = []

        def unavailable() -> HardwareProviderUnavailable:
            return HardwareProviderUnavailable("missing", "detail")

        monitor = HardwareMonitor(
            HardwareConfig(preferred_provider=mode),  # type: ignore[arg-type]
            _store(),
            lambda state: None,
            nvml_factory=_factory(unavailable, calls, "nvml"),
            nvidia_smi_factory=_factory(unavailable, calls, "smi"),
        )
        assert monitor.begin_refresh()
        result = await monitor.refresh()
        assert not result.success
        assert monitor.state.hardware_status is HardwareStatus.UNAVAILABLE
        assert monitor.state.hardware_snapshot is not None
        assert monitor.state.hardware_snapshot.cpu.utilisation_percent == 20
        assert calls == (["nvml"] if mode == "nvml" else ["smi"])
        await monitor.aclose()

    asyncio.run(scenario())


def test_status_transitions_retain_snapshot_and_server_fields_on_failures() -> None:
    async def scenario() -> None:
        store = _store()
        provider = _FakeProvider(
            "fixture",
            [
                _snapshot(),
                _snapshot(error="Partial GPU metrics available", second=1),
                HardwareCollectionError("Collection failed", "private detail"),
                HardwareProviderUnavailable("Provider unavailable", "driver detail"),
            ],
        )
        states: list[ApplicationState] = []
        monitor = HardwareMonitor(
            HardwareConfig(), store, states.append, provider=provider
        )
        assert monitor.begin_refresh()
        assert store.state.hardware_is_refreshing
        assert (await monitor.refresh()).success
        assert store.state.hardware_status is HardwareStatus.AVAILABLE
        available_snapshot = store.state.hardware_snapshot

        assert monitor.begin_refresh()
        assert store.state.hardware_status is HardwareStatus.AVAILABLE
        assert (await monitor.refresh()).success
        assert store.state.hardware_status is HardwareStatus.DEGRADED
        degraded_snapshot = store.state.hardware_snapshot
        assert degraded_snapshot is not available_snapshot

        assert monitor.begin_refresh()
        failure = await monitor.refresh()
        assert not failure.success
        assert store.state.hardware_status is HardwareStatus.ERROR
        assert store.state.hardware_snapshot is degraded_snapshot
        assert store.state.hardware_last_error == "Collection failed"

        assert monitor.begin_refresh()
        await monitor.refresh()
        assert store.state.hardware_status is HardwareStatus.UNAVAILABLE
        assert store.state.hardware_snapshot is degraded_snapshot
        assert store.state.selected_server_id == "server"
        assert store.state.server_status is ServerStatus.CONNECTING
        assert all(state.selected_server_id == "server" for state in states)
        await monitor.aclose()

    asyncio.run(scenario())


def test_reservation_rejects_overlap_and_unreserved_refresh_skips() -> None:
    async def scenario() -> None:
        provider = _FakeProvider("fixture", [_snapshot()], gate=True)
        monitor = HardwareMonitor(
            HardwareConfig(), _store(), lambda state: None, provider=provider
        )
        skipped = await monitor.refresh()
        assert skipped.skipped
        assert provider.collect_count == 0
        assert monitor.begin_refresh()
        assert not monitor.begin_refresh()
        task = asyncio.create_task(monitor.refresh())
        await provider.started.wait()
        assert not monitor.begin_refresh()
        overlap = await monitor.refresh()
        assert overlap.skipped
        provider.release.set()
        assert (await task).success
        assert provider.collect_count == 1
        await monitor.aclose()

    asyncio.run(scenario())


def test_unexpected_error_is_sanitized_and_gpu_count_logs_only_changes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        caplog.set_level(logging.INFO, logger="modeltop.services.hardware_monitor")
        provider = _FakeProvider(
            "fixture",
            [
                _snapshot(gpu_count=1),
                _snapshot(gpu_count=1, second=1),
                _snapshot(gpu_count=2, second=2),
                RuntimeError("secret internal failure"),
            ],
        )
        monitor = HardwareMonitor(
            HardwareConfig(), _store(), lambda state: None, provider=provider
        )
        for _ in range(4):
            assert monitor.begin_refresh()
            await monitor.refresh()
        assert caplog.text.count("Detected GPU count changed") == 2
        assert "secret internal failure" in caplog.text
        assert monitor.state.hardware_last_error == (
            "Unexpected hardware collection failure"
        )
        await monitor.aclose()

    asyncio.run(scenario())


def test_disabled_monitor_never_reserves_or_closes_unselected_provider() -> None:
    async def scenario() -> None:
        provider = _FakeProvider("fixture", [_snapshot()])
        monitor = HardwareMonitor(
            HardwareConfig(enabled=False),
            _store(),
            lambda state: None,
            provider=provider,
        )
        assert not monitor.begin_refresh()
        assert (await monitor.refresh()).skipped
        assert provider.collect_count == 0
        await monitor.aclose()
        assert provider.close_count == 0

    asyncio.run(scenario())
