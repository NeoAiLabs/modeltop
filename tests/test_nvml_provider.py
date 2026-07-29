"""Deterministic NVML provider tests with a structural fake binding."""

import asyncio
import logging
import threading
from types import SimpleNamespace
from typing import cast

import pytest

import modeltop.hardware.nvml as nvml_module
from modeltop.hardware.base import (
    HardwareCollectionError,
    HardwareProviderUnavailable,
    SystemMetricsCollector,
    SystemMetricsResult,
)
from modeltop.hardware.models import CpuMetrics, MemoryMetrics
from modeltop.hardware.nvml import NvmlHardwareProvider


class _NvmlError(Exception):
    pass


class _LibraryNotFound(_NvmlError):
    pass


class _DriverNotLoaded(_NvmlError):
    pass


class _VersionMismatch(_NvmlError):
    pass


class _NoPermission(_NvmlError):
    pass


class _NotSupported(_NvmlError):
    pass


class _FunctionNotFound(_NvmlError):
    pass


class _GpuIsLost(_NvmlError):
    pass


class _FakeBinding:
    NVMLError = _NvmlError
    NVMLError_LibraryNotFound = _LibraryNotFound
    NVMLError_DriverNotLoaded = _DriverNotLoaded
    NVMLError_LibRmVersionMismatch = _VersionMismatch
    NVMLError_NoPermission = _NoPermission
    NVMLError_NotSupported = _NotSupported
    NVMLError_FunctionNotFound = _FunctionNotFound
    NVMLError_GpuIsLost = _GpuIsLost
    NVML_TEMPERATURE_GPU = 0

    def __init__(self) -> None:
        self.devices: list[dict[str, object]] = [
            {
                "name": b"NVIDIA A",
                "uuid": b"gpu-0",
                "utilisation": 25,
                "used": 2 * 1024**3,
                "total": 8 * 1024**3,
                "temperature": 55,
                "draw": 125_500,
                "limit": 250_000,
                "fan": 120,
            }
        ]
        self.fail_init: Exception | None = None
        self.failures: dict[tuple[int, str], Exception] = {}
        self.init_count = 0
        self.shutdown_count = 0
        self.temperature_sensors: list[int] = []
        self.block_started: threading.Event | None = None
        self.block_release: threading.Event | None = None

    def nvmlInit(self) -> None:
        self.init_count += 1
        if self.fail_init is not None:
            raise self.fail_init

    def nvmlShutdown(self) -> None:
        self.shutdown_count += 1

    def nvmlDeviceGetCount(self) -> int:
        return len(self.devices)

    def nvmlDeviceGetHandleByIndex(self, index: int) -> object:
        if self.block_started is not None and self.block_release is not None:
            self.block_started.set()
            self.block_release.wait()
        self._raise(index, "handle")
        return index

    def nvmlDeviceGetName(self, handle: object) -> str | bytes:
        index = cast(int, handle)
        self._raise(index, "name")
        return cast(str | bytes, self.devices[index]["name"])

    def nvmlDeviceGetUUID(self, handle: object) -> str | bytes:
        index = cast(int, handle)
        self._raise(index, "uuid")
        return cast(str | bytes, self.devices[index]["uuid"])

    def nvmlDeviceGetUtilizationRates(self, handle: object) -> SimpleNamespace:
        index = cast(int, handle)
        self._raise(index, "utilisation")
        return SimpleNamespace(gpu=self.devices[index]["utilisation"])

    def nvmlDeviceGetMemoryInfo(self, handle: object) -> SimpleNamespace:
        index = cast(int, handle)
        self._raise(index, "memory")
        return SimpleNamespace(
            used=self.devices[index]["used"], total=self.devices[index]["total"]
        )

    def nvmlDeviceGetTemperatureV(self, handle: object, sensor: int) -> int:
        index = cast(int, handle)
        self.temperature_sensors.append(sensor)
        self._raise(index, "temperature")
        return cast(int, self.devices[index]["temperature"])

    def nvmlDeviceGetPowerUsage(self, handle: object) -> int:
        index = cast(int, handle)
        self._raise(index, "power draw")
        return cast(int, self.devices[index]["draw"])

    def nvmlDeviceGetEnforcedPowerLimit(self, handle: object) -> int:
        index = cast(int, handle)
        self._raise(index, "power limit")
        return cast(int, self.devices[index]["limit"])

    def nvmlDeviceGetFanSpeed(self, handle: object) -> int:
        index = cast(int, handle)
        self._raise(index, "fan speed")
        return cast(int, self.devices[index]["fan"])

    def _raise(self, index: int, metric: str) -> None:
        failure = self.failures.get((index, metric))
        if failure is not None:
            raise failure


class _FakeSystemCollector:
    def __init__(self, *, partial: bool = False) -> None:
        self.partial = partial
        self.calls = 0
        self.thread_ids: list[int] = []

    def collect(self) -> SystemMetricsResult:
        self.calls += 1
        self.thread_ids.append(threading.get_ident())
        return SystemMetricsResult(
            CpuMetrics(10.0, 8, 4, 1.0, 2.0, 3.0),
            MemoryMetrics(4, 8, 50.0),
            self.partial,
        )


def _install(
    monkeypatch: pytest.MonkeyPatch, binding: _FakeBinding
) -> _FakeSystemCollector:
    monkeypatch.setattr(nvml_module, "_nvml", binding)
    return _FakeSystemCollector()


def test_successful_one_and_two_gpu_extraction_and_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        binding = _FakeBinding()
        binding.devices.append(
            {
                "name": "NVIDIA B",
                "uuid": "gpu-1",
                "utilisation": 75,
                "used": 3,
                "total": 9,
                "temperature": 65,
                "draw": 50_000,
                "limit": 100_000,
                "fan": 40,
            }
        )
        collector = _install(monkeypatch, binding)
        owner = threading.get_ident()
        provider = await NvmlHardwareProvider.create(
            cast(SystemMetricsCollector, collector)
        )
        snapshot = await provider.collect()
        assert provider.name == "NVML"
        assert len(snapshot.gpus) == 2
        assert snapshot.gpus[0].name == "NVIDIA A"
        assert snapshot.gpus[0].uuid == "gpu-0"
        assert snapshot.gpus[0].power_draw_watts == 125.5
        assert snapshot.gpus[0].power_limit_watts == 250.0
        assert snapshot.gpus[0].fan_speed_percent == 120.0
        assert snapshot.gpus[1].name == "NVIDIA B"
        assert snapshot.error is None
        assert binding.temperature_sensors == [0, 0]
        assert collector.thread_ids == [owner]
        await provider.close()
        await provider.close()
        assert binding.init_count == 1
        assert binding.shutdown_count == 1

    asyncio.run(scenario())


def test_capability_gaps_warn_once_and_system_partial_combines(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def scenario() -> None:
        binding = _FakeBinding()
        binding.failures[(0, "memory")] = _NotSupported("missing")
        binding.failures[(0, "fan speed")] = _NoPermission("denied")
        collector = _install(monkeypatch, binding)
        collector.partial = True
        caplog.set_level(logging.WARNING, logger="modeltop.hardware.nvml")
        provider = await NvmlHardwareProvider.create(
            cast(SystemMetricsCollector, collector)
        )
        first = await provider.collect()
        second = await provider.collect()
        assert first.error == "Partial hardware metrics available"
        assert first.gpus[0].memory_used_bytes is None
        assert first.gpus[0].memory_total_bytes is None
        assert second.gpus[0].fan_speed_percent is None
        assert caplog.text.count("GPU 0 memory unavailable") == 1
        assert caplog.text.count("GPU 0 fan speed unavailable") == 1
        await provider.close()

    asyncio.run(scenario())


def test_fatal_lost_gpu_and_zero_devices_during_refresh_retain_readable_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        binding = _FakeBinding()
        collector = _install(monkeypatch, binding)
        provider = await NvmlHardwareProvider.create(
            cast(SystemMetricsCollector, collector)
        )
        binding.failures[(0, "utilisation")] = _GpuIsLost("lost")
        with pytest.raises(HardwareCollectionError, match="NVML collection failed"):
            await provider.collect()
        binding.failures.clear()
        binding.devices.clear()
        with pytest.raises(HardwareCollectionError, match="No NVIDIA GPU detected"):
            await provider.collect()
        await provider.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (_LibraryNotFound("missing"), "NVML unavailable"),
        (_DriverNotLoaded("driver"), "NVIDIA driver unavailable"),
        (_VersionMismatch("version"), "NVIDIA driver unavailable"),
        (_NoPermission("denied"), "NVML permission denied"),
        (_NvmlError("other"), "NVML initialisation failed"),
        (ValueError("unexpected"), "NVML initialisation failed"),
    ],
)
def test_initialization_failure_mappings(
    monkeypatch: pytest.MonkeyPatch, failure: Exception, message: str
) -> None:
    async def scenario() -> None:
        binding = _FakeBinding()
        binding.fail_init = failure
        collector = _install(monkeypatch, binding)
        with pytest.raises(HardwareProviderUnavailable, match=message) as caught:
            await NvmlHardwareProvider.create(cast(SystemMetricsCollector, collector))
        assert str(caught.value) == message
        assert "unexpected" not in caught.value.user_message
        assert binding.shutdown_count == 0

    asyncio.run(scenario())


def test_zero_devices_at_startup_shuts_down_before_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        binding = _FakeBinding()
        binding.devices.clear()
        collector = _install(monkeypatch, binding)
        with pytest.raises(HardwareProviderUnavailable, match="No NVIDIA GPU detected"):
            await NvmlHardwareProvider.create(cast(SystemMetricsCollector, collector))
        assert binding.shutdown_count == 1

    asyncio.run(scenario())


def test_cancellation_waits_for_thread_before_guarded_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        binding = _FakeBinding()
        binding.block_started = threading.Event()
        binding.block_release = threading.Event()
        collector = _install(monkeypatch, binding)
        provider = await NvmlHardwareProvider.create(
            cast(SystemMetricsCollector, collector)
        )
        task = asyncio.create_task(provider.collect())
        while not binding.block_started.is_set():
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        binding.block_release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await provider.close()
        assert binding.shutdown_count == 1

    asyncio.run(scenario())
