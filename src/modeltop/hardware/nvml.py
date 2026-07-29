"""NVIDIA Management Library hardware provider."""

import asyncio
import logging
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from typing import Protocol, cast

import pynvml  # pyright: ignore[reportMissingTypeStubs]

from modeltop.hardware.base import (
    HardwareCollectionError,
    HardwareProvider,
    HardwareProviderUnavailable,
    SystemMetricsCollector,
)
from modeltop.hardware.models import GpuMetrics, HardwareSnapshot

logger = logging.getLogger(__name__)


class _Utilization(Protocol):
    gpu: int


class _MemoryInfo(Protocol):
    used: int
    total: int


class _NvmlBinding(Protocol):
    NVMLError: type[Exception]
    NVMLError_LibraryNotFound: type[Exception]
    NVMLError_DriverNotLoaded: type[Exception]
    NVMLError_LibRmVersionMismatch: type[Exception]
    NVMLError_NoPermission: type[Exception]
    NVMLError_NotSupported: type[Exception]
    NVMLError_FunctionNotFound: type[Exception]
    NVMLError_GpuIsLost: type[Exception]
    NVML_TEMPERATURE_GPU: int

    def nvmlInit(self) -> None: ...
    def nvmlShutdown(self) -> None: ...
    def nvmlDeviceGetCount(self) -> int: ...
    def nvmlDeviceGetHandleByIndex(self, index: int) -> object: ...
    def nvmlDeviceGetName(self, handle: object) -> str | bytes: ...
    def nvmlDeviceGetUUID(self, handle: object) -> str | bytes: ...
    def nvmlDeviceGetUtilizationRates(self, handle: object) -> _Utilization: ...
    def nvmlDeviceGetMemoryInfo(self, handle: object) -> _MemoryInfo: ...
    def nvmlDeviceGetTemperatureV(self, handle: object, sensor: int) -> int: ...
    def nvmlDeviceGetPowerUsage(self, handle: object) -> int: ...
    def nvmlDeviceGetEnforcedPowerLimit(self, handle: object) -> int: ...
    def nvmlDeviceGetFanSpeed(self, handle: object) -> int: ...


_nvml = cast(_NvmlBinding, pynvml)


async def _shielded_thread[T](operation: Callable[[], T]) -> T:
    task = asyncio.create_task(asyncio.to_thread(operation))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        with suppress(Exception):
            await task
        raise


class NvmlHardwareProvider(HardwareProvider):
    """Collect NVIDIA GPU metrics through one long-lived NVML session."""

    name = "NVML"

    def __init__(
        self,
        system_collector: SystemMetricsCollector,
        binding: _NvmlBinding,
    ) -> None:
        self._system_collector = system_collector
        self._binding = binding
        self._lock = asyncio.Lock()
        self._initialized = True
        self._closed = False
        self._warned_metrics: set[tuple[int, str]] = set()

    @classmethod
    async def create(
        cls, system_collector: SystemMetricsCollector
    ) -> "NvmlHardwareProvider":
        """Initialize NVML once and reject machines without NVIDIA GPUs."""
        binding = _nvml
        initialized = False

        def initialize() -> int:
            nonlocal initialized
            binding.nvmlInit()
            initialized = True
            try:
                return int(binding.nvmlDeviceGetCount())
            except Exception:
                try:
                    binding.nvmlShutdown()
                finally:
                    initialized = False
                raise

        task = asyncio.create_task(asyncio.to_thread(initialize))
        try:
            count = await asyncio.shield(task)
        except asyncio.CancelledError:
            with suppress(Exception):
                await task
            if initialized:
                try:
                    await _shielded_thread(binding.nvmlShutdown)
                except Exception:
                    logger.exception(
                        "NVML shutdown failed after cancelled initialisation"
                    )
            raise
        except Exception as error:
            raise cls._initialization_error(binding, error) from error

        if count == 0:
            try:
                await _shielded_thread(binding.nvmlShutdown)
            except binding.NVMLError as error:
                logger.warning("NVML shutdown after zero devices failed: %s", error)
            raise HardwareProviderUnavailable(
                "No NVIDIA GPU detected", "NVML reported zero devices"
            )

        logger.info("NVML initialised with %d GPU(s)", count)
        return cls(system_collector, binding)

    @staticmethod
    def _initialization_error(
        binding: _NvmlBinding, error: Exception
    ) -> HardwareProviderUnavailable:
        detail = f"{type(error).__name__}: {error}"
        if isinstance(error, binding.NVMLError_LibraryNotFound):
            message = "NVML unavailable"
        elif isinstance(
            error,
            (
                binding.NVMLError_DriverNotLoaded,
                binding.NVMLError_LibRmVersionMismatch,
            ),
        ):
            message = "NVIDIA driver unavailable"
        elif isinstance(error, binding.NVMLError_NoPermission):
            message = "NVML permission denied"
        elif isinstance(error, binding.NVMLError):
            message = "NVML initialisation failed"
        else:
            message = "NVML initialisation failed"
        logger.warning("NVML initialisation failed: %s", detail)
        return HardwareProviderUnavailable(message, detail)

    async def collect(self) -> HardwareSnapshot:
        """Collect system metrics on-loop and NVML metrics off-loop."""
        system = self._system_collector.collect()
        async with self._lock:
            if self._closed or not self._initialized:
                raise HardwareProviderUnavailable(
                    "NVML unavailable", "NVML provider is closed"
                )
            gpus, gpu_partial = await _shielded_thread(self._collect_gpus)

        if gpu_partial and system.partial:
            error = "Partial hardware metrics available"
        elif gpu_partial:
            error = "Partial GPU metrics available"
        elif system.partial:
            error = "Partial system metrics available"
        else:
            error = None
        return HardwareSnapshot(
            provider_name=self.name,
            gpus=gpus,
            cpu=system.cpu,
            memory=system.memory,
            collected_at=datetime.now(UTC),
            error=error,
        )

    def _collect_gpus(self) -> tuple[tuple[GpuMetrics, ...], bool]:
        try:
            count = int(self._binding.nvmlDeviceGetCount())
            if count == 0:
                raise HardwareCollectionError(
                    "No NVIDIA GPU detected",
                    "NVML reported zero devices during refresh",
                )
            gpus: list[GpuMetrics] = []
            partial = False
            for index in range(count):
                handle = self._binding.nvmlDeviceGetHandleByIndex(index)
                name = self._normalize_text(self._binding.nvmlDeviceGetName(handle))
                uuid = self._normalize_text(self._binding.nvmlDeviceGetUUID(handle))

                utilisation, missing = self._optional_metric(
                    index,
                    "utilisation",
                    lambda handle=handle: float(
                        self._binding.nvmlDeviceGetUtilizationRates(handle).gpu
                    ),
                )
                partial |= missing
                memory, missing = self._optional_metric(
                    index,
                    "memory",
                    lambda handle=handle: self._binding.nvmlDeviceGetMemoryInfo(handle),
                )
                partial |= missing
                temperature, missing = self._optional_metric(
                    index,
                    "temperature",
                    lambda handle=handle: float(
                        self._binding.nvmlDeviceGetTemperatureV(
                            handle, self._binding.NVML_TEMPERATURE_GPU
                        )
                    ),
                )
                partial |= missing
                power_draw, missing = self._optional_metric(
                    index,
                    "power draw",
                    lambda handle=handle: (
                        float(self._binding.nvmlDeviceGetPowerUsage(handle)) / 1000
                    ),
                )
                partial |= missing
                power_limit, missing = self._optional_metric(
                    index,
                    "power limit",
                    lambda handle=handle: (
                        float(self._binding.nvmlDeviceGetEnforcedPowerLimit(handle))
                        / 1000
                    ),
                )
                partial |= missing
                fan_speed, missing = self._optional_metric(
                    index,
                    "fan speed",
                    lambda handle=handle: float(
                        self._binding.nvmlDeviceGetFanSpeed(handle)
                    ),
                )
                partial |= missing
                gpus.append(
                    GpuMetrics(
                        index=index,
                        name=name,
                        uuid=uuid,
                        utilisation_percent=utilisation,
                        memory_used_bytes=None if memory is None else int(memory.used),
                        memory_total_bytes=None
                        if memory is None
                        else int(memory.total),
                        temperature_celsius=temperature,
                        power_draw_watts=power_draw,
                        power_limit_watts=power_limit,
                        fan_speed_percent=fan_speed,
                    )
                )
            return tuple(gpus), partial
        except HardwareCollectionError:
            raise
        except self._binding.NVMLError as error:
            detail = f"{type(error).__name__}: {error}"
            raise HardwareCollectionError("NVML collection failed", detail) from error

    def _optional_metric[T](
        self, index: int, metric: str, operation: Callable[[], T]
    ) -> tuple[T | None, bool]:
        try:
            return operation(), False
        except (
            self._binding.NVMLError_NotSupported,
            self._binding.NVMLError_FunctionNotFound,
            self._binding.NVMLError_NoPermission,
        ) as error:
            key = (index, metric)
            if key not in self._warned_metrics:
                self._warned_metrics.add(key)
                logger.warning(
                    "GPU %d %s unavailable through NVML: %s",
                    index,
                    metric,
                    type(error).__name__,
                )
            return None, True

    @staticmethod
    def _normalize_text(value: str | bytes) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    async def close(self) -> None:
        """Wait for collection to unwind and shut NVML down exactly once."""
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            if not self._initialized:
                return
            try:
                await _shielded_thread(self._binding.nvmlShutdown)
            except self._binding.NVMLError as error:
                logger.warning("NVML shutdown failed: %s", error)
            finally:
                self._initialized = False
            logger.info("NVML provider shut down")
