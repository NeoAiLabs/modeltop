"""Hardware provider contracts and local system metrics collection."""

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar, cast

import psutil

from modeltop.hardware.models import CpuMetrics, HardwareSnapshot, MemoryMetrics

logger = logging.getLogger(__name__)
_T = TypeVar("_T")
_EXPECTED_SYSTEM_ERRORS = (psutil.Error, OSError, RuntimeError)


class HardwareProvider(ABC):
    """One local hardware snapshot provider."""

    name: str

    @abstractmethod
    async def collect(self) -> HardwareSnapshot:
        """Collect one coherent hardware snapshot."""

    @abstractmethod
    async def close(self) -> None:
        """Release provider resources idempotently."""


class HardwareProviderUnavailable(Exception):
    """A selected provider cannot operate on this machine."""

    def __init__(self, user_message: str, detail: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


class HardwareCollectionError(Exception):
    """One hardware collection attempt failed."""

    def __init__(self, user_message: str, detail: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


@dataclass(frozen=True, slots=True)
class SystemMetricsResult:
    """Local CPU and memory sample with degradation information."""

    cpu: CpuMetrics
    memory: MemoryMetrics
    partial: bool


class SystemMetricsCollector:
    """Collect non-blocking psutil metrics on one owning thread."""

    def __init__(self) -> None:
        self._owner_thread_id = threading.get_ident()
        self._prime_failed = False
        try:
            psutil.cpu_percent(interval=None)
        except _EXPECTED_SYSTEM_ERRORS as error:
            self._prime_failed = True
            logger.warning("Unable to prime CPU metrics: %s", error)
        except Exception:
            self._prime_failed = True
            logger.exception("Unexpected failure while priming CPU metrics")

    def collect(self) -> SystemMetricsResult:
        """Collect one sample without blocking or changing threads."""
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError("System metrics must be collected on the owner thread")

        partial = self._prime_failed
        utilisation, failed = self._collect_value(
            "CPU utilisation", lambda: float(psutil.cpu_percent(interval=None))
        )
        partial |= failed
        logical_count, failed = self._collect_count(True)
        partial |= failed
        physical_count, failed = self._collect_count(False)
        partial |= failed
        loads, failed = self._collect_load_average()
        partial |= failed
        memory, failed = self._collect_memory()
        partial |= failed

        load_1m, load_5m, load_15m = loads if loads is not None else (None, None, None)
        return SystemMetricsResult(
            cpu=CpuMetrics(
                utilisation_percent=utilisation,
                logical_core_count=logical_count,
                physical_core_count=physical_count,
                load_average_1m=load_1m,
                load_average_5m=load_5m,
                load_average_15m=load_15m,
            ),
            memory=memory,
            partial=partial,
        )

    def _collect_count(self, logical: bool) -> tuple[int | None, bool]:
        value, failed = self._collect_value(
            f"{'logical' if logical else 'physical'} CPU count",
            lambda: psutil.cpu_count(logical=logical),
        )
        return value, failed or value is None

    def _collect_memory(self) -> tuple[MemoryMetrics, bool]:
        value, failed = self._collect_value("memory", psutil.virtual_memory)
        if value is None:
            return MemoryMetrics(None, None, None), True
        try:
            total = int(value.total)
            available = int(value.available)
            return (
                MemoryMetrics(
                    used_bytes=total - available,
                    total_bytes=total,
                    utilisation_percent=float(value.percent),
                ),
                failed,
            )
        except _EXPECTED_SYSTEM_ERRORS as error:
            logger.warning("Unable to collect memory metrics: %s", error)
        except Exception:
            logger.exception("Unexpected failure while reading memory metrics")
        return MemoryMetrics(None, None, None), True

    def _collect_load_average(
        self,
    ) -> tuple[tuple[float, float, float] | None, bool]:
        getloadavg = cast(
            "Callable[[], tuple[float, float, float]] | None",
            getattr(psutil, "getloadavg", None),
        )
        if getloadavg is None:
            return None, True
        value, failed = self._collect_value("load average", getloadavg)
        if value is None:
            return None, True
        return (float(value[0]), float(value[1]), float(value[2])), failed

    def _collect_value(
        self, label: str, operation: Callable[[], _T]
    ) -> tuple[_T | None, bool]:
        try:
            return operation(), False
        except _EXPECTED_SYSTEM_ERRORS as error:
            logger.warning("Unable to collect %s: %s", label, error)
        except Exception:
            logger.exception("Unexpected failure while collecting %s", label)
        return None, True
