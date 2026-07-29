"""CPU and memory fallback for unavailable GPU providers."""

from datetime import UTC, datetime

from modeltop.hardware.base import HardwareProvider, SystemMetricsCollector
from modeltop.hardware.models import HardwareSnapshot


class UnavailableHardwareProvider(HardwareProvider):
    """Collect system metrics when no configured GPU provider can operate."""

    name = "unavailable"

    def __init__(self, reason: str, system_collector: SystemMetricsCollector) -> None:
        self._reason = reason
        self._system_collector = system_collector
        self._closed = False

    async def collect(self) -> HardwareSnapshot:
        """Return useful CPU and RAM values with a stable unavailable reason."""
        system = self._system_collector.collect()
        return HardwareSnapshot(
            provider_name=self.name,
            gpus=(),
            cpu=system.cpu,
            memory=system.memory,
            collected_at=datetime.now(UTC),
            error=self._reason,
        )

    async def close(self) -> None:
        """Close idempotently; this provider owns no external resource."""
        self._closed = True
