"""Independent local hardware monitoring service."""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from modeltop.hardware.base import (
    HardwareCollectionError,
    HardwareProvider,
    HardwareProviderUnavailable,
    SystemMetricsCollector,
)
from modeltop.hardware.nvidia_smi import NvidiaSmiHardwareProvider
from modeltop.hardware.nvml import NvmlHardwareProvider
from modeltop.hardware.unavailable import UnavailableHardwareProvider
from modeltop.models import HardwareConfig
from modeltop.state import (
    ApplicationState,
    ApplicationStateStore,
    HardwareStatus,
)

logger = logging.getLogger(__name__)

type ProviderFactory = Callable[[SystemMetricsCollector], Awaitable[HardwareProvider]]


@dataclass(frozen=True, slots=True)
class HardwareRefreshResult:
    """Summary of one attempted or skipped hardware refresh."""

    success: bool
    skipped: bool
    message: str
    gpu_count: int


class HardwareMonitor:
    """Select one provider and merge hardware-only state transitions."""

    def __init__(
        self,
        config: HardwareConfig,
        state_store: ApplicationStateStore,
        on_state_change: Callable[[ApplicationState], None],
        *,
        provider: HardwareProvider | None = None,
        nvml_factory: ProviderFactory = NvmlHardwareProvider.create,
        nvidia_smi_factory: ProviderFactory = NvidiaSmiHardwareProvider.create,
    ) -> None:
        self._config = config
        self._state_store = state_store
        self._on_state_change = on_state_change
        self._provider = (
            provider
            if config.enabled and config.preferred_provider != "disabled"
            else None
        )
        self._nvml_factory = nvml_factory
        self._nvidia_smi_factory = nvidia_smi_factory
        self._system_collector: SystemMetricsCollector | None = None
        self._refresh_pending = False
        self._selection_logged = False
        self._last_gpu_count: int | None = None
        self._closed = False
        self._provider_closed = False
        self._enabled = config.enabled and config.preferred_provider != "disabled"

    @property
    def state(self) -> ApplicationState:
        """Return the latest shared application state."""
        return self._state_store.state

    @property
    def provider(self) -> HardwareProvider | None:
        """Return the selected provider, if initialization has completed."""
        return self._provider

    def _publish(
        self, transform: Callable[[ApplicationState], ApplicationState]
    ) -> ApplicationState:
        state = self._state_store.update(transform)
        self._on_state_change(state)
        return state

    def begin_refresh(self) -> bool:
        """Synchronously reserve one hardware refresh without overlap."""
        if (
            not self._enabled
            or self._closed
            or self._refresh_pending
            or self.state.hardware_is_refreshing
        ):
            return False
        self._refresh_pending = True
        if self.state.hardware_snapshot is None:
            self._publish(
                lambda state: replace(
                    state,
                    hardware_status=HardwareStatus.INITIALISING,
                    hardware_last_error=None,
                    hardware_is_refreshing=True,
                )
            )
        else:
            self._publish(lambda state: replace(state, hardware_is_refreshing=True))
        return True

    async def refresh(self) -> HardwareRefreshResult:
        """Run one reserved collection and preserve prior data on failure."""
        if not self._refresh_pending:
            return HardwareRefreshResult(
                success=False,
                skipped=True,
                message="Hardware refresh already in progress.",
                gpu_count=self._current_gpu_count(),
            )
        self._refresh_pending = False
        try:
            provider = await self._get_provider()
            snapshot = await provider.collect()
        except HardwareProviderUnavailable as error:
            logger.warning("Hardware provider unavailable: %s", error.detail)
            return self._finish_failure(HardwareStatus.UNAVAILABLE, error.user_message)
        except HardwareCollectionError as error:
            logger.warning("Hardware collection failed: %s", error.detail)
            return self._finish_failure(HardwareStatus.ERROR, error.user_message)
        except Exception:
            logger.exception("Unexpected hardware collection failure")
            return self._finish_failure(
                HardwareStatus.ERROR, "Unexpected hardware collection failure"
            )

        gpu_count = len(snapshot.gpus)
        if self._last_gpu_count != gpu_count:
            logger.info("Detected GPU count changed to %d", gpu_count)
            self._last_gpu_count = gpu_count
        if provider.name == "unavailable":
            status = HardwareStatus.UNAVAILABLE
            success = False
        elif snapshot.error is not None:
            status = HardwareStatus.DEGRADED
            success = True
        else:
            status = HardwareStatus.AVAILABLE
            success = True
        self._publish(
            lambda state: replace(
                state,
                hardware_status=status,
                hardware_snapshot=snapshot,
                hardware_last_refresh_time=snapshot.collected_at,
                hardware_last_error=snapshot.error,
                hardware_is_refreshing=False,
            )
        )
        return HardwareRefreshResult(
            success=success,
            skipped=False,
            message=snapshot.error or "Hardware metrics refreshed.",
            gpu_count=gpu_count,
        )

    async def _get_provider(self) -> HardwareProvider:
        if self._provider is not None:
            if not self._selection_logged:
                logger.info("Selected hardware provider %s", self._provider.name)
                self._selection_logged = True
            return self._provider

        collector = self._system_collector
        if collector is None:
            collector = SystemMetricsCollector()
            self._system_collector = collector

        candidates: tuple[tuple[str, ProviderFactory], ...]
        if self._config.preferred_provider == "auto":
            candidates = (
                ("NVML", self._nvml_factory),
                ("nvidia-smi", self._nvidia_smi_factory),
            )
        elif self._config.preferred_provider == "nvml":
            candidates = (("NVML", self._nvml_factory),)
        else:
            candidates = (("nvidia-smi", self._nvidia_smi_factory),)

        reasons: list[str] = []
        for index, (name, factory) in enumerate(candidates):
            try:
                provider = await factory(collector)
            except HardwareProviderUnavailable as error:
                reasons.append(error.user_message)
                logger.warning(
                    "Hardware provider %s unavailable: %s", name, error.detail
                )
                if index + 1 < len(candidates):
                    logger.info(
                        "Falling back from %s to %s", name, candidates[index + 1][0]
                    )
                continue
            logger.info("Initialised hardware provider %s", name)
            self._provider = provider
            logger.info("Selected hardware provider %s", provider.name)
            self._selection_logged = True
            return provider

        reason = "; ".join(reasons) or "Hardware monitoring unavailable"
        provider = UnavailableHardwareProvider(reason, collector)
        self._provider = provider
        logger.info("Selected hardware provider %s: %s", provider.name, reason)
        self._selection_logged = True
        return provider

    def _finish_failure(
        self, status: HardwareStatus, message: str
    ) -> HardwareRefreshResult:
        attempt_time = datetime.now(UTC)
        state = self._publish(
            lambda state: replace(
                state,
                hardware_status=status,
                hardware_last_refresh_time=attempt_time,
                hardware_last_error=message,
                hardware_is_refreshing=False,
            )
        )
        gpu_count = (
            len(state.hardware_snapshot.gpus)
            if state.hardware_snapshot is not None
            else 0
        )
        return HardwareRefreshResult(
            success=False,
            skipped=False,
            message=message,
            gpu_count=gpu_count,
        )

    def _current_gpu_count(self) -> int:
        snapshot = self.state.hardware_snapshot
        return 0 if snapshot is None else len(snapshot.gpus)

    async def aclose(self) -> None:
        """Close the selected or injected provider exactly once."""
        if self._closed:
            return
        self._closed = True
        provider = self._provider
        if provider is not None and not self._provider_closed:
            self._provider_closed = True
            await provider.close()
            logger.info("Hardware provider %s closed", provider.name)
        logger.info("Hardware monitor shut down")
