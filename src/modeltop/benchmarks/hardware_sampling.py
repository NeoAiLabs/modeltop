"""Shared sampling of fresh cached local-hardware snapshots."""

import asyncio
import logging
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime

from modeltop.hardware.models import HardwareSnapshot

logger = logging.getLogger(__name__)


async def sample_hardware_snapshots(
    *,
    read_snapshot: Callable[[], HardwareSnapshot | None],
    measured_phase_started_at: datetime,
    stop_event: asyncio.Event,
    interval_seconds: float,
    append_sample: Callable[[HardwareSnapshot], None],
) -> None:
    """Append fresh unique cached samples until the measured phase stops."""
    seen: set[datetime] = set()
    while not stop_event.is_set():
        try:
            sample = read_snapshot()
            if (
                sample is not None
                and sample.collected_at >= measured_phase_started_at
                and sample.collected_at not in seen
            ):
                seen.add(sample.collected_at)
                append_sample(sample)
        except Exception:
            logger.warning("Benchmark hardware snapshot read failed")
        with suppress(TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
