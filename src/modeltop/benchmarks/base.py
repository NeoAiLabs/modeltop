"""Reusable contracts for asynchronous benchmarks."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from modeltop.hardware.models import HardwareSnapshot


@dataclass(frozen=True, slots=True)
class BenchmarkContext:
    """Runtime facilities shared by benchmark implementations."""

    benchmark_id: str
    started_at: datetime
    monotonic_clock: Callable[[], float]
    utc_now: Callable[[], datetime]
    read_hardware_snapshot: Callable[[], HardwareSnapshot | None]


class Benchmark[ResultT](ABC):
    """Interface implemented by cancellable asynchronous benchmarks."""

    @abstractmethod
    async def run(self, context: BenchmarkContext) -> ResultT:
        """Run the benchmark to a terminal result."""

    @abstractmethod
    def request_cancellation(self) -> None:
        """Request cooperative cancellation of the benchmark."""
