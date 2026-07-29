"""Immutable hardware metrics, aggregation, and display formatting."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

_MEBIBYTE = 1024**2
_GIBIBYTE = 1024**3
_TEBIBYTE = 1024**4


@dataclass(frozen=True, slots=True)
class GpuMetrics:
    """Metrics collected for one local NVIDIA GPU."""

    index: int
    name: str
    uuid: str
    utilisation_percent: float | None
    memory_used_bytes: int | None
    memory_total_bytes: int | None
    temperature_celsius: float | None
    power_draw_watts: float | None
    power_limit_watts: float | None
    fan_speed_percent: float | None


@dataclass(frozen=True, slots=True)
class CpuMetrics:
    """Non-blocking local CPU metrics."""

    utilisation_percent: float | None
    logical_core_count: int | None
    physical_core_count: int | None
    load_average_1m: float | None
    load_average_5m: float | None
    load_average_15m: float | None


@dataclass(frozen=True, slots=True)
class MemoryMetrics:
    """Local system memory metrics."""

    used_bytes: int | None
    total_bytes: int | None
    utilisation_percent: float | None


@dataclass(frozen=True, slots=True)
class HardwareSnapshot:
    """One coherent local hardware sample."""

    provider_name: str
    gpus: tuple[GpuMetrics, ...]
    cpu: CpuMetrics
    memory: MemoryMetrics
    collected_at: datetime
    error: str | None

    def __post_init__(self) -> None:
        offset = self.collected_at.utcoffset()
        if offset is None or offset != timedelta(0):
            raise ValueError("collected_at must be timezone-aware UTC")


@dataclass(frozen=True, slots=True)
class GpuSummary:
    """Display-oriented aggregate of every detected GPU."""

    count: int
    display_name: str
    utilisation_percent: float | None
    memory_used_bytes: int | None
    memory_total_bytes: int | None
    temperature_celsius: float | None
    power_draw_watts: float | None
    power_limit_watts: float | None


def summarize_gpus(gpus: tuple[GpuMetrics, ...]) -> GpuSummary:
    """Aggregate GPUs without presenting incomplete values as complete totals."""
    count = len(gpus)
    if count == 0:
        display_name = "No GPUs"
    elif count == 1:
        display_name = gpus[0].name
    elif all(gpu.name == gpus[0].name for gpu in gpus[1:]):
        display_name = f"{count} × {gpus[0].name}"  # noqa: RUF001
    else:
        display_name = f"{count} GPUs"

    utilisation = _complete_floats(
        tuple(gpu.utilisation_percent for gpu in gpus), "average"
    )
    memory_used = _complete_ints(tuple(gpu.memory_used_bytes for gpu in gpus))
    memory_total = _complete_ints(tuple(gpu.memory_total_bytes for gpu in gpus))
    temperatures = tuple(gpu.temperature_celsius for gpu in gpus)
    temperature = (
        max(value for value in temperatures if value is not None)
        if temperatures and all(value is not None for value in temperatures)
        else None
    )
    power_draw = _complete_floats(tuple(gpu.power_draw_watts for gpu in gpus), "sum")
    power_limit = _complete_floats(tuple(gpu.power_limit_watts for gpu in gpus), "sum")
    return GpuSummary(
        count=count,
        display_name=display_name,
        utilisation_percent=utilisation,
        memory_used_bytes=memory_used,
        memory_total_bytes=memory_total,
        temperature_celsius=temperature,
        power_draw_watts=power_draw,
        power_limit_watts=power_limit,
    )


def _complete_ints(values: tuple[int | None, ...]) -> int | None:
    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _complete_floats(values: tuple[float | None, ...], operation: str) -> float | None:
    if not values or any(value is None for value in values):
        return None
    total = sum(value for value in values if value is not None)
    return total / len(values) if operation == "average" else total


def format_bytes(value: int | None) -> str:
    """Format a byte count using binary MB, GB, or TB."""
    if value is None:
        return "--"
    divisor, unit = _byte_unit(value)
    return f"{value / divisor:.1f} {unit}"


def format_byte_pair(
    used: int | None, total: int | None, *, compact: bool = False
) -> str:
    """Format two byte counts using the largest known side's shared unit."""
    if used is None and total is None:
        return "--"
    reference = max(value for value in (used, total) if value is not None)
    divisor, unit = _byte_unit(reference)
    used_text = "--" if used is None else f"{used / divisor:.1f}"
    total_text = "--" if total is None else f"{total / divisor:.1f}"
    separator = "/" if compact else " / "
    return f"{used_text}{separator}{total_text} {unit}"


def _byte_unit(value: int) -> tuple[int, str]:
    magnitude = abs(value)
    if magnitude >= _TEBIBYTE:
        return _TEBIBYTE, "TB"
    if magnitude >= _GIBIBYTE:
        return _GIBIBYTE, "GB"
    return _MEBIBYTE, "MB"


def format_percentage(value: float | None) -> str:
    """Format a percentage or the shared missing-value marker."""
    return "--" if value is None else f"{value:.0f}%"


def format_temperature(value: float | None) -> str:
    """Format Celsius or the shared missing-value marker."""
    return "--" if value is None else f"{value:.0f}°C"


def format_watts(value: float | None) -> str:
    """Format watts or the shared missing-value marker."""
    return "--" if value is None else f"{value:.0f} W"


def format_elapsed(timestamp: datetime | None, *, now: datetime | None = None) -> str:
    """Format the elapsed whole-unit age of a UTC timestamp."""
    if timestamp is None:
        return "--"
    current = datetime.now(UTC) if now is None else now
    seconds = max(0, int((current - timestamp).total_seconds()))
    if seconds < 1:
        return "just now"
    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    return f"{hours // 24}d ago"


def truncate_device_name(name: str, max_length: int) -> str:
    """Truncate a device name with one ellipsis within the given width."""
    if max_length <= 0:
        return ""
    if len(name) <= max_length:
        return name
    if max_length == 1:
        return "…"
    return f"{name[: max_length - 1]}…"
