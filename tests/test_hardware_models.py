"""Hardware value, aggregation, and formatting tests."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from modeltop.hardware.models import (
    CpuMetrics,
    GpuMetrics,
    HardwareSnapshot,
    MemoryMetrics,
    format_byte_pair,
    format_bytes,
    format_elapsed,
    format_percentage,
    format_temperature,
    format_watts,
    summarize_gpus,
    truncate_device_name,
)


def _gpu(
    index: int,
    name: str,
    *,
    utilisation: float | None = 20.0,
    used: int | None = 2 * 1024**3,
    total: int | None = 8 * 1024**3,
    temperature: float | None = 50.0,
    draw: float | None = 100.0,
    limit: float | None = 200.0,
) -> GpuMetrics:
    return GpuMetrics(
        index=index,
        name=name,
        uuid=f"gpu-{index}",
        utilisation_percent=utilisation,
        memory_used_bytes=used,
        memory_total_bytes=total,
        temperature_celsius=temperature,
        power_draw_watts=draw,
        power_limit_watts=limit,
        fan_speed_percent=30.0,
    )


def test_byte_and_pair_formatting_boundaries_and_missing_sides() -> None:
    assert format_bytes(None) == "--"
    assert format_bytes(512 * 1024) == "0.5 MB"
    assert format_bytes(1024**2) == "1.0 MB"
    assert format_bytes(1024**3) == "1.0 GB"
    assert format_bytes(2 * 1024**4) == "2.0 TB"
    assert format_byte_pair(None, None) == "--"
    used = int(18.2 * 1024**3)
    assert format_byte_pair(used, 24 * 1024**3) == "18.2 / 24.0 GB"
    assert format_byte_pair(used, 24 * 1024**3, compact=True) == ("18.2/24.0 GB")
    assert format_byte_pair(None, 24 * 1024**3) == "-- / 24.0 GB"
    assert format_byte_pair(512 * 1024**2, None) == "512.0 / -- MB"


def test_scalar_elapsed_and_truncation_formatting_boundaries() -> None:
    assert format_percentage(None) == "--"
    assert format_temperature(None) == "--"
    assert format_watts(None) == "--"
    assert format_percentage(49.6) == "50%"
    assert format_temperature(64.6) == "65°C"
    assert format_watts(249.6) == "250 W"

    now = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)
    assert format_elapsed(None, now=now) == "--"
    assert format_elapsed(now, now=now) == "just now"
    assert format_elapsed(now - timedelta(seconds=59), now=now) == "59s ago"
    assert format_elapsed(now - timedelta(seconds=60), now=now) == "1m ago"
    assert format_elapsed(now - timedelta(hours=2), now=now) == "2h ago"
    assert format_elapsed(now - timedelta(days=3), now=now) == "3d ago"
    assert format_elapsed(now + timedelta(seconds=5), now=now) == "just now"

    assert truncate_device_name("GPU", 3) == "GPU"
    assert truncate_device_name("NVIDIA GB10", 8) == "NVIDIA …"
    assert truncate_device_name("GPU", 1) == "…"
    assert truncate_device_name("GPU", 0) == ""


def test_gpu_aggregation_names_and_complete_numeric_policy() -> None:
    one = summarize_gpus((_gpu(0, "NVIDIA A"),))
    assert one.count == 1
    assert one.display_name == "NVIDIA A"
    assert one.utilisation_percent == 20.0

    identical = summarize_gpus(
        (
            _gpu(
                0,
                "NVIDIA A",
                utilisation=20,
                used=2,
                total=8,
                temperature=50,
                draw=100,
                limit=200,
            ),
            _gpu(
                1,
                "NVIDIA A",
                utilisation=40,
                used=3,
                total=9,
                temperature=60,
                draw=110,
                limit=210,
            ),
        )
    )
    assert identical.display_name == "2 × NVIDIA A"  # noqa: RUF001
    assert identical.utilisation_percent == 30
    assert identical.memory_used_bytes == 5
    assert identical.memory_total_bytes == 17
    assert identical.temperature_celsius == 60
    assert identical.power_draw_watts == 210
    assert identical.power_limit_watts == 410

    mixed = summarize_gpus((_gpu(0, "A"), _gpu(1, "B")))
    assert mixed.display_name == "2 GPUs"


def test_incomplete_aggregates_are_suppressed_independently() -> None:
    summary = summarize_gpus(
        (
            _gpu(0, "A"),
            _gpu(
                1,
                "A",
                utilisation=None,
                used=None,
                total=None,
                temperature=None,
                draw=None,
                limit=None,
            ),
        )
    )
    assert summary.utilisation_percent is None
    assert summary.memory_used_bytes is None
    assert summary.memory_total_bytes is None
    assert summary.temperature_celsius is None
    assert summary.power_draw_watts is None
    assert summary.power_limit_watts is None


def test_snapshot_requires_timezone_aware_utc_collection_time() -> None:
    cpu = CpuMetrics(None, None, None, None, None, None)
    memory = MemoryMetrics(None, None, None)
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        HardwareSnapshot("fixture", (), cpu, memory, datetime(2026, 1, 1), None)
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        HardwareSnapshot(
            "fixture",
            (),
            cpu,
            memory,
            datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=1))),
            None,
        )
