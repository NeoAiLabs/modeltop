"""Non-blocking nvidia-smi hardware provider."""

import asyncio
import csv
import shutil
from contextlib import suppress
from datetime import UTC, datetime
from io import StringIO

from modeltop.hardware.base import (
    HardwareCollectionError,
    HardwareProvider,
    HardwareProviderUnavailable,
    SystemMetricsCollector,
)
from modeltop.hardware.models import GpuMetrics, HardwareSnapshot

_QUERY_ARGUMENTS = (
    "nvidia-smi",
    "--query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,fan.speed",
    "--format=csv,noheader,nounits",
)
_MEBIBYTE = 1024**2
_MAX_OUTPUT_BYTES = 1024**2
_TERMINATE_GRACE_SECONDS = 0.2
_MISSING_VALUES = {"", "N/A", "[N/A]"}


def parse_nvidia_smi_csv(output: str) -> tuple[GpuMetrics, ...]:
    """Parse the exact ten-column query emitted by this provider."""
    if "\ufffd" in output:
        raise HardwareCollectionError(
            "Invalid nvidia-smi output", "Output contains invalid UTF-8"
        )
    try:
        rows = list(csv.reader(StringIO(output), skipinitialspace=True))
    except csv.Error as error:
        raise HardwareCollectionError(
            "Invalid nvidia-smi output", f"CSV parsing failed: {error}"
        ) from error
    rows = [row for row in rows if row]
    if not rows:
        raise HardwareCollectionError(
            "No NVIDIA GPU detected", "nvidia-smi returned no GPU rows"
        )

    gpus: list[GpuMetrics] = []
    for row_number, row in enumerate(rows, start=1):
        if len(row) != 10:
            raise HardwareCollectionError(
                "Invalid nvidia-smi output",
                f"Row {row_number} contains {len(row)} columns instead of 10",
            )
        cells = tuple(cell.strip() for cell in row)
        try:
            index = int(cells[0])
        except ValueError as error:
            raise HardwareCollectionError(
                "Invalid nvidia-smi output", f"Row {row_number} has an invalid index"
            ) from error
        if not cells[1]:
            raise HardwareCollectionError(
                "Invalid nvidia-smi output", f"Row {row_number} has no GPU name"
            )
        gpus.append(
            GpuMetrics(
                index=index,
                name=cells[1],
                uuid=cells[2],
                utilisation_percent=_optional_number(
                    cells[3], "utilisation", row_number
                ),
                memory_used_bytes=_optional_mebibytes(
                    cells[4], "memory used", row_number
                ),
                memory_total_bytes=_optional_mebibytes(
                    cells[5], "memory total", row_number
                ),
                temperature_celsius=_optional_number(
                    cells[6], "temperature", row_number
                ),
                power_draw_watts=_optional_number(cells[7], "power draw", row_number),
                power_limit_watts=_optional_number(cells[8], "power limit", row_number),
                fan_speed_percent=_optional_number(cells[9], "fan speed", row_number),
            )
        )
    return tuple(gpus)


def _optional_number(value: str, metric: str, row_number: int) -> float | None:
    if value in _MISSING_VALUES:
        return None
    try:
        return float(value)
    except ValueError as error:
        raise HardwareCollectionError(
            "Invalid nvidia-smi output",
            f"Row {row_number} has an invalid {metric} value",
        ) from error


def _optional_mebibytes(value: str, metric: str, row_number: int) -> int | None:
    number = _optional_number(value, metric, row_number)
    return None if number is None else int(number * _MEBIBYTE)


class NvidiaSmiHardwareProvider(HardwareProvider):
    """Collect NVIDIA GPU metrics with one bounded child process per refresh."""

    name = "nvidia-smi"

    def __init__(
        self,
        system_collector: SystemMetricsCollector,
        *,
        timeout_seconds: float = 3.0,
    ) -> None:
        self._system_collector = system_collector
        self._timeout_seconds = timeout_seconds
        self._closed = False

    @classmethod
    async def create(
        cls, system_collector: SystemMetricsCollector
    ) -> "NvidiaSmiHardwareProvider":
        """Verify the query command exists without launching it."""
        executable = shutil.which("nvidia-smi")
        if executable is None:
            raise HardwareProviderUnavailable(
                "nvidia-smi not found", "nvidia-smi is not present on PATH"
            )
        return cls(system_collector)

    async def collect(self) -> HardwareSnapshot:
        """Collect system data on-loop and GPU data through a reaped subprocess."""
        if self._closed:
            raise HardwareProviderUnavailable(
                "nvidia-smi unavailable", "nvidia-smi provider is closed"
            )
        system = self._system_collector.collect()
        try:
            process = await asyncio.create_subprocess_exec(
                *_QUERY_ARGUMENTS,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as error:
            raise HardwareProviderUnavailable(
                "nvidia-smi not found", f"FileNotFoundError: {error}"
            ) from error
        except PermissionError as error:
            raise HardwareProviderUnavailable(
                "nvidia-smi permission denied", f"PermissionError: {error}"
            ) from error
        except OSError as error:
            raise HardwareCollectionError(
                "nvidia-smi failed", f"{type(error).__name__}: {error}"
            ) from error

        io_task = asyncio.create_task(process.communicate())
        try:
            stdout, stderr = await asyncio.wait_for(
                asyncio.shield(io_task), self._timeout_seconds
            )
        except TimeoutError as error:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.terminate()
            try:
                await asyncio.wait_for(
                    asyncio.shield(io_task), _TERMINATE_GRACE_SECONDS
                )
            except TimeoutError:
                if process.returncode is None:
                    with suppress(ProcessLookupError):
                        process.kill()
            with suppress(Exception, asyncio.CancelledError):
                await io_task
            raise HardwareCollectionError(
                "Hardware collection timed out",
                f"nvidia-smi exceeded {self._timeout_seconds:g} seconds",
            ) from error
        except asyncio.CancelledError:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
            with suppress(Exception, asyncio.CancelledError):
                await io_task
            raise

        if process.returncode != 0:
            raise HardwareCollectionError(
                "nvidia-smi failed",
                f"nvidia-smi exited with code {process.returncode}",
            )
        if len(stdout) > _MAX_OUTPUT_BYTES or len(stderr) > _MAX_OUTPUT_BYTES:
            raise HardwareCollectionError(
                "Invalid nvidia-smi output", "nvidia-smi output exceeded the size limit"
            )
        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
        if "\ufffd" in stdout_text or "\ufffd" in stderr_text:
            raise HardwareCollectionError(
                "Invalid nvidia-smi output", "nvidia-smi emitted invalid UTF-8"
            )

        gpus = parse_nvidia_smi_csv(stdout_text)
        gpu_partial = any(
            value is None
            for gpu in gpus
            for value in (
                gpu.utilisation_percent,
                gpu.memory_used_bytes,
                gpu.memory_total_bytes,
                gpu.temperature_celsius,
                gpu.power_draw_watts,
                gpu.power_limit_watts,
                gpu.fan_speed_percent,
            )
        )
        if gpu_partial and system.partial:
            snapshot_error = "Partial hardware metrics available"
        elif gpu_partial:
            snapshot_error = "Partial GPU metrics available"
        elif system.partial:
            snapshot_error = "Partial system metrics available"
        else:
            snapshot_error = None
        return HardwareSnapshot(
            provider_name=self.name,
            gpus=gpus,
            cpu=system.cpu,
            memory=system.memory,
            collected_at=datetime.now(UTC),
            error=snapshot_error,
        )

    async def close(self) -> None:
        """Close idempotently; child processes are owned by each collection."""
        self._closed = True
