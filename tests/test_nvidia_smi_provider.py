"""Deterministic nvidia-smi parsing and subprocess lifecycle tests."""

import asyncio
from typing import cast

import pytest

import modeltop.hardware.nvidia_smi as smi_module
from modeltop.hardware.base import (
    HardwareCollectionError,
    HardwareProviderUnavailable,
    SystemMetricsCollector,
    SystemMetricsResult,
)
from modeltop.hardware.models import CpuMetrics, MemoryMetrics
from modeltop.hardware.nvidia_smi import (
    NvidiaSmiHardwareProvider,
    parse_nvidia_smi_csv,
)

_ROW = "0, NVIDIA GB10, GPU-0, 25, 1024, 8192, 55, 125.5, 250, 120\n"


class _FakeSystemCollector:
    def __init__(self, *, partial: bool = False) -> None:
        self.partial = partial
        self.calls = 0

    def collect(self) -> SystemMetricsResult:
        self.calls += 1
        return SystemMetricsResult(
            CpuMetrics(10.0, 8, 4, 1.0, 2.0, 3.0),
            MemoryMetrics(4, 8, 50.0),
            self.partial,
        )


class _FakeProcess:
    def __init__(
        self,
        stdout: bytes = _ROW.encode(),
        stderr: bytes = b"",
        *,
        returncode: int | None = 0,
        gated: bool = False,
        terminate_releases: bool = True,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.gated = gated
        self.terminate_releases = terminate_releases
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.terminate_count = 0
        self.kill_count = 0

    async def communicate(self) -> tuple[bytes, bytes]:
        self.started.set()
        if self.gated:
            await self.release.wait()
        return self.stdout, self.stderr

    def terminate(self) -> None:
        self.terminate_count += 1
        if self.terminate_releases:
            self.returncode = -15
            self.release.set()

    def kill(self) -> None:
        self.kill_count += 1
        self.returncode = -9
        self.release.set()


def _provider(
    collector: _FakeSystemCollector | None = None, *, timeout: float = 3.0
) -> NvidiaSmiHardwareProvider:
    return NvidiaSmiHardwareProvider(
        cast(SystemMetricsCollector, collector or _FakeSystemCollector()),
        timeout_seconds=timeout,
    )


def _install_process(
    monkeypatch: pytest.MonkeyPatch,
    process: _FakeProcess,
    captured: list[tuple[tuple[str, ...], dict[str, object]]] | None = None,
) -> None:
    async def create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProcess:
        if captured is not None:
            captured.append((args, kwargs))
        return process

    monkeypatch.setattr(
        smi_module.asyncio, "create_subprocess_exec", create_subprocess_exec
    )


def test_csv_parser_handles_quotes_two_gpus_missing_values_and_mib() -> None:
    output = (
        '0, "NVIDIA, A", GPU-0, 25, 1024, 8192, 55, 125.5, 250, 120\n'
        "1, NVIDIA B, GPU-1, [N/A], N/A, , 65, 50, N/A, N/A\n"
    )
    gpus = parse_nvidia_smi_csv(output)
    assert len(gpus) == 2
    assert gpus[0].name == "NVIDIA, A"
    assert gpus[0].memory_used_bytes == 1024 * 1024**2
    assert gpus[0].memory_total_bytes == 8192 * 1024**2
    assert gpus[0].fan_speed_percent == 120
    assert gpus[1].utilisation_percent is None
    assert gpus[1].memory_used_bytes is None
    assert gpus[1].memory_total_bytes is None
    assert gpus[1].power_limit_watts is None


@pytest.mark.parametrize(
    "output",
    [
        "",
        "0, only, three\n",
        "bad, GPU, UUID, 1, 2, 3, 4, 5, 6, 7\n",
        "0, , UUID, 1, 2, 3, 4, 5, 6, 7\n",
        "0, GPU, UUID, unexpected, 2, 3, 4, 5, 6, 7\n",
        "0, GPU, UUID, 1, 2, 3, 4, 5, 6, 7\ufffd\n",
    ],
)
def test_csv_parser_rejects_empty_malformed_and_nonnumeric_output(
    output: str,
) -> None:
    with pytest.raises(HardwareCollectionError):
        parse_nvidia_smi_csv(output)


def test_collect_uses_exact_command_and_reports_partial_system_and_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        process = _FakeProcess(b"0, GPU, UUID, N/A, 1024, 8192, 55, 125, 250, N/A\n")
        captured: list[tuple[tuple[str, ...], dict[str, object]]] = []
        _install_process(monkeypatch, process, captured)
        collector = _FakeSystemCollector(partial=True)
        snapshot = await _provider(collector).collect()
        assert captured[0][0] == (
            "nvidia-smi",
            "--query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,fan.speed",
            "--format=csv,noheader,nounits",
        )
        assert captured[0][1]["stdin"] is asyncio.subprocess.DEVNULL
        assert captured[0][1]["stdout"] is asyncio.subprocess.PIPE
        assert captured[0][1]["stderr"] is asyncio.subprocess.PIPE
        assert snapshot.provider_name == "nvidia-smi"
        assert snapshot.error == "Partial hardware metrics available"
        assert collector.calls == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("error", "message", "exception_type"),
    [
        (
            FileNotFoundError("missing"),
            "nvidia-smi not found",
            HardwareProviderUnavailable,
        ),
        (
            PermissionError("denied"),
            "nvidia-smi permission denied",
            HardwareProviderUnavailable,
        ),
        (OSError("failed"), "nvidia-smi failed", HardwareCollectionError),
    ],
)
def test_spawn_error_mappings(
    monkeypatch: pytest.MonkeyPatch,
    error: OSError,
    message: str,
    exception_type: type[Exception],
) -> None:
    async def scenario() -> None:
        async def create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProcess:
            raise error

        monkeypatch.setattr(
            smi_module.asyncio, "create_subprocess_exec", create_subprocess_exec
        )
        with pytest.raises(exception_type, match=message):
            await _provider().collect()

    asyncio.run(scenario())


def test_create_checks_executable_without_spawning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        def missing(command: str) -> str | None:
            assert command == "nvidia-smi"
            return None

        def present(command: str) -> str | None:
            assert command == "nvidia-smi"
            return "/usr/bin/nvidia-smi"

        monkeypatch.setattr(smi_module.shutil, "which", missing)
        with pytest.raises(HardwareProviderUnavailable, match="nvidia-smi not found"):
            await NvidiaSmiHardwareProvider.create(
                cast(SystemMetricsCollector, _FakeSystemCollector())
            )
        monkeypatch.setattr(smi_module.shutil, "which", present)
        provider = await NvidiaSmiHardwareProvider.create(
            cast(SystemMetricsCollector, _FakeSystemCollector())
        )
        assert provider.name == "nvidia-smi"
        await provider.close()
        await provider.close()

    asyncio.run(scenario())


def test_nonzero_invalid_utf8_and_oversized_output_are_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        process = _FakeProcess(b"secret", b"private", returncode=7)
        _install_process(monkeypatch, process)
        with pytest.raises(
            HardwareCollectionError, match="nvidia-smi failed"
        ) as caught:
            await _provider().collect()
        assert caught.value.detail == "nvidia-smi exited with code 7"
        assert "secret" not in caught.value.detail

        invalid = _FakeProcess(_ROW.encode() + b"\xff")
        _install_process(monkeypatch, invalid)
        with pytest.raises(HardwareCollectionError, match="Invalid nvidia-smi output"):
            await _provider().collect()

        oversized = _FakeProcess(b"x" * (1024**2 + 1))
        _install_process(monkeypatch, oversized)
        with pytest.raises(HardwareCollectionError, match="Invalid nvidia-smi output"):
            await _provider().collect()

    asyncio.run(scenario())


def test_timeout_terminates_then_kills_and_reaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        graceful = _FakeProcess(returncode=None, gated=True, terminate_releases=True)
        _install_process(monkeypatch, graceful)
        with pytest.raises(
            HardwareCollectionError, match="Hardware collection timed out"
        ):
            await _provider(timeout=0).collect()
        assert graceful.terminate_count == 1
        assert graceful.kill_count == 0
        assert graceful.release.is_set()

        stubborn = _FakeProcess(returncode=None, gated=True, terminate_releases=False)
        _install_process(monkeypatch, stubborn)
        monkeypatch.setattr(smi_module, "_TERMINATE_GRACE_SECONDS", 0)
        with pytest.raises(
            HardwareCollectionError, match="Hardware collection timed out"
        ):
            await _provider(timeout=0).collect()
        assert stubborn.terminate_count == 1
        assert stubborn.kill_count == 1
        assert stubborn.release.is_set()

    asyncio.run(scenario())


def test_cancellation_kills_drains_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        process = _FakeProcess(returncode=None, gated=True)
        _install_process(monkeypatch, process)
        task = asyncio.create_task(_provider().collect())
        await process.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert process.kill_count == 1
        assert process.release.is_set()

    asyncio.run(scenario())
