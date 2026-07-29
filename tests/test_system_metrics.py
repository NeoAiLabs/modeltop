"""Deterministic psutil system metrics collection tests."""

import threading
from types import SimpleNamespace

import psutil
import pytest

import modeltop.hardware.base as base_module
from modeltop.hardware.base import SystemMetricsCollector


def _install_successful_psutil(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cpu_values: list[float] | None = None,
) -> tuple[list[float | None], list[int]]:
    values = iter(cpu_values or [99.0, 25.0])
    intervals: list[float | None] = []
    thread_ids: list[int] = []

    def cpu_percent(*, interval: float | None) -> float:
        intervals.append(interval)
        thread_ids.append(threading.get_ident())
        return next(values)

    def cpu_count(*, logical: bool) -> int:
        thread_ids.append(threading.get_ident())
        return 16 if logical else 8

    def virtual_memory() -> SimpleNamespace:
        thread_ids.append(threading.get_ident())
        return SimpleNamespace(total=1000, available=250, percent=75.0)

    def getloadavg() -> tuple[float, float, float]:
        thread_ids.append(threading.get_ident())
        return (1.25, 2.5, 3.75)

    monkeypatch.setattr(base_module.psutil, "cpu_percent", cpu_percent)
    monkeypatch.setattr(base_module.psutil, "cpu_count", cpu_count)
    monkeypatch.setattr(base_module.psutil, "virtual_memory", virtual_memory)
    monkeypatch.setattr(base_module.psutil, "getloadavg", getloadavg)
    return intervals, thread_ids


def test_prime_is_discarded_and_later_sample_is_nonblocking_same_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intervals, thread_ids = _install_successful_psutil(monkeypatch)
    owner = threading.get_ident()
    collector = SystemMetricsCollector()
    result = collector.collect()

    assert result.cpu.utilisation_percent == 25.0
    assert result.cpu.logical_core_count == 16
    assert result.cpu.physical_core_count == 8
    assert result.cpu.load_average_1m == 1.25
    assert result.cpu.load_average_5m == 2.5
    assert result.cpu.load_average_15m == 3.75
    assert result.memory.used_bytes == 750
    assert result.memory.total_bytes == 1000
    assert result.memory.utilisation_percent == 75.0
    assert not result.partial
    assert intervals == [None, None]
    assert thread_ids and set(thread_ids) == {owner}


def test_unsupported_load_average_is_partial_without_losing_other_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_successful_psutil(monkeypatch)
    monkeypatch.delattr(base_module.psutil, "getloadavg")
    result = SystemMetricsCollector().collect()
    assert result.partial
    assert result.cpu.load_average_1m is None
    assert result.cpu.load_average_5m is None
    assert result.cpu.load_average_15m is None
    assert result.cpu.utilisation_percent == 25.0
    assert result.memory.used_bytes == 750


def test_cpu_count_and_memory_failures_are_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intervals, _ = _install_successful_psutil(monkeypatch)

    def cpu_count(*, logical: bool) -> int:
        if logical:
            raise psutil.Error("logical failed")
        return 8

    def virtual_memory() -> SimpleNamespace:
        raise OSError("memory failed")

    monkeypatch.setattr(base_module.psutil, "cpu_count", cpu_count)
    monkeypatch.setattr(base_module.psutil, "virtual_memory", virtual_memory)
    result = SystemMetricsCollector().collect()

    assert result.partial
    assert result.cpu.utilisation_percent == 25.0
    assert result.cpu.logical_core_count is None
    assert result.cpu.physical_core_count == 8
    assert result.memory.used_bytes is None
    assert result.memory.total_bytes is None
    assert result.memory.utilisation_percent is None
    assert intervals == [None, None]


def test_cpu_failure_does_not_discard_counts_memory_or_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_successful_psutil(monkeypatch)
    calls = 0

    def cpu_percent(*, interval: float | None) -> float:
        nonlocal calls
        assert interval is None
        calls += 1
        if calls == 1:
            return 99.0
        raise RuntimeError("sample failed")

    monkeypatch.setattr(base_module.psutil, "cpu_percent", cpu_percent)
    result = SystemMetricsCollector().collect()
    assert result.partial
    assert result.cpu.utilisation_percent is None
    assert result.cpu.logical_core_count == 16
    assert result.memory.used_bytes == 750


def test_collection_rejects_worker_thread_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_successful_psutil(monkeypatch)
    collector = SystemMetricsCollector()
    caught: list[BaseException] = []

    def collect_elsewhere() -> None:
        try:
            collector.collect()
        except BaseException as error:
            caught.append(error)

    thread = threading.Thread(target=collect_elsewhere)
    thread.start()
    thread.join()
    assert len(caught) == 1
    assert isinstance(caught[0], RuntimeError)
    assert str(caught[0]) == "System metrics must be collected on the owner thread"
