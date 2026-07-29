"""Validation tests for bounded Tool Calling state and result models."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from modeltop.benchmarks.models import (
    ToolCallingBenchmarkConfig,
    ToolCallingBenchmarkResult,
    ToolCallingBenchmarkStatus,
    ToolCallingCategoryScore,
    ToolCallingScenarioResult,
    ToolCallingScenarioStatus,
    tool_calling_benchmark_config_from_defaults,
)
from modeltop.models import BenchmarksConfig, ToolCallingBenchmarkDefaultsConfig
from modeltop.state import initial_application_state


def _scenario(
    scenario_id: str = "single_tool",
    *,
    category: str = "A",
) -> ToolCallingScenarioResult:
    return ToolCallingScenarioResult(
        scenario_id=scenario_id,
        category=category,
        title="Single tool selection",
        status=ToolCallingScenarioStatus.PASS,
        points=2,
        failure_kind=None,
        duration_seconds=1.25,
        ttft_ms=20.0,
        turn_count=1,
        prompt_tokens=10,
        completion_tokens=5,
        infrastructure_excluded=False,
    )


def _result() -> ToolCallingBenchmarkResult:
    now = datetime(2026, 3, 20, tzinfo=UTC)
    return ToolCallingBenchmarkResult(
        benchmark_id="tool-call-1",
        upstream_run_id="upstream-1",
        config_fingerprint="fingerprint",
        server_id="local",
        server_name="Local",
        server_endpoint="127.0.0.1:8000",
        model_id="model",
        backend="vLLM",
        integration_commit="7ec8fcf33943020349ff6df339834a7ef984da00",
        upstream_version="2.3.0",
        schema_version="1",
        config=ToolCallingBenchmarkConfig(suite="core"),
        started_at=now,
        completed_at=now,
        status=ToolCallingBenchmarkStatus.COMPLETED,
        cancelled=False,
        error_code=None,
        error_message=None,
        attempted_count=15,
        gradable_count=1,
        excluded_count=14,
        completion_rate_percent=6.7,
        final_score=100,
        total_points=2,
        max_points=2,
        rating="★★★★★ Excellent",
        category_k_gradable=False,
        safety_gate_passed=True,
        deployability=90,
        responsiveness=80,
        median_turn_ms=20.0,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        categories=(
            ToolCallingCategoryScore(
                category="A",
                label="Tool Selection",
                earned_points=2,
                max_points=2,
                percent=100.0,
                pass_count=1,
                partial_count=0,
                fail_count=0,
            ),
        ),
        scenarios=(_scenario(),),
        warnings=(),
        hardware_summary=None,
    )


def test_tool_calling_defaults_are_backward_compatible_and_full() -> None:
    defaults = BenchmarksConfig().tool_calling
    assert defaults == ToolCallingBenchmarkDefaultsConfig()
    runtime = tool_calling_benchmark_config_from_defaults(defaults)
    assert runtime.suite == "full"
    assert runtime.scenario_count == 69
    assert runtime.request_timeout_seconds == 120.0


def test_tool_calling_defaults_and_runtime_are_strict_and_frozen() -> None:
    with pytest.raises(ValidationError):
        ToolCallingBenchmarkDefaultsConfig.model_validate({"unknown": True})
    with pytest.raises(ValidationError):
        ToolCallingBenchmarkConfig.model_validate({"suite": "core", "unknown": True})
    with pytest.raises(ValidationError):
        ToolCallingBenchmarkConfig(request_timeout_seconds=float("inf"))
    with pytest.raises(ValidationError):
        ToolCallingBenchmarkConfig(request_timeout_seconds=0)

    config = ToolCallingBenchmarkConfig()
    with pytest.raises(ValidationError):
        config.suite = "core"  # pyright: ignore[reportAttributeAccessIssue]


def test_status_lifecycle_is_centralized() -> None:
    assert ToolCallingBenchmarkStatus.VALIDATING.is_active
    assert ToolCallingBenchmarkStatus.RUNNING.is_active
    assert ToolCallingBenchmarkStatus.CANCELLING.is_active
    assert not ToolCallingBenchmarkStatus.IDLE.is_active
    assert ToolCallingBenchmarkStatus.COMPLETED.is_terminal
    assert ToolCallingBenchmarkStatus.CANCELLED.is_terminal
    assert ToolCallingBenchmarkStatus.ERROR.is_terminal


def test_scenario_status_points_and_infrastructure_are_consistent() -> None:
    with pytest.raises(ValueError, match="status and points"):
        replace(_scenario(), status=ToolCallingScenarioStatus.FAIL)
    with pytest.raises(ValueError, match="infrastructure_excluded"):
        replace(_scenario(), infrastructure_excluded=True)

    excluded = replace(
        _scenario(),
        status=ToolCallingScenarioStatus.FAIL,
        points=0,
        failure_kind="timeout",
        infrastructure_excluded=True,
    )
    assert excluded.infrastructure_excluded
    with pytest.raises(FrozenInstanceError):
        excluded.points = 2  # pyright: ignore[reportAttributeAccessIssue]


def test_category_capacity_and_points_match_gradable_statuses() -> None:
    category = _result().categories[0]
    with pytest.raises(ValueError, match="capacity"):
        replace(category, max_points=4)
    with pytest.raises(ValueError, match="points must match"):
        replace(category, earned_points=1)


def test_terminal_result_rejects_duplicate_and_unbounded_rows() -> None:
    result = _result()
    with pytest.raises(ValueError, match="scenario IDs must be unique"):
        replace(result, scenarios=(result.scenarios[0], result.scenarios[0]))
    with pytest.raises(ValueError, match="warnings"):
        replace(result, warnings=("x" * 257,))
    with pytest.raises(ValueError, match="terminal status"):
        replace(result, status=ToolCallingBenchmarkStatus.RUNNING)


def test_application_state_includes_tool_calling_traffic_lane() -> None:
    state = initial_application_state("server", hardware_enabled=False)
    assert state.tool_calling_benchmark.config.suite == "full"
    assert not state.benchmark_is_active
    active_lane = replace(
        state.tool_calling_benchmark,
        status=ToolCallingBenchmarkStatus.RUNNING,
        active_benchmark_id="tool-call-1",
    )
    assert replace(state, tool_calling_benchmark=active_lane).benchmark_is_active
