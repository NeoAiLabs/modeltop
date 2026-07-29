"""Focused tests for the native tool-eval-bench adapter and executor."""

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

from modeltop.benchmarks.base import BenchmarkContext
from modeltop.benchmarks.models import (
    ToolCallingBenchmarkConfig,
    ToolCallingBenchmarkStatus,
)
from modeltop.benchmarks.tool_calling import (
    ALPHA,
    CONCURRENCY,
    ERROR_RATE,
    MAX_TURNS,
    REFERENCE_DATE,
    SEED,
    TEMPERATURE,
    ScenarioResultCallback,
    ScenarioStartCallback,
    ToolCallingBenchmark,
    UpstreamBenchmarkRunner,
    suite_registry,
)

_NOW = datetime(2026, 3, 20, tzinfo=UTC)
_CATEGORY_LABELS = {
    "A": "Tool Selection",
    "B": "Parameter Precision",
    "C": "Multi-Step Chains",
    "D": "Restraint & Refusal",
    "E": "Error Recovery",
    "F": "Localization",
    "G": "Structured Reasoning",
    "H": "Instruction Following",
    "I": "Context & State",
    "J": "Code Patterns",
    "K": "Safety & Boundaries",
    "L": "Toolset Scale",
    "M": "Autonomous Planning",
    "N": "Creative Composition",
    "O": "Structured Output",
}


def _context() -> BenchmarkContext:
    return BenchmarkContext(
        benchmark_id="tool-call-test",
        started_at=_NOW,
        monotonic_clock=lambda: 10.0,
        utc_now=lambda: _NOW,
        read_hardware_snapshot=lambda: None,
    )


def _build_envelope(
    *,
    suite: str,
    callback_rows: list[dict[str, object]],
    backend: str,
    model: str,
    timeout: float,
) -> dict[str, Any]:
    registry = suite_registry(suite)
    gradable = [row for row in callback_rows if row["failure_kind"] != "timeout"]
    excluded = sorted(
        cast(str, row["scenario_id"])
        for row in callback_rows
        if row["failure_kind"] == "timeout"
    )
    category_rows: list[dict[str, object]] = []
    for category in sorted({item[1] for item in registry}):
        rows = [
            row
            for row in gradable
            if next(item[1] for item in registry if item[0] == row["scenario_id"])
            == category
        ]
        if not rows:
            continue
        earned = sum(cast(int, row["points"]) for row in rows)
        category_rows.append(
            {
                "category": category,
                "label": _CATEGORY_LABELS[category],
                "earned": earned,
                "max": len(rows) * 2,
                "percent": round(earned / (len(rows) * 2) * 100),
                "pass_count": sum(row["status"] == "pass" for row in rows),
                "partial_count": sum(row["status"] == "partial" for row in rows),
                "fail_count": sum(row["status"] == "fail" for row in rows),
            }
        )
    total_points = sum(cast(int, row["points"]) for row in gradable)
    max_points = len(gradable) * 2
    final_score = round(total_points / max_points * 100) if max_points else 0
    total_tokens = sum(
        cast(int, row["prompt_tokens"]) + cast(int, row["completion_tokens"])
        for row in gradable
    )
    scenario_rows: list[dict[str, object]] = []
    for row in callback_rows:
        raw = {
            "scenario_id": row["scenario_id"],
            "status": row["status"],
            "points": row["points"],
            "summary": "SENSITIVE EVALUATOR PROSE",
            "raw_log": "SENSITIVE RAW TRACE",
            "tool_calls_made": ["danger(secret=true)"],
            "expected_behavior": "SENSITIVE EXPECTATION",
            "duration_seconds": row["duration_seconds"],
            "ttft_ms": row["ttft_ms"],
            "turn_count": row["turn_count"],
            "prompt_tokens": row["prompt_tokens"],
            "completion_tokens": row["completion_tokens"],
        }
        if row["failure_kind"] is not None:
            raw["failure_kind"] = row["failure_kind"]
        scenario_rows.append(raw)
    scores: dict[str, Any] = {
        "final_score": final_score,
        "total_points": total_points,
        "max_points": max_points,
        "rating": "★★★★★ Excellent" if final_score == 100 else "★ Poor",
        "category_scores": category_rows,
        "scenario_results": scenario_rows,
    }
    if excluded:
        scores["excluded_scenarios"] = excluded
        scores["completion_rate"] = round(len(gradable) / len(registry) * 100, 1)
    if total_tokens:
        scores["total_tokens"] = total_tokens
        scores["token_efficiency"] = 1.23
    if gradable:
        scores.update(
            {
                "deployability": 91,
                "responsiveness": 80,
                "median_turn_ms": 25.0,
                "alpha": ALPHA,
                "worst_category": "A Tool Selection (100%)",
                "worst_category_percent": 100,
            }
        )
    config = {
        "model": model,
        "backend": backend,
        "base_url": "http://127.0.0.1:8000/v1",
        "temperature": TEMPERATURE,
        "timeout_seconds": timeout,
        "max_turns": MAX_TURNS,
        "seed": SEED,
        "reference_date": REFERENCE_DATE,
        "scenario_count": len(registry),
        "scenario_ids": [item[0] for item in registry],
        "concurrency": CONCURRENCY,
        "error_rate": ERROR_RATE,
        "alpha": ALPHA,
        "extra_params": None,
        "weight_by_difficulty": False,
        "config_fingerprint": "opaque-fingerprint",
    }
    safety_warnings: list[str] = []
    return {
        "schema_version": "1",
        "tool_eval_bench_version": "2.3.0",
        "final_score": final_score,
        "rating": scores["rating"],
        "safety_warnings": safety_warnings,
        "deployability": scores.get("deployability"),
        "responsiveness": scores.get("responsiveness"),
        "total_scenarios": max_points // 2 if max_points else None,
        "run_id": "opaque-run-id",
        "status": "completed",
        "config": config,
        "scores": scores,
        "metadata": {
            "secret": "SENSITIVE METADATA",
            "request_url": "https://example.invalid/secret",
        },
        "safety_gate": {"passed": True, "warnings": safety_warnings},
        "report_path": "/tmp/SENSITIVE-report.md",
    }


def _runner(
    *,
    suite: str = "core",
    excluded: bool = False,
    mutate: Callable[[dict[str, Any]], None] | None = None,
    captured: dict[str, Any] | None = None,
    after_first: Callable[[], None] | None = None,
) -> UpstreamBenchmarkRunner:
    async def run(**kwargs: Any) -> dict[str, Any]:
        if captured is not None:
            captured.update(kwargs)
        registry = suite_registry(suite)
        on_start = cast(ScenarioStartCallback, kwargs["on_scenario_start"])
        on_result = cast(ScenarioResultCallback, kwargs["on_scenario_result"])
        callback_rows: list[dict[str, object]] = []
        for index, (scenario_id, category, title) in enumerate(registry):
            scenario = SimpleNamespace(
                id=scenario_id,
                category=SimpleNamespace(value=category),
                title=title,
            )
            await on_start(scenario, index, len(registry))
            row: dict[str, object] = {
                "scenario_id": scenario_id,
                "status": "fail" if excluded else "pass",
                "points": 0 if excluded else 2,
                "failure_kind": "timeout" if excluded else None,
                "duration_seconds": 1.0,
                "ttft_ms": None if excluded else 25.0,
                "turn_count": 0 if excluded else 1,
                "prompt_tokens": 0 if excluded else 10,
                "completion_tokens": 0 if excluded else 5,
            }
            result = SimpleNamespace(**row)
            result.status = SimpleNamespace(value=row["status"])
            await on_result(scenario, result, index, len(registry))
            callback_rows.append(row)
            if index == 0 and after_first is not None:
                after_first()
        envelope = _build_envelope(
            suite=suite,
            callback_rows=callback_rows,
            backend=cast(str, kwargs["backend"]),
            model=cast(str, kwargs["model"]),
            timeout=cast(float, kwargs["timeout_seconds"]),
        )
        if mutate is not None:
            mutate(envelope)
        return envelope

    return cast(UpstreamBenchmarkRunner, run)


def successful_upstream_runner() -> UpstreamBenchmarkRunner:
    """Return a deterministic all-pass Core-suite upstream runner."""
    return _runner()


def _benchmark(
    runner: UpstreamBenchmarkRunner,
    *,
    suite: str = "core",
    backend_hint: str | None = "vllm",
    progress: list[object] | None = None,
) -> ToolCallingBenchmark:
    async def publish(value: object) -> None:
        if progress is not None:
            progress.append(value)

    return ToolCallingBenchmark(
        config=ToolCallingBenchmarkConfig(
            suite=cast(Any, suite), request_timeout_seconds=37.5
        ),
        server_id="local",
        server_name="Local",
        server_base_url="http://127.0.0.1:8000/v1",
        server_endpoint="127.0.0.1:8000",
        model_id="test-model",
        backend="Actual Backend",
        backend_hint=backend_hint,
        api_key="SENSITIVE_API_KEY",
        progress_callback=publish,
        upstream_runner=runner,
        progress_interval_seconds=0.0,
    )


def test_exact_core_arguments_and_payload_free_normalization() -> None:
    captured: dict[str, Any] = {}
    progress: list[object] = []
    result = asyncio.run(
        _benchmark(_runner(captured=captured), progress=progress).run(_context())
    )

    assert result.status is ToolCallingBenchmarkStatus.COMPLETED
    assert result.attempted_count == result.gradable_count == 15
    assert len(result.scenarios) == 15
    assert result.final_score == 100
    assert result.total_points == result.max_points == 30
    assert result.completion_rate_percent == 100.0
    assert result.upstream_run_id == "opaque-run-id"
    assert result.config_fingerprint == "opaque-fingerprint"
    assert result.backend == "Actual Backend"
    assert result.total_tokens == 225
    assert result.prompt_tokens == 150
    assert result.completion_tokens == 75
    assert progress
    assert captured["api_key"] == "SENSITIVE_API_KEY"
    assert captured["scenarios"] is None
    assert captured["short"] is True
    assert captured["temperature"] == TEMPERATURE
    assert captured["timeout_seconds"] == 37.5
    assert captured["max_turns"] == MAX_TURNS
    assert captured["seed"] is SEED
    assert captured["reference_date"] == REFERENCE_DATE
    assert captured["concurrency"] == CONCURRENCY
    assert captured["error_rate"] == ERROR_RATE
    assert captured["alpha"] == ALPHA
    assert captured["extra_params"] is None
    assert captured["persist"] is False
    assert captured["output_dir"] is None
    retained = repr(result)
    for sentinel in (
        "SENSITIVE_API_KEY",
        "SENSITIVE EVALUATOR PROSE",
        "SENSITIVE RAW TRACE",
        "SENSITIVE METADATA",
        "SENSITIVE-report.md",
        "danger(secret=true)",
    ):
        assert sentinel not in retained


def test_full_mapping_and_unknown_backend_warning() -> None:
    captured: dict[str, Any] = {}
    result = asyncio.run(
        _benchmark(
            _runner(suite="full", captured=captured),
            suite="full",
            backend_hint="unknown-engine",
        ).run(_context())
    )
    assert result.status is ToolCallingBenchmarkStatus.COMPLETED
    assert result.attempted_count == len(result.scenarios) == 69
    assert captured["short"] is False
    assert captured["backend"] == "vllm"
    assert len(result.warnings) == 1
    assert "not recognized" in result.warnings[0]


def test_all_infrastructure_excluded_shape_preserves_coverage() -> None:
    result = asyncio.run(_benchmark(_runner(excluded=True)).run(_context()))
    assert result.status is ToolCallingBenchmarkStatus.COMPLETED
    assert result.gradable_count == 0
    assert result.excluded_count == 15
    assert result.completion_rate_percent == 0.0
    assert result.final_score == result.total_points == result.max_points == 0
    assert result.categories == ()
    assert result.deployability is None
    assert result.responsiveness is None
    assert result.total_tokens == 0
    assert not result.category_k_gradable


def test_schema_drift_and_duplicate_ids_are_fixed_protocol_errors() -> None:
    def mutate_schema(envelope: dict[str, Any]) -> None:
        envelope["schema_version"] = "2"

    schema_result = asyncio.run(
        _benchmark(_runner(mutate=mutate_schema)).run(_context())
    )
    assert schema_result.status is ToolCallingBenchmarkStatus.ERROR
    assert schema_result.error_code == "invalid_upstream_result"
    assert "incompatible" in cast(str, schema_result.error_message)

    def duplicate(envelope: dict[str, Any]) -> None:
        rows = cast(list[dict[str, Any]], envelope["scores"]["scenario_results"])
        rows[1]["scenario_id"] = rows[0]["scenario_id"]

    duplicate_result = asyncio.run(
        _benchmark(_runner(mutate=duplicate)).run(_context())
    )
    assert duplicate_result.error_code == "invalid_upstream_result"


def test_cancellation_before_and_during_work_returns_callback_only_rows() -> None:
    called = False

    async def should_not_run(**_kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        raise AssertionError

    before = _benchmark(cast(UpstreamBenchmarkRunner, should_not_run))
    before.request_cancellation()
    before_result = asyncio.run(before.run(_context()))
    assert before_result.status is ToolCallingBenchmarkStatus.CANCELLED
    assert before_result.scenarios == ()
    assert not called

    active: ToolCallingBenchmark | None = None

    def cancel() -> None:
        assert active is not None
        active.request_cancellation()

    active = _benchmark(_runner(after_first=cancel))
    during_result = asyncio.run(active.run(_context()))
    assert during_result.status is ToolCallingBenchmarkStatus.CANCELLED
    assert len(during_result.scenarios) == 1
    assert during_result.final_score is None
    assert during_result.categories == ()


def test_upstream_and_http_logs_are_suppressed_and_state_is_restored(
    caplog: Any,
) -> None:
    loggers = [
        logging.getLogger(name) for name in ("tool_eval_bench", "httpx", "httpcore")
    ]
    before = [
        (list(logger.handlers), logger.level, logger.disabled, logger.propagate)
        for logger in loggers
    ]

    base = _runner()

    async def noisy(**kwargs: Any) -> dict[str, Any]:
        for logger in loggers:
            logger.critical("SENSITIVE_BODY malformed_args secret_url")
        return await base(**kwargs)

    with caplog.at_level(logging.DEBUG):
        result = asyncio.run(
            _benchmark(cast(UpstreamBenchmarkRunner, noisy)).run(_context())
        )
    assert result.status is ToolCallingBenchmarkStatus.COMPLETED
    assert "SENSITIVE_BODY" not in caplog.text
    after = [
        (list(logger.handlers), logger.level, logger.disabled, logger.propagate)
        for logger in loggers
    ]
    assert after == before


def test_top_level_exception_message_is_never_retained_or_logged(caplog: Any) -> None:
    async def broken(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("SENSITIVE_EXCEPTION_BODY")

    with caplog.at_level(logging.INFO, logger="modeltop.benchmarks.tool_calling"):
        result = asyncio.run(
            _benchmark(cast(UpstreamBenchmarkRunner, broken)).run(_context())
        )
    assert result.status is ToolCallingBenchmarkStatus.ERROR
    assert result.error_code == "upstream_failure"
    assert result.error_message == (
        "Tool Calling stopped because the benchmark library could not complete."
    )
    assert "SENSITIVE_EXCEPTION_BODY" not in caplog.text
    assert "RuntimeError" in caplog.text
