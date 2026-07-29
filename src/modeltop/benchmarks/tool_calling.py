"""Privacy-preserving boundary around the pinned tool-eval-bench library."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol, cast

from modeltop.benchmarks.base import Benchmark, BenchmarkContext
from modeltop.benchmarks.models import (
    ToolCallingBenchmarkConfig,
    ToolCallingBenchmarkProgress,
    ToolCallingBenchmarkResult,
    ToolCallingBenchmarkStatus,
    ToolCallingCategoryScore,
    ToolCallingErrorCode,
    ToolCallingFailureKind,
    ToolCallingScenarioProgress,
    ToolCallingScenarioResult,
    ToolCallingScenarioStatus,
)
from modeltop.benchmarks.statistics import summarize_hardware_samples
from modeltop.hardware.models import HardwareSnapshot

logger = logging.getLogger(__name__)

TOOL_EVAL_BENCH_COMMIT = "7ec8fcf33943020349ff6df339834a7ef984da00"
TOOL_EVAL_BENCH_SCHEMA_VERSION = "1"
TOOL_EVAL_BENCH_VERSION = "2.3.0"
REFERENCE_DATE = "2026-03-20"
TEMPERATURE = 0.0
MAX_TURNS = 8
SEED: None = None
CONCURRENCY = 1
ERROR_RATE = 0.0
ALPHA = 0.7

_UPSTREAM_LOGGERS = ("tool_eval_bench", "httpx", "httpcore")
_BACKEND_ALIASES = {
    "vllm": "vllm",
    "litellm": "litellm",
    "llamacpp": "llamacpp",
    "llama.cpp": "llamacpp",
    "llama_cpp": "llamacpp",
    "llama-cpp": "llamacpp",
}
BACKEND_FALLBACK_WARNING = (
    "The server backend is not recognized by tool-eval-bench; its generic "
    "OpenAI-compatible adapter is using the vLLM family label."
)
_FAILURE_KINDS = {
    "wrong_tool",
    "wrong_args",
    "missing_step",
    "forbidden_action",
    "budget_exceeded",
    "timeout",
    "connection_error",
    "server_error",
    "model_crash",
    "evaluator_error",
    "partial",
}
_INFRASTRUCTURE_FAILURE_KINDS = {
    "timeout",
    "connection_error",
    "server_error",
}
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
_ERROR_MESSAGES: dict[ToolCallingErrorCode, str] = {
    "dependency_unavailable": (
        "The pinned tool-eval-bench dependency is unavailable. "
        "Reinstall ModelTop's locked dependencies."
    ),
    "upstream_failure": (
        "Tool Calling stopped because the benchmark library could not complete."
    ),
    "invalid_upstream_result": (
        "Tool Calling rejected an incompatible benchmark result."
    ),
}

ScenarioStartCallback = Callable[[object, int, int], Awaitable[None]]
ScenarioResultCallback = Callable[[object, object, int, int], Awaitable[None]]


class UpstreamBenchmarkRunner(Protocol):
    """Injectable subset of tool_eval_bench.api.run_benchmark."""

    async def __call__(
        self,
        *,
        model: str,
        base_url: str,
        backend: str,
        api_key: str | None,
        scenarios: None,
        short: bool,
        temperature: float,
        timeout_seconds: float,
        max_turns: int,
        seed: int | None,
        reference_date: str,
        concurrency: int,
        error_rate: float,
        alpha: float,
        extra_params: None,
        on_scenario_start: ScenarioStartCallback,
        on_scenario_result: ScenarioResultCallback,
        persist: bool,
        output_dir: None,
    ) -> dict[str, Any]: ...


async def run_upstream_benchmark(
    *,
    model: str,
    base_url: str,
    backend: str,
    api_key: str | None,
    scenarios: None,
    short: bool,
    temperature: float,
    timeout_seconds: float,
    max_turns: int,
    seed: int | None,
    reference_date: str,
    concurrency: int,
    error_rate: float,
    alpha: float,
    extra_params: None,
    on_scenario_start: ScenarioStartCallback,
    on_scenario_result: ScenarioResultCallback,
    persist: bool,
    output_dir: None,
) -> dict[str, Any]:
    """Import and invoke the optional upstream package only for an active run."""
    from tool_eval_bench.api import run_benchmark

    return await run_benchmark(
        model=model,
        base_url=base_url,
        backend=backend,
        api_key=api_key,
        scenarios=scenarios,
        short=short,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        max_turns=max_turns,
        seed=seed,
        reference_date=reference_date,
        concurrency=concurrency,
        error_rate=error_rate,
        alpha=alpha,
        extra_params=extra_params,
        on_scenario_start=on_scenario_start,
        on_scenario_result=on_scenario_result,
        persist=persist,
        output_dir=output_dir,
    )


def map_upstream_backend(backend_hint: str | None) -> tuple[str, tuple[str, ...]]:
    """Return a validated adapter-family label without changing actual backend data."""
    normalized = (backend_hint or "").strip().lower()
    mapped = _BACKEND_ALIASES.get(normalized)
    if mapped is not None:
        return mapped, ()
    return "vllm", (BACKEND_FALLBACK_WARNING,)


def suite_registry(suite: str) -> tuple[tuple[str, str, str], ...]:
    """Load bounded identity fields for the pinned public Core or Full registry."""
    from tool_eval_bench.evals.scenarios import ALL_SCENARIOS, SCENARIOS

    definitions = SCENARIOS if suite == "core" else ALL_SCENARIOS
    return tuple(
        (scenario.id, scenario.category.value, scenario.title)
        for scenario in definitions
    )


@contextmanager
def suppress_upstream_logs() -> Generator[None]:
    """Suppress payload-bearing upstream logs and restore exact logger state."""
    saved: list[tuple[logging.Logger, list[logging.Handler], int, bool, bool]] = []
    try:
        for name in _UPSTREAM_LOGGERS:
            logger = logging.getLogger(name)
            saved.append(
                (
                    logger,
                    list(logger.handlers),
                    logger.level,
                    logger.disabled,
                    logger.propagate,
                )
            )
            logger.handlers = [logging.NullHandler()]
            logger.setLevel(logging.CRITICAL + 1)
            logger.disabled = True
            logger.propagate = False
        yield
    finally:
        for logger, handlers, level, disabled, propagate in saved:
            logger.handlers = handlers
            logger.setLevel(level)
            logger.disabled = disabled
            logger.propagate = propagate


class ToolCallingProtocolError(Exception):
    """Internal marker for a rejected upstream schema-1 envelope."""


def _protocol_error() -> ToolCallingProtocolError:
    return ToolCallingProtocolError()


def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _protocol_error()
    untyped = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in untyped):
        raise _protocol_error()
    return cast(dict[str, Any], untyped)


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise _protocol_error()
    return cast(list[object], value)


def _string(value: object, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise _protocol_error()
    return value


def _integer(value: object, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _protocol_error()
    if maximum is not None and value > maximum:
        raise _protocol_error()
    return value


def _number(
    value: object,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _protocol_error()
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise _protocol_error()
    if maximum is not None and result > maximum:
        raise _protocol_error()
    return result


def _optional_number(value: object) -> float | None:
    if value is None:
        return None
    return _number(value)


def _enum_string(value: object) -> str:
    raw = getattr(value, "value", value)
    return _string(raw, maximum=64)


def _attribute(value: object, name: str) -> object:
    try:
        return getattr(value, name)
    except AttributeError:
        raise _protocol_error() from None


@dataclass(frozen=True, slots=True)
class _ParsedAggregates:
    upstream_run_id: str
    config_fingerprint: str
    upstream_version: str
    schema_version: str
    completion_rate_percent: float
    final_score: int
    total_points: int
    max_points: int
    rating: str
    safety_gate_passed: bool
    deployability: int | None
    responsiveness: int | None
    median_turn_ms: float | None
    total_tokens: int
    categories: tuple[ToolCallingCategoryScore, ...]


class ToolCallingBenchmark(Benchmark[ToolCallingBenchmarkResult]):
    """Execute and normalize one native tool-eval-bench run."""

    def __init__(
        self,
        *,
        config: ToolCallingBenchmarkConfig,
        server_id: str,
        server_name: str,
        server_base_url: str,
        server_endpoint: str,
        model_id: str,
        backend: str,
        backend_hint: str | None,
        api_key: str | None,
        progress_callback: Callable[[ToolCallingBenchmarkProgress], Awaitable[None]],
        upstream_runner: UpstreamBenchmarkRunner = run_upstream_benchmark,
        progress_interval_seconds: float = 0.1,
    ) -> None:
        self._config = config
        self._server_id = server_id
        self._server_name = server_name
        self._server_base_url = server_base_url
        self._server_endpoint = server_endpoint
        self._model_id = model_id
        self._backend = backend
        self._backend_hint = backend_hint
        self._api_key = api_key
        self._progress_callback = progress_callback
        self._upstream_runner = upstream_runner
        self._progress_interval_seconds = progress_interval_seconds
        self._cancel_requested = False
        self._context: BenchmarkContext | None = None
        self._registry: tuple[tuple[str, str, str], ...] = ()
        self._scenario_results: list[ToolCallingScenarioResult] = []
        self._hardware_samples: list[HardwareSnapshot] = []
        self._current_scenario: ToolCallingScenarioProgress | None = None
        self._last_progress_at: float | None = None
        self._started_at_monotonic = 0.0
        self._mapped_backend = ""
        self._warnings: tuple[str, ...] = ()

    def request_cancellation(self) -> None:
        """Request cancellation at the next callback boundary."""
        self._cancel_requested = True

    async def run(self, context: BenchmarkContext) -> ToolCallingBenchmarkResult:
        """Run the official selected suite and discard the raw envelope."""
        self._context = context
        self._started_at_monotonic = context.monotonic_clock()
        self._mapped_backend, self._warnings = map_upstream_backend(self._backend_hint)
        try:
            self._registry = suite_registry(self._config.suite)
            if len(self._registry) != self._config.scenario_count:
                raise _protocol_error()
            if self._cancel_requested:
                raise asyncio.CancelledError
            with suppress_upstream_logs():
                envelope = await self._upstream_runner(
                    model=self._model_id,
                    base_url=self._server_base_url,
                    backend=self._mapped_backend,
                    api_key=self._api_key,
                    scenarios=None,
                    short=self._config.suite == "core",
                    temperature=TEMPERATURE,
                    timeout_seconds=self._config.request_timeout_seconds,
                    max_turns=MAX_TURNS,
                    seed=SEED,
                    reference_date=REFERENCE_DATE,
                    concurrency=CONCURRENCY,
                    error_rate=ERROR_RATE,
                    alpha=ALPHA,
                    extra_params=None,
                    on_scenario_start=self._on_scenario_start,
                    on_scenario_result=self._on_scenario_result,
                    persist=False,
                    output_dir=None,
                )
            if self._cancel_requested:
                raise asyncio.CancelledError
            parsed = self._parse_envelope(envelope)
            result = self._completed_result(parsed)
            logger.info(
                "Tool Calling benchmark=%s model=%s suite=%s status=%s "
                "attempted=%d gradable=%d excluded=%d duration=%.3f",
                context.benchmark_id,
                self._model_id,
                self._config.suite,
                result.status.value,
                result.attempted_count,
                result.gradable_count,
                result.excluded_count,
                result.wall_time_seconds,
            )
            return result
        except asyncio.CancelledError:
            result = self._partial_result(
                ToolCallingBenchmarkStatus.CANCELLED,
                error_code=None,
            )
            logger.info(
                "Tool Calling benchmark=%s model=%s suite=%s status=%s "
                "attempted=%d completed=%d duration=%.3f",
                context.benchmark_id,
                self._model_id,
                self._config.suite,
                result.status.value,
                result.attempted_count,
                len(result.scenarios),
                result.wall_time_seconds,
            )
            return result
        except (ImportError, ModuleNotFoundError) as error:
            return self._error_result("dependency_unavailable", error)
        except ToolCallingProtocolError as error:
            return self._error_result("invalid_upstream_result", error)
        except Exception as error:
            return self._error_result("upstream_failure", error)

    async def _on_scenario_start(
        self,
        scenario: object,
        index: int,
        total: int,
    ) -> None:
        if self._cancel_requested:
            raise asyncio.CancelledError
        expected = self._expected_callback_identity(scenario, index, total)
        context = self._require_context()
        self._current_scenario = ToolCallingScenarioProgress(
            scenario_id=expected[0],
            category=expected[1],
            title=expected[2],
            source_index=index,
            started_at_monotonic=context.monotonic_clock(),
        )
        await self._publish_progress(force=index == 0)

    async def _on_scenario_result(
        self,
        scenario: object,
        result: object,
        index: int,
        total: int,
    ) -> None:
        if self._cancel_requested:
            raise asyncio.CancelledError
        expected = self._expected_callback_identity(scenario, index, total)
        if (
            self._current_scenario is None
            or self._current_scenario.source_index != index
            or index != len(self._scenario_results)
        ):
            raise _protocol_error()
        scenario_id = _string(_attribute(result, "scenario_id"), maximum=128)
        if scenario_id != expected[0]:
            raise _protocol_error()
        status_text = _enum_string(_attribute(result, "status"))
        try:
            status = ToolCallingScenarioStatus(status_text)
        except ValueError:
            raise _protocol_error() from None
        points = _integer(_attribute(result, "points"), maximum=2)
        raw_failure_kind = _attribute(result, "failure_kind")
        failure_kind: ToolCallingFailureKind | None
        if raw_failure_kind is None:
            failure_kind = None
        else:
            parsed_failure_kind = _string(raw_failure_kind, maximum=64)
            if parsed_failure_kind not in _FAILURE_KINDS:
                raise _protocol_error()
            failure_kind = cast(ToolCallingFailureKind, parsed_failure_kind)
        snapshot = ToolCallingScenarioResult(
            scenario_id=expected[0],
            category=expected[1],
            title=expected[2],
            status=status,
            points=points,
            failure_kind=failure_kind,
            duration_seconds=_number(_attribute(result, "duration_seconds")),
            ttft_ms=_optional_number(_attribute(result, "ttft_ms")),
            turn_count=_integer(_attribute(result, "turn_count")),
            prompt_tokens=_integer(_attribute(result, "prompt_tokens")),
            completion_tokens=_integer(_attribute(result, "completion_tokens")),
            infrastructure_excluded=(
                status is ToolCallingScenarioStatus.FAIL
                and failure_kind in _INFRASTRUCTURE_FAILURE_KINDS
            ),
        )
        self._scenario_results.append(snapshot)
        hardware = self._require_context().read_hardware_snapshot()
        if hardware is not None:
            self._hardware_samples.append(hardware)
        self._current_scenario = None
        await self._publish_progress(force=index + 1 == total)

    def _expected_callback_identity(
        self,
        scenario: object,
        index: int,
        total: int,
    ) -> tuple[str, str, str]:
        if (
            type(index) is not int
            or type(total) is not int
            or total != len(self._registry)
            or not 0 <= index < total
        ):
            raise _protocol_error()
        expected = self._registry[index]
        actual = (
            _string(_attribute(scenario, "id"), maximum=128),
            _enum_string(_attribute(scenario, "category")),
            _string(_attribute(scenario, "title"), maximum=256),
        )
        if actual != expected:
            raise _protocol_error()
        return expected

    async def _publish_progress(self, *, force: bool) -> None:
        context = self._require_context()
        now = context.monotonic_clock()
        if (
            not force
            and self._last_progress_at is not None
            and now - self._last_progress_at < self._progress_interval_seconds
        ):
            return
        gradable = tuple(
            row for row in self._scenario_results if not row.infrastructure_excluded
        )
        await self._progress_callback(
            ToolCallingBenchmarkProgress(
                configured_count=self._config.scenario_count,
                completed_count=len(self._scenario_results),
                gradable_count=len(gradable),
                excluded_count=len(self._scenario_results) - len(gradable),
                pass_count=sum(
                    row.status is ToolCallingScenarioStatus.PASS for row in gradable
                ),
                partial_count=sum(
                    row.status is ToolCallingScenarioStatus.PARTIAL for row in gradable
                ),
                fail_count=sum(
                    row.status is ToolCallingScenarioStatus.FAIL for row in gradable
                ),
                current_scenario=self._current_scenario,
                elapsed_seconds=max(0.0, now - self._started_at_monotonic),
                cached_hardware=context.read_hardware_snapshot(),
            )
        )
        self._last_progress_at = now

    def _parse_envelope(self, envelope: object) -> _ParsedAggregates:
        data = _mapping(envelope)
        schema_version = _string(data.get("schema_version"), maximum=16)
        upstream_version = _string(data.get("tool_eval_bench_version"), maximum=64)
        if (
            schema_version != TOOL_EVAL_BENCH_SCHEMA_VERSION
            or upstream_version != TOOL_EVAL_BENCH_VERSION
            or data.get("status") != "completed"
        ):
            raise _protocol_error()
        run_id = _string(data.get("run_id"))
        config = _mapping(data.get("config"))
        self._validate_upstream_config(config)
        config_fingerprint = _string(config.get("config_fingerprint"))
        scores = _mapping(data.get("scores"))
        categories = self._parse_categories(scores)
        self._validate_envelope_scenarios(scores)

        gradable = tuple(
            row for row in self._scenario_results if not row.infrastructure_excluded
        )
        excluded_ids = sorted(
            row.scenario_id
            for row in self._scenario_results
            if row.infrastructure_excluded
        )
        raw_excluded = scores.get("excluded_scenarios")
        if excluded_ids:
            parsed_excluded = [
                _string(value, maximum=128) for value in _list(raw_excluded)
            ]
            if parsed_excluded != excluded_ids:
                raise _protocol_error()
            completion_rate = _number(scores.get("completion_rate"), maximum=100.0)
        else:
            if "excluded_scenarios" in scores or "completion_rate" in scores:
                raise _protocol_error()
            completion_rate = 100.0
        expected_completion = round(
            len(gradable) / self._config.scenario_count * 100,
            1,
        )
        if completion_rate != expected_completion:
            raise _protocol_error()

        total_points = _integer(scores.get("total_points"))
        max_points = _integer(scores.get("max_points"))
        final_score = _integer(scores.get("final_score"), maximum=100)
        expected_score = round(total_points / max_points * 100) if max_points else 0
        if (
            max_points != len(gradable) * 2
            or total_points != sum(row.points for row in gradable)
            or final_score != expected_score
            or sum(category.max_points for category in categories) != max_points
            or sum(category.earned_points for category in categories) != total_points
        ):
            raise _protocol_error()
        rating = _string(scores.get("rating"))
        safety_gate = _mapping(data.get("safety_gate"))
        safety_passed = safety_gate.get("passed")
        safety_warnings = _list(safety_gate.get("warnings"))
        nested_warnings = scores.get("safety_warnings")
        if nested_warnings is None:
            nested_warning_count = 0
        else:
            nested_warning_count = len(_list(nested_warnings))
        if (
            not isinstance(safety_passed, bool)
            or safety_passed != (len(safety_warnings) == 0)
            or len(safety_warnings) != nested_warning_count
        ):
            raise _protocol_error()

        deployability, responsiveness, median_turn_ms = self._parse_deployability(
            scores
        )
        expected_total_tokens = sum(
            row.prompt_tokens + row.completion_tokens for row in gradable
        )
        if expected_total_tokens:
            total_tokens = _integer(scores.get("total_tokens"))
            if total_tokens != expected_total_tokens:
                raise _protocol_error()
        else:
            if "total_tokens" in scores:
                raise _protocol_error()
            total_tokens = 0
        self._validate_top_level_mirrors(
            data,
            final_score=final_score,
            rating=rating,
            max_points=max_points,
            deployability=deployability,
            responsiveness=responsiveness,
            warning_count=nested_warning_count,
        )
        return _ParsedAggregates(
            upstream_run_id=run_id,
            config_fingerprint=config_fingerprint,
            upstream_version=upstream_version,
            schema_version=schema_version,
            completion_rate_percent=completion_rate,
            final_score=final_score,
            total_points=total_points,
            max_points=max_points,
            rating=rating,
            safety_gate_passed=safety_passed,
            deployability=deployability,
            responsiveness=responsiveness,
            median_turn_ms=median_turn_ms,
            total_tokens=total_tokens,
            categories=categories,
        )

    def _validate_upstream_config(self, config: dict[str, Any]) -> None:
        expected_ids = [identity[0] for identity in self._registry]
        scenario_ids = [
            _string(value, maximum=128) for value in _list(config.get("scenario_ids"))
        ]
        expected_values: tuple[tuple[str, object], ...] = (
            ("model", self._model_id),
            ("backend", self._mapped_backend),
            ("scenario_count", self._config.scenario_count),
            ("temperature", TEMPERATURE),
            ("timeout_seconds", self._config.request_timeout_seconds),
            ("max_turns", MAX_TURNS),
            ("seed", SEED),
            ("reference_date", REFERENCE_DATE),
            ("concurrency", CONCURRENCY),
            ("error_rate", ERROR_RATE),
            ("alpha", ALPHA),
            ("extra_params", None),
            ("weight_by_difficulty", False),
        )
        if scenario_ids != expected_ids or any(
            config.get(key) != expected for key, expected in expected_values
        ):
            raise _protocol_error()

    def _parse_categories(
        self,
        scores: dict[str, Any],
    ) -> tuple[ToolCallingCategoryScore, ...]:
        rows: list[ToolCallingCategoryScore] = []
        gradable_by_category: dict[str, list[ToolCallingScenarioResult]] = {}
        for scenario in self._scenario_results:
            if not scenario.infrastructure_excluded:
                gradable_by_category.setdefault(scenario.category, []).append(scenario)
        for raw in _list(scores.get("category_scores")):
            item = _mapping(raw)
            category = _string(item.get("category"), maximum=1)
            label = _string(item.get("label"), maximum=128)
            if label != _CATEGORY_LABELS.get(category):
                raise _protocol_error()
            category_rows = gradable_by_category.get(category)
            if category_rows is None:
                raise _protocol_error()
            row = ToolCallingCategoryScore(
                category=category,
                label=label,
                earned_points=_integer(item.get("earned")),
                max_points=_integer(item.get("max")),
                percent=_number(item.get("percent"), maximum=100.0),
                pass_count=_integer(item.get("pass_count")),
                partial_count=_integer(item.get("partial_count")),
                fail_count=_integer(item.get("fail_count")),
            )
            if (
                row.max_points != len(category_rows) * 2
                or row.earned_points != sum(value.points for value in category_rows)
                or row.pass_count
                != sum(
                    value.status is ToolCallingScenarioStatus.PASS
                    for value in category_rows
                )
                or row.partial_count
                != sum(
                    value.status is ToolCallingScenarioStatus.PARTIAL
                    for value in category_rows
                )
                or row.fail_count
                != sum(
                    value.status is ToolCallingScenarioStatus.FAIL
                    for value in category_rows
                )
                or row.percent != round(row.earned_points / row.max_points * 100)
            ):
                raise _protocol_error()
            rows.append(row)
        expected_categories = tuple(sorted(gradable_by_category))
        if tuple(row.category for row in rows) != expected_categories:
            raise _protocol_error()
        return tuple(rows)

    def _validate_envelope_scenarios(self, scores: dict[str, Any]) -> None:
        raw_results = _list(scores.get("scenario_results"))
        if len(raw_results) != len(self._registry) or len(
            self._scenario_results
        ) != len(self._registry):
            raise _protocol_error()
        seen: set[str] = set()
        for raw, callback in zip(raw_results, self._scenario_results, strict=True):
            item = _mapping(raw)
            scenario_id = _string(item.get("scenario_id"), maximum=128)
            failure_kind = item.get("failure_kind")
            if failure_kind is not None:
                failure_kind = _string(failure_kind, maximum=64)
            if (
                scenario_id in seen
                or scenario_id != callback.scenario_id
                or item.get("status") != callback.status.value
                or item.get("points") != callback.points
                or failure_kind != callback.failure_kind
            ):
                raise _protocol_error()
            seen.add(scenario_id)

    def _parse_deployability(
        self,
        scores: dict[str, Any],
    ) -> tuple[int | None, int | None, float | None]:
        keys = ("deployability", "responsiveness", "median_turn_ms", "alpha")
        present = tuple(key in scores for key in keys)
        if not any(present):
            return None, None, None
        if not all(present) or scores.get("alpha") != ALPHA:
            raise _protocol_error()
        return (
            _integer(scores.get("deployability"), maximum=100),
            _integer(scores.get("responsiveness"), maximum=100),
            _number(scores.get("median_turn_ms")),
        )

    def _validate_top_level_mirrors(
        self,
        data: dict[str, Any],
        *,
        final_score: int,
        rating: str,
        max_points: int,
        deployability: int | None,
        responsiveness: int | None,
        warning_count: int,
    ) -> None:
        top_warnings = _list(data.get("safety_warnings"))
        expected_total_scenarios = max_points // 2 if max_points else None
        if (
            data.get("final_score") != final_score
            or data.get("rating") != rating
            or len(top_warnings) != warning_count
            or data.get("deployability") != deployability
            or data.get("responsiveness") != responsiveness
            or data.get("total_scenarios") != expected_total_scenarios
        ):
            raise _protocol_error()

    def _completed_result(
        self,
        parsed: _ParsedAggregates,
    ) -> ToolCallingBenchmarkResult:
        context = self._require_context()
        gradable = tuple(
            row for row in self._scenario_results if not row.infrastructure_excluded
        )
        prompt_tokens = sum(row.prompt_tokens for row in gradable)
        completion_tokens = sum(row.completion_tokens for row in gradable)
        if prompt_tokens + completion_tokens != parsed.total_tokens:
            raise _protocol_error()
        return ToolCallingBenchmarkResult(
            benchmark_id=context.benchmark_id,
            upstream_run_id=parsed.upstream_run_id,
            config_fingerprint=parsed.config_fingerprint,
            server_id=self._server_id,
            server_name=self._server_name,
            server_endpoint=self._server_endpoint,
            model_id=self._model_id,
            backend=self._backend,
            integration_commit=TOOL_EVAL_BENCH_COMMIT,
            upstream_version=parsed.upstream_version,
            schema_version=parsed.schema_version,
            config=self._config,
            started_at=context.started_at,
            completed_at=context.utc_now(),
            status=ToolCallingBenchmarkStatus.COMPLETED,
            cancelled=False,
            error_code=None,
            error_message=None,
            attempted_count=self._config.scenario_count,
            gradable_count=len(gradable),
            excluded_count=len(self._scenario_results) - len(gradable),
            completion_rate_percent=parsed.completion_rate_percent,
            final_score=parsed.final_score,
            total_points=parsed.total_points,
            max_points=parsed.max_points,
            rating=parsed.rating,
            category_k_gradable=any(row.category == "K" for row in gradable),
            safety_gate_passed=parsed.safety_gate_passed,
            deployability=parsed.deployability,
            responsiveness=parsed.responsiveness,
            median_turn_ms=parsed.median_turn_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=parsed.total_tokens,
            categories=parsed.categories,
            scenarios=tuple(self._scenario_results),
            warnings=self._warnings,
            hardware_summary=summarize_hardware_samples(self._hardware_samples),
        )

    def _partial_result(
        self,
        status: ToolCallingBenchmarkStatus,
        *,
        error_code: ToolCallingErrorCode | None,
    ) -> ToolCallingBenchmarkResult:
        context = self._require_context()
        gradable = tuple(
            row for row in self._scenario_results if not row.infrastructure_excluded
        )
        prompt_tokens = sum(row.prompt_tokens for row in gradable)
        completion_tokens = sum(row.completion_tokens for row in gradable)
        return ToolCallingBenchmarkResult(
            benchmark_id=context.benchmark_id,
            upstream_run_id=None,
            config_fingerprint=None,
            server_id=self._server_id,
            server_name=self._server_name,
            server_endpoint=self._server_endpoint,
            model_id=self._model_id,
            backend=self._backend,
            integration_commit=TOOL_EVAL_BENCH_COMMIT,
            upstream_version=None,
            schema_version=None,
            config=self._config,
            started_at=context.started_at,
            completed_at=context.utc_now(),
            status=status,
            cancelled=status is ToolCallingBenchmarkStatus.CANCELLED,
            error_code=error_code,
            error_message=_ERROR_MESSAGES[error_code] if error_code else None,
            attempted_count=self._config.scenario_count,
            gradable_count=len(gradable),
            excluded_count=len(self._scenario_results) - len(gradable),
            completion_rate_percent=None,
            final_score=None,
            total_points=None,
            max_points=None,
            rating=None,
            category_k_gradable=any(row.category == "K" for row in gradable),
            safety_gate_passed=None,
            deployability=None,
            responsiveness=None,
            median_turn_ms=None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            categories=(),
            scenarios=tuple(self._scenario_results),
            warnings=self._warnings,
            hardware_summary=summarize_hardware_samples(self._hardware_samples),
        )

    def _error_result(
        self,
        code: ToolCallingErrorCode,
        error: BaseException,
    ) -> ToolCallingBenchmarkResult:
        result = self._partial_result(
            ToolCallingBenchmarkStatus.ERROR,
            error_code=code,
        )
        logger.info(
            "Tool Calling benchmark=%s model=%s suite=%s status=%s "
            "attempted=%d completed=%d duration=%.3f error_code=%s "
            "exception_class=%s",
            result.benchmark_id,
            self._model_id,
            self._config.suite,
            result.status.value,
            result.attempted_count,
            len(result.scenarios),
            result.wall_time_seconds,
            code,
            type(error).__name__,
        )
        return result

    def _require_context(self) -> BenchmarkContext:
        if self._context is None:
            raise RuntimeError("Tool Calling benchmark context is unavailable")
        return self._context
