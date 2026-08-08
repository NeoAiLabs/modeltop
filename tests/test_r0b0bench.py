"""Contract, runner-boundary, and atomic lifecycle tests for r0b0bench."""
# pyright: reportPrivateUsage=false

import asyncio
import json
import stat
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from modeltop.benchmarks.models import (
    ContextBenchmarkStatus,
    R0b0benchBenchmarkConfig,
    R0b0benchBenchmarkStatus,
    R0b0benchLaneResult,
    R0b0benchLaneStatus,
    R0b0benchMetric,
)
from modeltop.benchmarks.r0b0bench import (
    R0B0BENCH_COMMIT,
    R0B0BENCH_REPORT_SCHEMA,
    R0B0BENCH_VERSION,
    R0b0benchPreparedRun,
    R0b0benchRunner,
    R0b0benchRunnerError,
    R0b0benchRunnerReport,
    R0b0benchRunnerRequest,
    SubprocessR0b0benchRunner,
    _parse_report,
)
from modeltop.benchmarks.r0b0bench_contract import (
    R0B0BENCH_SYSTEMS_ORDER,
    r0b0bench_ordered_selection,
    r0b0bench_profile_lanes,
)
from modeltop.models import DiscoveredModel, ServerConfig
from modeltop.services.r0b0bench import (
    R0b0benchBenchmarkOperationError,
    R0b0benchBenchmarkService,
)
from modeltop.state import (
    ApplicationStateStore,
    ServerStatus,
    initial_application_state,
)

_RUN_ID = "r0b0bench-20260804T120000Z-deadbeef"


def _config() -> R0b0benchBenchmarkConfig:
    return R0b0benchBenchmarkConfig(
        profile="core-subset",
        selected_lanes=("canary",),
        request_timeout_seconds=12.5,
    )


def _request(
    tmp_path: Path, *, api_key: str | None = "EMPTY"
) -> R0b0benchRunnerRequest:
    return R0b0benchRunnerRequest(
        benchmark_id=_RUN_ID,
        config=_config(),
        base_url="http://127.0.0.1:8000/v1",
        model_id="org/model",
        output_root=tmp_path,
        api_key=api_key,
    )


def _lane(
    lane_id: str = "canary",
    *,
    status: R0b0benchLaneStatus = R0b0benchLaneStatus.PASS,
    infra_errors: int = 0,
) -> R0b0benchLaneResult:
    metrics = (
        (
            R0b0benchMetric("passed", True, None),
            R0b0benchMetric("cases", 1, "count"),
        )
        if lane_id == "canary"
        else ()
    )
    return R0b0benchLaneResult(cast(Any, lane_id), status, infra_errors, 0.25, metrics)


def test_profile_selection_and_model_invariants_fail_closed() -> None:
    assert r0b0bench_profile_lanes("systems") == R0B0BENCH_SYSTEMS_ORDER
    assert r0b0bench_ordered_selection("core", ("gsm8k", "canary", "latency")) == (
        "canary",
        "latency",
        "gsm8k",
    )

    with pytest.raises(ValidationError, match="at least one"):
        R0b0benchBenchmarkConfig(selected_lanes=())
    with pytest.raises(ValidationError, match="unique"):
        R0b0benchBenchmarkConfig(selected_lanes=("canary", "canary"))
    with pytest.raises(ValidationError, match="unavailable"):
        R0b0benchBenchmarkConfig(profile="systems", selected_lanes=("qa",))
    with pytest.raises(ValidationError, match="cannot be combined"):
        R0b0benchBenchmarkConfig(profile="core", selected_lanes=("latency", "perf"))
    with pytest.raises(ValueError, match="ratio"):
        R0b0benchMetric("accuracy", 1.1, "ratio")
    with pytest.raises(ValueError, match="invalid for the lane"):
        R0b0benchLaneResult(
            "canary",
            R0b0benchLaneStatus.PASS,
            0,
            0.1,
            (R0b0benchMetric("ttft_mean", 1.0, "ms"),),
        )


def test_prepare_rejects_credentials_and_uses_private_allowlisted_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = SubprocessR0b0benchRunner()
    with pytest.raises(
        R0b0benchRunnerError, match="does not support authenticated"
    ) as captured:
        asyncio.run(runner.prepare(_request(tmp_path, api_key="TOP_SECRET")))
    assert captured.value.code == "unsupported_authenticated_endpoint"
    assert "TOP_SECRET" not in repr(captured.value)

    async def probe(*_args: object, **_kwargs: object) -> tuple[int, bytes]:
        return 0, f"r0b0bench {R0B0BENCH_VERSION}\n".encode()

    monkeypatch.setattr("modeltop.benchmarks.r0b0bench._probe", probe)
    monkeypatch.setenv("OPENAI_API_KEY", "PARENT_SECRET")
    monkeypatch.setenv("UNRELATED_SECRET", "DO_NOT_COPY")
    prepared = asyncio.run(runner.prepare(_request(tmp_path)))
    mode = stat.S_IMODE(prepared.run_directory.stat().st_mode)
    assert mode == 0o700
    assert prepared.run_directory.parent == tmp_path.resolve()
    assert prepared.child_environment["OPENAI_API_KEY"] == "EMPTY"
    assert prepared.child_environment["R0B0BENCH_SERVED_MODEL"] == "org/model"
    assert "UNRELATED_SECRET" not in prepared.child_environment
    assert "PARENT_SECRET" not in repr(prepared)


def _raw_lane(lane_id: str, summary: dict[str, object]) -> dict[str, object]:
    return {
        "lane_id": lane_id,
        "status": "PASS",
        "summary": summary,
        "artifacts": {},
        "infra_errors": 0,
        "imported": False,
        "elapsed_s": 0.25,
    }


def _write_report_fixture(
    prepared: R0b0benchPreparedRun, *, extra_report_field: bool = False
) -> Path:
    summaries: dict[str, dict[str, object]] = {
        "canary": {"passed": True, "n": 1},
        "latency": {"stream": {"ttft_ms_mean": 12.0}, "failed": 0},
        "concurrency": {
            "rows": [
                {
                    "concurrency": 2,
                    "aggregate_output_tok_s": 44.0,
                    "completed": 2,
                    "failed": 0,
                }
            ]
        },
        "throughput": {
            "decode": {"median_client_output_tok_s": 21.0},
            "prefill": {"median_server_prompt_tok_s": 90.0},
            "failed": 0,
        },
    }
    lanes: list[dict[str, object]] = []
    for lane_id in prepared.request.config.selected_lanes:
        raw = _raw_lane(lane_id, summaries[lane_id])
        lane_directory = prepared.run_directory / "lanes" / lane_id
        lane_directory.mkdir(parents=True, exist_ok=True)
        (lane_directory / "lane_result.json").write_text(json.dumps(raw))
        lanes.append(raw)
    report: dict[str, object] = {
        "schema_version": R0B0BENCH_REPORT_SCHEMA,
        "r0b0bench_version": R0B0BENCH_VERSION,
        "run_id": prepared.request.benchmark_id,
        "profile": prepared.request.config.profile,
        "base_url": prepared.request.base_url,
        "model": prepared.request.model_id,
        "systems_lanes": list(R0B0BENCH_SYSTEMS_ORDER),
        "invalid_for_publish": True,
        "started_utc": "2026-08-04T12:00:00+00:00",
        "elapsed_s": 1.0,
        "lanes": lanes,
        "infra_errors_total": 0,
    }
    if extra_report_field:
        report["unexpected"] = "drift"
    report_path = prepared.run_directory / "report.json"
    report_path.write_text(json.dumps(report))
    return report_path


def test_report_parser_normalizes_allowlisted_metrics_and_rejects_drift(
    tmp_path: Path,
) -> None:
    config = R0b0benchBenchmarkConfig()
    run_directory = tmp_path / _RUN_ID
    run_directory.mkdir(mode=0o700)
    prepared = R0b0benchPreparedRun(
        R0b0benchRunnerRequest(
            _RUN_ID,
            config,
            "http://127.0.0.1:8000/v1",
            "org/model",
            tmp_path,
            "EMPTY",
        ),
        {},
        run_directory,
    )
    report_path = _write_report_fixture(prepared)
    report = _parse_report(prepared, report_path)
    assert report.invalid_for_publish
    assert tuple(row.lane_id for row in report.lanes) == config.selected_lanes
    assert report.lanes[1].metrics[0] == R0b0benchMetric("ttft_mean", 12.0, "ms")
    assert report.lanes[2].metrics[:2] == (
        R0b0benchMetric("peak_level", 2, "count"),
        R0b0benchMetric("peak_aggregate_output_rate", 44.0, "tokens/s"),
    )
    assert not any("summary" in repr(row) for row in report.lanes)

    _write_report_fixture(prepared, extra_report_field=True)
    with pytest.raises(R0b0benchRunnerError) as captured:
        _parse_report(prepared, report_path)
    assert captured.value.code == "invalid_upstream_result"


def _store() -> ApplicationStateStore:
    return ApplicationStateStore(
        replace(
            initial_application_state("server", hardware_enabled=False),
            server_status=ServerStatus.ONLINE,
            selected_model_id="org/model",
            available_models=(DiscoveredModel(id="org/model", owned_by="vllm"),),
        )
    )


class _SuccessfulRunner:
    def __init__(self, root: Path) -> None:
        self.root = root

    async def prepare(self, request: R0b0benchRunnerRequest) -> R0b0benchPreparedRun:
        run_directory = self.root / request.benchmark_id
        run_directory.mkdir(mode=0o700)
        return R0b0benchPreparedRun(request, {}, run_directory)

    async def run(
        self,
        prepared: R0b0benchPreparedRun,
        on_start: Any,
        on_finish: Any,
    ) -> R0b0benchRunnerReport:
        row = _lane()
        await on_start("canary", 1, 1)
        await on_finish(row, 1, 1)
        return R0b0benchRunnerReport(
            upstream_run_id=prepared.request.benchmark_id,
            upstream_version=R0B0BENCH_VERSION,
            schema_version=R0B0BENCH_REPORT_SCHEMA,
            profile="core-subset",
            model_id="org/model",
            elapsed_seconds=0.5,
            invalid_for_publish=True,
            infra_errors_total=0,
            lanes=(row,),
            unstarted_lanes=(),
            run_directory=prepared.run_directory,
            cancelled=False,
        )


def _service(
    tmp_path: Path,
    runner: R0b0benchRunner,
    store: ApplicationStateStore | None = None,
) -> tuple[R0b0benchBenchmarkService, ApplicationStateStore]:
    actual_store = store or _store()
    return (
        R0b0benchBenchmarkService(
            actual_store,
            ServerConfig(
                id="server",
                name="Server",
                base_url="http://127.0.0.1:8000/v1",
                api_key="EMPTY",
                backend_hint="vllm",
            ),
            lambda _state: None,
            runner=runner,
            output_root=tmp_path,
            benchmark_id_factory=lambda _started: _RUN_ID,
        ),
        actual_store,
    )


def test_service_reserves_runs_and_publishes_one_terminal_result(
    tmp_path: Path,
) -> None:
    service, store = _service(
        tmp_path, cast(R0b0benchRunner, _SuccessfulRunner(tmp_path))
    )
    pending = service.begin_benchmark(_config())
    assert store.state.r0b0bench_benchmark.status is R0b0benchBenchmarkStatus.VALIDATING
    assert store.state.benchmark_is_active
    with pytest.raises(R0b0benchBenchmarkOperationError, match="already running"):
        service.begin_benchmark(_config())

    result = asyncio.run(service.run_benchmark(pending))
    assert result.status is R0b0benchBenchmarkStatus.COMPLETED
    assert result.upstream_commit == R0B0BENCH_COMMIT
    assert result.completed_count == 1
    assert result.warning_codes == ("filtered_selection",)
    assert result.invalid_for_publish
    assert store.state.r0b0bench_benchmark.latest_result is result
    assert store.state.r0b0bench_benchmark.active_benchmark_id is None
    assert not store.state.benchmark_is_active


def test_service_error_retains_completed_validated_lanes(tmp_path: Path) -> None:
    class FailAfterCompletedRunner(_SuccessfulRunner):
        async def run(
            self,
            prepared: R0b0benchPreparedRun,
            on_start: Any,
            on_finish: Any,
        ) -> R0b0benchRunnerReport:
            for index, lane_id in enumerate(("canary", "latency"), start=1):
                row = _lane(lane_id)
                await on_start(lane_id, index, 4)
                await on_finish(row, index, 4)
            raise R0b0benchRunnerError("upstream_failure")

    service, store = _service(
        tmp_path, cast(R0b0benchRunner, FailAfterCompletedRunner(tmp_path))
    )
    pending = service.begin_benchmark(R0b0benchBenchmarkConfig())
    result = asyncio.run(service.run_benchmark(pending))

    assert result.status is R0b0benchBenchmarkStatus.ERROR
    assert result.error_code == "upstream_failure"
    assert result.completed_count == 2
    assert tuple(row.lane_id for row in result.lanes) == ("canary", "latency")
    assert result.unstarted_lanes == ("concurrency", "throughput")
    assert result.run_directory == tmp_path / _RUN_ID
    assert store.state.r0b0bench_benchmark.latest_result is result


def test_service_cancellation_flushes_partial_progress_and_releases_lane(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()

    class BlockingRunner(_SuccessfulRunner):
        async def run(
            self,
            prepared: R0b0benchPreparedRun,
            on_start: Any,
            on_finish: Any,
        ) -> R0b0benchRunnerReport:
            row = _lane()
            await on_start("canary", 1, 1)
            await on_finish(row, 1, 1)
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    async def scenario() -> None:
        service, store = _service(
            tmp_path, cast(R0b0benchRunner, BlockingRunner(tmp_path))
        )
        pending = service.begin_benchmark(_config())
        task = asyncio.create_task(service.run_benchmark(pending))
        await started.wait()
        service.request_cancellation(pending.benchmark_id)
        result = await task
        assert result.status is R0b0benchBenchmarkStatus.CANCELLED
        assert result.completed_count == 1
        assert result.warning_codes == ("filtered_selection", "cancelled_partial")
        assert result.run_directory is not None
        assert store.state.r0b0bench_benchmark.latest_result is result
        assert not store.state.benchmark_is_active

    asyncio.run(scenario())


def test_service_rejects_active_sibling_lane_before_runner_io(tmp_path: Path) -> None:
    store = _store()
    store.update(
        lambda state: replace(
            state,
            context_benchmark=replace(
                state.context_benchmark,
                status=ContextBenchmarkStatus.RUNNING,
                active_benchmark_id="context",
            ),
        )
    )
    service, _ = _service(
        tmp_path, cast(R0b0benchRunner, _SuccessfulRunner(tmp_path)), store
    )
    with pytest.raises(R0b0benchBenchmarkOperationError, match="Context"):
        service.begin_benchmark(_config())
