"""Sequential Drafter orchestration, telemetry, cancel, and exclusion."""

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from modeltop.api.chat import (
    ChatStreamEvent,
    ContentDelta,
    GenerationFinished,
    StreamDone,
    UsageUpdate,
)
from modeltop.api.errors import HTTPResponseError
from modeltop.benchmarks.models import (
    DrafterBenchmarkConfig,
    DrafterBenchmarkStatus,
    SpeedTestConfig,
    SpeedTestStatus,
)
from modeltop.chat.models import ChatMessage, GenerationSettings
from modeltop.models import DiscoveredModel, ServerConfig
from modeltop.services.drafter_benchmark import (
    DrafterBenchmarkOperationError,
    DrafterBenchmarkService,
)
from modeltop.services.generation import GenerationService
from modeltop.services.speed_test import SpeedTestOperationError, SpeedTestService
from modeltop.state import (
    ApplicationState,
    ApplicationStateStore,
    ServerStatus,
    initial_application_state,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.1
        return self.value


class _ScriptedClient:
    def __init__(self, scripts: list[list[ChatStreamEvent | Exception]]) -> None:
        self.scripts = scripts
        self.requests: list[
            tuple[str, Sequence[ChatMessage], GenerationSettings, float | None]
        ] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def stream_chat_completion(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        settings: GenerationSettings,
        *,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[ChatStreamEvent]:
        self.requests.append((model, messages, settings, timeout_seconds))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            for item in self.scripts.pop(0):
                await asyncio.sleep(0)
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            self.in_flight -= 1


class _GatedClient:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.closed = False

    async def stream_chat_completion(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        settings: GenerationSettings,
        *,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[ChatStreamEvent]:
        del model, messages, settings, timeout_seconds
        try:
            yield ContentDelta("partial")
            self.entered.set()
            await asyncio.Event().wait()
        finally:
            self.closed = True


def _success(
    text: str = "answer",
    *,
    draft: int | None = 6,
    accepted: int | None = 4,
    rate: float | None = 4 / 6,
) -> list[ChatStreamEvent | Exception]:
    return [
        ContentDelta(text),
        UsageUpdate(
            10,
            4,
            14,
            draft_tokens=draft,
            accepted_tokens=accepted,
            acceptance_rate=rate,
        ),
        GenerationFinished("stop", True),
        StreamDone(),
    ]


def _success_no_telemetry(text: str = "answer") -> list[ChatStreamEvent | Exception]:
    return _success(text, draft=None, accepted=None, rate=None)


def _store(*, online: bool = True, model: bool = True) -> ApplicationStateStore:
    state = initial_application_state("server", hardware_enabled=False)
    return ApplicationStateStore(
        replace(
            state,
            server_status=ServerStatus.ONLINE if online else ServerStatus.OFFLINE,
            selected_model_id="model" if model else None,
            available_models=(
                (DiscoveredModel(id="model", owned_by="vllm"),) if model else ()
            ),
        )
    )


def _service(
    client: _ScriptedClient | _GatedClient,
    store: ApplicationStateStore,
    *,
    on_state_change: Callable[[ApplicationState], None] | None = None,
) -> DrafterBenchmarkService:
    server = ServerConfig(
        id="server",
        name="Local",
        base_url="http://localhost:8000/v1",
        backend_hint=None,
    )
    return DrafterBenchmarkService(
        GenerationService(client, clock=_Clock()),
        store,
        server,
        on_state_change or (lambda _state: None),
        utc_now=lambda: datetime(2026, 7, 27, tzinfo=UTC),
        benchmark_id_factory=lambda _started_at: f"drafter-{id(client):x}",
    )


def test_warmups_measured_telemetry_and_aggregates() -> None:
    async def scenario() -> None:
        client = _ScriptedClient(
            [_success("warm"), _success("one"), _success("two"), _success("three")]
        )
        states: list[ApplicationState] = []
        service = _service(client, _store(), on_state_change=states.append)
        config = DrafterBenchmarkConfig(
            prompt="drafter prompt",
            warmup_runs=1,
            measured_runs=3,
            max_tokens=64,
            temperature=0.0,
            top_p=1.0,
            seed=7,
            request_timeout_seconds=30,
        )
        result = await service.run_benchmark(service.begin_benchmark(config))

        assert client.max_in_flight == 1
        assert result.status is DrafterBenchmarkStatus.COMPLETED
        assert [run.warmup for run in result.run_results] == [True, False, False, False]
        assert result.successful_runs == 3
        assert result.acceptance_rate.count == 3
        assert result.acceptance_rate.mean == pytest.approx(4 / 6)
        assert result.draft_tokens.mean == 6
        assert result.accepted_tokens.mean == 4
        assert result.speculative_telemetry_available
        assert result.observations == ()
        assert service.state.drafter_benchmark.latest_result == result
        assert service.state.drafter_benchmark.active_benchmark_id is None
        assert any(
            state.drafter_benchmark.progress is not None
            and state.drafter_benchmark.progress.latest_metrics is not None
            for state in states
        )
        assert all(request[0] == "model" for request in client.requests)
        assert all(
            request[1] == (ChatMessage("user", "drafter prompt"),)
            for request in client.requests
        )

    asyncio.run(scenario())


def test_missing_telemetry_observation() -> None:
    async def scenario() -> None:
        client = _ScriptedClient([_success_no_telemetry(), _success_no_telemetry()])
        service = _service(client, _store())
        result = await service.run_benchmark(
            service.begin_benchmark(
                DrafterBenchmarkConfig(warmup_runs=0, measured_runs=2)
            )
        )
        assert result.status is DrafterBenchmarkStatus.COMPLETED
        assert not result.speculative_telemetry_available
        assert len(result.observations) == 1
        assert result.observations[0].code == "speculative_telemetry_unavailable"
        assert result.acceptance_rate.count == 0

    asyncio.run(scenario())


def test_continue_on_error_and_stop_on_error() -> None:
    async def scenario() -> None:
        failure = HTTPResponseError("Readable failure", "safe detail")
        stop_client = _ScriptedClient([_success(), [failure], _success()])
        stop_service = _service(stop_client, _store())
        stopped = await stop_service.run_benchmark(
            stop_service.begin_benchmark(
                DrafterBenchmarkConfig(warmup_runs=1, measured_runs=2)
            )
        )
        assert stopped.status is DrafterBenchmarkStatus.FAILED
        assert len(stopped.run_results) == 2
        assert stopped.error == "Readable failure"

        continue_client = _ScriptedClient([[failure], _success(), [failure]])
        continue_service = _service(continue_client, _store())
        continued = await continue_service.run_benchmark(
            continue_service.begin_benchmark(
                DrafterBenchmarkConfig(
                    warmup_runs=0,
                    measured_runs=3,
                    continue_on_error=True,
                )
            )
        )
        assert continued.status is DrafterBenchmarkStatus.COMPLETED_WITH_ERRORS
        assert continued.successful_runs == 1
        assert continued.failed_runs == 2

    asyncio.run(scenario())


def test_cancel_mid_request_records_partial() -> None:
    async def scenario() -> None:
        client = _GatedClient()
        service = _service(client, _store())
        pending = service.begin_benchmark(
            DrafterBenchmarkConfig(warmup_runs=0, measured_runs=1)
        )
        task = asyncio.create_task(service.run_benchmark(pending))
        await client.entered.wait()
        assert service.request_cancellation(pending.benchmark_id)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        lane = service.state.drafter_benchmark
        assert lane.status is DrafterBenchmarkStatus.CANCELLED
        assert lane.latest_result is not None
        assert lane.latest_result.cancelled_runs == 1
        assert client.closed

    asyncio.run(scenario())


def test_preflight_rejects_busy_offline_and_missing_model() -> None:
    offline = _service(_ScriptedClient([]), _store(online=False))
    with pytest.raises(DrafterBenchmarkOperationError, match="offline"):
        offline.begin_benchmark(DrafterBenchmarkConfig())

    missing = _service(_ScriptedClient([]), _store(model=False))
    with pytest.raises(DrafterBenchmarkOperationError, match="Select an available"):
        missing.begin_benchmark(DrafterBenchmarkConfig())

    store = _store()
    active = _service(_ScriptedClient([_success()]), store)
    pending = active.begin_benchmark(
        DrafterBenchmarkConfig(warmup_runs=0, measured_runs=1)
    )
    with pytest.raises(DrafterBenchmarkOperationError, match="already running"):
        active.begin_benchmark(DrafterBenchmarkConfig())
    active.cancel_reservation(pending)


def test_mutual_exclusion_with_speed_test() -> None:
    async def scenario() -> None:
        store = _store()
        gated = _GatedClient()
        drafter = _service(gated, store)
        pending = drafter.begin_benchmark(
            DrafterBenchmarkConfig(warmup_runs=0, measured_runs=1)
        )
        task = asyncio.create_task(drafter.run_benchmark(pending))
        await gated.entered.wait()

        speed = SpeedTestService(
            GenerationService(_ScriptedClient([]), clock=_Clock()),
            store,
            ServerConfig(id="server", name="Local", base_url="http://localhost:8000"),
            lambda state: None,
        )
        with pytest.raises(SpeedTestOperationError, match="Drafter"):
            speed.begin_test(SpeedTestConfig())

        assert drafter.request_cancellation(pending.benchmark_id)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Reverse: speed active blocks drafter.
        speed_store = _store()
        speed_service = SpeedTestService(
            GenerationService(_GatedClient(), clock=_Clock()),
            speed_store,
            ServerConfig(id="server", name="Local", base_url="http://localhost:8000"),
            lambda state: None,
            utc_now=lambda: datetime(2026, 7, 27, tzinfo=UTC),
            run_id_factory=lambda started_at: "speed-active",
        )
        speed_pending = speed_service.begin_test(
            SpeedTestConfig(warmup_runs=0, measured_runs=1)
        )
        assert speed_store.state.speed_test.status is SpeedTestStatus.PREPARING
        blocked = DrafterBenchmarkService(
            GenerationService(_ScriptedClient([]), clock=_Clock()),
            speed_store,
            ServerConfig(id="server", name="Local", base_url="http://localhost:8000"),
            lambda state: None,
        )
        with pytest.raises(DrafterBenchmarkOperationError, match="Speed Test"):
            blocked.begin_benchmark(DrafterBenchmarkConfig())
        speed_service.cancel_reservation(speed_pending)

    asyncio.run(scenario())
