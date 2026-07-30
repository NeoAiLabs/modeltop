"""End-to-end Context benchmark and service lifecycle contracts."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace

from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Static

from modeltop.api.chat import (
    ChatStreamEvent,
    ContentDelta,
    GenerationFinished,
    ResponseStarted,
    UsageUpdate,
)
from modeltop.api.errors import ContextLimitError
from modeltop.benchmarks.models import ContextBenchmarkConfig, ContextBenchmarkStatus
from modeltop.chat.metrics import CharacterTokenCounter
from modeltop.chat.models import ChatMessage, GenerationSettings
from modeltop.models import DiscoveredModel, ServerConfig
from modeltop.services.context_benchmark import ContextBenchmarkService
from modeltop.services.generation import GenerationService
from modeltop.state import (
    ApplicationStateStore,
    ServerStatus,
    initial_application_state,
)
from modeltop.widgets.context_results import ContextResultsPanel


class _ContextClient:
    def __init__(self) -> None:
        self.calls = 0
        self.messages: list[tuple[ChatMessage, ...]] = []
        self.settings: list[GenerationSettings] = []

    async def stream_chat_completion(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        settings: GenerationSettings,
        *,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[ChatStreamEvent]:
        del model, timeout_seconds
        self.calls += 1
        self.messages.append(tuple(messages))
        self.settings.append(settings)
        assert settings.stream
        marker_values = tuple(
            line.split(": ", 1)[1]
            for line in messages[-1].content.splitlines()
            if line.startswith("MODELTOP_RETRIEVAL_KEY: ")
        )
        content = "\n".join(marker_values) if marker_values else "ok"
        yield ResponseStarted(200)
        yield ContentDelta(content)
        yield UsageUpdate(512, 2, 514)
        yield GenerationFinished("stop", True)


class _ThresholdClient:
    def __init__(self, maximum_prompt_tokens: int) -> None:
        self.maximum_prompt_tokens = maximum_prompt_tokens
        self.calls: list[int] = []
        self.counter = CharacterTokenCounter()

    async def stream_chat_completion(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        settings: GenerationSettings,
        *,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[ChatStreamEvent]:
        del model, settings, timeout_seconds
        prompt_tokens = self.counter.count_messages(messages).total_tokens
        self.calls.append(prompt_tokens)
        if prompt_tokens > self.maximum_prompt_tokens:
            raise ContextLimitError(
                "Conversation exceeds the server context limit",
                "fixture context rejection",
                status_code=400,
            )
        yield ResponseStarted(200)
        yield ContentDelta("ok")
        yield UsageUpdate(prompt_tokens, 2, prompt_tokens + 2)
        yield GenerationFinished("stop", True)


class _GatedClient:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream_chat_completion(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        settings: GenerationSettings,
        *,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[ChatStreamEvent]:
        del model, messages, settings, timeout_seconds
        yield ResponseStarted(200)
        self.started.set()
        await self.release.wait()
        yield ContentDelta("late")
        yield GenerationFinished("stop", True)


class _FirstTimeoutClient:
    def __init__(self) -> None:
        self.calls = 0

    async def stream_chat_completion(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        settings: GenerationSettings,
        *,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[ChatStreamEvent]:
        del model, messages, settings, timeout_seconds
        self.calls += 1
        yield ResponseStarted(200)
        if self.calls == 1:
            await asyncio.sleep(1.0)
        yield ContentDelta("ok")
        yield UsageUpdate(512, 2, 514)
        yield GenerationFinished("stop", True)


def _service(
    client: _ContextClient | _ThresholdClient | _GatedClient | _FirstTimeoutClient,
) -> tuple[ContextBenchmarkService, ApplicationStateStore]:
    state = replace(
        initial_application_state("server", hardware_enabled=False),
        server_status=ServerStatus.ONLINE,
        selected_model_id="model",
        available_models=(DiscoveredModel(id="model", owned_by="vllm"),),
    )
    store = ApplicationStateStore(state)
    service = ContextBenchmarkService(
        GenerationService(client, publish_interval_seconds=0.0),
        store,
        ServerConfig(id="server", name="Server", base_url="http://127.0.0.1:8000/v1"),
        lambda state: None,
        benchmark_id_factory=lambda started: "context-benchmark",
    )
    return service, store


def test_fixed_context_service_omits_warmups_and_releases_reservation() -> None:
    async def scenario() -> None:
        client = _ContextClient()
        service, store = _service(client)
        config = ContextBenchmarkConfig(
            mode="fixed",
            target_lengths=(512,),
            repetitions_per_length=2,
            warmup_requests=1,
            maximum_output_tokens=8,
            request_timeout_seconds=1.0,
            delay_between_lengths_seconds=0.0,
            thinking_mode="disabled",
        )
        pending = service.begin_benchmark(config)
        assert store.state.context_benchmark.status is ContextBenchmarkStatus.VALIDATING
        result = await service.run_benchmark(pending)
        assert result.status is ContextBenchmarkStatus.COMPLETED
        assert client.calls == 3
        assert len(result.lengths) == 1
        assert len(result.lengths[0].requests) == 2
        assert result.lengths[0].successful_requests == 2
        assert store.state.context_benchmark.latest_result is result
        assert store.state.context_benchmark.active_benchmark_id is None
        assert store.state.active_generation_id is None
        assert all(
            call[-1].content.endswith("Use the available response budget.")
            for call in client.messages
        )
        assert all(settings.enable_thinking is False for settings in client.settings)

    asyncio.run(scenario())


def test_context_cancellation_before_network_is_terminal() -> None:
    async def scenario() -> None:
        client = _ContextClient()
        service, store = _service(client)
        pending = service.begin_benchmark(
            ContextBenchmarkConfig(
                mode="fixed", target_lengths=(512,), warmup_requests=0
            )
        )
        service.cancel_reservation(pending)
        result = store.state.context_benchmark.latest_result
        assert result is not None
        assert result.status is ContextBenchmarkStatus.CANCELLED
        assert client.calls == 0
        assert store.state.context_benchmark.status is ContextBenchmarkStatus.CANCELLED
        assert store.state.context_benchmark.active_benchmark_id is None

    asyncio.run(scenario())


def test_context_result_rendering_does_not_expose_retrieval_values() -> None:
    async def scenario() -> None:
        client = _ContextClient()
        service, _store = _service(client)
        pending = service.begin_benchmark(
            ContextBenchmarkConfig(
                mode="retrieval",
                target_lengths=(512,),
                repetitions_per_length=1,
                warmup_requests=0,
                retrieval_enabled=True,
                retrieval_positions=("middle",),
                retrieval_key="private-key-1234",
                delay_between_lengths_seconds=0.0,
            )
        )
        result = await service.run_benchmark(pending)
        score = result.lengths[0].requests[0].retrieval_results[0]
        assert score.status == "pass"
        assert score.expected_value == "private-key-1234"

        class ResultApp(App[None]):
            def compose(self) -> ComposeResult:
                yield ContextResultsPanel()

        app = ResultApp()
        async with app.run_test(size=(100, 30)):
            panel = app.query_one(ContextResultsPanel)
            panel.update_result(result)
            rendered = "\n".join(
                value.plain
                if isinstance(value := widget.render(), Text)
                else str(value)
                for widget in panel.query(Static)
            )
            assert "MODELTOP_RETRIEVAL_KEY" in rendered
            assert "PASS" in rendered
            assert "private-key-1234" not in rendered

    asyncio.run(scenario())


def test_probe_execution_advances_only_consistent_bounds() -> None:
    async def scenario() -> None:
        client = _ThresholdClient(maximum_prompt_tokens=1200)
        service, _store = _service(client)
        pending = service.begin_benchmark(
            ContextBenchmarkConfig(
                mode="probe",
                target_lengths=(512,),
                repetitions_per_length=2,
                warmup_requests=0,
                maximum_output_tokens=8,
                probe_start_tokens=512,
                probe_maximum_tokens=2048,
                probe_resolution_tokens=256,
                delay_between_lengths_seconds=0.0,
            )
        )
        result = await service.run_benchmark(pending)
        assert result.status is ContextBenchmarkStatus.COMPLETED
        assert result.probe_bounds is not None
        assert result.probe_bounds.highest_confirmed_success == 1024
        assert result.probe_bounds.first_confirmed_rejection == 1280
        assert result.probe_bounds.attempted_targets == (
            512,
            1024,
            2048,
            1536,
            1280,
        )
        assert len(client.calls) == 10

    asyncio.run(scenario())


def test_sweep_early_stops_after_fully_rejected_length() -> None:
    async def scenario() -> None:
        client = _ThresholdClient(maximum_prompt_tokens=700)
        service, _store = _service(client)
        pending = service.begin_benchmark(
            ContextBenchmarkConfig(
                mode="sweep",
                target_lengths=(512, 1024, 2048),
                repetitions_per_length=2,
                warmup_requests=0,
                maximum_output_tokens=8,
                delay_between_lengths_seconds=0.0,
            )
        )
        result = await service.run_benchmark(pending)
        assert tuple(length.target_length for length in result.lengths) == (512, 1024)
        assert result.lengths[-1].context_rejected_requests == 2
        assert result.lengths[-1].early_stopped
        assert all(prompt_tokens < 2048 for prompt_tokens in client.calls)

    asyncio.run(scenario())


def test_timeout_policy_controls_later_measured_requests() -> None:
    async def run(continue_after_timeout: bool) -> tuple[int, int, int]:
        client = _FirstTimeoutClient()
        service, _store = _service(client)
        pending = service.begin_benchmark(
            ContextBenchmarkConfig(
                mode="fixed",
                target_lengths=(512,),
                repetitions_per_length=2,
                warmup_requests=0,
                request_timeout_seconds=0.01,
                continue_after_timeout=continue_after_timeout,
            )
        )
        result = await service.run_benchmark(pending)
        length = result.lengths[0]
        return client.calls, length.timed_out_requests, length.successful_requests

    assert asyncio.run(run(False)) == (1, 1, 0)
    assert asyncio.run(run(True)) == (2, 1, 1)


def test_active_network_cancellation_closes_context_lane() -> None:
    async def scenario() -> None:
        client = _GatedClient()
        service, store = _service(client)
        pending = service.begin_benchmark(
            ContextBenchmarkConfig(
                mode="fixed",
                target_lengths=(512,),
                warmup_requests=0,
                request_timeout_seconds=5.0,
            )
        )
        task = asyncio.create_task(service.run_benchmark(pending))
        await asyncio.wait_for(client.started.wait(), timeout=1.0)
        service.request_cancellation(pending.benchmark_id)
        result = await asyncio.wait_for(task, timeout=1.0)
        assert result.status is ContextBenchmarkStatus.CANCELLED
        assert store.state.context_benchmark.active_benchmark_id is None
        assert store.state.active_generation_id is None

    asyncio.run(scenario())
