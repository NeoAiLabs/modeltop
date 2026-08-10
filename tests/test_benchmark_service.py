"""Behavioral coverage for bounded Concurrency benchmark lifecycle."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace

import pytest

from modeltop.api.chat import (
    ChatStreamEvent,
    ContentDelta,
    GenerationFinished,
    ResponseStarted,
    UsageUpdate,
)
from modeltop.api.errors import RateLimitError
from modeltop.benchmarks.models import (
    ConcurrencyBenchmarkConfig,
    ConcurrencyBenchmarkStatus,
)
from modeltop.chat.models import ChatMessage, GenerationSettings
from modeltop.models import DiscoveredModel, ServerConfig
from modeltop.services.benchmark_service import BenchmarkService
from modeltop.services.generation import GenerationService
from modeltop.state import (
    ApplicationStateStore,
    ServerStatus,
    initial_application_state,
)


class _ConcurrentClient:
    def __init__(
        self,
        *,
        delay: float = 0.01,
        gate: asyncio.Event | None = None,
        timeout_every: int = 0,
    ) -> None:
        self.delay = delay
        self.gate = gate
        self.timeout_every = timeout_every
        self.active = 0
        self.peak_active = 0
        self.calls = 0
        self.messages: list[tuple[ChatMessage, ...]] = []
        self.settings: list[GenerationSettings] = []
        self.reached_peak = asyncio.Event()

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
        call = self.calls
        self.messages.append(tuple(messages))
        self.settings.append(settings)
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        if self.peak_active >= 2:
            self.reached_peak.set()
        try:
            yield ResponseStarted(200)
            if self.timeout_every and call % self.timeout_every == 0:
                await asyncio.sleep(1.0)
            elif self.gate is not None:
                await self.gate.wait()
            else:
                await asyncio.sleep(self.delay)
            yield ContentDelta(f"answer-{call}")
            yield UsageUpdate(5, 10, 15)
            yield GenerationFinished("stop", True)
        finally:
            self.active -= 1


class _FailingClient:
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
        if self.calls < 0:
            yield ResponseStarted(200)
        raise RateLimitError(
            "Server rate limit reached",
            "fixture rate limit",
            status_code=429,
        )


def _service(
    client: _ConcurrentClient | _FailingClient,
) -> tuple[BenchmarkService, ApplicationStateStore]:
    state = initial_application_state("server", hardware_enabled=False)
    state = replace(
        state,
        server_status=ServerStatus.ONLINE,
        selected_model_id="model",
        available_models=(DiscoveredModel(id="model", owned_by="vllm"),),
    )
    store = ApplicationStateStore(state)
    server = ServerConfig(
        id="server",
        name="Server",
        base_url="http://127.0.0.1:8000/v1",
        api_key="secret",
    )
    service = BenchmarkService(
        GenerationService(client, publish_interval_seconds=0.0),
        store,
        server,
        0.01,
        lambda updated: None,
        benchmark_id_factory=lambda started: "benchmark",
    )
    return service, store


def test_fixed_benchmark_enforces_bound_and_preserves_request_contract() -> None:
    async def scenario() -> None:
        client = _ConcurrentClient()
        service, store = _service(client)
        config = ConcurrencyBenchmarkConfig(
            mode="fixed",
            concurrency_levels=(3,),
            requests_per_level=8,
            warmup_requests=0,
            prompt="fixed prompt",
            system_prompt=" system ",
            max_tokens=64,
            temperature=0.0,
            top_p=1.0,
            seed=42,
            request_timeout_seconds=1.0,
            thinking_mode="disabled",
            delay_between_levels_seconds=0.0,
        )
        pending = service.begin_benchmark(config)
        result = await service.run_benchmark(pending)
        assert result.status is ConcurrencyBenchmarkStatus.COMPLETED
        assert client.peak_active == 3
        assert client.peak_active <= config.concurrency_levels[0]
        level = result.levels[0]
        assert level.attempted_requests == level.successful_requests == 8
        assert level.total_completion_tokens == 80
        assert level.aggregate_output_tokens_per_second > 0
        assert level.requests_per_second > 0
        assert level.token_count_mode == "exact"
        assert [request.sequence_number for request in level.requests] == list(
            range(1, 9)
        )
        assert all(settings.enable_thinking is False for settings in client.settings)
        assert all(
            request.total_latency_seconds is not None for request in level.requests
        )
        assert any(request.queue_wait_seconds > 0 for request in level.requests[3:])
        assert all(
            messages
            == (
                ChatMessage("system", "system"),
                ChatMessage(
                    "user",
                    f"fixed prompt\n\n[concurrency-request measured 3/{sequence}]",
                ),
            )
            for sequence, messages in enumerate(client.messages, start=1)
        )
        assert all(
            settings.max_tokens == 64 and settings.stream
            for settings in client.settings
        )
        assert store.state.concurrency_benchmark.latest_result == result
        assert store.state.active_generation_id is None

    asyncio.run(scenario())


def test_unique_prompt_suffix_can_be_disabled_for_cache_hit_runs() -> None:
    async def scenario() -> None:
        client = _ConcurrentClient()
        service, _ = _service(client)
        config = ConcurrencyBenchmarkConfig(
            mode="fixed",
            concurrency_levels=(2,),
            requests_per_level=3,
            warmup_requests=0,
            prompt="cached prompt",
            unique_prompt_suffix_per_request=False,
            request_timeout_seconds=1.0,
            delay_between_levels_seconds=0.0,
        )
        result = await service.run_benchmark(service.begin_benchmark(config))
        assert result.status is ConcurrencyBenchmarkStatus.COMPLETED
        assert client.messages == [(ChatMessage("user", "cached prompt"),)] * 3

    asyncio.run(scenario())


def test_warmups_are_excluded_and_sweep_levels_do_not_overlap() -> None:
    async def scenario() -> None:
        client = _ConcurrentClient(delay=0.002)
        service, _ = _service(client)
        config = ConcurrencyBenchmarkConfig(
            mode="sweep",
            concurrency_levels=(1, 2),
            requests_per_level=3,
            warmup_requests=1,
            request_timeout_seconds=1.0,
            delay_between_levels_seconds=0.01,
        )
        result = await service.run_benchmark(service.begin_benchmark(config))
        assert client.calls == 8
        assert [level.concurrency for level in result.levels] == [1, 2]
        assert [level.attempted_requests for level in result.levels] == [3, 3]
        assert client.peak_active == 2
        prompts = [messages[-1].content for messages in client.messages]
        assert len(prompts) == len(set(prompts))
        assert prompts[0].endswith("[concurrency-request warmup 1/1]")
        assert prompts[1].endswith("[concurrency-request measured 1/1]")

    asyncio.run(scenario())


def test_independent_timeouts_release_slots_and_remain_isolated() -> None:
    async def scenario() -> None:
        client = _ConcurrentClient(delay=0.001, timeout_every=2)
        service, _ = _service(client)
        config = ConcurrencyBenchmarkConfig(
            mode="fixed",
            concurrency_levels=(2,),
            requests_per_level=6,
            warmup_requests=0,
            request_timeout_seconds=0.02,
            delay_between_levels_seconds=0.0,
        )
        result = await service.run_benchmark(service.begin_benchmark(config))
        level = result.levels[0]
        assert level.successful_requests == 3
        assert level.timed_out_requests == 3
        assert level.failed_requests == 0
        assert client.active == 0
        assert client.peak_active == 2

    asyncio.run(scenario())


def test_active_cancellation_retains_partial_rows_and_closes_streams() -> None:
    async def scenario() -> None:
        gate = asyncio.Event()
        client = _ConcurrentClient(gate=gate)
        service, store = _service(client)
        config = ConcurrencyBenchmarkConfig(
            mode="fixed",
            concurrency_levels=(2,),
            requests_per_level=8,
            warmup_requests=0,
            request_timeout_seconds=10.0,
            delay_between_levels_seconds=0.0,
        )
        pending = service.begin_benchmark(config)
        task = asyncio.create_task(service.run_benchmark(pending))
        await asyncio.wait_for(client.reached_peak.wait(), timeout=1.0)
        service.request_cancellation(pending.benchmark_id)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        lane = store.state.concurrency_benchmark
        assert lane.status is ConcurrencyBenchmarkStatus.CANCELLED
        assert lane.latest_result is not None
        assert lane.latest_result.cancelled
        assert (
            lane.latest_result.error == "Benchmark cancelled — partial results retained"
        )
        assert client.active == 0
        if lane.latest_result.levels:
            level = lane.latest_result.levels[0]
            assert level.attempted_requests <= 2
            assert level.cancelled_requests == level.attempted_requests
            assert level.configured_requests - level.attempted_requests >= 6

    asyncio.run(scenario())


def test_all_failed_warmups_error_and_first_eight_failures_stop_sweep() -> None:
    async def scenario() -> None:
        warmup_client = _FailingClient()
        warmup_service, _ = _service(warmup_client)
        warmup_config = ConcurrencyBenchmarkConfig(
            mode="fixed",
            concurrency_levels=(2,),
            requests_per_level=4,
            warmup_requests=2,
            request_timeout_seconds=1.0,
            delay_between_levels_seconds=0.0,
        )
        warmup_result = await warmup_service.run_benchmark(
            warmup_service.begin_benchmark(warmup_config)
        )
        assert warmup_result.status is ConcurrencyBenchmarkStatus.ERROR
        assert warmup_result.levels == ()
        assert warmup_client.calls == 2

        client = _FailingClient()
        service, _ = _service(client)
        config = ConcurrencyBenchmarkConfig(
            mode="sweep",
            concurrency_levels=(4, 8),
            requests_per_level=20,
            warmup_requests=0,
            request_timeout_seconds=1.0,
            delay_between_levels_seconds=0.0,
        )
        result = await service.run_benchmark(service.begin_benchmark(config))
        assert result.status is ConcurrencyBenchmarkStatus.COMPLETED
        assert len(result.levels) == 1
        level = result.levels[0]
        assert level.early_stopped
        assert level.attempted_requests >= 8
        assert level.attempted_requests < level.configured_requests
        assert all(not request.success for request in level.requests[:8])
        assert any(
            observation.code == "first_failures" for observation in result.observations
        )

    asyncio.run(scenario())


def test_between_level_delay_is_immediately_cancellable() -> None:
    async def scenario() -> None:
        client = _ConcurrentClient(delay=0.001)
        service, store = _service(client)
        config = ConcurrencyBenchmarkConfig(
            mode="sweep",
            concurrency_levels=(1, 2),
            requests_per_level=1,
            warmup_requests=0,
            request_timeout_seconds=1.0,
            delay_between_levels_seconds=10.0,
        )
        pending = service.begin_benchmark(config)
        task = asyncio.create_task(service.run_benchmark(pending))
        for _ in range(200):
            if (
                store.state.concurrency_benchmark.status
                is ConcurrencyBenchmarkStatus.BETWEEN_LEVELS
            ):
                break
            await asyncio.sleep(0.005)
        service.request_cancellation(pending.benchmark_id)
        result = await asyncio.wait_for(task, timeout=1.0)
        assert result.status is ConcurrencyBenchmarkStatus.CANCELLED
        assert len(result.levels) == 1
        assert client.calls == 1

    asyncio.run(scenario())
