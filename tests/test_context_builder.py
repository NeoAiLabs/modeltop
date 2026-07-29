"""Bounded deterministic Context prompt construction contracts."""

import asyncio

from modeltop.benchmarks.context_builder import build_context_prompt
from modeltop.benchmarks.context_retrieval import (
    RETRIEVAL_INSTRUCTION,
    RetrievalMarkerSpec,
    RetrievalPromptSpec,
)
from modeltop.benchmarks.models import ContextBenchmarkConfig
from modeltop.chat.metrics import CharacterTokenCounter


def test_token_prompt_is_deterministic_and_within_tolerance() -> None:
    async def scenario() -> None:
        config = ContextBenchmarkConfig(
            mode="fixed", target_lengths=(512,), warmup_requests=0
        )
        first = await build_context_prompt(config, 512, CharacterTokenCounter())
        second = await build_context_prompt(config, 512, CharacterTokenCounter())
        assert first.messages == second.messages
        assert abs(first.measurement.builder_difference) <= 6
        assert first.measurement.estimated
        assert first.measurement.counter_name == "character-estimate"

    asyncio.run(scenario())


def test_character_prompt_hits_exact_visible_target() -> None:
    async def scenario() -> None:
        config = ContextBenchmarkConfig(
            mode="fixed",
            target_lengths=(1200,),
            context_unit="characters",
            warmup_requests=0,
        )
        prompt = await build_context_prompt(config, 1200, CharacterTokenCounter())
        assert prompt.measurement.visible_content_characters == 1200
        assert prompt.measurement.builder_difference == 0

    asyncio.run(scenario())


def test_retrieval_marker_is_single_and_inside_target() -> None:
    async def scenario() -> None:
        config = ContextBenchmarkConfig(
            mode="retrieval",
            target_lengths=(512,),
            retrieval_enabled=True,
            warmup_requests=0,
        )
        spec = RetrievalPromptSpec(
            (
                RetrievalMarkerSpec(
                    "MODELTOP_RETRIEVAL_KEY", "amber-cloud-1234", "middle"
                ),
            ),
            RETRIEVAL_INSTRUCTION,
        )
        prompt = await build_context_prompt(
            config, 512, CharacterTokenCounter(), retrieval_prompt=spec
        )
        content = prompt.messages[-1].content
        assert content.count("MODELTOP_RETRIEVAL_KEY: amber-cloud-1234") == 1
        assert content.endswith(RETRIEVAL_INSTRUCTION)
        assert prompt.measurement.local_prompt_tokens <= 512 + 6

    asyncio.run(scenario())


def test_build_cancellation_propagates() -> None:
    async def scenario() -> None:
        config = ContextBenchmarkConfig(
            mode="fixed", target_lengths=(200_000,), warmup_requests=0
        )
        task = asyncio.create_task(
            build_context_prompt(config, 200_000, CharacterTokenCounter())
        )
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return
        raise AssertionError("prompt build ignored cancellation")

    asyncio.run(scenario())
