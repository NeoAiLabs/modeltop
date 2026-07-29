"""Deterministic, bounded construction of Context benchmark prompts."""

import asyncio
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass

from modeltop.benchmarks.context_retrieval import (
    RetrievalPromptSpec,
    insert_retrieval_markers,
)
from modeltop.benchmarks.models import (
    ContextBenchmarkConfig,
    ContextPromptBuildProgress,
    ContextPromptMeasurement,
)
from modeltop.benchmarks.token_budget import (
    calculate_token_budget,
    effective_context_output_budget,
)
from modeltop.chat.metrics import TokenCounter, count_chat_messages, token_counter_name
from modeltop.chat.models import ChatMessage

CONTEXT_SYSTEM_MESSAGE = (
    "You are a precise assistant. Follow the user's final instruction."
)
CONTEXT_PERFORMANCE_INSTRUCTION = (
    "Using the context above, describe how the records are organised. "
    "Use the available response budget."
)
TARGET_FRAGMENT_CHARACTERS = 512
BUILD_BATCH_FRAGMENTS = 32
MAX_BUILD_FRAGMENTS = 32768
MAX_BUILD_ITERATIONS = 64
RETRIEVAL_PREVIEW_CHARACTERS = 512
_PROGRESS_INTERVAL_SECONDS = 0.1

_BUILT_IN_CORPUS = (
    "Aster Station keeps civic records in chronological ledgers. Each ledger "
    "groups entries by district, then assigns a stable record number.",
    "The Northbridge archive describes fictional waterways, maintenance dates, "
    "and responsible crews without referring to real people or places.",
    "Orchard Library catalogues imaginary expeditions by year, region, vehicle, "
    "and specimen class so a reader can compare related records.",
    "Harbor Workshop stores inspection notes in numbered sections. Every section "
    "contains a category, a sequence, and a concise observation.",
)


@dataclass(frozen=True, slots=True)
class BuiltContextPrompt:
    """Ephemeral complete messages plus bounded measurement metadata."""

    messages: tuple[ChatMessage, ...]
    measurement: ContextPromptMeasurement
    section_boundaries: tuple[int, ...]
    warnings: tuple[str, ...]


type BuildProgressCallback = Callable[[ContextPromptBuildProgress], None]


def _synthetic_fragment(index: int, rng: random.Random) -> str:
    district = ("north", "south", "east", "west")[rng.randrange(4)]
    category = ("archive", "garden", "harbor", "transit")[rng.randrange(4)]
    value = rng.randrange(100000, 999999)
    prefix = (
        f"RECORD {index:07d} | district={district} | category={category} | "
        f"value={value}. "
    )
    detail = (
        "This fictional entry is ordered first by district, then by category, "
        "and finally by its ascending record number. "
    )
    repeated = (prefix + detail) * math.ceil(
        TARGET_FRAGMENT_CHARACTERS / (len(prefix) + len(detail))
    )
    return repeated[:TARGET_FRAGMENT_CHARACTERS]


def _repeated_fragment(index: int, base_text: str) -> str:
    prefix = f"SECTION {index:07d}\n"
    available = TARGET_FRAGMENT_CHARACTERS - len(prefix)
    repeats = math.ceil(available / len(base_text))
    return prefix + (base_text * repeats)[:available]


def _corpus_fragment(index: int) -> str:
    prefix = f"PASSAGE {index:07d}\n"
    prose = _BUILT_IN_CORPUS[(index - 1) % len(_BUILT_IN_CORPUS)]
    available = TARGET_FRAGMENT_CHARACTERS - len(prefix)
    repeats = math.ceil(available / len(prose))
    return prefix + (prose * repeats)[:available]


def _make_fragment(
    config: ContextBenchmarkConfig, index: int, rng: random.Random
) -> str:
    if config.content_source == "synthetic":
        return _synthetic_fragment(index, rng)
    if config.content_source == "repeated_text":
        assert config.base_text is not None
        return _repeated_fragment(index, config.base_text)
    return _corpus_fragment(index)


def _measurement(
    *,
    target_length: int,
    config: ContextBenchmarkConfig,
    counter: TokenCounter,
    body: str,
    instruction: str,
    messages: tuple[ChatMessage, ...],
) -> ContextPromptMeasurement:
    complete = count_chat_messages(counter, messages)
    body_tokens = counter.count(body)
    system_tokens = counter.count(CONTEXT_SYSTEM_MESSAGE)
    instruction_tokens = counter.count(instruction)
    component_total = body_tokens + system_tokens + instruction_tokens
    overhead = complete.total_tokens - component_total
    if overhead < 0:
        overhead_value: int | None = None
    else:
        overhead_value = overhead
    visible_characters = sum(len(message.content) for message in messages)
    measured_units = (
        complete.total_tokens if config.context_unit == "tokens" else visible_characters
    )
    return ContextPromptMeasurement(
        requested_length=target_length,
        requested_unit=config.context_unit,
        visible_content_characters=visible_characters,
        body_tokens=body_tokens,
        system_tokens=system_tokens,
        instruction_tokens=instruction_tokens,
        template_overhead_tokens=overhead_value,
        local_prompt_tokens=complete.total_tokens,
        server_prompt_tokens=None,
        builder_difference=measured_units - target_length,
        server_token_difference=None,
        server_token_difference_percent=None,
        counter_name=token_counter_name(counter),
        estimated=complete.estimated,
    )


async def build_context_prompt(
    config: ContextBenchmarkConfig,
    target_length: int,
    counter: TokenCounter,
    *,
    instruction: str = CONTEXT_PERFORMANCE_INSTRUCTION,
    retrieval_prompt: RetrievalPromptSpec | None = None,
    absolute_attempt: int = 0,
    on_progress: BuildProgressCallback | None = None,
) -> BuiltContextPrompt:
    """Build a deterministic prompt within the configured target and safety bounds."""
    if retrieval_prompt is not None:
        instruction = retrieval_prompt.instruction
    budget = calculate_token_budget(
        target_length,
        config.context_unit,
        CONTEXT_SYSTEM_MESSAGE,
        instruction,
        effective_context_output_budget(config),
        config.maximum_context_test_tokens,
        counter,
    )
    tolerance = max(
        1,
        math.ceil(target_length * config.prompt_target_tolerance_percent / 100),
    )
    suffix = f"\nRUN {absolute_attempt}" if config.unique_prompt_suffix_per_run else ""
    desired_body_characters = (
        budget.body_target_units
        if config.context_unit == "characters"
        else max(TARGET_FRAGMENT_CHARACTERS, budget.body_target_units * 6)
    )
    maximum_body_characters = min(
        MAX_BUILD_FRAGMENTS * TARGET_FRAGMENT_CHARACTERS,
        desired_body_characters
        if config.context_unit == "characters"
        else max(desired_body_characters, budget.body_target_units * 10),
    )
    fragments: list[str] = []
    boundaries: list[int] = []
    rng = random.Random(config.random_seed)
    current_characters = 0
    last_publish = 0.0

    def publish(measured: int, iteration: int, *, force: bool = False) -> None:
        nonlocal last_publish
        if on_progress is None:
            return
        now = time.monotonic()
        if not force and now - last_publish < _PROGRESS_INTERVAL_SECONDS:
            return
        last_publish = now
        on_progress(
            ContextPromptBuildProgress(
                target_length=target_length,
                context_unit=config.context_unit,
                measured_count=max(0, measured),
                fragment_count=len(fragments),
                iteration=iteration,
                percentage=min(100.0, max(0.0, measured / target_length * 100)),
            )
        )

    publish(0, 0, force=True)
    try:
        while current_characters < maximum_body_characters:
            if len(fragments) >= MAX_BUILD_FRAGMENTS:
                raise ValueError(
                    "Unable to reach the requested prompt target within "
                    "the fragment limit."
                )
            fragment = _make_fragment(config, len(fragments) + 1, rng)
            fragments.append(fragment)
            current_characters += len(fragment)
            boundaries.append(current_characters)
            if len(fragments) % BUILD_BATCH_FRAGMENTS == 0:
                estimated = (
                    current_characters
                    if config.context_unit == "characters"
                    else budget.fixed_prompt_tokens + current_characters // 4
                )
                publish(estimated, 0)
                await asyncio.sleep(0)

        body_pool = "".join(fragments)
        low = 1
        high = len(body_pool)
        candidate_lengths: list[int]
        if config.context_unit == "characters" and retrieval_prompt is None:
            candidate_lengths = [budget.body_target_units - len(suffix)]
        else:
            candidate_lengths = []
        closest: (
            tuple[int, str, tuple[ChatMessage, ...], ContextPromptMeasurement] | None
        ) = None
        iteration = 0
        while iteration < MAX_BUILD_ITERATIONS:
            iteration += 1
            if candidate_lengths:
                body_length = candidate_lengths.pop()
            elif low <= high:
                body_length = (low + high) // 2
            else:
                break
            if body_length < 1:
                raise ValueError(
                    "Target prompt length is too small for required messages."
                )
            filler_body = body_pool[:body_length]
            if retrieval_prompt is None:
                body = filler_body + suffix
            else:
                section_values: list[str] = []
                remaining = body_length
                for fragment in fragments:
                    if remaining <= 0:
                        break
                    section_values.append(fragment[:remaining])
                    remaining -= len(section_values[-1])
                if any(
                    marker.key in filler_body for marker in retrieval_prompt.markers
                ):
                    raise ValueError(
                        "The configured retrieval key already occurs in filler."
                    )
                marked_body, _ = insert_retrieval_markers(
                    tuple(section_values),
                    retrieval_prompt,
                    random_seed=config.random_seed,
                    target=target_length,
                    absolute_attempt=absolute_attempt,
                )
                body = marked_body + suffix
            user_content = body + "\n\n" + instruction
            messages = (
                ChatMessage("system", CONTEXT_SYSTEM_MESSAGE),
                ChatMessage("user", user_content),
            )
            measurement = _measurement(
                target_length=target_length,
                config=config,
                counter=counter,
                body=body,
                instruction=instruction,
                messages=messages,
            )
            if measurement.local_prompt_tokens > budget.maximum_prompt_tokens:
                difference = measurement.local_prompt_tokens - target_length
                high = body_length - 1
            else:
                difference = measurement.builder_difference
                rank = abs(difference)
                if closest is None or rank < closest[0]:
                    closest = (rank, body, messages, measurement)
                if config.context_unit == "characters" or rank <= tolerance:
                    break
                if difference < 0:
                    low = body_length + 1
                else:
                    high = body_length - 1
            measured = (
                measurement.local_prompt_tokens
                if config.context_unit == "tokens"
                else measurement.visible_content_characters
            )
            publish(measured, iteration)
            await asyncio.sleep(0)

        if closest is None:
            raise ValueError(
                "Unable to construct a prompt below the configured context "
                "safety maximum."
            )
        _, final_body, final_messages, final_measurement = closest
        if (
            config.context_unit == "characters"
            and final_measurement.builder_difference != 0
        ):
            raise ValueError(
                "Unable to construct the exact requested character target."
            )
        warnings: tuple[str, ...] = ()
        if (
            config.context_unit == "tokens"
            and abs(final_measurement.builder_difference) > tolerance
        ):
            warnings = (
                "Prompt builder used the closest safe candidate outside "
                "the configured tolerance.",
            )
        used_length = len(final_body) - len(suffix)
        used_boundaries = tuple(
            boundary for boundary in boundaries if boundary <= used_length
        )
        measured = (
            final_measurement.local_prompt_tokens
            if config.context_unit == "tokens"
            else final_measurement.visible_content_characters
        )
        publish(measured, iteration, force=True)
        return BuiltContextPrompt(
            messages=final_messages,
            measurement=final_measurement,
            section_boundaries=used_boundaries,
            warnings=warnings,
        )
    except MemoryError as error:
        publish(0, 0, force=True)
        raise ValueError(
            "Unable to construct the requested prompt within available memory."
        ) from error
    finally:
        fragments.clear()
        boundaries.clear()
