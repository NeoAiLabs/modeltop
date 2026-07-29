"""Context prompt budgeting with explicit output reserve and provenance."""

import math
from dataclasses import dataclass

from modeltop.benchmarks.models import ContextBenchmarkConfig, ContextUnit
from modeltop.chat.metrics import TokenCounter, count_chat_messages, token_counter_name
from modeltop.chat.models import ChatMessage


@dataclass(frozen=True, slots=True)
class ContextTokenBudget:
    """Preflight body budget for one complete Context prompt target."""

    target_length: int
    context_unit: ContextUnit
    body_target_units: int
    fixed_visible_characters: int
    fixed_prompt_tokens: int
    template_overhead_tokens: int | None
    output_budget_tokens: int
    maximum_prompt_tokens: int
    counter_name: str
    estimated: bool


def effective_context_output_budget(config: ContextBenchmarkConfig) -> int:
    """Return the output reserve active for the selected Context mode."""
    if config.mode == "retrieval":
        return config.retrieval_maximum_output_tokens
    return config.maximum_output_tokens


def calculate_token_budget(
    target_length: int,
    context_unit: ContextUnit,
    system_message: str,
    instruction: str,
    output_budget: int,
    maximum_context: int,
    counter: TokenCounter,
) -> ContextTokenBudget:
    """Calculate body space while reserving complete-message and output overhead."""
    separator = "\n\n"
    fixed_visible_characters = len(system_message) + len(separator) + len(instruction)
    fixed_messages = (
        ChatMessage("system", system_message),
        ChatMessage("user", separator + instruction),
    )
    fixed_measurement = count_chat_messages(counter, fixed_messages)
    maximum_prompt_tokens = maximum_context - output_budget
    if maximum_prompt_tokens < 1:
        raise ValueError(
            "Prompt plus reserved output exceeds the configured context safety maximum."
        )

    if context_unit == "tokens":
        if target_length + output_budget > maximum_context:
            raise ValueError(
                "Prompt plus reserved output exceeds the configured "
                "context safety maximum."
            )
        body_target_units = target_length - fixed_measurement.total_tokens
    else:
        body_target_units = target_length - fixed_visible_characters
        # Character mode is exact for visible content but must conservatively reserve
        # the dependency-free chat wrapper estimate before allocating a body.
        estimated_complete_tokens = math.ceil(target_length / 4) + 11
        if estimated_complete_tokens + output_budget > maximum_context:
            raise ValueError(
                "Prompt plus reserved output exceeds the configured "
                "context safety maximum."
            )

    if body_target_units < 1:
        raise ValueError("Target prompt length is too small for required messages.")

    return ContextTokenBudget(
        target_length=target_length,
        context_unit=context_unit,
        body_target_units=body_target_units,
        fixed_visible_characters=fixed_visible_characters,
        fixed_prompt_tokens=fixed_measurement.total_tokens,
        template_overhead_tokens=fixed_measurement.template_overhead_tokens,
        output_budget_tokens=output_budget,
        maximum_prompt_tokens=maximum_prompt_tokens,
        counter_name=token_counter_name(counter),
        estimated=fixed_measurement.estimated,
    )
