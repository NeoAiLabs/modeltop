"""Immutable chat domain values."""

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, cast

type ChatRole = Literal["system", "user", "assistant", "tool"]

type ThinkingMode = Literal["server_default", "disabled"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One ordered OpenAI-compatible chat message."""

    role: ChatRole
    content: str

    def __post_init__(self) -> None:
        role = cast(object, self.role)
        content = cast(object, self.content)
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported chat role: {role}")
        if not isinstance(content, str):
            raise TypeError("Chat message content must be a string")


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """Validated generation controls shared by UI and API layers."""

    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 1024
    seed: int | None = None
    stream: bool = True
    enable_thinking: bool | None = None

    def __post_init__(self) -> None:
        temperature = cast(object, self.temperature)
        top_p = cast(object, self.top_p)
        max_tokens = cast(object, self.max_tokens)
        seed = cast(object, self.seed)
        stream = cast(object, self.stream)
        enable_thinking = cast(object, self.enable_thinking)
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise TypeError("Temperature must be a number")
        if not math.isfinite(temperature) or not 0.0 <= temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
            raise TypeError("Top-p must be a number")
        if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
            raise ValueError("Top-p must be greater than 0.0 and at most 1.0")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise TypeError("Max tokens must be an integer")
        if max_tokens < 1:
            raise ValueError("Max tokens must be at least 1")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("Seed must be an integer or empty")
        if not isinstance(stream, bool):
            raise TypeError("Stream must be a boolean")
        if enable_thinking is not None and not isinstance(enable_thinking, bool):
            raise TypeError("Enable thinking must be a boolean or empty")

        object.__setattr__(self, "temperature", float(temperature))
        object.__setattr__(self, "top_p", float(top_p))


class GenerationStatus(StrEnum):
    """Lifecycle state of the latest generation."""

    IDLE = "idle"
    STARTING = "starting"
    STREAMING = "streaming"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """Measured and token-derived values for one generation."""

    request_started_at: float
    first_token_at: float | None = None
    completed_at: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    prompt_tokens_estimated: bool = False
    completion_tokens_estimated: bool = False
    total_tokens_estimated: bool = False
    draft_tokens: int | None = None
    accepted_tokens: int | None = None
    acceptance_rate: float | None = None
    ttft_ms: float | None = None
    active_generation_duration_s: float | None = None
    total_duration_s: float | None = None
    output_tokens_per_second: float | None = None
    inter_token_latency_ms: float | None = None
    finish_reason: str | None = None
    streamed: bool = True
    cancelled: bool = False
