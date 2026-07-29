"""Immutable conversation session and ordering policy."""

from dataclasses import dataclass, field, replace

from modeltop.chat.models import ChatMessage, GenerationSettings

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass(frozen=True, slots=True)
class ChatSession:
    """Conversation history plus in-memory generation preferences."""

    messages: tuple[ChatMessage, ...] = ()
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    show_system_prompt: bool = False
    settings: GenerationSettings = field(default_factory=GenerationSettings)
    server_id: str | None = None
    model_id: str | None = None

    def append(self, *messages: ChatMessage) -> "ChatSession":
        """Append ordered messages without mutating the current snapshot."""
        return replace(self, messages=(*self.messages, *messages))

    def clear(self) -> "ChatSession":
        """Clear history while preserving all preferences and capture metadata."""
        return replace(self, messages=())

    def request_context(self, user_prompt: str) -> tuple[ChatMessage, ...]:
        """Build exact API context with at most one leading system message."""
        if not user_prompt.strip():
            raise ValueError("Prompt must not be blank")
        context: list[ChatMessage] = []
        if system_prompt := self.system_prompt.strip():
            context.append(ChatMessage("system", system_prompt))
        context.extend(self.messages)
        context.append(ChatMessage("user", user_prompt))
        return tuple(context)
