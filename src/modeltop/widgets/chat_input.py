"""Multiline chat composer with explicit typed commands."""

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.widgets import Static, TextArea

from modeltop.messages import ClearConversationRequested, PromptSubmitted


class PromptTextArea(TextArea):
    """Text area whose send/newline keys are local to the focused editor."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "send", "Send", show=False, priority=True),
        Binding("ctrl+enter", "send", "Send", show=False, priority=True),
        Binding("shift+enter", "newline", "Newline", show=False, priority=True),
        Binding("alt+enter", "newline", "Newline", show=False, priority=True),
        Binding("ctrl+k", "clear_conversation", "Clear", show=False, priority=True),
    ]

    def action_send(self) -> None:
        self.post_message(PromptSubmitted(self.text))

    def action_newline(self) -> None:
        self.insert("\n")

    def action_clear_conversation(self) -> None:
        self.post_message(ClearConversationRequested())


class ChatInput(Vertical):
    """Prompt editor that preserves text until reservation succeeds."""

    DEFAULT_CSS = """
    ChatInput {
        width: 1fr;
        height: 6;
        border: solid $border-blurred;
        background: $catppuccin-base;
        padding: 0 1;
    }

    ChatInput #chat-input-label {
        height: 1;
        color: $primary;
        text-style: bold;
    }

    ChatInput #chat-prompt {
        width: 1fr;
        height: 4;
        border: none;
        background: $background;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(
            "PROMPT  Enter/Ctrl+Enter send · Shift/Alt+Enter newline",
            id="chat-input-label",
        )
        yield PromptTextArea(id="chat-prompt", language=None, soft_wrap=True)

    @property
    def editor(self) -> TextArea:
        return self.query_one("#chat-prompt", TextArea)

    def clear_after_submit(self) -> None:
        self.editor.clear()

    def set_generating(self, generating: bool) -> None:
        self.editor.disabled = generating

    def focus_prompt(self) -> None:
        self.editor.focus()
