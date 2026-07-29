"""Scrollable chat transcript with one Markdown stream per assistant turn."""

from textual.containers import VerticalScroll
from textual.widgets import Markdown, Static
from textual.widgets._markdown import MarkdownStream

from modeltop.chat.models import ChatMessage
from modeltop.chat.session import ChatSession
from modeltop.state import ApplicationState


class ChatHistory(VerticalScroll):
    """Render ordered turns and incrementally append the active assistant response."""

    DEFAULT_CSS = """
    ChatHistory {
        width: 1fr;
        height: 1fr;
        min-height: 4;
        border: solid #2d3b49;
        background: #0b0f14;
        padding: 0 1;
    }

    ChatHistory .chat-role {
        width: 1fr;
        height: 1;
        margin-top: 1;
        color: #5da9e9;
        text-style: bold;
    }

    ChatHistory .chat-user,
    ChatHistory .chat-system {
        width: 1fr;
        height: auto;
        color: #d8dee9;
    }

    ChatHistory Markdown {
        width: 1fr;
        height: auto;
        background: transparent;
        padding: 0;
    }

    ChatHistory #chat-empty {
        width: 1fr;
        height: 1fr;
        content-align: center middle;
        color: #7f8c9a;
    }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(can_focus=True, id=id)
        self._session: ChatSession | None = None
        self._active_generation_id: int | None = None
        self._stream_text = ""
        self._markdown_stream: MarkdownStream | None = None

    async def update_state(self, state: ApplicationState) -> None:
        """Reconcile committed turns and append only new active response text."""
        session_changed = state.chat_session != self._session
        generation_changed = state.active_generation_id != self._active_generation_id
        if session_changed or generation_changed:
            await self._rebuild(state)
        if state.active_generation_id is not None:
            response = state.current_response
            if response.startswith(self._stream_text):
                delta = response[len(self._stream_text) :]
                if delta and self._markdown_stream is not None:
                    await self._markdown_stream.write(delta)
                    self._stream_text = response
            elif response != self._stream_text:
                await self._rebuild(state)
                if response and self._markdown_stream is not None:
                    await self._markdown_stream.write(response)
                    self._stream_text = response

    async def clear_stream(self) -> None:
        stream, self._markdown_stream = self._markdown_stream, None
        if stream is not None:
            await stream.stop()

    async def _rebuild(self, state: ApplicationState) -> None:
        was_at_bottom = self.is_vertical_scroll_end
        await self.clear_stream()
        await self.remove_children()
        session = state.chat_session
        visible = bool(session.messages) or (
            session.show_system_prompt and bool(session.system_prompt.strip())
        )
        if not visible and state.active_generation_id is None:
            await self.mount(
                Static(
                    "Start a conversation with the selected model.",
                    id="chat-empty",
                    markup=False,
                )
            )
        else:
            if session.show_system_prompt and (
                system_prompt := session.system_prompt.strip()
            ):
                await self._mount_plain("SYSTEM", system_prompt, "chat-system")
            for message in session.messages:
                await self._mount_message(message)
        self._stream_text = ""
        self._active_generation_id = state.active_generation_id
        if state.active_generation_id is not None:
            await self.mount(Static("ASSISTANT", classes="chat-role", markup=False))
            markdown = Markdown("")
            await self.mount(markdown)
            self._markdown_stream = Markdown.get_stream(markdown)
        self._session = session
        if was_at_bottom:
            self.anchor()
        else:
            self.anchor(False)

    async def _mount_message(self, message: ChatMessage) -> None:
        if message.role == "assistant":
            await self.mount(Static("ASSISTANT", classes="chat-role", markup=False))
            await self.mount(Markdown(message.content))
        else:
            await self._mount_plain(message.role.upper(), message.content, "chat-user")

    async def _mount_plain(self, role: str, content: str, css_class: str) -> None:
        await self.mount(Static(role, classes="chat-role", markup=False))
        await self.mount(Static(content, classes=css_class, markup=False))

    async def on_unmount(self) -> None:
        await self.clear_stream()
