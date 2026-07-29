"""Inline Chat workspace view; not a pushed Textual screen."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from modeltop.chat.models import GenerationStatus
from modeltop.state import ApplicationState, ServerStatus
from modeltop.widgets.chat_history import ChatHistory
from modeltop.widgets.chat_input import ChatInput
from modeltop.widgets.generation_metrics import GenerationMetricsView
from modeltop.widgets.generation_settings import GenerationSettingsPanel


class ChatView(Vertical):
    """Compose status, preferences, transcript, metrics, and composer."""

    DEFAULT_CSS = """
    ChatView {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
        background: #0b0f14;
        overflow: hidden;
    }

    ChatView #chat-title {
        width: 1fr;
        height: 1;
        color: #5da9e9;
        text-style: bold;
        text-wrap: nowrap;
        text-overflow: clip;
    }

    ChatView #chat-status {
        width: 1fr;
        height: 1;
        color: #7f8c9a;
        text-wrap: nowrap;
        text-overflow: clip;
    }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._loaded_preferences: tuple[object, str, bool] | None = None
        self._was_generating = False
        self._history: ChatHistory | None = None

    def compose(self) -> ComposeResult:
        yield Static("CHAT PLAYGROUND", id="chat-title")
        yield Static("Waiting for server and model selection.", id="chat-status")
        yield GenerationSettingsPanel(id="generation-settings")
        yield ChatHistory(id="chat-history")
        yield GenerationMetricsView(id="generation-metrics")
        yield ChatInput(id="chat-input")

    def on_mount(self) -> None:
        self._history = self.query_one("#chat-history", ChatHistory)

    async def update_state(self, state: ApplicationState) -> None:
        server = state.selected_server_id or "--"
        model = state.selected_model_id or "--"
        eligibility = self._eligibility(state)
        self.query_one("#chat-status", Static).update(
            f"SERVER {server} · MODEL {model} · {eligibility}"
        )
        preferences = (
            state.chat_session.settings,
            state.chat_session.system_prompt,
            state.chat_session.show_system_prompt,
        )
        if preferences != self._loaded_preferences:
            self.settings_panel.load_session(state.chat_session)
            self._loaded_preferences = preferences

        generating = state.active_generation_id is not None or state.benchmark_is_active
        self.composer.set_generating(generating)
        self.settings_panel.set_generating(generating)
        await self.history.update_state(state)
        self.metrics.update_state(state)
        if self._was_generating and not generating and state.active_view == "chat":
            self.composer.focus_prompt()
        self._was_generating = generating

    @property
    def composer(self) -> ChatInput:
        return self.query_one("#chat-input", ChatInput)

    @property
    def history(self) -> ChatHistory:
        return self.query_one("#chat-history", ChatHistory)

    @property
    def settings_panel(self) -> GenerationSettingsPanel:
        return self.query_one("#generation-settings", GenerationSettingsPanel)

    @property
    def metrics(self) -> GenerationMetricsView:
        return self.query_one("#generation-metrics", GenerationMetricsView)

    def focus_prompt(self) -> None:
        self.composer.focus_prompt()

    def toggle_settings(self) -> None:
        self.settings_panel.toggle()

    async def stop_stream(self) -> None:
        if self._history is not None:
            await self._history.clear_stream()

    @staticmethod
    def _eligibility(state: ApplicationState) -> str:
        if state.benchmark_is_active:
            return "Chat is unavailable while a benchmark is running"
        if state.active_generation_id is not None:
            return "GENERATING · Esc cancels"
        if state.server_status is not ServerStatus.ONLINE:
            return "OFFLINE · refresh before sending"
        if state.selected_model_id is None:
            return "NO MODEL · select a model"
        if state.generation_status is GenerationStatus.ERROR:
            return f"ERROR · {state.generation_error or 'generation failed'}"
        if state.generation_status is GenerationStatus.CANCELLED:
            return "CANCELLED · ready to send again"
        if state.generation_notice:
            return "READY · non-stream fallback used"
        return "READY · Enter sends"
