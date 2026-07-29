"""Textual terminal dashboard for local LLM benchmarking."""

import asyncio
import logging
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import ClassVar, cast

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal
from textual.screen import ModalScreen, Screen
from textual.timer import Timer
from textual.widgets import Button, ContentSwitcher, OptionList, Static
from textual.worker import Worker, WorkerError, WorkerState

from modeltop.api.client import OpenAICompatibleClient
from modeltop.benchmarks.models import (
    ConcurrencyBenchmarkConfig,
    ConcurrencyBenchmarkResult,
    ContextBenchmarkConfig,
    ContextBenchmarkResult,
    DrafterBenchmarkConfig,
    DrafterBenchmarkResult,
    SpeedTestConfig,
    SpeedTestResult,
    SpeedTestStatus,
    ToolCallingBenchmarkConfig,
    ToolCallingBenchmarkResult,
    concurrency_benchmark_config_from_defaults,
    context_benchmark_config_from_defaults,
    drafter_benchmark_config_from_defaults,
    tool_calling_benchmark_config_from_defaults,
)
from modeltop.benchmarks.tool_calling import UpstreamBenchmarkRunner
from modeltop.config import configure_logging
from modeltop.constants import DEFAULT_TITLE
from modeltop.events import DashboardStateChanged
from modeltop.hardware.base import HardwareProvider
from modeltop.messages import (
    ClearConversationRequested,
    ConcurrencyBenchmarkEditRequested,
    ConcurrencyBenchmarkRunAgainRequested,
    ConcurrencyBenchmarkStartRequested,
    ContextBenchmarkEditRequested,
    ContextBenchmarkRunAgainRequested,
    ContextBenchmarkStartRequested,
    DrafterBenchmarkEditRequested,
    DrafterBenchmarkRunAgainRequested,
    DrafterBenchmarkStartRequested,
    PromptSubmitted,
    SettingsSubmitted,
    SpeedTestCancelRequested,
    SpeedTestCopySummaryRequested,
    SpeedTestExportRequested,
    SpeedTestResultSelected,
    SpeedTestRunAgainRequested,
    SpeedTestStartRequested,
    ToolCallingBenchmarkEditRequested,
    ToolCallingBenchmarkRunAgainRequested,
    ToolCallingBenchmarkStartRequested,
)
from modeltop.models import ModelTopConfig, ServerConfig
from modeltop.screens.chat import ChatView
from modeltop.screens.concurrency import ConcurrencyView
from modeltop.screens.context import ContextView
from modeltop.screens.drafter import DrafterView
from modeltop.screens.results import ResultsView
from modeltop.screens.settings import SettingsView
from modeltop.screens.speed_test import SpeedTestView
from modeltop.screens.tool_calling import ToolCallingView
from modeltop.services.benchmark_service import (
    BenchmarkOperationError,
    BenchmarkService,
    PendingConcurrencyBenchmark,
)
from modeltop.services.chat import (
    ChatOperationError,
    DashboardChatService,
    PendingGeneration,
)
from modeltop.services.configuration import (
    ConfigurationLoadError,
    load_configuration,
)
from modeltop.services.context_benchmark import (
    ContextBenchmarkOperationError,
    ContextBenchmarkService,
    PendingContextBenchmark,
)
from modeltop.services.drafter_benchmark import (
    DrafterBenchmarkOperationError,
    DrafterBenchmarkService,
    PendingDrafterBenchmark,
)
from modeltop.services.generation import (
    GenerationFailed,
    GenerationOutcome,
    GenerationService,
)
from modeltop.services.hardware_monitor import HardwareMonitor, HardwareRefreshResult
from modeltop.services.model_discovery import ModelDiscoveryService
from modeltop.services.result_export import (
    ResultExportError,
    export_speed_test_result,
    format_speed_test_summary,
)
from modeltop.services.server_monitor import RefreshResult, ServerMonitor
from modeltop.services.speed_test import (
    PendingSpeedTest,
    SpeedTestOperationError,
    SpeedTestService,
)
from modeltop.services.tool_calling import (
    PendingToolCallingBenchmark,
    ToolCallingBenchmarkOperationError,
    ToolCallingBenchmarkService,
)
from modeltop.state import (
    ActiveView,
    ApplicationState,
    ApplicationStateStore,
    initial_application_state,
)
from modeltop.widgets.footer import StatusFooter
from modeltop.widgets.header import HeaderBar
from modeltop.widgets.sidebar import BenchmarkSidebar
from modeltop.widgets.workspace import Workspace


class ShortcutHelp(ModalScreen[None]):
    """Display the keyboard shortcuts."""

    DEFAULT_CSS = """
    ShortcutHelp {
        align: center middle;
        background: #0b0f14 80%;
    }

    ShortcutHelp #shortcut-help {
        width: 52;
        max-width: 90%;
        height: 20;
        padding: 1 2;
        border: solid #5da9e9;
        background: #111820;
        color: #d8dee9;
        text-align: left;
        text-wrap: nowrap;
        text-overflow: clip;
    }
    """
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape,?", "dismiss", "Close", show=False),
        Binding("q", "quit_app", "Quit", show=False),
    ]

    def compose(self) -> ComposeResult:
        """Compose the non-focusable shortcut panel."""
        shortcuts = Text(
            "KEYBOARD SHORTCUTS\n\n"
            "↑ / ↓  Move selection · Tab cycle focus\n"
            "Enter  Select / start / send · Shift/Alt+Enter newline\n"
            "Esc  Cancel active work / back\n"
            "R  Run or Run Again in benchmark workspaces\n"
            "E  Edit Tool/Context/Concurrency · export Speed result\n"
            "C  Copy Speed result summary\n"
            "Ctrl+K  Clear chat · Ctrl+G chat settings\n"
            "?  Toggle help\n"
            "Q  Quit outside inputs · Ctrl+Q always quit\n"
        )
        shortcuts.stylize("#5da9e9 bold", 0, len("KEYBOARD SHORTCUTS"))
        yield Static(shortcuts, id="shortcut-help")

    def action_quit_app(self) -> None:
        cast(App[None], self.app).exit()  # pyright: ignore[reportUnknownMemberType]


class ConfigurationErrorScreen(Screen[None]):
    """Display a startup configuration failure without dashboard resources."""

    AUTO_FOCUS = ""
    DEFAULT_CSS = """
    ConfigurationErrorScreen {
        align: center middle;
        background: #0b0f14;
    }

    ConfigurationErrorScreen #configuration-error {
        width: 72;
        max-width: 90%;
        height: auto;
        padding: 1 2;
        border: solid #e06c75;
        background: #111820;
        color: #d8dee9;
        text-align: center;
        text-wrap: wrap;
        text-overflow: clip;
    }
    """

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        """Compose the sanitized startup guidance."""
        content = Text(
            "CONFIGURATION ERROR\n\n"
            f"{self._message}\n\n"
            "Set MODELTOP_CONFIG or fix the selected YAML file.\n\n"
            "Q Quit"
        )
        content.stylize("#e06c75 bold", 0, len("CONFIGURATION ERROR"))
        yield Static(content, id="configuration-error")


class LargeContextConfirmation(ModalScreen[bool]):
    """Keyboard-operable warning before reserving a large Context run."""

    DEFAULT_CSS = """
    LargeContextConfirmation {
        align: center middle;
        background: #0b0f14 80%;
    }
    LargeContextConfirmation #large-context-dialog {
        width: 64;
        max-width: 90%;
        height: auto;
        padding: 1 2;
        border: solid #e5c07b;
        background: #111820;
    }
    LargeContextConfirmation #large-context-actions {
        height: 3;
        align-horizontal: center;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Static(
            "Large context tests may consume significant VRAM "
            "and take several minutes.",
            id="large-context-dialog",
            markup=False,
        )
        yield Horizontal(
            Button("Continue", id="large-context-confirm", variant="warning"),
            Button("Cancel", id="large-context-cancel"),
            id="large-context-actions",
        )

    def on_mount(self) -> None:
        self.query_one("#large-context-confirm", Button).focus()

    @on(Button.Pressed, "#large-context-confirm")
    def confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(True)

    @on(Button.Pressed, "#large-context-cancel")
    def cancel_button(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)


class FullToolCallingConfirmation(ModalScreen[bool]):
    """Keyboard-operable warning before reserving the official Full suite."""

    DEFAULT_CSS = """
    FullToolCallingConfirmation {
        align: center middle;
        background: #0b0f14 80%;
    }
    FullToolCallingConfirmation #full-tool-calling-dialog {
        width: 68;
        max-width: 90%;
        height: auto;
        padding: 1 2;
        border: solid #e5c07b;
        background: #111820;
    }
    FullToolCallingConfirmation #full-tool-calling-actions {
        height: 3;
        align-horizontal: center;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Static(
            "Full runs 69 multi-turn scenarios. Each scenario can issue several "
            "requests, so the run may take substantial time.",
            id="full-tool-calling-dialog",
            markup=False,
        )
        yield Horizontal(
            Button("Continue", id="full-tool-calling-confirm", variant="warning"),
            Button("Cancel", id="full-tool-calling-cancel"),
            id="full-tool-calling-actions",
        )

    def on_mount(self) -> None:
        self.query_one("#full-tool-calling-confirm", Button).focus()

    @on(Button.Pressed, "#full-tool-calling-confirm")
    def confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(True)

    @on(Button.Pressed, "#full-tool-calling-cancel")
    def cancel_button(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)


class ModelTopApp(App[None]):
    """The ModelTop terminal dashboard application."""

    TITLE = DEFAULT_TITLE
    AUTO_FOCUS = "#sidebar-menu"
    CSS = """
    Screen {
        background: #0b0f14;
        color: #d8dee9;
        overflow: hidden;
    }

    .warning {
        color: #e5c07b;
    }

    .error {
        color: #e06c75;
    }

    #workspace-switcher {
        width: 1fr;
        height: 1fr;
    }
    """
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("?", "show_help", "Help", show=False),
        Binding("r", "run_or_rerun", "Run", show=False),
        Binding("ctrl+l", "refresh_dashboard", "Refresh", show=False),
        Binding("e", "export_result", "Edit or export", show=False),
        Binding("c", "copy_summary", "Copy summary", show=False),
        Binding("escape", "cancel_generation", "Cancel", show=False),
        Binding("ctrl+k", "clear_chat", "Clear chat", show=False),
        Binding("ctrl+g", "toggle_chat_settings", "Chat settings", show=False),
    ]

    def __init__(
        self,
        config: ModelTopConfig | None = None,
        *,
        client: OpenAICompatibleClient | None = None,
        hardware_provider: HardwareProvider | None = None,
        tool_calling_runner: UpstreamBenchmarkRunner | None = None,
    ) -> None:
        super().__init__()
        self._chat_view: ChatView | None = None
        self._speed_test_view: SpeedTestView | None = None
        self._concurrency_view: ConcurrencyView | None = None
        self._context_view: ContextView | None = None
        self._tool_calling_view: ToolCallingView | None = None
        self._drafter_view: DrafterView | None = None
        self._results_view: ResultsView | None = None
        self._settings_view: SettingsView | None = None
        self._configuration_source: Path | None = None
        self._configuration_error: ConfigurationLoadError | None = None
        self._config: ModelTopConfig | None = None
        self._server: ServerConfig | None = None
        self._api_client: OpenAICompatibleClient | None = None
        self._generation_service: GenerationService | None = None
        self._chat_service: DashboardChatService | None = None
        self._speed_test_service: SpeedTestService | None = None
        self._benchmark_service: BenchmarkService | None = None
        self._context_benchmark_service: ContextBenchmarkService | None = None
        self._tool_calling_service: ToolCallingBenchmarkService | None = None
        self._drafter_service: DrafterBenchmarkService | None = None
        self._monitor: ServerMonitor | None = None
        self._state_store: ApplicationStateStore | None = None
        self._hardware_monitor: HardwareMonitor | None = None
        self._refresh_timer: Timer | None = None
        self._refresh_worker: Worker[RefreshResult] | None = None
        self._hardware_timer: Timer | None = None
        self._hardware_worker: Worker[HardwareRefreshResult] | None = None
        self._context_hardware_timer: Timer | None = None
        self._chat_worker: Worker[GenerationOutcome | None] | None = None
        self._speed_test_worker: Worker[SpeedTestResult | None] | None = None
        self._benchmark_worker: Worker[ConcurrencyBenchmarkResult | None] | None = None
        self._context_worker: Worker[ContextBenchmarkResult | None] | None = None
        self._tool_calling_worker: Worker[ToolCallingBenchmarkResult | None] | None = (
            None
        )
        self._drafter_worker: Worker[DrafterBenchmarkResult | None] | None = None
        self._pending_chat: PendingGeneration | None = None
        self._pending_speed_test: PendingSpeedTest | None = None
        self._pending_benchmark: PendingConcurrencyBenchmark | None = None
        self._pending_context: PendingContextBenchmark | None = None
        self._pending_tool_calling: PendingToolCallingBenchmark | None = None
        self._pending_drafter: PendingDrafterBenchmark | None = None
        self._shutting_down = False
        self.dashboard_state: ApplicationState | None = None

        if config is None:
            try:
                loaded = load_configuration()
            except ConfigurationLoadError as error:
                self._configuration_error = error
                logging.getLogger(__name__).error(
                    "Configuration loading failed: %s", error.detail
                )
                return
            config = loaded.config
            self._configuration_source = loaded.source_path
        self._config = config

        server_id = config.application.default_server
        self._server = next(
            (server for server in config.servers if server.id == server_id),
            config.servers[0],
        )
        hardware_enabled = (
            config.hardware.enabled and config.hardware.preferred_provider != "disabled"
        )
        self._state_store = ApplicationStateStore(
            initial_application_state(
                self._server.id,
                hardware_enabled=hardware_enabled,
                concurrency_config=concurrency_benchmark_config_from_defaults(
                    config.benchmarks.concurrency
                ),
                context_config=context_benchmark_config_from_defaults(
                    config.benchmarks.context
                ),
                tool_calling_config=tool_calling_benchmark_config_from_defaults(
                    config.benchmarks.tool_calling
                ),
                drafter_config=drafter_benchmark_config_from_defaults(
                    config.benchmarks.drafter
                ),
            )
        )
        api_client = client or OpenAICompatibleClient(
            self._server.base_url,
            self._server.api_key,
            config.application.request_timeout_seconds,
        )
        self._api_client = api_client
        self._generation_service = GenerationService(api_client)
        discovery = ModelDiscoveryService(api_client)
        self._monitor = ServerMonitor(
            self._server,
            discovery,
            self._state_store,
            self._handle_server_state_change,
        )
        self._chat_service = DashboardChatService(
            self._generation_service,
            self._state_store,
            self._handle_dashboard_state_change,
        )
        self._speed_test_service = SpeedTestService(
            self._generation_service,
            self._state_store,
            self._server,
            self._handle_dashboard_state_change,
        )
        self._benchmark_service = BenchmarkService(
            self._generation_service,
            self._state_store,
            self._server,
            config.hardware.refresh_interval_seconds,
            self._handle_dashboard_state_change,
        )
        self._context_benchmark_service = ContextBenchmarkService(
            self._generation_service,
            self._state_store,
            self._server,
            self._handle_dashboard_state_change,
        )
        if tool_calling_runner is None:
            self._tool_calling_service = ToolCallingBenchmarkService(
                self._state_store,
                self._server,
                self._handle_dashboard_state_change,
            )
        else:
            self._tool_calling_service = ToolCallingBenchmarkService(
                self._state_store,
                self._server,
                self._handle_dashboard_state_change,
                upstream_runner=tool_calling_runner,
            )
        self._drafter_service = DrafterBenchmarkService(
            self._generation_service,
            self._state_store,
            self._server,
            self._handle_dashboard_state_change,
        )
        if hardware_enabled:
            self._hardware_monitor = HardwareMonitor(
                config.hardware,
                self._state_store,
                self._handle_hardware_state_change,
                provider=hardware_provider,
            )
        self.dashboard_state = self._monitor.state
        self._refresh_interval_seconds = config.application.refresh_interval_seconds
        self._hardware_refresh_interval_seconds = (
            config.hardware.refresh_interval_seconds
        )

    def get_default_screen(self) -> Screen[object]:
        """Use a standalone error screen when configuration cannot load."""
        if self._configuration_error is not None:
            return cast(
                Screen[object],
                ConfigurationErrorScreen(self._configuration_error.user_message),
            )
        return cast(Screen[object], Screen(id="_default"))

    def compose(self) -> ComposeResult:
        """Compose the dashboard only for a valid configuration."""
        if self._configuration_error is not None:
            return
        config = self._config
        server = self._server
        if config is None or server is None:
            return
        yield HeaderBar()
        yield Horizontal(
            BenchmarkSidebar(),
            ContentSwitcher(
                Workspace(id="overview-workspace"),
                ChatView(id="chat-workspace"),
                SpeedTestView(id="speed-test-workspace"),
                ConcurrencyView(id="concurrency-workspace"),
                ContextView(id="context-workspace"),
                ToolCallingView(id="tool-calling-workspace"),
                DrafterView(id="drafter-workspace"),
                ResultsView(id="results-workspace"),
                SettingsView(
                    config,
                    server,
                    self._configuration_source,
                    id="settings-workspace",
                ),
                id="workspace-switcher",
                initial="overview-workspace",
            ),
            id="dashboard-body",
        )
        yield StatusFooter()

    async def on_mount(self) -> None:
        """Render initial state and start independent runtime lanes."""
        if self._monitor is None:
            return
        self._chat_view = self.query_one(ChatView)
        self._speed_test_view = self.query_one(SpeedTestView)
        self._concurrency_view = self.query_one(ConcurrencyView)
        self._context_view = self.query_one(ContextView)
        self._tool_calling_view = self.query_one(ToolCallingView)
        self._drafter_view = self.query_one(DrafterView)
        self._results_view = self.query_one(ResultsView)
        self._settings_view = self.query_one(SettingsView)
        self._render_server_state(self._monitor.state)
        self._render_hardware_state(self._monitor.state)
        await self.query_one(ChatView).update_state(self._monitor.state)
        self.query_one(SpeedTestView).update_state(self._monitor.state)
        self.query_one(ConcurrencyView).update_state(self._monitor.state)
        self.query_one(ContextView).config_panel.load_config(
            self._monitor.state.context_benchmark.config
        )
        self.query_one(ContextView).update_state(self._monitor.state)
        self.query_one(ToolCallingView).config_panel.load_config(
            self._monitor.state.tool_calling_benchmark.config
        )
        self.query_one(ToolCallingView).update_state(self._monitor.state)
        self.query_one(DrafterView).config_panel.load_config(
            self._monitor.state.drafter_benchmark.config
        )
        self.query_one(DrafterView).update_state(self._monitor.state)
        self.query_one(ResultsView).update_state(self._monitor.state)
        self.query_one(SettingsView).update_state(self._monitor.state)
        self._launch_refresh(
            manual=False,
            preserve_online_on_failure=False,
        )
        self._refresh_timer = self.set_interval(
            self._refresh_interval_seconds,
            self._automatic_refresh,
        )
        if self._hardware_monitor is not None:
            self._launch_hardware_refresh()
            self._hardware_timer = self.set_interval(
                self._hardware_refresh_interval_seconds,
                self._automatic_hardware_refresh,
            )

    def _handle_server_state_change(self, state: ApplicationState) -> None:
        self.dashboard_state = state
        self._render_server_state(state)
        self.post_message(DashboardStateChanged())

    def _handle_hardware_state_change(self, state: ApplicationState) -> None:
        self.dashboard_state = state
        self._render_hardware_state(state)
        self.post_message(DashboardStateChanged())

    def _handle_dashboard_state_change(self, state: ApplicationState) -> None:
        self.dashboard_state = state
        self.post_message(DashboardStateChanged())

    def _render_server_state(self, state: ApplicationState) -> None:
        server = self._server
        if server is None:
            return
        self.query_one(HeaderBar).update_state(state, server)
        self.query_one(Workspace).update_state(state)
        self.query_one(StatusFooter).update_state(state, server)

    def _render_hardware_state(self, state: ApplicationState) -> None:
        server = self._server
        if server is None:
            return
        self.query_one(HeaderBar).update_state(state, server)
        self.query_one(Workspace).update_hardware_state(state)
        self.query_one(StatusFooter).update_state(state, server)

    async def on_dashboard_state_changed(self, event: DashboardStateChanged) -> None:
        event.stop()
        store = self._state_store
        server = self._server
        if store is None or server is None:
            return
        if len(self.query(HeaderBar)) == 0:
            return
        state = store.state
        self.dashboard_state = state
        self.query_one(HeaderBar).update_state(state, server)
        self.query_one(StatusFooter).update_state(state, server)
        switcher = self.query_one("#workspace-switcher", ContentSwitcher)
        switcher.current = f"{state.active_view}-workspace"
        await self.query_one(ChatView).update_state(state)
        self.query_one(SpeedTestView).update_state(state)
        self.query_one(ConcurrencyView).update_state(state)
        self.query_one(ContextView).update_state(state)
        self.query_one(ToolCallingView).update_state(state)
        self.query_one(DrafterView).update_state(state)
        self.query_one(ResultsView).update_state(state)
        self.query_one(SettingsView).update_state(state)

    def _automatic_refresh(self) -> None:
        if not self.is_running:
            return
        state = self.dashboard_state
        if state is not None and state.benchmark_is_active:
            return
        self._launch_refresh(
            manual=False,
            preserve_online_on_failure=True,
        )

    def _launch_refresh(
        self,
        *,
        manual: bool,
        preserve_online_on_failure: bool = False,
    ) -> bool:
        monitor = self._monitor
        if monitor is None:
            return False
        worker = self._refresh_worker
        if worker is not None and not worker.is_finished:
            if manual:
                self.notify("Refresh already in progress.", title="Refresh")
            return False
        if not monitor.begin_refresh(
            preserve_online_on_failure=preserve_online_on_failure
        ):
            if manual:
                self.notify("Refresh already in progress.", title="Refresh")
            return False
        self._refresh_worker = self._refresh_server(manual)
        return True

    def _automatic_hardware_refresh(self) -> None:
        if not self.is_running:
            return
        self._launch_hardware_refresh()

    def _launch_hardware_refresh(self) -> bool:
        monitor = self._hardware_monitor
        if monitor is None:
            return False
        worker = self._hardware_worker
        if worker is not None and not worker.is_finished:
            return False
        if not monitor.begin_refresh():
            return False
        self._hardware_worker = self._refresh_hardware()
        return True

    @work(
        name="hardware-refresh",
        group="hardware-refresh",
        exit_on_error=False,
    )
    async def _refresh_hardware(self) -> HardwareRefreshResult:
        monitor = self._hardware_monitor
        if monitor is None:
            return HardwareRefreshResult(
                success=False,
                skipped=True,
                message="Hardware refresh unavailable.",
                gpu_count=0,
            )
        return await monitor.refresh()

    @work(
        name="server-refresh",
        group="server-refresh",
        exit_on_error=False,
    )
    async def _refresh_server(self, manual: bool) -> RefreshResult:
        monitor = self._monitor
        if monitor is None:
            return RefreshResult(
                success=False,
                skipped=True,
                message="Refresh unavailable.",
                model_count=0,
                latency_ms=None,
            )
        result = await monitor.refresh()
        if manual:
            self.notify(
                result.message,
                title="Refresh",
                severity="information" if result.success else "error",
            )
        return result

    def action_show_help(self) -> None:
        """Open the keyboard shortcut reference."""
        self.push_screen(ShortcutHelp())

    def action_cancel_generation(self) -> None:
        """Cancel active work first; otherwise navigate back within benchmark views."""
        state = self.dashboard_state
        if state is None:
            return
        if state.tool_calling_benchmark.is_active:
            self._cancel_tool_calling_benchmark()
            return
        if state.drafter_benchmark.is_active:
            self._cancel_drafter_benchmark()
            return
        if state.context_benchmark.is_active:
            self._cancel_context_benchmark()
            return
        if state.concurrency_benchmark.is_active:
            self._cancel_concurrency_benchmark()
            return
        if state.speed_test.is_active:
            self._cancel_speed_test()
            return
        if state.active_generation_id is not None:
            worker = self._chat_worker
            pending = self._pending_chat
            if worker is not None and not worker.is_finished:
                was_pending = worker.state is WorkerState.PENDING
                worker.cancel()
                if (
                    was_pending
                    and pending is not None
                    and self._chat_service is not None
                ):
                    self._chat_service.cancel_reservation(pending.generation_id)
            elif pending is not None and self._chat_service is not None:
                self._chat_service.cancel_reservation(pending.generation_id)
            return
        if (
            state.active_view == "concurrency"
            and state.concurrency_benchmark.is_terminal
        ):
            self.query_one(ConcurrencyView).show_config(
                state.concurrency_benchmark.config
            )
            return
        if state.active_view == "context" and state.context_benchmark.is_terminal:
            self.query_one(ContextView).show_config(state.context_benchmark.config)
            return
        if (
            state.active_view == "tool-calling"
            and state.tool_calling_benchmark.is_terminal
        ):
            self.query_one(ToolCallingView).show_config(
                state.tool_calling_benchmark.config
            )
            return
        if state.active_view == "drafter" and state.drafter_benchmark.is_terminal:
            self.query_one(DrafterView).show_config(state.drafter_benchmark.config)
            return
        if state.active_view == "speed-test" and state.speed_test.is_terminal:
            self._reset_speed_test_ready(state.speed_test.config)
            return
        if state.active_view == "results":
            if state.speed_test.selected_result_id is not None:
                self._select_result(None)
            else:
                self._set_active_view("speed-test")

    def _reset_speed_test_ready(self, config: SpeedTestConfig) -> None:
        store = self._state_store
        if store is None:
            return
        state = store.update(
            lambda current: replace(
                current,
                speed_test=replace(
                    current.speed_test,
                    config=config,
                    status=SpeedTestStatus.IDLE,
                    run_id=None,
                    current_phase=None,
                    current_run=0,
                    phase_total=0,
                    latest_metrics=None,
                    live_output_preview="",
                    run_results=(),
                    last_error=None,
                ),
            )
        )
        self.query_one(SpeedTestView).show_config(config)
        self._handle_dashboard_state_change(state)

    def _cancel_concurrency_benchmark(self) -> None:
        service = self._benchmark_service
        pending = self._pending_benchmark
        if service is None:
            return
        benchmark_id = pending.benchmark_id if pending is not None else None
        service.request_cancellation(benchmark_id)
        worker = self._benchmark_worker
        if worker is not None and not worker.is_finished:
            was_pending = worker.state is WorkerState.PENDING
            worker.cancel()
            if was_pending and pending is not None:
                service.cancel_reservation(pending)
        elif pending is not None:
            service.cancel_reservation(pending)

    def _cancel_context_benchmark(self) -> None:
        service = self._context_benchmark_service
        pending = self._pending_context
        if service is None:
            return
        benchmark_id = pending.benchmark_id if pending is not None else None
        service.request_cancellation(benchmark_id)
        worker = self._context_worker
        if worker is not None and not worker.is_finished:
            was_pending = worker.state is WorkerState.PENDING
            worker.cancel()
            if was_pending and pending is not None:
                service.cancel_reservation(pending)
        elif pending is not None:
            service.cancel_reservation(pending)

    def _cancel_tool_calling_benchmark(self) -> None:
        service = self._tool_calling_service
        pending = self._pending_tool_calling
        if service is None:
            return
        benchmark_id = pending.benchmark_id if pending is not None else None
        service.request_cancellation(benchmark_id)
        worker = self._tool_calling_worker
        if worker is not None and not worker.is_finished:
            was_pending = worker.state is WorkerState.PENDING
            worker.cancel()
            if was_pending and pending is not None:
                service.cancel_reservation(pending)
        elif pending is not None:
            service.cancel_reservation(pending)

    def _cancel_drafter_benchmark(self) -> None:
        service = self._drafter_service
        pending = self._pending_drafter
        if service is None:
            return
        benchmark_id = pending.benchmark_id if pending is not None else None
        service.request_cancellation(benchmark_id)
        worker = self._drafter_worker
        if worker is not None and not worker.is_finished:
            was_pending = worker.state is WorkerState.PENDING
            worker.cancel()
            if was_pending and pending is not None:
                service.cancel_reservation(pending)
        elif pending is not None:
            service.cancel_reservation(pending)

    def _cancel_speed_test(self) -> None:
        service = self._speed_test_service
        pending = self._pending_speed_test
        if service is None:
            return
        run_id = pending.run_id if pending is not None else None
        service.request_cancellation(run_id)
        worker = self._speed_test_worker
        if worker is not None and not worker.is_finished:
            was_pending = worker.state is WorkerState.PENDING
            worker.cancel()
            if was_pending and pending is not None:
                service.cancel_reservation(pending)
        elif pending is not None:
            service.cancel_reservation(pending)

    def action_clear_chat(self) -> None:
        state = self.dashboard_state
        if state is not None and state.active_view == "chat":
            self._clear_chat()

    def action_toggle_chat_settings(self) -> None:
        state = self.dashboard_state
        if state is not None and state.active_view == "chat":
            self.query_one(ChatView).toggle_settings()

    def action_run_or_rerun(self) -> None:
        """Run the visible draft or rerun the visible immutable terminal result."""
        state = self.dashboard_state
        if state is None:
            return
        if (
            state.active_view == "tool-calling"
            and not state.tool_calling_benchmark.is_active
        ):
            result = state.tool_calling_benchmark.latest_result
            if state.tool_calling_benchmark.is_terminal and result is not None:
                self._request_tool_calling_benchmark(result.config)
            else:
                config = self.query_one(ToolCallingView).config_panel.parse_config(
                    notify=True
                )
                if config is not None:
                    self._request_tool_calling_benchmark(config)
            return
        if state.active_view == "drafter" and not state.drafter_benchmark.is_active:
            result = state.drafter_benchmark.latest_result
            if state.drafter_benchmark.is_terminal and result is not None:
                self._start_drafter_benchmark(result.config)
            else:
                config = self.query_one(DrafterView).config_panel.parse_config(
                    notify=True
                )
                if config is not None:
                    self._start_drafter_benchmark(config)
            return
        if state.active_view == "context" and not state.context_benchmark.is_active:
            result = state.context_benchmark.latest_result
            if state.context_benchmark.is_terminal and result is not None:
                self._request_context_benchmark(result.config)
            else:
                config = self.query_one(ContextView).config_panel.parse_config(
                    notify=True
                )
                if config is not None:
                    self._request_context_benchmark(config)
            return
        if (
            state.active_view == "concurrency"
            and not state.concurrency_benchmark.is_active
        ):
            result = state.concurrency_benchmark.latest_result
            config = (
                result.config
                if state.concurrency_benchmark.is_terminal and result is not None
                else state.concurrency_benchmark.config
            )
            self._start_concurrency_benchmark(config)
            return
        if state.active_view == "speed-test" and state.speed_test.is_terminal:
            latest = state.speed_test.latest_result
            if latest is not None:
                self._run_result_again(latest.run_id)
        elif state.active_view == "results":
            run_id = state.speed_test.selected_result_id
            if run_id is not None:
                self._run_result_again(run_id)

    def action_refresh_dashboard(self) -> None:
        """Reset intervals and refresh lanes without model polling during load."""
        state = self.dashboard_state
        if self._refresh_timer is not None:
            self._refresh_timer.reset()
        if self._hardware_timer is not None:
            self._hardware_timer.reset()
        if state is None or not state.benchmark_is_active:
            self._launch_refresh(
                manual=True,
                preserve_online_on_failure=False,
            )
        self._launch_hardware_refresh()

    def action_export_result(self) -> None:
        state = self.dashboard_state
        if (
            state is not None
            and state.active_view == "context"
            and state.context_benchmark.is_terminal
        ):
            self.query_one(ContextView).show_config(state.context_benchmark.config)
            return
        if (
            state is not None
            and state.active_view == "tool-calling"
            and state.tool_calling_benchmark.is_terminal
        ):
            self.query_one(ToolCallingView).show_config(
                state.tool_calling_benchmark.config
            )
            return
        if (
            state is not None
            and state.active_view == "drafter"
            and state.drafter_benchmark.is_terminal
        ):
            self.query_one(DrafterView).show_config(state.drafter_benchmark.config)
            return
        if (
            state is not None
            and state.active_view == "concurrency"
            and state.concurrency_benchmark.is_terminal
        ):
            self.query_one(ConcurrencyView).show_config(
                state.concurrency_benchmark.config
            )
            return
        result = self._current_result()
        if result is not None:
            self._export_result(result)

    def action_copy_summary(self) -> None:
        result = self._current_result()
        if result is not None:
            self._copy_result_summary(result)

    def _current_result(self) -> SpeedTestResult | None:
        state = self.dashboard_state
        if state is None:
            return None
        if state.active_view == "results":
            run_id = state.speed_test.selected_result_id
            return state.speed_test.result_by_id(run_id) if run_id is not None else None
        if state.active_view == "speed-test" and state.speed_test.is_terminal:
            return state.speed_test.latest_result
        return None

    def _export_result(self, result: SpeedTestResult) -> None:
        try:
            path = export_speed_test_result(result)
        except ResultExportError as error:
            self.notify(error.user_message, title="Export failed", severity="error")
            return
        self.notify(f"Saved {path}", title="Speed Test")

    def _copy_result_summary(self, result: SpeedTestResult) -> None:
        summary = format_speed_test_summary(result)
        self.copy_to_clipboard(summary)
        self.notify("Copy requested", title="Speed Test")

    @on(PromptSubmitted)
    def submit_chat_prompt(self, event: PromptSubmitted) -> None:
        event.stop()
        service = self._chat_service
        if service is None:
            return
        worker = self._chat_worker
        if worker is not None and not worker.is_finished:
            self.notify(
                "A generation is already in progress.",
                title="Chat",
                severity="error",
            )
            return
        try:
            pending = service.begin_generation(event.prompt)
        except ChatOperationError as error:
            self.notify(error.user_message, title="Chat", severity="error")
            return
        self.query_one(ChatView).composer.clear_after_submit()
        self._pending_chat = pending
        self._chat_worker = self._generate_chat(pending)

    @work(
        name="chat-generation",
        group="chat-generation",
        exit_on_error=False,
        exclusive=False,
    )
    async def _generate_chat(
        self, pending: PendingGeneration
    ) -> GenerationOutcome | None:
        service = self._chat_service
        if service is None:
            return None
        try:
            try:
                return await service.generate(pending)
            except GenerationFailed as error:
                self.notify(
                    error.error.user_message,
                    title="Generation failed",
                    severity="error",
                )
                return None
            except asyncio.CancelledError:
                raise
            except Exception:
                self.notify(
                    "Generation failed",
                    title="Generation failed",
                    severity="error",
                )
                raise
        finally:
            if self._pending_chat == pending:
                self._pending_chat = None

    @on(ToolCallingBenchmarkStartRequested)
    def start_tool_calling_benchmark(
        self,
        event: ToolCallingBenchmarkStartRequested,
    ) -> None:
        event.stop()
        self._request_tool_calling_benchmark(event.config)

    @on(ToolCallingBenchmarkRunAgainRequested)
    def run_tool_calling_benchmark_again(
        self,
        event: ToolCallingBenchmarkRunAgainRequested,
    ) -> None:
        event.stop()
        state = self.dashboard_state
        result = (
            state.tool_calling_benchmark.latest_result if state is not None else None
        )
        if result is None:
            self.notify(
                "Tool Calling result is no longer available.",
                title="Tool Calling",
                severity="error",
            )
            return
        self._request_tool_calling_benchmark(result.config)

    @on(ToolCallingBenchmarkEditRequested)
    def edit_tool_calling_benchmark(
        self,
        event: ToolCallingBenchmarkEditRequested,
    ) -> None:
        event.stop()
        state = self.dashboard_state
        if state is not None:
            self.query_one(ToolCallingView).show_config(
                state.tool_calling_benchmark.config
            )

    def _request_tool_calling_benchmark(
        self,
        config: ToolCallingBenchmarkConfig,
    ) -> None:
        if config.suite == "full":
            self.push_screen(
                FullToolCallingConfirmation(),
                lambda confirmed: (
                    self._start_tool_calling_benchmark(config) if confirmed else None
                ),
            )
            return
        self._start_tool_calling_benchmark(config)

    def _start_tool_calling_benchmark(
        self,
        config: ToolCallingBenchmarkConfig,
    ) -> None:
        service = self._tool_calling_service
        if service is None:
            return
        worker = self._tool_calling_worker
        if worker is not None and not worker.is_finished:
            self.notify(
                "A Tool Calling benchmark is already running.",
                title="Tool Calling",
                severity="error",
            )
            return
        try:
            pending = service.begin_benchmark(config)
        except ToolCallingBenchmarkOperationError as error:
            self.notify(
                error.user_message,
                title="Tool Calling",
                severity="error",
            )
            return
        self._pending_tool_calling = pending
        self.query_one(ToolCallingView).prepare_run(config)
        self._tool_calling_worker = self._run_tool_calling_benchmark(pending)

    @work(
        name="tool-calling-benchmark",
        group="tool-calling-benchmark",
        exit_on_error=False,
        exclusive=False,
    )
    async def _run_tool_calling_benchmark(
        self,
        pending: PendingToolCallingBenchmark,
    ) -> ToolCallingBenchmarkResult | None:
        service = self._tool_calling_service
        if service is None:
            return None
        try:
            return await service.run_benchmark(pending)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logging.getLogger(__name__).info(
                "Tool Calling worker failed benchmark=%s exception_class=%s",
                pending.benchmark_id,
                type(error).__name__,
            )
            self.notify(
                "Tool Calling benchmark failed",
                title="Tool Calling",
                severity="error",
            )
            raise
        finally:
            if self._pending_tool_calling == pending:
                self._pending_tool_calling = None
            if not self._shutting_down:
                if self._refresh_timer is not None:
                    self._refresh_timer.reset()
                self._launch_refresh(
                    manual=False,
                    preserve_online_on_failure=True,
                )

    @on(DrafterBenchmarkStartRequested)
    def start_drafter_benchmark(
        self,
        event: DrafterBenchmarkStartRequested,
    ) -> None:
        event.stop()
        self._start_drafter_benchmark(event.config)

    @on(DrafterBenchmarkRunAgainRequested)
    def run_drafter_benchmark_again(
        self,
        event: DrafterBenchmarkRunAgainRequested,
    ) -> None:
        event.stop()
        state = self.dashboard_state
        result = state.drafter_benchmark.latest_result if state is not None else None
        if result is None:
            self.notify(
                "Drafter result is no longer available.",
                title="Drafter",
                severity="error",
            )
            return
        self._start_drafter_benchmark(result.config)

    @on(DrafterBenchmarkEditRequested)
    def edit_drafter_benchmark(
        self,
        event: DrafterBenchmarkEditRequested,
    ) -> None:
        event.stop()
        state = self.dashboard_state
        if state is not None:
            self.query_one(DrafterView).show_config(state.drafter_benchmark.config)

    def _start_drafter_benchmark(self, config: DrafterBenchmarkConfig) -> None:
        service = self._drafter_service
        if service is None:
            return
        worker = self._drafter_worker
        if worker is not None and not worker.is_finished:
            self.notify(
                "A Drafter benchmark is already running.",
                title="Drafter",
                severity="error",
            )
            return
        try:
            pending = service.begin_benchmark(config)
        except DrafterBenchmarkOperationError as error:
            self.notify(
                error.user_message,
                title="Drafter",
                severity="error",
            )
            return
        self._pending_drafter = pending
        self.query_one(DrafterView).prepare_run(config)
        self._drafter_worker = self._run_drafter_benchmark(pending)

    @work(
        name="drafter-benchmark",
        group="drafter-benchmark",
        exit_on_error=False,
        exclusive=False,
    )
    async def _run_drafter_benchmark(
        self,
        pending: PendingDrafterBenchmark,
    ) -> DrafterBenchmarkResult | None:
        service = self._drafter_service
        if service is None:
            return None
        try:
            return await service.run_benchmark(pending)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logging.getLogger(__name__).info(
                "Drafter worker failed benchmark=%s exception_class=%s",
                pending.benchmark_id,
                type(error).__name__,
            )
            self.notify(
                "Drafter benchmark failed",
                title="Drafter",
                severity="error",
            )
            raise
        finally:
            if self._pending_drafter == pending:
                self._pending_drafter = None
            if not self._shutting_down:
                if self._refresh_timer is not None:
                    self._refresh_timer.reset()
                self._launch_refresh(
                    manual=False,
                    preserve_online_on_failure=True,
                )

    @on(ContextBenchmarkStartRequested)
    def start_context_benchmark(self, event: ContextBenchmarkStartRequested) -> None:
        event.stop()
        self._request_context_benchmark(event.config)

    @on(ContextBenchmarkRunAgainRequested)
    def run_context_benchmark_again(
        self, event: ContextBenchmarkRunAgainRequested
    ) -> None:
        event.stop()
        state = self.dashboard_state
        result = state.context_benchmark.latest_result if state is not None else None
        if result is None:
            self.notify(
                "Context result is no longer available.",
                title="Context Length",
                severity="error",
            )
            return
        self._request_context_benchmark(result.config)

    @on(ContextBenchmarkEditRequested)
    def edit_context_benchmark(self, event: ContextBenchmarkEditRequested) -> None:
        event.stop()
        state = self.dashboard_state
        if state is not None:
            self.query_one(ContextView).show_config(state.context_benchmark.config)

    @staticmethod
    def _context_requires_confirmation(config: ContextBenchmarkConfig) -> bool:
        largest = (
            config.probe_maximum_tokens
            if config.mode == "probe"
            else max(config.target_lengths)
        )
        threshold = config.warning_threshold_tokens
        return (
            largest >= threshold
            if config.context_unit == "tokens"
            else largest >= threshold * 4
        )

    def _request_context_benchmark(self, config: ContextBenchmarkConfig) -> None:
        if self._context_requires_confirmation(config):
            self.push_screen(
                LargeContextConfirmation(),
                lambda confirmed: (
                    self._start_context_benchmark(config) if confirmed else None
                ),
            )
            return
        self._start_context_benchmark(config)

    def _start_context_benchmark(self, config: ContextBenchmarkConfig) -> None:
        service = self._context_benchmark_service
        if service is None:
            return
        worker = self._context_worker
        if worker is not None and not worker.is_finished:
            self.notify(
                "A Context benchmark is already running.",
                title="Context Length",
                severity="error",
            )
            return
        try:
            pending = service.begin_benchmark(config)
        except ContextBenchmarkOperationError as error:
            self.notify(error.user_message, title="Context Length", severity="error")
            return
        self._pending_context = pending
        self.query_one(ContextView).prepare_run(config)
        if self._context_hardware_timer is not None:
            self._context_hardware_timer.stop()
        if self._hardware_monitor is not None:
            self._context_hardware_timer = self.set_interval(
                config.hardware_sample_interval_seconds,
                self._automatic_hardware_refresh,
            )
        self._context_worker = self._run_context_benchmark(pending)

    @work(
        name="context-benchmark",
        group="context-benchmark",
        exit_on_error=False,
        exclusive=False,
    )
    async def _run_context_benchmark(
        self, pending: PendingContextBenchmark
    ) -> ContextBenchmarkResult | None:
        service = self._context_benchmark_service
        if service is None:
            return None
        try:
            return await service.run_benchmark(pending)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.notify(
                "Context benchmark failed",
                title="Context Length",
                severity="error",
            )
            raise
        finally:
            if self._pending_context == pending:
                self._pending_context = None
            if self._context_hardware_timer is not None:
                self._context_hardware_timer.stop()
                self._context_hardware_timer = None
            if not self._shutting_down:
                if self._hardware_timer is not None:
                    self._hardware_timer.reset()
                if self._refresh_timer is not None:
                    self._refresh_timer.reset()
                self._launch_refresh(
                    manual=False,
                    preserve_online_on_failure=True,
                )

    @on(ConcurrencyBenchmarkStartRequested)
    def start_concurrency_benchmark(
        self, event: ConcurrencyBenchmarkStartRequested
    ) -> None:
        event.stop()
        self._start_concurrency_benchmark(event.config)

    @on(ConcurrencyBenchmarkRunAgainRequested)
    def run_concurrency_benchmark_again(
        self, event: ConcurrencyBenchmarkRunAgainRequested
    ) -> None:
        event.stop()
        state = self.dashboard_state
        result = (
            state.concurrency_benchmark.latest_result if state is not None else None
        )
        if result is None:
            self.notify(
                "Concurrency result is no longer available.",
                title="Concurrency",
                severity="error",
            )
            return
        self._start_concurrency_benchmark(result.config)

    @on(ConcurrencyBenchmarkEditRequested)
    def edit_concurrency_benchmark(
        self, event: ConcurrencyBenchmarkEditRequested
    ) -> None:
        event.stop()
        state = self.dashboard_state
        if state is not None:
            self.query_one(ConcurrencyView).show_config(
                state.concurrency_benchmark.config
            )

    def _start_concurrency_benchmark(self, config: ConcurrencyBenchmarkConfig) -> None:
        service = self._benchmark_service
        if service is None:
            return
        worker = self._benchmark_worker
        if worker is not None and not worker.is_finished:
            self.notify(
                "A Concurrency benchmark is already running.",
                title="Concurrency",
                severity="error",
            )
            return
        try:
            pending = service.begin_benchmark(config)
        except BenchmarkOperationError as error:
            self.notify(error.user_message, title="Concurrency", severity="error")
            return
        self._pending_benchmark = pending
        self.query_one(ConcurrencyView).prepare_run(config)
        self._benchmark_worker = self._run_concurrency_benchmark(pending)

    @work(
        name="concurrency-benchmark",
        group="concurrency-benchmark",
        exit_on_error=False,
        exclusive=False,
    )
    async def _run_concurrency_benchmark(
        self, pending: PendingConcurrencyBenchmark
    ) -> ConcurrencyBenchmarkResult | None:
        service = self._benchmark_service
        if service is None:
            return None
        try:
            return await service.run_benchmark(pending)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.notify(
                "Concurrency benchmark failed",
                title="Concurrency",
                severity="error",
            )
            raise
        finally:
            if self._pending_benchmark == pending:
                self._pending_benchmark = None
            if not self._shutting_down:
                if self._refresh_timer is not None:
                    self._refresh_timer.reset()
                self._launch_refresh(
                    manual=False,
                    preserve_online_on_failure=True,
                )

    @on(SpeedTestStartRequested)
    def start_speed_test(self, event: SpeedTestStartRequested) -> None:
        event.stop()
        self._start_speed_test(event.config)

    def _start_speed_test(self, config: SpeedTestConfig) -> None:
        service = self._speed_test_service
        if service is None:
            return
        worker = self._speed_test_worker
        if worker is not None and not worker.is_finished:
            self.notify(
                "A Speed Test is already running.",
                title="Speed Test",
                severity="error",
            )
            return
        try:
            pending = service.begin_test(config)
        except SpeedTestOperationError as error:
            self.notify(error.user_message, title="Speed Test", severity="error")
            return
        self._pending_speed_test = pending
        self.query_one(SpeedTestView).prepare_run(config)
        self._speed_test_worker = self._run_speed_test(pending)

    @work(
        name="speed-test",
        group="speed-test",
        exit_on_error=False,
        exclusive=False,
    )
    async def _run_speed_test(
        self, pending: PendingSpeedTest
    ) -> SpeedTestResult | None:
        service = self._speed_test_service
        if service is None:
            return None
        try:
            return await service.run_test(pending)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.notify(
                "Speed Test failed",
                title="Speed Test",
                severity="error",
            )
            raise
        finally:
            if self._pending_speed_test == pending:
                self._pending_speed_test = None

    @on(SpeedTestCancelRequested)
    def cancel_speed_test(self, event: SpeedTestCancelRequested) -> None:
        event.stop()
        self._cancel_speed_test()

    @on(SpeedTestRunAgainRequested)
    def run_speed_test_again(self, event: SpeedTestRunAgainRequested) -> None:
        event.stop()
        self._run_result_again(event.run_id)

    def _run_result_again(self, run_id: str) -> None:
        state = self.dashboard_state
        if state is None:
            return
        result = state.speed_test.result_by_id(run_id)
        if result is None:
            self.notify("Speed Test result is no longer available.", severity="error")
            return
        worker = self._speed_test_worker
        if worker is not None and not worker.is_finished:
            self.notify("A Speed Test is already running.", severity="error")
            return
        self._select_result(None)
        self._set_active_view("speed-test")
        self.query_one(SpeedTestView).show_config(result.config)
        self._start_speed_test(result.config)

    @on(SpeedTestResultSelected)
    def select_speed_result(self, event: SpeedTestResultSelected) -> None:
        event.stop()
        self._select_result(event.run_id)

    @on(SpeedTestExportRequested)
    def export_speed_result(self, event: SpeedTestExportRequested) -> None:
        event.stop()
        state = self.dashboard_state
        result = (
            state.speed_test.result_by_id(event.run_id) if state is not None else None
        )
        if result is not None:
            self._export_result(result)

    @on(SpeedTestCopySummaryRequested)
    def copy_speed_summary(self, event: SpeedTestCopySummaryRequested) -> None:
        event.stop()
        state = self.dashboard_state
        result = (
            state.speed_test.result_by_id(event.run_id) if state is not None else None
        )
        if result is not None:
            self._copy_result_summary(result)

    @on(ClearConversationRequested)
    def clear_chat_from_widget(self, event: ClearConversationRequested) -> None:
        event.stop()
        self._clear_chat()

    def _clear_chat(self) -> None:
        service = self._chat_service
        if service is None:
            return
        try:
            service.clear_conversation()
        except ChatOperationError as error:
            self.notify(error.user_message, title="Chat", severity="error")

    @on(SettingsSubmitted)
    def apply_chat_settings(self, event: SettingsSubmitted) -> None:
        event.stop()
        service = self._chat_service
        if service is None:
            return
        try:
            service.update_preferences(
                event.settings,
                event.system_prompt,
                event.show_system_prompt,
            )
        except ChatOperationError as error:
            self.notify(error.user_message, title="Chat", severity="error")
            return
        self.notify("Generation settings applied.", title="Chat")

    @on(OptionList.OptionSelected, "#sidebar-menu")
    def select_workspace(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        option_id = event.option.id
        if option_id not in {
            "overview",
            "chat",
            "speed-test",
            "concurrency",
            "context",
            "tool-calling",
            "drafter",
            "results",
            "settings",
        }:
            return
        view = cast(ActiveView, option_id)
        if view == "results":
            self._select_result(None)
        self._set_active_view(view)
        if view == "chat":
            self.call_after_refresh(self.query_one(ChatView).focus_prompt)
        elif view == "speed-test":
            self.call_after_refresh(
                self.query_one(SpeedTestView).config_panel.focus_presets
            )
        elif view == "concurrency":
            state = self.dashboard_state
            if state is not None and not state.concurrency_benchmark.is_active:
                self.call_after_refresh(
                    self.query_one(ConcurrencyView).config_panel.focus_mode
                )
        elif view == "context":
            state = self.dashboard_state
            if state is not None and not state.context_benchmark.is_active:
                self.call_after_refresh(
                    self.query_one(ContextView).config_panel.focus_mode
                )
        elif view == "tool-calling":
            state = self.dashboard_state
            if state is not None and not state.tool_calling_benchmark.is_active:
                self.call_after_refresh(
                    self.query_one(ToolCallingView).config_panel.focus_suite
                )
        elif view == "drafter":
            state = self.dashboard_state
            if state is not None and not state.drafter_benchmark.is_active:
                self.call_after_refresh(
                    self.query_one(DrafterView).config_panel.focus_prompt
                )
        elif view == "results":
            self.call_after_refresh(self.query_one(ResultsView).focus_list)

    def _set_active_view(self, view: ActiveView) -> None:
        store = self._state_store
        if store is None:
            return
        state = store.update(lambda current: replace(current, active_view=view))
        self._handle_dashboard_state_change(state)

    def _select_result(self, run_id: str | None) -> None:
        store = self._state_store
        if store is None:
            return

        def transform(state: ApplicationState) -> ApplicationState:
            if run_id is not None and state.speed_test.result_by_id(run_id) is None:
                return state
            return replace(
                state,
                speed_test=replace(
                    state.speed_test,
                    selected_result_id=run_id,
                ),
            )

        self._handle_dashboard_state_change(store.update(transform))

    @on(OptionList.OptionSelected)
    def select_discovered_model(self, event: OptionList.OptionSelected) -> None:
        """Select only options emitted by the discovered-model selector."""
        if event.option_list.id != "model-selector":
            return
        event.stop()
        monitor = self._monitor
        state = self.dashboard_state
        if monitor is None or state is None:
            return
        if not 0 <= event.option_index < len(state.available_models):
            return
        model_id = state.available_models[event.option_index].id
        monitor.select_model(model_id)

    async def on_unmount(self) -> None:
        """Cancel all lanes, flush UI streams, then close app-owned resources."""
        self._shutting_down = True
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
        if self._hardware_timer is not None:
            self._hardware_timer.stop()
        if self._context_hardware_timer is not None:
            self._context_hardware_timer.stop()
        tool_calling_service = self._tool_calling_service
        tool_calling_pending = self._pending_tool_calling
        try:
            if tool_calling_service is not None and tool_calling_pending is not None:
                tool_calling_service.request_cancellation(
                    tool_calling_pending.benchmark_id
                )
                tool_calling_worker = self._tool_calling_worker
                if (
                    tool_calling_worker is None
                    or tool_calling_worker.state is WorkerState.PENDING
                ):
                    tool_calling_service.cancel_reservation(tool_calling_pending)
        except Exception as error:
            logging.getLogger(__name__).error(
                "Cleanup failed resource=tool-calling-reservation error=%s",
                type(error).__name__,
            )

        drafter_service = self._drafter_service
        drafter_pending = self._pending_drafter
        try:
            if drafter_service is not None and drafter_pending is not None:
                drafter_service.request_cancellation(drafter_pending.benchmark_id)
                drafter_worker = self._drafter_worker
                if (
                    drafter_worker is None
                    or drafter_worker.state is WorkerState.PENDING
                ):
                    drafter_service.cancel_reservation(drafter_pending)
        except Exception as error:
            logging.getLogger(__name__).error(
                "Cleanup failed resource=drafter-reservation error=%s",
                type(error).__name__,
            )

        context_service = self._context_benchmark_service
        context_pending = self._pending_context
        try:
            if context_service is not None and context_pending is not None:
                context_service.request_cancellation(context_pending.benchmark_id)
                context_worker = self._context_worker
                if (
                    context_worker is None
                    or context_worker.state is WorkerState.PENDING
                ):
                    context_service.cancel_reservation(context_pending)
        except Exception as error:
            logging.getLogger(__name__).error(
                "Cleanup failed resource=context-reservation error=%s",
                type(error).__name__,
            )
        benchmark_service = self._benchmark_service
        benchmark_pending = self._pending_benchmark
        try:
            if benchmark_service is not None and benchmark_pending is not None:
                benchmark_service.request_cancellation(benchmark_pending.benchmark_id)
                benchmark_worker = self._benchmark_worker
                if (
                    benchmark_worker is None
                    or benchmark_worker.state is WorkerState.PENDING
                ):
                    benchmark_service.cancel_reservation(benchmark_pending)
        except Exception as error:
            logging.getLogger(__name__).error(
                "Cleanup failed resource=concurrency-reservation error=%s",
                type(error).__name__,
            )
        speed_service = self._speed_test_service
        speed_pending = self._pending_speed_test
        try:
            if speed_service is not None and speed_pending is not None:
                speed_service.request_cancellation(speed_pending.run_id)
                speed_worker = self._speed_test_worker
                if speed_worker is None or speed_worker.state is WorkerState.PENDING:
                    speed_service.cancel_reservation(speed_pending)
        except Exception as error:
            logging.getLogger(__name__).error(
                "Cleanup failed resource=speed-test-reservation error=%s",
                type(error).__name__,
            )
        chat_pending = self._pending_chat
        chat_worker = self._chat_worker
        try:
            if (
                chat_pending is not None
                and self._chat_service is not None
                and (chat_worker is None or chat_worker.state is WorkerState.PENDING)
            ):
                self._chat_service.cancel_reservation(chat_pending.generation_id)
        except Exception as error:
            logging.getLogger(__name__).error(
                "Cleanup failed resource=chat-reservation error=%s",
                type(error).__name__,
            )

        workers: tuple[Worker[object] | None, ...] = (
            cast(Worker[object] | None, self._tool_calling_worker),
            cast(Worker[object] | None, self._drafter_worker),
            cast(Worker[object] | None, self._context_worker),
            cast(Worker[object] | None, self._benchmark_worker),
            cast(Worker[object] | None, self._speed_test_worker),
            cast(Worker[object] | None, self._chat_worker),
            cast(Worker[object] | None, self._refresh_worker),
            cast(Worker[object] | None, self._hardware_worker),
        )
        for worker in workers:
            if worker is not None and not worker.is_finished:
                worker.cancel()
        for worker in workers:
            if worker is not None:
                with suppress(WorkerError, asyncio.CancelledError):
                    await worker.wait()

        if self._chat_view is not None:
            try:
                await self._chat_view.stop_stream()
            except Exception as error:
                logging.getLogger(__name__).error(
                    "Cleanup failed resource=chat-render error=%s",
                    type(error).__name__,
                )
        if self._hardware_monitor is not None:
            try:
                await self._hardware_monitor.aclose()
            except Exception as error:
                logging.getLogger(__name__).error(
                    "Cleanup failed resource=hardware-monitor error=%s",
                    type(error).__name__,
                )
        if self._api_client is not None:
            try:
                await self._api_client.aclose()
            except Exception as error:
                logging.getLogger(__name__).error(
                    "Cleanup failed resource=api-client error=%s",
                    type(error).__name__,
                )


def run() -> None:
    """Launch ModelTop with file logging configured."""
    configure_logging()
    logging.getLogger(__name__).info("Starting ModelTop")
    ModelTopApp().run()
