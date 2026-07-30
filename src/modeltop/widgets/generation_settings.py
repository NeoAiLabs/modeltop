"""Compact validated in-memory generation settings panel."""

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Label, Switch

from modeltop.chat.models import GenerationSettings
from modeltop.chat.session import ChatSession
from modeltop.messages import SettingsSubmitted


class GenerationSettingsPanel(Vertical):
    """Editable settings that publish only fully validated values."""

    DEFAULT_CSS = """
    GenerationSettingsPanel {
        display: none;
        width: 1fr;
        height: 7;
        border: solid $border-blurred;
        background: $catppuccin-base;
        padding: 0 1;
    }

    GenerationSettingsPanel.visible { display: block; }
    GenerationSettingsPanel .settings-row { width: 1fr; height: 2; }
    GenerationSettingsPanel Label { width: auto; height: 1; margin: 1 1 0 0; }
    GenerationSettingsPanel Input { width: 10; height: 1; margin: 0 2 0 0; }
    GenerationSettingsPanel #system-prompt { width: 1fr; }
    GenerationSettingsPanel Switch { width: 8; height: 1; }
    GenerationSettingsPanel Button { min-width: 10; height: 1; }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(classes="settings-row"):
            yield Label("TEMP")
            yield Input(id="temperature", type="number")
            yield Label("TOP-P")
            yield Input(id="top-p", type="number")
            yield Label("MAX")
            yield Input(id="max-tokens", type="integer")
            yield Label("SEED")
            yield Input(id="seed", placeholder="none", type="integer")
        with Horizontal(classes="settings-row"):
            yield Label("SYSTEM")
            yield Input(id="system-prompt")
        with Horizontal(classes="settings-row"):
            yield Label("SHOW SYSTEM")
            yield Switch(id="show-system")
            yield Label("DISABLE THINKING")
            yield Switch(id="disable-thinking")
            yield Button("APPLY", id="apply-settings", variant="primary")

    @property
    def is_open(self) -> bool:
        return self.has_class("visible")

    def toggle(self) -> None:
        self.set_class(not self.is_open, "visible")
        if self.is_open:
            self.query_one("#temperature", Input).focus()

    def load_session(self, session: ChatSession) -> None:
        settings = session.settings
        self.query_one("#temperature", Input).value = str(settings.temperature)
        self.query_one("#top-p", Input).value = str(settings.top_p)
        self.query_one("#max-tokens", Input).value = str(settings.max_tokens)
        self.query_one("#seed", Input).value = (
            "" if settings.seed is None else str(settings.seed)
        )
        self.query_one("#system-prompt", Input).value = session.system_prompt
        self.query_one("#show-system", Switch).value = session.show_system_prompt
        self.query_one("#disable-thinking", Switch).value = (
            settings.enable_thinking is False
        )

    def set_generating(self, generating: bool) -> None:
        for control in self.query("Input, Switch, Button"):
            control.disabled = generating

    @on(Button.Pressed, "#apply-settings")
    def apply_settings(self) -> None:
        try:
            seed_text = self.query_one("#seed", Input).value.strip()
            settings = GenerationSettings(
                temperature=float(self.query_one("#temperature", Input).value),
                top_p=float(self.query_one("#top-p", Input).value),
                max_tokens=int(self.query_one("#max-tokens", Input).value),
                seed=int(seed_text) if seed_text else None,
                enable_thinking=(
                    False if self.query_one("#disable-thinking", Switch).value else None
                ),
            )
        except (TypeError, ValueError) as error:
            self.notify(
                str(error), title="Invalid generation settings", severity="error"
            )
            return
        self.post_message(
            SettingsSubmitted(
                settings,
                self.query_one("#system-prompt", Input).value,
                self.query_one("#show-system", Switch).value,
            )
        )
