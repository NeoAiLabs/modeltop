"""Focused keyboard-first Context configuration widget coverage."""

import asyncio

from textual.app import App, ComposeResult
from textual.widgets import Input, OptionList

from modeltop.benchmarks.models import ContextBenchmarkConfig
from modeltop.widgets.context_configuration import ContextConfigurationPanel


class _ContextWidgetApp(App[None]):
    def compose(self) -> ComposeResult:
        yield ContextConfigurationPanel(id="context-config")


def test_context_configuration_preserves_mode_drafts_and_private_values() -> None:
    async def scenario() -> None:
        app = _ContextWidgetApp()
        async with app.run_test(size=(100, 38)) as pilot:
            panel = app.query_one(ContextConfigurationPanel)
            panel.load_config(
                ContextBenchmarkConfig(
                    mode="fixed",
                    target_lengths=(4096,),
                    warmup_requests=0,
                    content_source="repeated_text",
                    base_text="private repeated source",
                )
            )
            fixed = panel.parse_config()
            assert fixed is not None
            assert fixed.target_lengths == (4096,)
            assert fixed.base_text == "private repeated source"

            mode = panel.query_one("#context-mode", OptionList)
            mode.focus()
            await pilot.press("down", "enter")
            await pilot.pause()
            panel.query_one("#context-lengths", Input).value = "8192, 1024, 4096"
            sweep = panel.parse_config()
            assert sweep is not None
            assert sweep.mode == "sweep"
            assert sweep.target_lengths == (1024, 4096, 8192)

            await pilot.press("up", "enter")
            await pilot.pause()
            restored = panel.parse_config()
            assert restored is not None
            assert restored.mode == "fixed"
            assert restored.target_lengths == (4096,)

    asyncio.run(scenario())


def test_retrieval_fields_parse_without_rendering_key_value() -> None:
    async def scenario() -> None:
        app = _ContextWidgetApp()
        async with app.run_test(size=(100, 38)):
            panel = app.query_one(ContextConfigurationPanel)
            panel.load_config(
                ContextBenchmarkConfig(
                    mode="retrieval",
                    target_lengths=(2048,),
                    retrieval_enabled=True,
                    retrieval_positions=("beginning", "end"),
                    retrieval_key="private-key-1234",
                    warmup_requests=0,
                )
            )
            parsed = panel.parse_config()
            assert parsed is not None
            assert parsed.retrieval_positions == ("beginning", "end")
            assert parsed.retrieval_key == "private-key-1234"

    asyncio.run(scenario())
