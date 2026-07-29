"""Focused widget coverage for Concurrency configuration and request tables."""

import asyncio
from dataclasses import replace

from textual.app import App, ComposeResult
from textual.widgets import Input, OptionList

from modeltop.benchmarks.models import (
    ConcurrencyBenchmarkConfig,
    ConcurrencyRequestProgress,
)
from modeltop.widgets.benchmark_configuration import BenchmarkConfigurationPanel
from modeltop.widgets.request_table import RequestTable


class _WidgetApp(App[None]):
    def compose(self) -> ComposeResult:
        yield BenchmarkConfigurationPanel(id="config")
        yield RequestTable(id="requests")


def test_configuration_fixed_and_sweep_drafts_and_run_plan() -> None:
    async def scenario() -> None:
        app = _WidgetApp()
        async with app.run_test(size=(80, 24)) as pilot:
            panel = app.query_one(BenchmarkConfigurationPanel)
            panel.load_config(
                ConcurrencyBenchmarkConfig(
                    mode="fixed",
                    concurrency_levels=(4,),
                    requests_per_level=8,
                    warmup_requests=1,
                )
            )
            fixed = panel.parse_config()
            assert fixed is not None
            assert fixed.mode == "fixed"
            assert fixed.concurrency_levels == (4,)
            assert "maximum 4 simultaneous" in str(
                panel.query_one("#concurrency-run-plan").render()
            )
            mode = panel.query_one("#concurrency-mode", OptionList)
            mode.focus()
            await pilot.press("down", "enter")
            await pilot.pause()
            panel.query_one("#concurrency-levels", Input).value = "8, 2, 4"
            sweep = panel.parse_config()
            assert sweep is not None
            assert sweep.concurrency_levels == (2, 4, 8)
            await pilot.press("up", "enter")
            await pilot.pause()
            restored = panel.parse_config()
            assert restored is not None
            assert restored.concurrency_levels == (4,)

    asyncio.run(scenario())


def test_request_table_retains_500_rows_and_updates_one_in_place() -> None:
    async def scenario() -> None:
        app = _WidgetApp()
        async with app.run_test(size=(80, 24)) as pilot:
            table = app.query_one(RequestTable)
            rows = tuple(
                ConcurrencyRequestProgress(
                    request_id=f"benchmark-c8-r{sequence:04d}",
                    concurrency_level=8,
                    sequence_number=sequence,
                    state="queued",
                    queued_at=0.0,
                    started_at=None,
                    latest_metrics=None,
                    error=None,
                )
                for sequence in range(1, 501)
            )
            table.update_requests(rows)
            assert table.row_count == 500
            keys_before = tuple(table.rows)
            changed = replace(rows[249], state="error", error="safe failure")
            table.update_requests((changed,))
            assert table.row_count == 500
            assert tuple(table.rows) == keys_before
            assert table.get_cell(changed.request_id, "STATE") == "ERROR"
            assert table.get_cell(changed.request_id, "ERROR") == "safe failure"
            table.scroll_end(animate=False)
            table.scroll_right(animate=False)
            await pilot.resize_terminal(60, 20)
            await pilot.resize_terminal(100, 30)

    asyncio.run(scenario())


def test_request_table_follows_new_rows_until_reader_scrolls_away() -> None:
    async def scenario() -> None:
        app = _WidgetApp()
        async with app.run_test(size=(80, 24)) as pilot:
            table = app.query_one(RequestTable)

            def queued(sequence: int) -> ConcurrencyRequestProgress:
                return ConcurrencyRequestProgress(
                    request_id=f"benchmark-c4-r{sequence:04d}",
                    concurrency_level=4,
                    sequence_number=sequence,
                    state="queued",
                    queued_at=0.0,
                    started_at=None,
                    latest_metrics=None,
                    error=None,
                )

            table.update_requests(tuple(queued(sequence) for sequence in range(1, 41)))
            await pilot.pause()
            assert table.is_vertical_scroll_end

            table.update_requests((queued(41),))
            await pilot.pause()
            assert table.is_vertical_scroll_end

            table.scroll_home(animate=False)
            await pilot.pause()
            table.update_requests((queued(42),))
            await pilot.pause()
            assert table.scroll_offset.y == 0
            assert not table.is_vertical_scroll_end

            table.scroll_end(animate=False)
            await pilot.pause()
            table.update_requests((queued(43),))
            await pilot.pause()
            assert table.is_vertical_scroll_end

    asyncio.run(scenario())
