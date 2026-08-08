"""Textual Pilot coverage for the Drafter benchmark workspace."""

# pyright: reportPrivateUsage=false

import asyncio
import json
from collections.abc import AsyncIterator
from typing import cast

import httpx
import pytest
from textual.widgets import (
    Button,
    ContentSwitcher,
    DataTable,
    Input,
    OptionList,
    Static,
)

from modeltop.api.client import OpenAICompatibleClient
from modeltop.app import ModelTopApp
from modeltop.benchmarks.models import DrafterBenchmarkConfig, DrafterBenchmarkStatus
from modeltop.screens.drafter import DrafterView
from modeltop.state import ServerStatus
from modeltop.widgets.drafter_configuration import DrafterConfigurationPanel
from modeltop.widgets.drafter_results import DrafterResultsPanel
from modeltop.widgets.footer import StatusFooter
from tests.test_app import (
    _config,
    _plain_render,
    _ScriptedHardwareProvider,
    _wait_for_status,
)


class _GatedDrafterStream(httpx.AsyncByteStream):
    def __init__(self, first: str = "partial") -> None:
        self.first = first
        self.first_sent = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        payload = {
            "choices": [{"delta": {"content": self.first}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(payload)}\n\n".encode()
        self.first_sent.set()
        await self.release.wait()
        yield (
            b'data: {"choices":[{"delta":{"content":" done"},'
            b'"finish_reason":null}]}\n\n'
        )
        yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield (
            b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":4,'
            b'"total_tokens":14,"draft_tokens":6,"accepted_tokens":4,'
            b'"acceptance_rate":0.6666666666666666}}\n\n'
        )
        yield b"data: [DONE]\n\n"

    async def aclose(self) -> None:
        self.closed = True


class _ImmediateDrafterStream(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield (
            b'data: {"choices":[{"delta":{"content":"answer"},'
            b'"finish_reason":null}]}\n\n'
        )
        yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield (
            b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":4,'
            b'"total_tokens":14,"completion_tokens_details":'
            b'{"accepted_prediction_tokens":4,"rejected_prediction_tokens":2}}}\n\n'
        )
        yield b"data: [DONE]\n\n"

    async def aclose(self) -> None:
        return None


class _NoTelemetryDrafterStream(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'
        yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield (
            b'data: {"choices":[],"usage":{"prompt_tokens":10,'
            b'"completion_tokens":4,"total_tokens":14}}\n\n'
        )
        yield b"data: [DONE]\n\n"

    async def aclose(self) -> None:
        return None


class _DrafterTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        streams: list[httpx.AsyncByteStream] | None = None,
        metrics_snapshots: list[str] | None = None,
    ) -> None:
        self.streams = list(streams or [])
        self.model_requests = 0
        self.chat_requests: list[dict[str, object]] = []
        self.metrics_snapshots = list(metrics_snapshots or [])
        self.close_count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/metrics"):
            payload = self.metrics_snapshots.pop(0) if self.metrics_snapshots else ""
            return httpx.Response(200, text=payload)
        if request.method == "GET":
            self.model_requests += 1
            return httpx.Response(200, json={"data": [{"id": "model"}]})
        payload = cast(dict[str, object], json.loads(request.content))
        self.chat_requests.append(payload)
        stream = self.streams.pop(0) if self.streams else _ImmediateDrafterStream()
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
        )

    async def aclose(self) -> None:
        self.close_count += 1


def _app(transport: _DrafterTransport) -> ModelTopApp:
    client = OpenAICompatibleClient(
        "http://server/prefix/v1", None, 5, transport=transport
    )
    return ModelTopApp(
        _config(refresh_interval=3600),
        client=client,
        hardware_provider=_ScriptedHardwareProvider(),
    )


async def _wait_for_drafter_status(
    app: ModelTopApp,
    status: DrafterBenchmarkStatus,
) -> None:
    for _ in range(200):
        state = app.dashboard_state
        if state is not None and state.drafter_benchmark.status is status:
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"Drafter did not reach {status}")


def test_sidebar_configure_run_results_rerun_edit_and_escape() -> None:
    async def scenario() -> None:
        transport = _DrafterTransport()
        app = _app(transport)
        async with app.run_test(size=(120, 36)) as pilot:
            await _wait_for_status(app, pilot, ServerStatus.ONLINE)
            menu = app.query_one("#sidebar-menu", OptionList)
            menu.highlighted = 7
            menu.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert app.dashboard_state is not None
            assert app.dashboard_state.active_view == "drafter"

            view = app.query_one("#drafter-workspace", DrafterView)
            panel = view.query_one(DrafterConfigurationPanel)
            assert panel.query_one("#drafter-warmups", Input).value == "1"
            assert panel.query_one("#drafter-measured", Input).value == "5"

            config = DrafterBenchmarkConfig(
                prompt="drafter app prompt",
                warmup_runs=0,
                measured_runs=2,
                max_tokens=32,
            )
            panel.load_config(config)
            panel.query_one("#drafter-run", Button).press()
            await _wait_for_drafter_status(app, DrafterBenchmarkStatus.COMPLETED)
            await pilot.pause()

            state = app.dashboard_state
            assert state is not None
            result = state.drafter_benchmark.latest_result
            assert result is not None
            assert result.successful_runs == 2
            assert result.acceptance_rate.count == 2
            assert result.draft_tokens.mean == 6
            assert result.accepted_tokens.mean == 4
            assert result.acceptance_rate.mean == pytest.approx(4 / 6)
            assert result.speculative_telemetry_available
            switcher = app.query_one("#drafter-view-switcher", ContentSwitcher)
            assert switcher.current == "drafter-result-panel"
            results_panel = view.query_one(DrafterResultsPanel)
            rendered = "\n".join(
                _plain_render(widget) for widget in results_panel.query(Static)
            )
            assert "Mean acceptance: 0.67" in rendered
            assert "Mean draft tokens: 6.0" in rendered
            assert "Mean accepted tokens: 4.0" in rendered
            table = cast(
                DataTable[str],
                results_panel.query_one("#drafter-result-runs", DataTable),
            )
            assert table.row_count == 2
            table_rendered = _plain_render(table)
            assert table_rendered.count("6") >= 2
            assert table_rendered.count("4") >= 2
            assert table_rendered.count("0.67") == 2
            assert "DRAFTER" in _plain_render(app.query_one("#header-subtitle", Static))
            footer = _plain_render(app.query_one(StatusFooter))
            assert "acc 0.67" in footer

            first_id = result.benchmark_id
            app.action_run_or_rerun()
            await _wait_for_drafter_status(app, DrafterBenchmarkStatus.COMPLETED)
            await pilot.pause()
            rerun_state = app.dashboard_state
            assert rerun_state is not None
            rerun_result = rerun_state.drafter_benchmark.latest_result
            assert rerun_result is not None
            assert rerun_result.benchmark_id != first_id

            app.action_export_result()
            await pilot.pause()
            assert switcher.current == "drafter-config-panel"
            edited = panel.parse_config()
            assert edited is not None
            assert edited.measured_runs == 2

            await pilot.press("escape")
            await pilot.pause()
            assert switcher.current == "drafter-config-panel"

        assert transport.close_count == 1
        assert len(transport.chat_requests) == 4

    asyncio.run(scenario())


def test_vllm_metric_fallback_renders_counter_derived_acceptance_rate() -> None:
    async def scenario() -> None:
        metrics = [
            (
                'vllm:spec_decode_num_draft_tokens_total{model_name="model"} 0\n'
                'vllm:spec_decode_num_accepted_tokens_total{model_name="model"} 0'
            ),
            (
                'vllm:spec_decode_num_draft_tokens_total{model_name="model"} 8\n'
                'vllm:spec_decode_num_accepted_tokens_total{model_name="model"} 6'
            ),
            (
                'vllm:spec_decode_num_draft_tokens_total{model_name="model"} 8\n'
                'vllm:spec_decode_num_accepted_tokens_total{model_name="model"} 6'
            ),
            (
                'vllm:spec_decode_num_draft_tokens_total{model_name="model"} 20\n'
                'vllm:spec_decode_num_accepted_tokens_total{model_name="model"} 15'
            ),
        ]
        transport = _DrafterTransport(
            [_NoTelemetryDrafterStream(), _NoTelemetryDrafterStream()],
            metrics,
        )
        app = _app(transport)
        async with app.run_test(size=(120, 36)) as pilot:
            await _wait_for_status(app, pilot, ServerStatus.ONLINE)
            app._set_active_view("drafter")
            view = app.query_one(DrafterView)
            view.show_config(
                DrafterBenchmarkConfig(
                    prompt="counter fallback", warmup_runs=0, measured_runs=2
                )
            )
            view.query_one("#drafter-run", Button).press()
            await _wait_for_drafter_status(app, DrafterBenchmarkStatus.COMPLETED)
            await pilot.pause()

            state = app.dashboard_state
            assert state is not None
            result = state.drafter_benchmark.latest_result
            assert result is not None
            assert [run.draft_tokens for run in result.run_results] == [8, 12]
            assert [run.accepted_tokens for run in result.run_results] == [6, 9]
            assert result.draft_tokens.mean == 10
            assert result.accepted_tokens.mean == 7.5
            assert result.acceptance_rate.mean == pytest.approx(0.75)
            results_panel = view.query_one(DrafterResultsPanel)
            rendered = "\n".join(
                _plain_render(widget) for widget in results_panel.query(Static)
            )
            assert "Mean acceptance: 0.75" in rendered
            table = cast(
                DataTable[str],
                results_panel.query_one("#drafter-result-runs", DataTable),
            )
            assert _plain_render(table).count("0.75") == 2

        assert transport.metrics_snapshots == []
        assert transport.close_count == 1

    asyncio.run(scenario())


def test_cancel_mid_run_and_mutual_exclusion_message() -> None:
    async def scenario() -> None:
        stream = _GatedDrafterStream()
        transport = _DrafterTransport([stream])
        app = _app(transport)
        async with app.run_test(size=(100, 30)) as pilot:
            await _wait_for_status(app, pilot, ServerStatus.ONLINE)
            app._set_active_view("drafter")
            view = app.query_one(DrafterView)
            view.show_config(
                DrafterBenchmarkConfig(
                    prompt="cancel me",
                    warmup_runs=0,
                    measured_runs=1,
                )
            )
            view.query_one("#drafter-run", Button).press()
            await asyncio.wait_for(stream.first_sent.wait(), timeout=3)
            await pilot.pause()
            running = app.dashboard_state
            assert running is not None
            assert running.drafter_benchmark.is_active
            assert "DRAFTER" in _plain_render(app.query_one("#header-subtitle", Static))

            await pilot.press("escape")
            await _wait_for_drafter_status(app, DrafterBenchmarkStatus.CANCELLED)
            await pilot.pause()
            state = app.dashboard_state
            assert state is not None
            lane = state.drafter_benchmark
            assert lane.active_benchmark_id is None
            assert lane.latest_result is not None
            assert lane.latest_result.status is DrafterBenchmarkStatus.CANCELLED
            assert stream.closed

        assert transport.close_count == 1

    asyncio.run(scenario())
