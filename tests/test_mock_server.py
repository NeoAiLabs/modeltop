"""Real-socket tests for the standard-library development mock server."""
# pyright: reportPrivateUsage=false

import asyncio
import json
import sys
import threading
import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar, cast

import httpx
import pytest
from scripts.mock_server import (
    ChatMode,
    ModelsRequestHandler,
    _arguments,
)

from modeltop.benchmarks.base import BenchmarkContext
from modeltop.benchmarks.models import (
    R0b0benchBenchmarkConfig,
    R0b0benchLaneResult,
    R0b0benchLaneStatus,
    ToolCallingBenchmarkConfig,
    ToolCallingBenchmarkStatus,
)
from modeltop.benchmarks.r0b0bench import (
    R0b0benchRunnerRequest,
    SubprocessR0b0benchRunner,
)
from modeltop.benchmarks.r0b0bench_contract import R0b0benchLaneId
from modeltop.benchmarks.tool_calling import ToolCallingBenchmark


@contextmanager
def _server(
    mode: ChatMode = "normal",
    *,
    first_token_delay: float = 0.0,
    chunk_delay: float = 0.0,
    output_chunks: int = 6,
    failure_every_n: int = 3,
    timeout_every_n: int = 2,
    disconnect_every_n: int = 2,
    timeout_delay: float = 0.1,
    context_limit: int = 32768,
) -> Generator[str]:
    configured_failure_every_n = failure_every_n
    configured_timeout_every_n = timeout_every_n
    configured_disconnect_every_n = disconnect_every_n

    class Handler(ModelsRequestHandler):
        chat_mode = mode
        stream_delay_seconds = 0.0
        first_token_delay_seconds = first_token_delay
        chunk_delay_seconds = chunk_delay
        output_chunk_count = output_chunks
        failure_every_n = configured_failure_every_n
        timeout_every_n = configured_timeout_every_n
        disconnect_every_n = configured_disconnect_every_n
        timeout_delay_seconds = timeout_delay
        context_limit_tokens = context_limit
        prompt_hash_counts: ClassVar[dict[str, int]] = {}
        total_chat_requests = 0
        models_requests = 0
        active_requests = 0
        peak_active_requests = 0

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = cast(tuple[str, int], server.server_address)
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive()
        deadline = time.monotonic() + 1.0
        while True:
            with Handler.chat_request_lock:
                active = Handler.active_requests
            if active == 0 or time.monotonic() >= deadline:
                break
            time.sleep(0.001)
        assert active == 0


def _payload(
    *,
    stream: bool,
    seed: int | None = None,
    max_tokens: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": "modeltop/mock-small",
        "messages": [{"role": "user", "content": "private prompt"}],
        "stream": stream,
    }
    if seed is not None:
        payload["seed"] = seed
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return payload


def _custom_payload(
    content: str, *, stream: bool, max_tokens: int = 32
) -> dict[str, object]:
    return {
        "model": "modeltop/mock-small",
        "messages": [{"role": "user", "content": content}],
        "stream": stream,
        "max_tokens": max_tokens,
    }


def _tool_payload(
    *,
    stream: bool | None,
    messages: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": "modeltop/mock-small",
        "messages": messages or [{"role": "user", "content": "Send one notification."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "send_notification",
                    "description": "Send a notification.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "email": {"type": "string", "format": "email"},
                            "count": {"type": "integer", "minimum": 1},
                        },
                        "required": ["email", "count"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }
    if stream is not None:
        payload["stream"] = stream
    return payload


def _debug(client: httpx.Client, base_url: str) -> dict[str, int]:
    response = client.get(f"{base_url}/debug/concurrency")
    response.raise_for_status()
    return cast(dict[str, int], response.json())


def test_models_endpoint_remains_exact_and_methods_are_validated() -> None:
    with _server() as base_url, httpx.Client(timeout=3) as client:
        models = client.get(f"{base_url}/v1/models")
        assert models.status_code == 200
        assert [item["id"] for item in models.json()["data"]] == [
            "modeltop/mock-small",
            "modeltop/mock-large",
        ]
        assert client.get(f"{base_url}/v1/models/").status_code == 404
        assert client.post(f"{base_url}/missing", json={}).status_code == 404
        assert client.put(f"{base_url}/v1/chat/completions").status_code == 405
        assert (
            client.post(
                f"{base_url}/v1/chat/completions",
                content=b"{",
                headers={"content-type": "application/json"},
            ).status_code
            == 400
        )
        assert (
            client.post(f"{base_url}/v1/chat/completions", json={}).status_code == 400
        )
        assert _debug(client, base_url) == {
            "active_requests": 0,
            "peak_active_requests": 0,
            "total_chat_requests": 0,
            "models_requests": 1,
        }


def test_streaming_frames_content_usage_finish_and_done() -> None:
    with _server() as base_url, httpx.Client(timeout=3) as client:
        with client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json=_payload(stream=True),
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream"
            frames = [
                frame for frame in "".join(response.iter_text()).split("\n\n") if frame
            ]
        assert frames[-1] == "data: [DONE]"
        payloads = [json.loads(frame.removeprefix("data: ")) for frame in frames[:-1]]
        content = "".join(
            choice["delta"].get("content", "")
            for payload in payloads
            for choice in payload.get("choices", [])
        )
        assert content == "Mock response from modeltop/mock-small."
        assert any(
            choice.get("finish_reason") == "stop"
            for payload in payloads
            for choice in payload.get("choices", [])
        )
        usage_payload = payloads[-1]
        assert usage_payload["choices"] == []
        assert usage_payload["usage"]["total_tokens"] > 0


def test_non_stream_completion_is_deterministic() -> None:
    with _server() as base_url, httpx.Client(timeout=3) as client:
        response = client.post(
            f"{base_url}/v1/chat/completions", json=_payload(stream=False)
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["choices"][0]["message"] == {
            "role": "assistant",
            "content": "Mock response from modeltop/mock-small.",
        }
        assert payload["choices"][0]["finish_reason"] == "stop"
        assert payload["usage"]["total_tokens"] > 0


@pytest.mark.parametrize(
    ("mode", "stream", "status", "body_marker"),
    [
        ("no-usage", True, 200, "[DONE]"),
        ("no-stream", True, 400, "not supported"),
        ("no-stream", False, 200, "Mock response"),
        ("malformed", True, 200, "data: {"),
        ("malformed", False, 200, "{"),
        ("disconnect", True, 200, "Mock response"),
        ("error", True, 500, "mock server error"),
        ("slow", True, 200, "[DONE]"),
        ("variable", True, 200, "[DONE]"),
        ("rate-limit", True, 429, "mock rate limit"),
        ("concurrency-degradation", True, 200, "[DONE]"),
    ],
)
def test_chat_modes(
    mode: ChatMode, stream: bool, status: int, body_marker: str
) -> None:
    with _server(mode) as base_url, httpx.Client(timeout=3) as client:
        response = client.post(
            f"{base_url}/v1/chat/completions", json=_payload(stream=stream)
        )
        assert response.status_code == status
        assert body_marker in response.text
        if mode == "no-usage" and stream:
            assert '"usage"' not in response.text
        if mode == "disconnect":
            assert "[DONE]" not in response.text
            assert 'finish_reason":"stop' not in response.text


def test_tool_calls_stream_round_trip_and_optional_stream_flag() -> None:
    with _server("tool-calling") as base_url, httpx.Client(timeout=3) as client:
        first = client.post(
            f"{base_url}/v1/chat/completions",
            json=_tool_payload(stream=True),
        )
        assert first.status_code == 200
        frames = [frame for frame in first.text.split("\n\n") if frame]
        assert frames[-1] == "data: [DONE]"
        payloads = [json.loads(frame.removeprefix("data: ")) for frame in frames[:-1]]
        tool_delta = payloads[0]["choices"][0]["delta"]["tool_calls"][0]
        assert tool_delta["id"] == "call_modeltop_1"
        assert tool_delta["function"] == {
            "name": "send_notification",
            "arguments": ('{"count":1,"email":"modeltop@example.com"}'),
        }
        assert payloads[1]["choices"][0]["finish_reason"] == "tool_calls"
        assert payloads[-1]["usage"]["total_tokens"] > 0

        follow_up = _tool_payload(
            stream=None,
            messages=[
                {"role": "user", "content": "Send one notification."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_delta["id"],
                            "type": "function",
                            "function": tool_delta["function"],
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"status":"sent"}',
                    "tool_call_id": tool_delta["id"],
                },
            ],
        )
        completed = client.post(
            f"{base_url}/v1/chat/completions",
            json=follow_up,
        )
        assert completed.status_code == 200
        assert completed.json()["choices"][0]["message"]["content"] == (
            "Tool result received."
        )


def test_tool_failure_modes_and_validation_are_deterministic() -> None:
    with _server("tool-malformed-arguments") as base_url:
        malformed = httpx.post(
            f"{base_url}/v1/chat/completions",
            json=_tool_payload(stream=False),
            timeout=3,
        )
        function = malformed.json()["choices"][0]["message"]["tool_calls"][0][
            "function"
        ]
        assert function == {
            "name": "send_notification",
            "arguments": "not-json",
        }

    with _server("tool-refusal") as base_url:
        refusal = httpx.post(
            f"{base_url}/v1/chat/completions",
            json=_tool_payload(stream=True),
            timeout=3,
        )
        assert "I cannot call that tool." in refusal.text
        assert '"tool_calls"' not in refusal.text

    with (
        _server("tool-timeout", timeout_delay=0.1) as base_url,
        pytest.raises(httpx.ReadTimeout),
    ):
        httpx.post(
            f"{base_url}/v1/chat/completions",
            json=_tool_payload(stream=False),
            timeout=0.02,
        )

    with _server() as base_url:
        invalid = _tool_payload(stream=False)
        invalid["tools"] = [{"type": "function", "function": {"name": 7}}]
        response = httpx.post(
            f"{base_url}/v1/chat/completions",
            json=invalid,
            timeout=3,
        )
        assert response.status_code == 400
        assert response.json() == {"error": "Invalid tools"}


def test_core_tool_calling_benchmark_completes_against_mock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        monkeypatch.chdir(tmp_path)

        async def publish(_value: object) -> None:
            return None

        with _server("tool-calling") as base_url:
            now = datetime(2026, 1, 1, tzinfo=UTC)
            benchmark = ToolCallingBenchmark(
                config=ToolCallingBenchmarkConfig(
                    suite="core",
                    request_timeout_seconds=3,
                ),
                server_id="mock",
                server_name="Mock",
                server_base_url=f"{base_url}/v1",
                server_endpoint=base_url.removeprefix("http://"),
                model_id="modeltop/mock-small",
                backend="vLLM",
                backend_hint="vllm",
                api_key=None,
                progress_callback=publish,
                progress_interval_seconds=0,
            )
            result = await benchmark.run(
                BenchmarkContext(
                    benchmark_id="mock-tool-calling",
                    started_at=now,
                    monotonic_clock=time.monotonic,
                    utc_now=lambda: now,
                    read_hardware_snapshot=lambda: None,
                )
            )

        assert result.status is ToolCallingBenchmarkStatus.COMPLETED
        assert result.error_code is None
        assert result.attempted_count == result.gradable_count == 15
        assert result.excluded_count == 0
        assert len(result.scenarios) == 15
        assert result.final_score == 7
        assert not (tmp_path / "data").exists()
        assert not (tmp_path / "runs").exists()

    asyncio.run(scenario())


def test_slow_modes_are_ordered_seeded_and_bounded() -> None:
    with (
        _server("slow-first", first_token_delay=0.05, output_chunks=8) as base_url,
        httpx.Client(timeout=3) as client,
    ):
        started = time.perf_counter()
        with client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json=_payload(stream=True, seed=42, max_tokens=3),
        ) as response:
            lines = response.iter_lines()
            first = next(line for line in lines if line)
            elapsed = time.perf_counter() - started
            remaining = [line for line in lines if line]
        assert elapsed >= 0.04
        frames = [first, *remaining]
        assert frames[-1] == "data: [DONE]"
        payloads = [json.loads(frame.removeprefix("data: ")) for frame in frames[:-1]]
        content_frames = [
            choice["delta"]["content"]
            for payload in payloads
            for choice in payload.get("choices", [])
            if "content" in choice["delta"]
        ]
        assert len(content_frames) == 3
        assert payloads[-1]["usage"]["completion_tokens"] > 0

        repeated = client.post(
            f"{base_url}/v1/chat/completions",
            json=_payload(stream=False, seed=42, max_tokens=3),
        ).json()
        assert repeated["choices"][0]["message"]["content"] == "".join(content_frames)

    with _server("slow-decode", chunk_delay=0.01, output_chunks=4) as base_url:
        response = httpx.post(
            f"{base_url}/v1/chat/completions",
            json=_payload(stream=True, seed=7),
            timeout=3,
        )
        assert response.text.count('"content"') == 4
        assert 'finish_reason":"stop"' in response.text
        assert response.text.rstrip().endswith("data: [DONE]")

    with _server(
        "slow",
        first_token_delay=0.03,
        chunk_delay=0.01,
        output_chunks=2,
    ) as base_url:
        started = time.perf_counter()
        response = httpx.post(
            f"{base_url}/v1/chat/completions",
            json=_payload(stream=True, seed=7),
            timeout=3,
        )
        assert time.perf_counter() - started >= 0.04
        assert response.text.count('"content"') == 2


def test_error_second_fails_once_then_recovers() -> None:
    with _server("error-second") as base_url, httpx.Client(timeout=3) as client:
        statuses = [
            client.post(
                f"{base_url}/v1/chat/completions",
                json=_payload(stream=False),
            ).status_code
            for _ in range(3)
        ]
    assert statuses == [200, 500, 200]


def test_debug_counters_track_parallel_requests_and_model_polls() -> None:
    with _server(
        "slow",
        first_token_delay=0.4,
        output_chunks=1,
    ) as base_url:
        url = f"{base_url}/v1/chat/completions"
        with httpx.Client(timeout=3) as client:
            assert client.get(f"{base_url}/v1/models").status_code == 200
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [
                    pool.submit(
                        httpx.post,
                        url,
                        json=_payload(stream=True, seed=index),
                        timeout=3,
                    )
                    for index in range(4)
                ]
                deadline = time.perf_counter() + 2
                counts = _debug(client, base_url)
                while counts["active_requests"] < 4:
                    assert time.perf_counter() < deadline
                    time.sleep(0.01)
                    counts = _debug(client, base_url)
                assert counts == {
                    "active_requests": 4,
                    "peak_active_requests": 4,
                    "total_chat_requests": 4,
                    "models_requests": 1,
                }
                responses = [future.result(timeout=3) for future in futures]
            assert all(response.status_code == 200 for response in responses)
            assert _debug(client, base_url) == {
                "active_requests": 0,
                "peak_active_requests": 4,
                "total_chat_requests": 4,
                "models_requests": 1,
            }


def test_rate_limit_and_periodic_failures_are_exact() -> None:
    with _server("rate-limit") as base_url, httpx.Client(timeout=3) as client:
        statuses = [
            client.post(
                f"{base_url}/v1/chat/completions",
                json=_payload(stream=False),
            ).status_code
            for _ in range(3)
        ]
        assert statuses == [429, 429, 429]
        assert _debug(client, base_url)["total_chat_requests"] == 3

    with (
        _server(
            "fail-every-n",
            failure_every_n=3,
        ) as base_url,
        httpx.Client(timeout=3) as client,
    ):
        statuses = [
            client.post(
                f"{base_url}/v1/chat/completions",
                json=_payload(stream=False),
            ).status_code
            for _ in range(7)
        ]
        assert statuses == [200, 200, 500, 200, 200, 500, 200]
        counts = _debug(client, base_url)
        assert counts["active_requests"] == 0
        assert counts["total_chat_requests"] == 7


def test_periodic_timeout_delays_headers_and_releases_cancelled_clients() -> None:
    with (
        _server(
            "timeout-every-n",
            timeout_every_n=2,
            timeout_delay=0.08,
        ) as base_url,
        httpx.Client(timeout=3) as client,
    ):
        durations: list[float] = []
        for _ in range(3):
            started = time.perf_counter()
            response = client.post(
                f"{base_url}/v1/chat/completions",
                json=_payload(stream=False),
            )
            durations.append(time.perf_counter() - started)
            assert response.status_code == 200
        assert durations[1] >= 0.07
        assert _debug(client, base_url)["total_chat_requests"] == 3

    with _server(
        "timeout-every-n",
        timeout_every_n=1,
        timeout_delay=0.12,
    ) as base_url:
        with pytest.raises(httpx.ReadTimeout):
            httpx.post(
                f"{base_url}/v1/chat/completions",
                json=_payload(stream=False),
                timeout=0.02,
            )
        with httpx.Client(timeout=1) as client:
            deadline = time.perf_counter() + 1
            counts = _debug(client, base_url)
            while counts["active_requests"]:
                assert time.perf_counter() < deadline
                time.sleep(0.01)
                counts = _debug(client, base_url)
            assert counts["peak_active_requests"] == 1
            assert counts["total_chat_requests"] == 1


def test_periodic_disconnect_affects_only_exact_multiples() -> None:
    with (
        _server(
            "disconnect-every-n",
            disconnect_every_n=2,
        ) as base_url,
        httpx.Client(timeout=3) as client,
    ):
        bodies = [
            client.post(
                f"{base_url}/v1/chat/completions",
                json=_payload(stream=True),
            ).text
            for _ in range(4)
        ]
        assert ["[DONE]" in body for body in bodies] == [True, False, True, False]
        assert bodies[1].count('"content"') == 1
        assert bodies[3].count('"content"') == 1
        counts = _debug(client, base_url)
        assert counts["active_requests"] == 0
        assert counts["total_chat_requests"] == 4


def test_variable_and_concurrency_degradation_timing_is_deterministic() -> None:
    handler = object.__new__(ModelsRequestHandler)
    handler.chat_mode = "variable"
    variable_delays = [handler._stream_delays(number, 1) for number in range(1, 7)]
    assert len(set(variable_delays[:3])) == 3
    assert variable_delays[:3] == variable_delays[3:]

    handler.output_chunk_count = 8
    first_output = handler._slow_pieces(42, 8, request_number=2)
    assert first_output == handler._slow_pieces(42, 8, request_number=2)

    handler.chat_mode = "concurrency-degradation"
    handler.chunk_delay_seconds = 0.02
    expected = [
        (1, (0.10, 0.02)),
        (2, (0.18, 0.02)),
        (4, (0.18, 0.04)),
        (5, (0.42, 0.05)),
        (8, (0.42, 0.08)),
        (9, (0.90, 0.09)),
        (16, (0.90, 0.16)),
        (17, (1.50, 0.17)),
    ]
    for active, delays in expected:
        assert handler._stream_delays(1, active) == pytest.approx(delays)


def test_request_logging_excludes_payloads_headers_and_content(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with _server() as base_url:
        response = httpx.post(
            f"{base_url}/v1/chat/completions",
            json=_payload(stream=False),
            headers={"Authorization": "Bearer unique-secret"},
            timeout=3,
        )
        assert response.status_code == 200
    output = capsys.readouterr().out
    assert output.splitlines() == ["POST /v1/chat/completions 200 active=1 peak=1"]
    assert "private prompt" not in output
    assert "unique-secret" not in output


def test_r0b0bench_canary_mode_runs_pinned_upstream_over_real_socket(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        with _server("r0b0bench-canary") as base_url:
            runner = SubprocessR0b0benchRunner()
            request = R0b0benchRunnerRequest(
                benchmark_id="r0b0bench-20260804T120000Z-deadbeef",
                config=R0b0benchBenchmarkConfig(
                    profile="core-subset",
                    selected_lanes=("canary",),
                    request_timeout_seconds=10,
                ),
                base_url=f"{base_url}/v1",
                model_id="modeltop/mock-small",
                output_root=tmp_path / "runs",
                api_key="EMPTY",
            )
            started: list[R0b0benchLaneId] = []
            finished: list[R0b0benchLaneResult] = []

            async def on_started(
                lane_id: R0b0benchLaneId,
                _index: int,
                _total: int,
            ) -> None:
                started.append(lane_id)

            async def on_finished(
                row: R0b0benchLaneResult,
                _index: int,
                _total: int,
            ) -> None:
                finished.append(row)

            prepared = await runner.prepare(request)
            report = await runner.run(prepared, on_started, on_finished)

        assert started == ["canary"]
        assert finished == list(report.lanes)
        assert report.schema_version == 2
        assert report.invalid_for_publish
        assert report.infra_errors_total == 0
        assert len(report.lanes) == 1
        lane = report.lanes[0]
        assert lane.status is R0b0benchLaneStatus.PASS
        metrics = {metric.name: metric.value for metric in lane.metrics}
        assert metrics == {"passed": True, "cases": 5}
        assert (report.run_directory / "report.json").is_file()

    asyncio.run(scenario())


def test_environment_selects_mode_and_all_timing_knobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["mock_server.py"])
    monkeypatch.setenv("MODELTOP_MOCK_MODE", "slow-decode")
    monkeypatch.setenv("MODELTOP_MOCK_FIRST_TOKEN_DELAY_SECONDS", "1.5")
    monkeypatch.setenv("MODELTOP_MOCK_CHUNK_DELAY_SECONDS", "0.02")
    monkeypatch.setenv("MODELTOP_MOCK_OUTPUT_CHUNK_COUNT", "17")
    monkeypatch.setenv("MODELTOP_MOCK_FAILURE_EVERY_N", "5")
    monkeypatch.setenv("MODELTOP_MOCK_TIMEOUT_EVERY_N", "6")
    monkeypatch.setenv("MODELTOP_MOCK_DISCONNECT_EVERY_N", "7")
    monkeypatch.setenv("MODELTOP_MOCK_TIMEOUT_DELAY_SECONDS", "0.4")
    arguments = _arguments()
    assert arguments.chat_mode == "slow-decode"
    assert arguments.first_token_delay_seconds == 1.5
    assert arguments.chunk_delay_seconds == 0.02
    assert arguments.output_chunk_count == 17
    assert arguments.failure_every_n == 5
    assert arguments.timeout_every_n == 6
    assert arguments.disconnect_every_n == 7
    assert arguments.timeout_delay_seconds == 0.4


def test_context_limit_and_silent_truncation_modes() -> None:
    long_prompt = (
        "MODELTOP_RETRIEVAL_KEY: amber-cloud-1000\n"
        + "x" * 600
        + "\nMODELTOP_RETRIEVAL_KEY: cobalt-river-2000\n"
        + "y" * 600
        + "\nMODELTOP_RETRIEVAL_KEY: jade-lake-3000"
    )
    with _server("context-limit", context_limit=128) as base_url:
        response = httpx.post(
            f"{base_url}/v1/chat/completions",
            json=_custom_payload(long_prompt, stream=False),
            timeout=3,
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "context_length_exceeded"

    with _server("silent-left-truncation", context_limit=128) as base_url:
        response = httpx.post(
            f"{base_url}/v1/chat/completions",
            json=_custom_payload(long_prompt, stream=False),
            timeout=3,
        )
        content = response.json()["choices"][0]["message"]["content"]
        assert "amber-cloud-1000" not in content
        assert "jade-lake-3000" in content

    with _server("silent-right-truncation", context_limit=128) as base_url:
        response = httpx.post(
            f"{base_url}/v1/chat/completions",
            json=_custom_payload(long_prompt, stream=False),
            timeout=3,
        )
        content = response.json()["choices"][0]["message"]["content"]
        assert "amber-cloud-1000" in content
        assert "jade-lake-3000" not in content


def test_context_timing_cache_and_malformed_usage_modes() -> None:
    handler = object.__new__(ModelsRequestHandler)
    handler.chat_mode = "slow-prefill"
    handler.first_token_delay_seconds = 1.0
    assert (
        handler._stream_delays(1, 1, prompt_tokens=8192, prompt_cache_hit=False)[0]
        == 0.25
    )
    assert (
        handler._stream_delays(1, 1, prompt_tokens=32768, prompt_cache_hit=False)[0]
        == 1.0
    )
    handler.chat_mode = "cache-second-request"
    handler.first_token_delay_seconds = 0.01
    assert (
        handler._stream_delays(1, 1, prompt_tokens=1024, prompt_cache_hit=False)[0]
        == 0.01
    )
    assert (
        handler._stream_delays(2, 1, prompt_tokens=1024, prompt_cache_hit=True)[0]
        == 0.0
    )
    with _server("malformed-usage") as base_url:
        response = httpx.post(
            f"{base_url}/v1/chat/completions",
            json=_payload(stream=False),
            timeout=3,
        )
        assert response.json()["usage"]["prompt_tokens"] == "invalid"


def test_canonical_mode_and_context_limit_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["mock_server.py", "--mode", "context-limit", "--context-limit", "4096"],
    )
    arguments = _arguments()
    assert arguments.chat_mode == "context-limit"
    assert arguments.context_limit == 4096


def test_r0b0bench_blocking_mode_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["mock_server.py", "--mode", "r0b0bench-blocking"],
    )
    assert _arguments().chat_mode == "r0b0bench-blocking"


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--failure-every-n", "0"),
        ("--timeout-every-n", "1000001"),
        ("--disconnect-every-n", "not-an-integer"),
        ("--timeout-delay-seconds", "61"),
    ],
)
def test_periodic_mode_flags_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    value: str,
) -> None:
    monkeypatch.setattr(sys, "argv", ["mock_server.py", flag, value])
    with pytest.raises(SystemExit):
        _arguments()


def test_drafter_usage_mode_emits_speculative_fields() -> None:
    with _server("drafter-usage") as base_url:
        response = httpx.post(
            f"{base_url}/v1/chat/completions",
            json=_payload(stream=False),
            timeout=3,
        )
    assert response.status_code == 200
    usage = response.json()["usage"]
    draft = usage["draft_tokens"]
    accepted = usage["accepted_tokens"]
    rate = usage["acceptance_rate"]
    assert isinstance(draft, int) and draft >= 1
    assert accepted == max(0, draft - 1)
    assert rate == pytest.approx(accepted / draft)
    assert draft == max(1, usage["completion_tokens"])
