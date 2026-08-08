"""Standard-library development OpenAI-compatible mock server."""

import argparse
import hashlib
import json
import math
import os
import random
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar, Literal, cast
from urllib.parse import urlsplit

_MODELS_RESPONSE: dict[str, object] = {
    "object": "list",
    "data": [
        {
            "id": "modeltop/mock-small",
            "object": "model",
            "created": 0,
            "owned_by": "modeltop",
        },
        {
            "id": "modeltop/mock-large",
            "object": "model",
            "created": 0,
            "owned_by": "modeltop",
        },
    ],
}
_MAX_REQUEST_BYTES = 2 * 1024 * 1024
_VARIABLE_DELAY_TIERS: tuple[tuple[float, float], ...] = (
    (0.03, 0.005),
    (0.08, 0.01),
    (0.15, 0.02),
)


type ChatMode = Literal[
    "normal",
    "no-usage",
    "no-stream",
    "malformed",
    "disconnect",
    "error",
    "slow-first",
    "slow-decode",
    "error-second",
    "slow",
    "variable",
    "rate-limit",
    "fail-every-n",
    "timeout-every-n",
    "disconnect-every-n",
    "concurrency-degradation",
    "context-limit",
    "silent-left-truncation",
    "silent-right-truncation",
    "slow-prefill",
    "cache-second-request",
    "timeout-large-context",
    "malformed-usage",
    "drafter-usage",
    "tool-calling",
    "tool-malformed-arguments",
    "tool-refusal",
    "tool-timeout",
    "r0b0bench-canary",
    "r0b0bench-blocking",
]
_GENERATED_MODES: frozenset[ChatMode] = frozenset(
    {
        "slow-first",
        "slow-decode",
        "slow",
        "variable",
        "concurrency-degradation",
    }
)


class ModelsRequestHandler(BaseHTTPRequestHandler):
    """Serve deterministic models and chat-completions endpoints."""

    protocol_version = "HTTP/1.1"
    chat_mode: ChatMode = "normal"
    stream_delay_seconds: float = 0.08
    first_token_delay_seconds: float = 0.75
    chunk_delay_seconds: float = 0.08
    output_chunk_count: int = 32
    failure_every_n: int = 3
    timeout_every_n: int = 2
    disconnect_every_n: int = 2
    timeout_delay_seconds: float = 5.0
    context_limit_tokens: int = 32768
    prompt_hash_counts: ClassVar[dict[str, int]] = {}

    total_chat_requests = 0
    models_requests = 0
    version_requests = 0
    health_requests = 0
    props_requests = 0
    authenticated_probe_requests = 0
    unauthenticated_probe_requests = 0
    response_format_requests = 0
    active_requests = 0
    peak_active_requests = 0
    chat_request_lock = threading.Lock()
    recorded_paths: ClassVar[list[str]] = []

    def _has_bearer_authorization(self) -> bool:
        header = self.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return False
        return bool(header[7:].strip())

    def do_GET(self) -> None:
        """Return exact read-only endpoints and JSON 404 otherwise."""
        path = urlsplit(self.path).path
        authorized = self._has_bearer_authorization()
        handler = type(self)
        with handler.chat_request_lock:
            handler.recorded_paths.append(path)
        if path == "/v1/models":
            with handler.chat_request_lock:
                handler.models_requests += 1
                if authorized:
                    handler.authenticated_probe_requests += 1
                else:
                    handler.unauthenticated_probe_requests += 1
            self._send_json(200, _MODELS_RESPONSE)
            return
        if path == "/version":
            with handler.chat_request_lock:
                handler.version_requests += 1
                if authorized:
                    handler.authenticated_probe_requests += 1
                else:
                    handler.unauthenticated_probe_requests += 1
            self._send_json(200, {"version": "0.0.0-modeltop-mock"})
            return
        if path == "/health":
            with handler.chat_request_lock:
                handler.health_requests += 1
                if authorized:
                    handler.authenticated_probe_requests += 1
                else:
                    handler.unauthenticated_probe_requests += 1
            # Unauthenticated llama.cpp health plus authenticated LiteLLM shape.
            payload: dict[str, object] = {"status": "ok"}
            if authorized:
                payload["litellm_version"] = "0.0.0-modeltop-mock"
            self._send_json(
                200,
                payload,
                extra_headers=(
                    {"x-litellm-version": "0.0.0-modeltop-mock"} if authorized else None
                ),
            )
            return
        if path == "/props":
            with handler.chat_request_lock:
                handler.props_requests += 1
                if authorized:
                    handler.authenticated_probe_requests += 1
                else:
                    handler.unauthenticated_probe_requests += 1
            self._send_json(
                200,
                {
                    "build_info": "b0-modeltop-mock",
                    "total_slots": 1,
                },
            )
            return
        if path == "/debug/concurrency":
            # Let just-finished handler threads publish their final active count.
            time.sleep(0.01)
            self._send_json(200, self._debug_counts())
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        """Handle exact generic chat completions with bounded request parsing."""
        if urlsplit(self.path).path != "/v1/chat/completions":
            self._send_json(404, {"error": "Not found"})
            return
        payload = self._read_chat_payload()
        if payload is None:
            return

        request_number, admitted_active = self._admit_chat_request()
        try:
            if self.chat_mode == "rate-limit":
                self._send_json(429, {"error": {"message": "mock rate limit"}})
                return
            if (
                self.chat_mode == "error"
                or (self.chat_mode == "error-second" and request_number == 2)
                or (
                    self.chat_mode == "fail-every-n"
                    and self._is_multiple(request_number, self.failure_every_n)
                )
            ):
                self._send_json(500, {"error": {"message": "mock server error"}})
                return
            if self.chat_mode == "timeout-every-n" and self._is_multiple(
                request_number, self.timeout_every_n
            ):
                time.sleep(self.timeout_delay_seconds)
            if self.chat_mode == "tool-timeout" and payload["tools"]:
                time.sleep(self.timeout_delay_seconds)

            stream = cast(bool, payload["stream"])
            if self.chat_mode == "no-stream" and stream:
                self._send_json(
                    400,
                    {"error": {"message": "stream is not supported by this mock mode"}},
                )
                return

            model = cast(str, payload["model"])
            raw_messages = cast(list[dict[str, object]], payload["messages"])
            messages = self._text_messages(raw_messages)
            seed = cast(int | None, payload["seed"])
            max_tokens = cast(int | None, payload["max_tokens"])
            reserve = max_tokens or 0
            prompt_tokens = self._approximate_prompt_tokens(messages)
            total_requested = prompt_tokens + reserve
            if (
                self.chat_mode == "context-limit"
                and total_requested > self.context_limit_tokens
            ):
                self._send_json(
                    400,
                    {
                        "error": {
                            "message": (
                                "Requested context length exceeds maximum context"
                            ),
                            "type": "invalid_request_error",
                            "code": "context_length_exceeded",
                            "param": "messages",
                            "requested_tokens": total_requested,
                            "limit_tokens": self.context_limit_tokens,
                        }
                    },
                )
                return
            if (
                self.chat_mode == "timeout-large-context"
                and total_requested > self.context_limit_tokens
            ):
                time.sleep(self.timeout_delay_seconds)
            if self.chat_mode == "r0b0bench-blocking":
                threading.Event().wait()
            if self._serve_tool_request(
                payload,
                raw_messages,
                messages,
                model=model,
                stream=stream,
                request_number=request_number,
            ):
                return

            retained_messages = self._retained_messages(
                messages, reserve, prompt_tokens
            )
            assignments = self._marker_assignments(retained_messages)
            if assignments:
                marker_values = [value for _, value in assignments]
                response_text = "\n".join(marker_values)
                pieces = [
                    ("" if index == 0 else "\n") + value
                    for index, value in enumerate(marker_values)
                ]
            elif self.chat_mode == "r0b0bench-canary":
                response_text = self._r0b0bench_canary_response(messages)
                pieces = [response_text]
            elif self.chat_mode in {
                "silent-left-truncation",
                "silent-right-truncation",
            } and self._marker_assignments(messages):
                response_text = "retrieval-marker-not-found"
                pieces = [response_text]
            elif self.chat_mode in _GENERATED_MODES:
                pieces = self._slow_pieces(
                    seed,
                    max_tokens,
                    request_number=request_number,
                )
                response_text = "".join(pieces)
            else:
                pieces = None
                response_text = f"Mock response from {model}."
            usage: dict[str, object] = self._usage(messages, response_text)
            if self.chat_mode == "malformed-usage":
                usage["prompt_tokens"] = "invalid"
            elif self.chat_mode == "drafter-usage":
                completion_tokens = cast(int, usage["completion_tokens"])
                draft_tokens = max(1, completion_tokens)
                accepted_tokens = max(0, draft_tokens - 1)
                usage["draft_tokens"] = draft_tokens
                usage["accepted_tokens"] = accepted_tokens
                usage["acceptance_rate"] = accepted_tokens / draft_tokens
            if stream:
                prompt_cache_hit = (
                    self._record_prompt_hash(messages) > 1
                    if self.chat_mode == "cache-second-request"
                    else False
                )
                first_token_delay, chunk_delay = self._stream_delays(
                    request_number,
                    admitted_active,
                    prompt_tokens=prompt_tokens,
                    prompt_cache_hit=prompt_cache_hit,
                )
                disconnect_after_first = self.chat_mode == "disconnect" or (
                    self.chat_mode == "disconnect-every-n"
                    and self._is_multiple(request_number, self.disconnect_every_n)
                )
                self._send_stream(
                    model,
                    response_text,
                    usage,
                    pieces=pieces,
                    first_token_delay_seconds=first_token_delay,
                    chunk_delay_seconds=chunk_delay,
                    disconnect_after_first=disconnect_after_first,
                )
            else:
                self._send_completion(model, response_text, usage)
        finally:
            self._release_chat_request()

    @staticmethod
    def _r0b0bench_canary_response(messages: list[dict[str, str]]) -> str:
        prompt = "\n".join(message["content"] for message in messages)
        if "R0B0BENCH_OK" in prompt:
            return "R0B0BENCH_OK"
        if "17乘以19" in prompt:
            return "323"
        if "keys alpha and beta" in prompt:
            return '{"alpha":2,"beta":3}'
        if "verification code" in prompt:
            return "A9Q7"
        return "R0B0BENCH_OK"

    def _admit_chat_request(self) -> tuple[int, int]:
        handler = type(self)
        with handler.chat_request_lock:
            handler.total_chat_requests += 1
            handler.active_requests += 1
            handler.peak_active_requests = max(
                handler.peak_active_requests,
                handler.active_requests,
            )
            return handler.total_chat_requests, handler.active_requests

    def _release_chat_request(self) -> None:
        handler = type(self)
        with handler.chat_request_lock:
            handler.active_requests -= 1

    def _debug_counts(self) -> dict[str, object]:
        handler = type(self)
        with handler.chat_request_lock:
            return {
                "active_requests": handler.active_requests,
                "peak_active_requests": handler.peak_active_requests,
                "total_chat_requests": handler.total_chat_requests,
                "models_requests": handler.models_requests,
            }

    @staticmethod
    def _approximate_prompt_tokens(messages: list[dict[str, str]]) -> int:
        characters = sum(
            sum(not character.isspace() for character in message["content"])
            for message in messages
        )
        return math.ceil(characters / 4) + 4 * len(messages) + 3

    @staticmethod
    def _marker_assignments(
        messages: list[dict[str, str]],
    ) -> list[tuple[str, str]]:
        pattern = re.compile(
            r"^(MODELTOP_RETRIEVAL_KEY|BEGIN_MARKER|MIDDLE_MARKER|END_MARKER): "
            r"([A-Za-z0-9-]+)$",
            re.MULTILINE,
        )
        return [
            (match.group(1), match.group(2))
            for message in messages
            for match in pattern.finditer(message["content"])
        ]

    def _retained_messages(
        self,
        messages: list[dict[str, str]],
        reserve: int,
        prompt_tokens: int,
    ) -> list[dict[str, str]]:
        if self.chat_mode not in {
            "silent-left-truncation",
            "silent-right-truncation",
        }:
            return messages
        available_tokens = max(0, self.context_limit_tokens - reserve)
        if prompt_tokens <= available_tokens:
            return messages
        wrapper_tokens = 4 * len(messages) + 3
        content_characters = max(0, available_tokens - wrapper_tokens) * 4
        combined = "\n".join(message["content"] for message in messages)
        retained = (
            combined[-content_characters:]
            if self.chat_mode == "silent-left-truncation" and content_characters > 0
            else combined[:content_characters]
        )
        return [{"role": "user", "content": retained}]

    def _record_prompt_hash(self, messages: list[dict[str, str]]) -> int:
        ordered = [(message["role"], message["content"]) for message in messages]
        digest = hashlib.sha256(
            json.dumps(ordered, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        handler = type(self)
        with handler.chat_request_lock:
            count = handler.prompt_hash_counts.get(digest, 0) + 1
            handler.prompt_hash_counts[digest] = count
        return count

    @staticmethod
    def _is_multiple(request_number: int, every_n: int) -> bool:
        return every_n > 0 and request_number % every_n == 0

    def _stream_delays(
        self,
        request_number: int,
        admitted_active: int,
        *,
        prompt_tokens: int = 0,
        prompt_cache_hit: bool = False,
    ) -> tuple[float, float]:
        if self.chat_mode == "slow-prefill":
            return (
                self.first_token_delay_seconds
                * min(prompt_tokens / self.context_limit_tokens, 1.0),
                0.0,
            )
        if self.chat_mode == "cache-second-request":
            return (
                0.0 if prompt_cache_hit else self.first_token_delay_seconds,
                0.0,
            )
        if self.chat_mode == "slow-first":
            return self.first_token_delay_seconds, 0.0
        if self.chat_mode == "slow-decode":
            return 0.0, self.chunk_delay_seconds
        if self.chat_mode == "slow":
            return self.first_token_delay_seconds, self.chunk_delay_seconds
        if self.chat_mode == "variable":
            return _VARIABLE_DELAY_TIERS[
                (request_number - 1) % len(_VARIABLE_DELAY_TIERS)
            ]
        if self.chat_mode == "concurrency-degradation":
            if admitted_active <= 1:
                first_token_delay = 0.10
            elif admitted_active <= 4:
                first_token_delay = 0.18
            elif admitted_active <= 8:
                first_token_delay = 0.42
            elif admitted_active <= 16:
                first_token_delay = 0.90
            else:
                first_token_delay = 1.50
            decode_multiplier = max(1.0, admitted_active / 2)
            return first_token_delay, self.chunk_delay_seconds * decode_multiplier
        return 0.0, self.stream_delay_seconds

    def do_PUT(self) -> None:
        self._method_not_allowed()

    def do_DELETE(self) -> None:
        self._method_not_allowed()

    def do_PATCH(self) -> None:
        self._method_not_allowed()

    def _method_not_allowed(self) -> None:
        self._send_json(405, {"error": "Method not allowed"})

    def _read_chat_payload(self) -> dict[str, object] | None:
        length_text = self.headers.get("Content-Length")
        try:
            length = int(length_text or "")
        except ValueError:
            self._send_json(400, {"error": "Invalid content length"})
            return None
        if length < 1 or length > _MAX_REQUEST_BYTES:
            self._send_json(
                413 if length > _MAX_REQUEST_BYTES else 400, {"error": "Invalid body"}
            )
            return None
        body = self.rfile.read(length)
        try:
            value: object = json.loads(body)
        except (UnicodeDecodeError, ValueError):
            self._send_json(400, {"error": "Malformed JSON"})
            return None
        if not isinstance(value, dict):
            self._send_json(400, {"error": "Request root must be an object"})
            return None
        payload = cast(dict[object, object], value)
        model = payload.get("model")
        messages_value = payload.get("messages")
        stream = payload.get("stream", False)
        seed = payload.get("seed")
        max_tokens = payload.get("max_tokens")
        tools_value = payload.get("tools")
        tool_choice = payload.get("tool_choice", "auto")
        parallel_tool_calls = payload.get("parallel_tool_calls")
        if not isinstance(model, str) or not model:
            self._send_json(400, {"error": "Invalid model"})
            return None
        if not isinstance(stream, bool):
            self._send_json(400, {"error": "Invalid stream value"})
            return None
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            self._send_json(400, {"error": "Invalid seed"})
            return None
        if max_tokens is not None and (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or not 1 <= max_tokens <= 32768
        ):
            self._send_json(400, {"error": "Invalid max_tokens"})
            return None
        if tools_value is not None and not self._valid_tools(tools_value):
            self._send_json(400, {"error": "Invalid tools"})
            return None
        if not isinstance(tool_choice, (str, dict)):
            self._send_json(400, {"error": "Invalid tool_choice"})
            return None
        if parallel_tool_calls is not None and not isinstance(
            parallel_tool_calls, bool
        ):
            self._send_json(400, {"error": "Invalid parallel_tool_calls"})
            return None
        if not isinstance(messages_value, list):
            self._send_json(400, {"error": "Invalid messages"})
            return None
        messages: list[dict[str, object]] = []
        for item in cast(list[object], messages_value):
            if not isinstance(item, dict):
                self._send_json(400, {"error": "Invalid message"})
                return None
            message = cast(dict[object, object], item)
            role = message.get("role")
            content = message.get("content")
            if (
                not isinstance(role, str)
                or (content is not None and not isinstance(content, str))
                or (
                    message.get("tool_call_id") is not None
                    and not isinstance(message.get("tool_call_id"), str)
                )
                or (
                    message.get("tool_calls") is not None
                    and not isinstance(message.get("tool_calls"), list)
                )
            ):
                self._send_json(400, {"error": "Invalid message"})
                return None
            messages.append({str(key): value for key, value in message.items()})
        return {
            "model": model,
            "messages": messages,
            "stream": stream,
            "seed": seed,
            "max_tokens": max_tokens,
            "tools": tools_value,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
        }

    @staticmethod
    def _valid_tools(value: object) -> bool:
        if not isinstance(value, list):
            return False
        for item in cast(list[object], value):
            if not isinstance(item, dict):
                return False
            tool = cast(dict[object, object], item)
            function = tool.get("function")
            if tool.get("type") != "function" or not isinstance(function, dict):
                return False
            function_fields = cast(dict[object, object], function)
            if not isinstance(function_fields.get("name"), str):
                return False
            parameters = function_fields.get("parameters")
            if parameters is not None and not isinstance(parameters, dict):
                return False
        return True

    @staticmethod
    def _text_messages(
        messages: list[dict[str, object]],
    ) -> list[dict[str, str]]:
        return [
            {
                "role": cast(str, message["role"]),
                "content": (
                    content
                    if isinstance((content := message.get("content")), str)
                    else ""
                ),
            }
            for message in messages
        ]

    def _serve_tool_request(
        self,
        payload: dict[str, object],
        raw_messages: list[dict[str, object]],
        messages: list[dict[str, str]],
        *,
        model: str,
        stream: bool,
        request_number: int,
    ) -> bool:
        tools_value = payload["tools"]
        if not isinstance(tools_value, list) or not tools_value:
            return False
        tools = cast(list[object], tools_value)
        tool_choice = payload["tool_choice"]
        if tool_choice == "none" or self.chat_mode == "tool-refusal":
            response_text = (
                "I cannot call that tool."
                if self.chat_mode == "tool-refusal"
                else "No tool call requested."
            )
            usage = self._usage(messages, response_text)
            if stream:
                self._send_stream(
                    model,
                    response_text,
                    usage,
                    pieces=[response_text],
                    first_token_delay_seconds=0.0,
                    chunk_delay_seconds=0.0,
                    disconnect_after_first=False,
                )
            else:
                self._send_completion(model, response_text, usage)
            return True
        if raw_messages and raw_messages[-1].get("role") == "tool":
            response_text = "Tool result received."
            usage = self._usage(messages, response_text)
            if stream:
                self._send_stream(
                    model,
                    response_text,
                    usage,
                    pieces=[response_text],
                    first_token_delay_seconds=0.0,
                    chunk_delay_seconds=0.0,
                    disconnect_after_first=False,
                )
            else:
                self._send_completion(model, response_text, usage)
            return True

        selected = cast(dict[object, object], tools[0])
        requested_name = self._requested_tool_name(tool_choice)
        if requested_name is not None:
            for candidate_value in tools:
                candidate = cast(dict[object, object], candidate_value)
                function = cast(dict[object, object], candidate["function"])
                if function["name"] == requested_name:
                    selected = candidate
                    break
        function = cast(dict[object, object], selected["function"])
        name = cast(str, function["name"])
        parameters = function.get("parameters", {})
        arguments = json.dumps(
            self._schema_value(parameters),
            separators=(",", ":"),
            sort_keys=True,
        )
        if self.chat_mode == "tool-malformed-arguments":
            arguments = "not-json"
        tool_call_id = f"call_modeltop_{request_number}"
        usage = self._usage(messages, f"{name} {arguments}")
        if stream:
            self._send_tool_call_stream(
                model,
                tool_call_id,
                name,
                arguments,
                usage,
            )
        else:
            self._send_tool_call_completion(
                model,
                tool_call_id,
                name,
                arguments,
                usage,
            )
        return True

    @staticmethod
    def _requested_tool_name(tool_choice: object) -> str | None:
        if not isinstance(tool_choice, dict):
            return None
        function = cast(dict[object, object], tool_choice).get("function")
        if not isinstance(function, dict):
            return None
        name = cast(dict[object, object], function).get("name")
        return name if isinstance(name, str) else None

    @classmethod
    def _schema_value(cls, schema_value: object, *, depth: int = 0) -> object:
        if not isinstance(schema_value, dict) or depth >= 8:
            return {}
        schema = cast(dict[object, object], schema_value)
        if "default" in schema:
            return schema["default"]
        enum = schema.get("enum")
        if isinstance(enum, list) and enum:
            return cast(list[object], enum)[0]
        alternatives = schema.get("oneOf") or schema.get("anyOf")
        if isinstance(alternatives, list) and alternatives:
            return cls._schema_value(
                cast(list[object], alternatives)[0],
                depth=depth + 1,
            )
        value_type = schema.get("type")
        if isinstance(value_type, list):
            value_type = next(
                (
                    candidate
                    for candidate in cast(list[object], value_type)
                    if isinstance(candidate, str) and candidate != "null"
                ),
                "null",
            )
        if value_type == "object" or isinstance(schema.get("properties"), dict):
            properties_value = schema.get("properties", {})
            if not isinstance(properties_value, dict):
                return {}
            properties = cast(dict[object, object], properties_value)
            return {
                str(key): cls._schema_value(value, depth=depth + 1)
                for key, value in properties.items()
            }
        if value_type == "array":
            return [cls._schema_value(schema.get("items", {}), depth=depth + 1)]
        if value_type == "integer":
            minimum = schema.get("minimum", 1)
            return minimum if isinstance(minimum, int) else 1
        if value_type == "number":
            minimum = schema.get("minimum", 1.0)
            return float(minimum) if isinstance(minimum, (int, float)) else 1.0
        if value_type == "boolean":
            return True
        if value_type == "null":
            return None
        string_format = schema.get("format")
        if string_format == "email":
            return "modeltop@example.com"
        if string_format == "date":
            return "2026-01-01"
        if string_format == "date-time":
            return "2026-01-01T00:00:00Z"
        return "modeltop"

    def _send_tool_call_completion(
        self,
        model: str,
        tool_call_id: str,
        name: str,
        arguments: str,
        usage: dict[str, object],
    ) -> None:
        self._send_json(
            200,
            {
                "id": "chatcmpl-modeltop-mock",
                "object": "chat.completion",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": arguments,
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": usage,
            },
        )

    def _send_tool_call_stream(
        self,
        model: str,
        tool_call_id: str,
        name: str,
        arguments: str,
        usage: dict[str, object],
    ) -> None:
        self.close_connection = True
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self._write_data(
                {
                    "id": "chatcmpl-modeltop-mock",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": tool_call_id,
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": arguments,
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            )
            self._write_data(
                {
                    "id": "chatcmpl-modeltop-mock",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "tool_calls",
                        }
                    ],
                }
            )
            self._write_data({"choices": [], "usage": usage})
            self._write_frame(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass
        finally:
            self._log_status(200)

    def _usage(
        self, messages: list[dict[str, str]], response: str
    ) -> dict[str, object]:
        prompt_tokens = self._approximate_prompt_tokens(messages)
        completion_characters = sum(not character.isspace() for character in response)
        completion_tokens = max(1, math.ceil(completion_characters / 4))
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _slow_pieces(
        self,
        seed: int | None,
        max_tokens: int | None,
        *,
        request_number: int = 0,
    ) -> list[str]:
        vocabulary = (
            "adaptive",
            "benchmark",
            "compute",
            "decode",
            "efficient",
            "generation",
            "latency",
            "model",
            "output",
            "request",
            "stream",
            "throughput",
            "token",
        )
        count = min(self.output_chunk_count, max_tokens or self.output_chunk_count)
        configured_seed = 0 if seed is None else seed
        if self.chat_mode in {"slow-first", "slow-decode"}:
            seed_material: int | str = configured_seed
        else:
            seed_material = f"{configured_seed}:{request_number}"
        generator = random.Random(seed_material)
        return [
            ("" if index == 0 else " ") + generator.choice(vocabulary)
            for index in range(count)
        ]

    def _send_completion(
        self, model: str, response_text: str, usage: dict[str, object]
    ) -> None:
        if self.chat_mode == "malformed":
            self._send_bytes(200, b"{", content_type="application/json")
            return
        payload: dict[str, object] = {
            "id": "chatcmpl-modeltop-mock",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
        }
        if self.chat_mode != "no-usage":
            payload["usage"] = usage
        self._send_json(200, payload)

    def _send_stream(
        self,
        model: str,
        response_text: str,
        usage: dict[str, object],
        *,
        pieces: list[str] | None,
        first_token_delay_seconds: float,
        chunk_delay_seconds: float,
        disconnect_after_first: bool,
    ) -> None:
        self.close_connection = True
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            if self.chat_mode == "malformed":
                self._write_frame(b"data: {\n\n")
                return
            output_pieces = pieces or ["Mock response", f" from {model}", "."]
            if first_token_delay_seconds:
                time.sleep(first_token_delay_seconds)
            for piece in output_pieces:
                self._write_data(
                    {
                        "id": "chatcmpl-modeltop-mock",
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": piece},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                if disconnect_after_first:
                    return
                if chunk_delay_seconds:
                    time.sleep(chunk_delay_seconds)
            self._write_data(
                {
                    "id": "chatcmpl-modeltop-mock",
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
            if self.chat_mode != "no-usage":
                self._write_data({"choices": [], "usage": usage})
            self._write_frame(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass
        finally:
            self._log_status(200)

    def _write_data(self, payload: dict[str, object]) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._write_frame(b"data: " + data + b"\n\n")

    def _write_frame(self, frame: bytes) -> None:
        self.wfile.write(frame)
        self.wfile.flush()

    def _send_json(
        self,
        status: int,
        payload: dict[str, object],
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._send_bytes(
            status,
            body,
            content_type="application/json",
            extra_headers=extra_headers,
        )

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        *,
        content_type: str,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.close_connection = True
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            for name, value in (extra_headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass
        finally:
            self._log_status(status)

    def _log_status(self, status: int) -> None:
        handler = type(self)
        with handler.chat_request_lock:
            active = handler.active_requests
            peak = handler.peak_active_requests
        print(
            f"{self.command} {urlsplit(self.path).path} {status} "
            f"active={active} peak={peak}",
            flush=True,
        )

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress BaseHTTPRequestHandler's duplicate request-line logging."""


def _nonnegative_seconds(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not 0 <= parsed <= 60:
        raise argparse.ArgumentTypeError("must be between 0 and 60 seconds")
    return parsed


def _bounded_chunk_count(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if not 1 <= parsed <= 4096:
        raise argparse.ArgumentTypeError("must be between 1 and 4096")
    return parsed


def _bounded_frequency(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if not 1 <= parsed <= 1_000_000:
        raise argparse.ArgumentTypeError("must be between 1 and 1000000")
    return parsed


def _bounded_context_limit(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if not 1 <= parsed <= 262144:
        raise argparse.ArgumentTypeError("must be between 1 and 262144")
    return parsed


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a mock OpenAI-compatible API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--stream-delay-seconds",
        type=_nonnegative_seconds,
        default=os.environ.get("MODELTOP_MOCK_STREAM_DELAY_SECONDS", "0.08"),
    )
    parser.add_argument(
        "--first-token-delay-seconds",
        type=_nonnegative_seconds,
        default=os.environ.get("MODELTOP_MOCK_FIRST_TOKEN_DELAY_SECONDS", "0.75"),
    )
    parser.add_argument(
        "--chunk-delay-seconds",
        type=_nonnegative_seconds,
        default=os.environ.get("MODELTOP_MOCK_CHUNK_DELAY_SECONDS", "0.08"),
    )
    parser.add_argument(
        "--output-chunk-count",
        type=_bounded_chunk_count,
        default=os.environ.get("MODELTOP_MOCK_OUTPUT_CHUNK_COUNT", "32"),
    )
    parser.add_argument(
        "--failure-every-n",
        type=_bounded_frequency,
        default=os.environ.get("MODELTOP_MOCK_FAILURE_EVERY_N", "3"),
    )
    parser.add_argument(
        "--timeout-every-n",
        type=_bounded_frequency,
        default=os.environ.get("MODELTOP_MOCK_TIMEOUT_EVERY_N", "2"),
    )
    parser.add_argument(
        "--disconnect-every-n",
        type=_bounded_frequency,
        default=os.environ.get("MODELTOP_MOCK_DISCONNECT_EVERY_N", "2"),
    )
    parser.add_argument(
        "--timeout-delay-seconds",
        type=_nonnegative_seconds,
        default=os.environ.get("MODELTOP_MOCK_TIMEOUT_DELAY_SECONDS", "5"),
    )
    parser.add_argument(
        "--context-limit",
        type=_bounded_context_limit,
        default=os.environ.get("MODELTOP_MOCK_CONTEXT_LIMIT", "32768"),
    )
    parser.add_argument(
        "--mode",
        "--chat-mode",
        dest="chat_mode",
        choices=(
            "normal",
            "no-usage",
            "no-stream",
            "malformed",
            "disconnect",
            "error",
            "slow-first",
            "slow-decode",
            "error-second",
            "slow",
            "variable",
            "rate-limit",
            "fail-every-n",
            "timeout-every-n",
            "disconnect-every-n",
            "concurrency-degradation",
            "context-limit",
            "silent-left-truncation",
            "silent-right-truncation",
            "slow-prefill",
            "cache-second-request",
            "timeout-large-context",
            "malformed-usage",
            "drafter-usage",
            "tool-calling",
            "tool-malformed-arguments",
            "tool-refusal",
            "tool-timeout",
            "r0b0bench-canary",
            "r0b0bench-blocking",
        ),
        default=os.environ.get("MODELTOP_MOCK_MODE", "normal"),
    )
    return parser.parse_args()


def main() -> None:
    """Run until interrupted and always close the listening socket."""
    arguments = _arguments()
    ModelsRequestHandler.stream_delay_seconds = arguments.stream_delay_seconds
    ModelsRequestHandler.chat_mode = arguments.chat_mode
    ModelsRequestHandler.first_token_delay_seconds = arguments.first_token_delay_seconds
    ModelsRequestHandler.chunk_delay_seconds = arguments.chunk_delay_seconds
    ModelsRequestHandler.output_chunk_count = arguments.output_chunk_count
    ModelsRequestHandler.failure_every_n = arguments.failure_every_n
    ModelsRequestHandler.timeout_every_n = arguments.timeout_every_n
    ModelsRequestHandler.disconnect_every_n = arguments.disconnect_every_n
    ModelsRequestHandler.timeout_delay_seconds = arguments.timeout_delay_seconds
    ModelsRequestHandler.context_limit_tokens = arguments.context_limit
    with ModelsRequestHandler.chat_request_lock:
        ModelsRequestHandler.total_chat_requests = 0
        ModelsRequestHandler.models_requests = 0
        ModelsRequestHandler.active_requests = 0
        ModelsRequestHandler.peak_active_requests = 0
        ModelsRequestHandler.prompt_hash_counts = {}
    server = ThreadingHTTPServer(
        (arguments.host, arguments.port),
        ModelsRequestHandler,
    )
    print(
        f"Mock OpenAI-compatible server listening on "
        f"http://{arguments.host}:{arguments.port}/v1",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping mock server", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
