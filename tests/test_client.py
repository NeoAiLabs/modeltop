"""Generic OpenAI-compatible API client tests."""

import asyncio
import errno
import logging
import socket
from collections.abc import Callable

import httpx
import pytest

from modeltop.api.client import OpenAICompatibleClient, normalize_base_url
from modeltop.api.errors import (
    APIClientError,
    AuthenticationError,
    HTTPResponseError,
    ModelsEndpointNotFoundError,
    ProtocolError,
    RequestTimeoutError,
    ServerConnectionError,
)


def test_normalize_base_url() -> None:
    """API roots preserve terminal v1 and gain it otherwise."""
    assert normalize_base_url(" http://server:8000/ ") == "http://server:8000/v1"
    assert normalize_base_url("http://server/root/v1/") == "http://server/root/v1"
    assert normalize_base_url("http://server/root") == "http://server/root/v1"


def test_models_request_headers_latency_and_unknown_fields() -> None:
    """The client sends one exact request and preserves unknown response fields."""

    async def scenario() -> None:
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["method"] = request.method
            seen["url"] = str(request.url)
            seen["accept"] = request.headers["Accept"]
            seen["authorization"] = request.headers["Authorization"]
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"id": "full/model", "unknown": {"nested": True}}],
                },
            )

        clock_values = iter((10.0, 10.125))
        client = OpenAICompatibleClient(
            "http://server/root/v1",
            " EMPTY ",
            3,
            transport=httpx.MockTransport(handler),
            clock=lambda: next(clock_values),
        )
        try:
            result = await client.list_models()
        finally:
            await client.aclose()

        assert seen == {
            "method": "GET",
            "url": "http://server/root/v1/models",
            "accept": "application/json",
            "authorization": "Bearer EMPTY",
        }
        assert result.status_code == 200
        assert result.latency_ms == 125.0
        assert result.data[0]["unknown"] == {"nested": True}

    asyncio.run(scenario())


@pytest.mark.parametrize("key", [None, "", "   "])
def test_blank_api_key_omits_authorization(key: str | None) -> None:
    """Missing and blank keys do not produce an Authorization header."""

    async def scenario() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "Authorization" not in request.headers
            return httpx.Response(200, json={"data": []})

        client = OpenAICompatibleClient(
            "http://server",
            key,
            3,
            transport=httpx.MockTransport(handler),
        )
        try:
            await client.list_models()
        finally:
            await client.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("response", "detail"),
    [
        (httpx.Response(200, content=b"{"), "malformed JSON"),
        (httpx.Response(200, json=[]), "root is not an object"),
        (httpx.Response(200, json={}), "data is not a list"),
        (httpx.Response(200, json={"data": None}), "data is not a list"),
        (httpx.Response(200, json={"data": {}}), "data is not a list"),
        (
            httpx.Response(200, json={"data": [{"id": "valid"}, 1]}),
            "non-object item",
        ),
    ],
)
def test_malformed_response_is_protocol_error(
    response: httpx.Response,
    detail: str,
) -> None:
    """Malformed JSON, envelopes, and items fail as readable protocol errors."""

    async def scenario() -> None:
        client = OpenAICompatibleClient(
            "http://server",
            None,
            3,
            transport=httpx.MockTransport(lambda request: response),
        )
        try:
            with pytest.raises(ProtocolError) as caught:
                await client.list_models()
        finally:
            await client.aclose()
        assert str(caught.value) == "Invalid response from server"
        assert detail in caught.value.detail
        assert caught.value.status_code is None

    asyncio.run(scenario())


def test_remote_protocol_failure_is_protocol_error() -> None:
    """Transport-level protocol failures do not become generic connection errors."""

    async def scenario() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.RemoteProtocolError("broken", request=request)

        client = OpenAICompatibleClient(
            "http://server", None, 3, transport=httpx.MockTransport(handler)
        )
        try:
            with pytest.raises(
                ProtocolError, match="Invalid response from server"
            ) as caught:
                await client.list_models()
        finally:
            await client.aclose()
        assert caught.value.status_code is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("status", "error_type", "message"),
    [
        (302, HTTPResponseError, "Server returned HTTP 302"),
        (401, AuthenticationError, "Authentication failed"),
        (403, AuthenticationError, "Authentication failed"),
        (404, ModelsEndpointNotFoundError, "Models endpoint not found"),
        (500, HTTPResponseError, "Server error (HTTP 500)"),
        (418, HTTPResponseError, "Server returned HTTP 418"),
    ],
)
def test_non_success_status_mapping(
    status: int,
    error_type: type[APIClientError],
    message: str,
) -> None:
    """Redirect, authentication, endpoint, and status failures map exactly."""

    async def scenario() -> None:
        client = OpenAICompatibleClient(
            "http://server",
            None,
            3,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(status, content=b"private body")
            ),
        )
        try:
            with pytest.raises(error_type) as caught:
                await client.list_models()
        finally:
            await client.aclose()
        assert str(caught.value) == message
        assert "private body" not in caught.value.detail
        assert caught.value.status_code == status

    asyncio.run(scenario())


def _raising_handler(
    factory: Callable[[httpx.Request], httpx.RequestError],
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        raise factory(request)

    return handler


def _read_timeout(request: httpx.Request) -> httpx.RequestError:
    return httpx.ReadTimeout("slow", request=request)


def _connect_error(request: httpx.Request) -> httpx.RequestError:
    return httpx.ConnectError("other", request=request)


def _read_error(request: httpx.Request) -> httpx.RequestError:
    return httpx.ReadError("read", request=request)


@pytest.mark.parametrize(
    ("factory", "error_type", "message"),
    [
        (
            _read_timeout,
            RequestTimeoutError,
            "Request timed out",
        ),
        (
            _connect_error,
            ServerConnectionError,
            "Unable to connect to server",
        ),
        (
            _read_error,
            ServerConnectionError,
            "Unable to connect to server",
        ),
    ],
)
def test_request_failure_mapping(
    factory: Callable[[httpx.Request], httpx.RequestError],
    error_type: type[Exception],
    message: str,
) -> None:
    """Timeout and remaining request failures have concise messages."""

    async def scenario() -> None:
        client = OpenAICompatibleClient(
            "http://server",
            None,
            3,
            transport=httpx.MockTransport(_raising_handler(factory)),
        )
        try:
            with pytest.raises(error_type, match=message) as caught:
                await client.list_models()
        finally:
            await client.aclose()
        assert getattr(caught.value, "status_code", None) is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("cause", "message"),
    [
        (socket.gaierror(socket.EAI_NONAME, "not found"), "Unable to resolve server"),
        (OSError(errno.ECONNREFUSED, "refused"), "Connection refused"),
    ],
)
def test_connect_cause_mapping(cause: OSError, message: str) -> None:
    """DNS and refusal causes are distinguished from other connect failures."""

    async def scenario() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("failed", request=request) from cause

        client = OpenAICompatibleClient(
            "http://server", None, 3, transport=httpx.MockTransport(handler)
        )
        try:
            with pytest.raises(ServerConnectionError, match=message) as caught:
                await client.list_models()
        finally:
            await client.aclose()
        assert caught.value.status_code is None

    asyncio.run(scenario())


class _TrackingTransport(httpx.AsyncBaseTransport):
    closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    async def aclose(self) -> None:
        self.closed = True


def test_client_closes_owned_http_pool() -> None:
    """Closing the API client closes its configured transport exactly once."""

    async def scenario() -> None:
        transport = _TrackingTransport()
        client = OpenAICompatibleClient("http://server", None, 3, transport=transport)
        await client.list_models()
        await client.aclose()
        assert transport.closed

    asyncio.run(scenario())


def test_logs_and_errors_exclude_keys_and_response_bodies(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Client diagnostics never copy authorization values or response content."""

    async def scenario() -> None:
        caplog.set_level(logging.INFO, logger="modeltop.api.client")
        client = OpenAICompatibleClient(
            "http://server",
            "unique-secret-key",
            3,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(500, content=b"unique-response-body")
            ),
        )
        try:
            with pytest.raises(HTTPResponseError) as caught:
                await client.list_models()
        finally:
            await client.aclose()
        combined = f"{caplog.text}\n{caught.value.detail}"
        assert "unique-secret-key" not in combined
        assert "Authorization" not in combined
        assert "unique-response-body" not in combined
        assert caught.value.status_code == 500

    asyncio.run(scenario())
