"""Model discovery parsing and selection-policy tests."""

import asyncio

import httpx
import pytest

from modeltop.api.client import OpenAICompatibleClient
from modeltop.api.errors import ProtocolError
from modeltop.services.model_discovery import ModelDiscoveryService


class _ResponseTransport(httpx.AsyncBaseTransport):
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.close_count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=self.payload)

    async def aclose(self) -> None:
        self.close_count += 1


def test_discovery_preserves_metadata_and_sorts_full_ids() -> None:
    """Validated metadata and full IDs survive deterministic alphabetical sorting."""

    async def scenario() -> None:
        transport = _ResponseTransport(
            {
                "data": [
                    {"id": "zeta/full", "owned_by": "owner", "created": 42},
                    {"id": "Alpha/full", "unknown": True},
                    {"id": "alpha/full"},
                ]
            }
        )
        client = OpenAICompatibleClient("http://server", None, 3, transport=transport)
        service = ModelDiscoveryService(client)
        result = await service.discover(None, None)
        assert [model.id for model in result.models] == [
            "Alpha/full",
            "alpha/full",
            "zeta/full",
        ]
        assert result.models[2].owned_by == "owner"
        assert result.models[2].created == 42
        assert result.selected_model_id == "Alpha/full"
        await client.aclose()
        assert transport.close_count == 1

    asyncio.run(scenario())


def test_empty_discovery_is_successful_without_selection() -> None:
    """A valid empty data tuple remains an online-capable discovery result."""

    async def scenario() -> None:
        client = OpenAICompatibleClient(
            "http://server",
            None,
            3,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"data": []})
            ),
        )
        service = ModelDiscoveryService(client)
        try:
            result = await service.discover("old", "configured")
        finally:
            await client.aclose()
        assert result.models == ()
        assert result.selected_model_id is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("previous", "configured", "expected"),
    [
        ("model-b", "model-a", "model-b"),
        ("missing", "model-b", "model-b"),
        (None, "missing", "model-a"),
        (None, None, "model-a"),
    ],
)
def test_selection_precedence(
    previous: str | None,
    configured: str | None,
    expected: str,
) -> None:
    """Prior, configured, first, and missing-default selection order is exact."""

    async def scenario() -> None:
        client = OpenAICompatibleClient(
            "http://server",
            None,
            3,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={"data": [{"id": "model-b"}, {"id": "model-a"}]},
                )
            ),
        )
        service = ModelDiscoveryService(client)
        try:
            result = await service.discover(previous, configured)
        finally:
            await client.aclose()
        assert result.selected_model_id == expected

    asyncio.run(scenario())


def test_invalid_model_rejects_whole_response_without_value_leakage() -> None:
    """One malformed item rejects discovery with field-only diagnostics."""

    async def scenario() -> None:
        invalid_value = "private unexpected value"
        client = OpenAICompatibleClient(
            "http://server",
            None,
            3,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "data": [
                            {"id": "valid"},
                            {"id": 123, "owned_by": invalid_value},
                        ]
                    },
                )
            ),
        )
        service = ModelDiscoveryService(client)
        try:
            with pytest.raises(ProtocolError) as caught:
                await service.discover(None, None)
        finally:
            await client.aclose()
        assert "data[1].id" in caught.value.detail
        assert invalid_value not in caught.value.detail
        assert str(caught.value) == "Invalid response from server"

    asyncio.run(scenario())
