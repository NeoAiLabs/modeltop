"""Server monitor refresh and state-transition tests."""

import asyncio
import errno
import logging
from dataclasses import replace

import httpx
import pytest

from modeltop.api.client import OpenAICompatibleClient
from modeltop.models import ServerConfig
from modeltop.services.model_discovery import DiscoveryResult, ModelDiscoveryService
from modeltop.services.server_monitor import ServerMonitor
from modeltop.state import (
    ApplicationState,
    ApplicationStateStore,
    HardwareStatus,
    ServerStatus,
    initial_application_state,
)


def _server(default_model: str | None = None) -> ServerConfig:
    return ServerConfig(
        id="server",
        name="Server",
        base_url="http://server/v1",
        default_model=default_model,
    )


def _state_store() -> ApplicationStateStore:
    store = ApplicationStateStore(
        initial_application_state("server", hardware_enabled=True)
    )
    store.update(
        lambda state: replace(
            state,
            hardware_status=HardwareStatus.DEGRADED,
            hardware_last_error="hardware marker",
        )
    )
    return store


def test_begin_refresh_is_synchronous_and_overlap_is_skipped() -> None:
    """Connecting publishes before I/O and one reservation permits one HTTP call."""

    async def scenario() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        request_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal request_count
            request_count += 1
            entered.set()
            await release.wait()
            return httpx.Response(200, json={"data": [{"id": "model"}]})

        client = OpenAICompatibleClient(
            "http://server", None, 3, transport=httpx.MockTransport(handler)
        )
        states: list[ApplicationState] = []
        monitor = ServerMonitor(
            _server(), ModelDiscoveryService(client), _state_store(), states.append
        )
        assert monitor.begin_refresh()
        assert monitor.state.server_status is ServerStatus.CONNECTING
        assert monitor.state.is_refreshing
        assert states[-1] is monitor.state
        assert not monitor.begin_refresh()

        assert monitor.state.hardware_status is HardwareStatus.DEGRADED
        assert monitor.state.hardware_last_error == "hardware marker"
        task = asyncio.create_task(monitor.refresh())
        await entered.wait()
        overlap = await monitor.refresh()
        assert overlap.skipped
        assert request_count == 1
        release.set()
        result = await task

        assert result.success
        assert monitor.state.server_status is ServerStatus.ONLINE
        assert not monitor.state.is_refreshing
        assert monitor.state.last_refresh_time is not None
        assert monitor.state.last_refresh_time.utcoffset() is not None
        await client.aclose()

    asyncio.run(scenario())


def test_online_empty_offline_error_and_recovery_retain_selection() -> None:
    """Failures hide stale latency but retain models and selection for recovery."""

    async def scenario() -> None:
        outcomes: list[httpx.Response | str] = [
            httpx.Response(
                200,
                json={"data": [{"id": "model-b"}, {"id": "model-a"}]},
            ),
            "refused",
            httpx.Response(401),
            httpx.Response(
                200,
                json={"data": [{"id": "model-a"}, {"id": "model-b"}]},
            ),
            httpx.Response(200, json={"data": []}),
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            outcome = outcomes.pop(0)
            if outcome == "refused":
                try:
                    raise OSError(errno.ECONNREFUSED, "refused")
                except OSError as cause:
                    raise httpx.ConnectError("failed", request=request) from cause
            assert isinstance(outcome, httpx.Response)
            return outcome

        client = OpenAICompatibleClient(
            "http://server", None, 3, transport=httpx.MockTransport(handler)
        )
        monitor = ServerMonitor(
            _server("model-b"),
            ModelDiscoveryService(client),
            _state_store(),
            lambda state: None,
        )

        assert monitor.begin_refresh()
        assert (await monitor.refresh()).success
        assert monitor.state.selected_model_id == "model-b"
        assert monitor.state.connection_latency_ms is not None
        assert monitor.select_model("model-a")
        confirmed_model_ids = tuple(
            model.id for model in monitor.state.available_models
        )
        confirmed_selection = monitor.state.selected_model_id
        confirmed_latency = monitor.state.connection_latency_ms
        confirmed_refresh_time = monitor.state.last_refresh_time

        assert monitor.begin_refresh()
        assert monitor.state.server_status is ServerStatus.ONLINE
        assert monitor.state.is_refreshing
        assert (
            tuple(model.id for model in monitor.state.available_models)
            == confirmed_model_ids
        )
        assert monitor.state.selected_model_id == confirmed_selection
        assert monitor.state.connection_latency_ms == confirmed_latency
        assert monitor.state.last_refresh_time == confirmed_refresh_time
        offline = await monitor.refresh()
        assert not offline.success
        assert monitor.state.server_status is ServerStatus.OFFLINE
        assert monitor.state.last_error == "Connection refused"
        assert not monitor.state.is_refreshing
        assert monitor.state.connection_latency_ms is None
        assert monitor.state.selected_model_id == "model-a"
        assert len(monitor.state.available_models) == 2

        assert monitor.begin_refresh()
        error = await monitor.refresh()
        assert not error.success
        assert monitor.state.server_status is ServerStatus.ERROR
        assert monitor.state.last_error == "Authentication failed"
        assert monitor.state.selected_model_id == "model-a"

        assert monitor.begin_refresh()
        recovered = await monitor.refresh()
        assert recovered.success
        assert monitor.state.server_status is ServerStatus.ONLINE
        assert monitor.state.selected_model_id == "model-a"
        assert monitor.state.last_error is None

        assert monitor.begin_refresh()
        empty = await monitor.refresh()
        assert empty.success
        assert empty.message == "Server online; no models discovered."
        assert monitor.state.server_status is ServerStatus.ONLINE
        assert monitor.state.available_models == ()
        assert monitor.state.selected_model_id is None
        await client.aclose()
        assert monitor.state.hardware_status is HardwareStatus.DEGRADED
        assert monitor.state.hardware_last_error == "hardware marker"

    asyncio.run(scenario())


def test_online_refresh_preserves_latest_selection_until_success() -> None:
    """A valid local selection made during discovery wins on success."""

    async def scenario() -> None:
        second_request_entered = asyncio.Event()
        release_second_request = asyncio.Event()
        request_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal request_count
            request_count += 1
            if request_count == 2:
                second_request_entered.set()
                await release_second_request.wait()
            return httpx.Response(200, json={"data": [{"id": "a"}, {"id": "b"}]})

        client = OpenAICompatibleClient(
            "http://server", None, 3, transport=httpx.MockTransport(handler)
        )
        monitor = ServerMonitor(
            _server(), ModelDiscoveryService(client), _state_store(), lambda state: None
        )
        assert monitor.begin_refresh()
        assert (await monitor.refresh()).success
        assert monitor.state.selected_model_id == "a"

        assert monitor.begin_refresh()
        task = asyncio.create_task(monitor.refresh())
        await second_request_entered.wait()
        assert monitor.select_model("b")
        release_second_request.set()
        assert (await task).success

        assert monitor.state.server_status is ServerStatus.ONLINE
        assert not monitor.state.is_refreshing
        assert monitor.state.selected_model_id == "b"
        assert request_count == 2
        await client.aclose()

    asyncio.run(scenario())


def test_model_selection_requires_online_available_id() -> None:
    """Selection is local, online-only, and limited to discovered IDs."""

    async def scenario() -> None:
        requests = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal requests
            requests += 1
            return httpx.Response(200, json={"data": [{"id": "a"}, {"id": "b"}]})

        client = OpenAICompatibleClient(
            "http://server", None, 3, transport=httpx.MockTransport(handler)
        )
        monitor = ServerMonitor(
            _server(),
            ModelDiscoveryService(client),
            _state_store(),
            lambda state: None,
        )
        assert not monitor.select_model("a")
        assert monitor.begin_refresh()
        await monitor.refresh()
        assert not monitor.select_model("missing")
        assert monitor.select_model("b")
        assert monitor.state.selected_model_id == "b"
        assert requests == 1
        await client.aclose()

    asyncio.run(scenario())


class _ExplodingDiscovery(ModelDiscoveryService):
    def __init__(self, client: OpenAICompatibleClient) -> None:
        super().__init__(client)

    async def discover(
        self,
        previous_selection: str | None,
        configured_default: str | None,
    ) -> DiscoveryResult:
        raise RuntimeError("unexpected monitor failure")


def test_automatic_failure_preservation_requires_confirmed_online_snapshot() -> None:
    """Only failures after a confirmed online snapshot preserve availability."""

    async def scenario() -> None:
        outcomes: list[httpx.Response | type[httpx.RequestError]] = [
            httpx.ReadError,
            httpx.Response(
                200,
                json={"data": [{"id": "model-a"}, {"id": "model-b"}]},
            ),
            httpx.ReadError,
            httpx.RemoteProtocolError,
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            outcome = outcomes.pop(0)
            if outcome is httpx.ReadError:
                raise httpx.ReadError("broken", request=request)
            if outcome is httpx.RemoteProtocolError:
                raise httpx.RemoteProtocolError("broken", request=request)
            assert isinstance(outcome, httpx.Response)
            return outcome

        client = OpenAICompatibleClient(
            "http://server", None, 3, transport=httpx.MockTransport(handler)
        )
        store = _state_store()
        monitor = ServerMonitor(
            _server(), ModelDiscoveryService(client), store, lambda state: None
        )

        assert monitor.begin_refresh(preserve_online_on_failure=True)
        initial_failure = await monitor.refresh()
        assert not initial_failure.success
        assert not initial_failure.skipped
        assert monitor.state.server_status is ServerStatus.OFFLINE
        assert monitor.state.available_models == ()

        assert monitor.begin_refresh()
        assert (await monitor.refresh()).success
        assert monitor.select_model("model-b")
        confirmed_model_ids = tuple(
            model.id for model in monitor.state.available_models
        )
        confirmed_selection = monitor.state.selected_model_id
        confirmed_latency = monitor.state.connection_latency_ms
        confirmed_refresh_time = monitor.state.last_refresh_time

        for expected_error in (
            "Unable to connect to server",
            "Invalid response from server",
        ):
            assert monitor.begin_refresh(preserve_online_on_failure=True)
            failure = await monitor.refresh()
            assert not failure.success
            assert not failure.skipped
            assert failure.message == expected_error
            assert failure.model_count == 2
            assert failure.latency_ms is None
            assert monitor.state.server_status is ServerStatus.ONLINE
            assert not monitor.state.is_refreshing
            assert (
                tuple(model.id for model in monitor.state.available_models)
                == confirmed_model_ids
            )
            assert monitor.state.selected_model_id == confirmed_selection
            assert monitor.state.connection_latency_ms == confirmed_latency
            assert monitor.state.last_refresh_time == confirmed_refresh_time
            assert monitor.state.last_error == expected_error

        exploding_monitor = ServerMonitor(
            _server(), _ExplodingDiscovery(client), store, lambda state: None
        )
        assert exploding_monitor.begin_refresh(preserve_online_on_failure=True)
        unexpected = await exploding_monitor.refresh()
        assert not unexpected.success
        assert not unexpected.skipped
        assert unexpected.message == "Unexpected refresh failure"
        assert unexpected.model_count == 2
        assert unexpected.latency_ms is None
        assert exploding_monitor.state.server_status is ServerStatus.ONLINE
        assert not exploding_monitor.state.is_refreshing
        assert (
            tuple(model.id for model in exploding_monitor.state.available_models)
            == confirmed_model_ids
        )
        assert exploding_monitor.state.selected_model_id == confirmed_selection
        assert exploding_monitor.state.connection_latency_ms == confirmed_latency
        assert exploding_monitor.state.last_refresh_time == confirmed_refresh_time
        assert exploding_monitor.state.last_error == "Unexpected refresh failure"
        await client.aclose()

    asyncio.run(scenario())


def test_unexpected_failure_is_logged_without_owning_client(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unexpected exceptions publish ERROR while client ownership stays external."""

    async def scenario() -> None:
        caplog.set_level(logging.ERROR, logger="modeltop.services.server_monitor")
        client = OpenAICompatibleClient(
            "http://server",
            None,
            3,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"data": []})
            ),
        )
        discovery = _ExplodingDiscovery(client)
        monitor = ServerMonitor(
            _server(), discovery, _state_store(), lambda state: None
        )
        assert monitor.begin_refresh()
        result = await monitor.refresh()
        assert not result.success
        assert result.message == "Unexpected refresh failure"
        assert monitor.state.server_status is ServerStatus.ERROR
        assert monitor.state.last_error == "Unexpected refresh failure"
        assert "unexpected monitor failure" in caplog.text
        await client.aclose()

    asyncio.run(scenario())
