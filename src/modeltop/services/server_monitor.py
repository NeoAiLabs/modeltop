"""Connectivity refresh orchestration and coherent state transitions."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from modeltop.api.errors import (
    AuthenticationError,
    HTTPResponseError,
    ModelsEndpointNotFoundError,
    ProtocolError,
    RequestTimeoutError,
    ServerConnectionError,
)
from modeltop.models import ServerConfig
from modeltop.services.model_discovery import ModelDiscoveryService
from modeltop.state import ApplicationState, ApplicationStateStore, ServerStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RefreshResult:
    """Summary of one attempted or skipped refresh."""

    success: bool
    skipped: bool
    message: str
    model_count: int
    latency_ms: float | None


class ServerMonitor:
    """Own state transitions for one configured server."""

    def __init__(
        self,
        server: ServerConfig,
        discovery: ModelDiscoveryService,
        state_store: ApplicationStateStore,
        on_state_change: Callable[[ApplicationState], None],
    ) -> None:
        self._server = server
        self._discovery = discovery
        self._state_store = state_store
        self._on_state_change = on_state_change
        self._refresh_pending = False
        self._preserve_online_on_failure = False

    @property
    def state(self) -> ApplicationState:
        """Return the latest immutable state snapshot."""
        return self._state_store.state

    def _publish(
        self, transform: Callable[[ApplicationState], ApplicationState]
    ) -> None:
        previous_status = self.state.server_status
        state = self._state_store.update(transform)
        logger.info(
            "Server %s state changed from %s to %s",
            self._server.id,
            previous_status,
            state.server_status,
        )
        self._on_state_change(state)

    def begin_refresh(self, *, preserve_online_on_failure: bool = False) -> bool:
        """Synchronously and atomically reserve a refresh operation."""
        if self.state.is_refreshing or self.state.benchmark_is_active:
            return False
        reserved = False

        def reserve(state: ApplicationState) -> ApplicationState:
            nonlocal reserved
            if state.is_refreshing or state.benchmark_is_active:
                return state
            reserved = True
            if state.server_status is ServerStatus.ONLINE:
                return replace(state, is_refreshing=True)
            return replace(
                state,
                server_status=ServerStatus.CONNECTING,
                connection_latency_ms=None,
                last_error=None,
                is_refreshing=True,
            )

        self._publish(reserve)
        if not reserved:
            return False
        self._preserve_online_on_failure = (
            preserve_online_on_failure
            and self.state.server_status is ServerStatus.ONLINE
        )
        self._refresh_pending = True
        return True

    async def refresh(self) -> RefreshResult:
        """Run a reserved refresh and publish its final state."""
        if not self._refresh_pending:
            return RefreshResult(
                success=False,
                skipped=True,
                message="Refresh already in progress.",
                model_count=len(self.state.available_models),
                latency_ms=None,
            )
        preserve_online_on_failure = self._preserve_online_on_failure
        self._refresh_pending = False
        self._preserve_online_on_failure = False
        logger.info("Refreshing server %s", self._server.id)
        try:
            result = await self._discovery.discover(
                self.state.selected_model_id,
                self._server.default_model,
            )
        except (ServerConnectionError, RequestTimeoutError) as error:
            return self._finish_expected_failure(
                ServerStatus.OFFLINE,
                error,
                preserve_online_on_failure=preserve_online_on_failure,
            )
        except (
            AuthenticationError,
            ModelsEndpointNotFoundError,
            HTTPResponseError,
            ProtocolError,
        ) as error:
            return self._finish_expected_failure(
                ServerStatus.ERROR,
                error,
                preserve_online_on_failure=preserve_online_on_failure,
            )
        except Exception:
            logger.exception(
                "Unexpected refresh failure for server %s", self._server.id
            )
            message = "Unexpected refresh failure"
            if preserve_online_on_failure:
                self._publish(
                    lambda state: replace(
                        state,
                        server_status=ServerStatus.ONLINE,
                        last_error=message,
                        is_refreshing=False,
                    )
                )
            else:
                self._publish(
                    lambda state: replace(
                        state,
                        server_status=ServerStatus.ERROR,
                        connection_latency_ms=None,
                        last_refresh_time=datetime.now(UTC),
                        last_error=message,
                        is_refreshing=False,
                    )
                )
            return RefreshResult(
                success=False,
                skipped=False,
                message=message,
                model_count=len(self.state.available_models),
                latency_ms=None,
            )

        model_count = len(result.models)
        logger.info("Discovered %d models from server %s", model_count, self._server.id)
        fresh_model_ids = {model.id for model in result.models}
        self._publish(
            lambda state: replace(
                state,
                selected_model_id=(
                    state.selected_model_id
                    if state.selected_model_id in fresh_model_ids
                    else result.selected_model_id
                ),
                server_status=ServerStatus.ONLINE,
                available_models=result.models,
                connection_latency_ms=result.latency_ms,
                last_refresh_time=datetime.now(UTC),
                last_error=None,
                is_refreshing=False,
            )
        )
        if model_count == 0:
            message = "Server online; no models discovered."
        else:
            noun = "model" if model_count == 1 else "models"
            message = f"Discovered {model_count} {noun} in {result.latency_ms:.0f} ms."
        return RefreshResult(
            success=True,
            skipped=False,
            message=message,
            model_count=model_count,
            latency_ms=result.latency_ms,
        )

    def _finish_expected_failure(
        self,
        status: ServerStatus,
        error: ServerConnectionError
        | RequestTimeoutError
        | AuthenticationError
        | ModelsEndpointNotFoundError
        | HTTPResponseError
        | ProtocolError,
        *,
        preserve_online_on_failure: bool,
    ) -> RefreshResult:
        logger.warning(
            "Refresh failed for server %s: %s",
            self._server.id,
            error.detail,
        )
        if preserve_online_on_failure:
            self._publish(
                lambda state: replace(
                    state,
                    server_status=ServerStatus.ONLINE,
                    last_error=error.user_message,
                    is_refreshing=False,
                )
            )
        else:
            self._publish(
                lambda state: replace(
                    state,
                    server_status=status,
                    connection_latency_ms=None,
                    last_refresh_time=datetime.now(UTC),
                    last_error=error.user_message,
                    is_refreshing=False,
                )
            )
        return RefreshResult(
            success=False,
            skipped=False,
            message=error.user_message,
            model_count=len(self.state.available_models),
            latency_ms=None,
        )

    def select_model(self, model_id: str) -> bool:
        """Atomically select an available model when no benchmark owns traffic."""
        selected = False

        def select(state: ApplicationState) -> ApplicationState:
            nonlocal selected
            if (
                state.benchmark_is_active
                or state.server_status is not ServerStatus.ONLINE
                or model_id not in {model.id for model in state.available_models}
            ):
                return state
            selected = True
            return replace(state, selected_model_id=model_id)

        self._publish(select)
        return selected
