"""Dashboard adapter for atomic chat state transitions."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace

from modeltop.chat.models import ChatMessage, GenerationSettings, GenerationStatus
from modeltop.services.generation import (
    GenerationCancelled,
    GenerationFailed,
    GenerationOutcome,
    GenerationProgress,
    GenerationRequest,
    GenerationService,
)
from modeltop.state import ApplicationState, ApplicationStateStore, ServerStatus

logger = logging.getLogger(__name__)


class ChatOperationError(Exception):
    """Readable synchronous rejection before network I/O."""

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


@dataclass(frozen=True, slots=True)
class PendingGeneration:
    """Reserved generation ID paired with its immutable runner request."""

    generation_id: int
    request: GenerationRequest


type StateCallback = Callable[[ApplicationState], None]


DEFAULT_CHAT_REQUEST_TIMEOUT_SECONDS = 300.0


class DashboardChatService:
    """Own chat-only transforms over the shared immutable state store."""

    def __init__(
        self,
        generation_service: GenerationService,
        state_store: ApplicationStateStore,
        on_state_change: StateCallback,
        *,
        request_timeout_seconds: float = DEFAULT_CHAT_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self._state_store = state_store
        self._on_state_change = on_state_change
        self._generation_service = generation_service
        self._request_timeout_seconds = request_timeout_seconds

    @property
    def state(self) -> ApplicationState:
        return self._state_store.state

    def _publish(
        self, transform: Callable[[ApplicationState], ApplicationState]
    ) -> ApplicationState:
        state = self._state_store.update(transform)
        self._on_state_change(state)
        return state

    def begin_generation(self, prompt: str) -> PendingGeneration:
        """Atomically validate, reserve, capture context, and append one user turn."""
        if not prompt.strip():
            raise ChatOperationError("Enter a nonblank prompt before sending.")

        pending: PendingGeneration | None = None

        def reserve(state: ApplicationState) -> ApplicationState:
            nonlocal pending
            if state.tool_calling_benchmark.is_active:
                raise ChatOperationError(
                    "Chat is unavailable while Tool Calling is running"
                )
            if state.r0b0bench_benchmark.is_active:
                raise ChatOperationError(
                    "Chat is unavailable while r0b0bench is running"
                )
            if state.context_benchmark.is_active:
                raise ChatOperationError(
                    "Chat is unavailable while a Context benchmark is running"
                )
            if state.concurrency_benchmark.is_active:
                raise ChatOperationError(
                    "Chat is unavailable while a benchmark is running"
                )
            if state.drafter_benchmark.is_active:
                raise ChatOperationError(
                    "Chat is unavailable while a Drafter benchmark is running"
                )
            if state.speed_test.is_active:
                raise ChatOperationError("A Speed Test is already running.")
            if state.active_generation_id is not None:
                raise ChatOperationError("A generation is already in progress.")
            if state.server_status is not ServerStatus.ONLINE:
                raise ChatOperationError("The selected server is offline.")
            server_id = state.selected_server_id
            model_id = state.selected_model_id
            if server_id is None or model_id is None:
                raise ChatOperationError("Select an available model before sending.")
            generation_id = state.generation_id + 1
            request_messages = state.chat_session.request_context(prompt)
            request = GenerationRequest(
                server_id=server_id,
                model_id=model_id,
                messages=request_messages,
                settings=state.chat_session.settings,
                request_timeout_seconds=self._request_timeout_seconds,
            )
            pending = PendingGeneration(generation_id, request)
            captured_session = replace(
                state.chat_session,
                server_id=server_id,
                model_id=model_id,
            ).append(ChatMessage("user", prompt))
            return replace(
                state,
                chat_session=captured_session,
                generation_status=GenerationStatus.STARTING,
                generation_metrics=None,
                generation_error=None,
                generation_notice=None,
                current_response="",
                generation_id=generation_id,
                active_generation_id=generation_id,
            )

        try:
            self._publish(reserve)
        except ChatOperationError:
            raise
        except Exception as error:
            if pending is not None:
                self._finish_unexpected(pending.generation_id)
            logger.error(
                "Chat reservation callback failed error=%s", type(error).__name__
            )
            raise
        if pending is None:
            raise RuntimeError("Generation reservation did not produce a request")
        return pending

    async def generate(self, pending: PendingGeneration) -> GenerationOutcome:
        """Run one reserved request and commit exactly one terminal assistant turn."""
        if self.state.active_generation_id != pending.generation_id:
            raise ChatOperationError("This generation is no longer active.")
        try:
            outcome = await self._generation_service.run(
                pending.request,
                lambda progress: self._handle_progress(pending.generation_id, progress),
            )
        except GenerationCancelled as error:
            self._finish_cancelled(pending.generation_id, error.outcome)
            raise asyncio.CancelledError from error
        except GenerationFailed as error:
            self._finish_error(
                pending.generation_id,
                error.outcome,
                error.error.user_message,
            )
            raise
        except Exception as error:
            self._finish_unexpected(pending.generation_id)
            logger.exception(
                "Unexpected Chat generation failure generation=%d error=%s",
                pending.generation_id,
                type(error).__name__,
            )
            raise

        self._finish_success(pending.generation_id, outcome)
        return outcome

    def _handle_progress(
        self, generation_id: int, progress: GenerationProgress
    ) -> None:
        def transform(state: ApplicationState) -> ApplicationState:
            if state.active_generation_id != generation_id:
                return state
            return replace(
                state,
                generation_status=GenerationStatus.STREAMING,
                generation_metrics=progress.metrics,
                generation_notice=progress.notice,
                current_response=progress.content,
            )

        self._publish(transform)

    def _finish_success(self, generation_id: int, outcome: GenerationOutcome) -> None:
        def transform(state: ApplicationState) -> ApplicationState:
            if state.active_generation_id != generation_id:
                return state
            return replace(
                state,
                chat_session=state.chat_session.append(
                    ChatMessage("assistant", outcome.content)
                ),
                generation_status=GenerationStatus.COMPLETED,
                generation_metrics=outcome.metrics,
                generation_error=None,
                generation_notice=outcome.notice,
                current_response="",
                active_generation_id=None,
            )

        self._publish(transform)

    def _finish_cancelled(self, generation_id: int, outcome: GenerationOutcome) -> None:
        def transform(state: ApplicationState) -> ApplicationState:
            if state.active_generation_id != generation_id:
                return state
            session = state.chat_session
            if outcome.content:
                session = session.append(ChatMessage("assistant", outcome.content))
            return replace(
                state,
                chat_session=session,
                generation_status=GenerationStatus.CANCELLED,
                generation_metrics=outcome.metrics,
                generation_error=None,
                generation_notice=outcome.notice,
                current_response="",
                active_generation_id=None,
            )

        self._publish(transform)

    def _finish_error(
        self,
        generation_id: int,
        outcome: GenerationOutcome,
        user_message: str,
    ) -> None:
        def transform(state: ApplicationState) -> ApplicationState:
            if state.active_generation_id != generation_id:
                return state
            session = state.chat_session
            if outcome.content:
                session = session.append(ChatMessage("assistant", outcome.content))
            return replace(
                state,
                chat_session=session,
                generation_status=GenerationStatus.ERROR,
                generation_metrics=outcome.metrics,
                generation_error=user_message,
                generation_notice=outcome.notice,
                current_response="",
                active_generation_id=None,
            )

        self._publish(transform)

    def cancel_reservation(self, generation_id: int) -> bool:
        """Clear a matching reservation if its worker never entered generation."""
        if self.state.active_generation_id != generation_id:
            return False

        def transform(state: ApplicationState) -> ApplicationState:
            if state.active_generation_id != generation_id:
                return state
            return replace(
                state,
                generation_status=GenerationStatus.CANCELLED,
                generation_error=None,
                generation_notice=None,
                current_response="",
                active_generation_id=None,
            )

        self._commit_cleanup(transform)
        return True

    def _finish_unexpected(self, generation_id: int) -> None:
        def transform(state: ApplicationState) -> ApplicationState:
            if state.active_generation_id != generation_id:
                return state
            session = state.chat_session
            if state.current_response:
                session = session.append(
                    ChatMessage("assistant", state.current_response)
                )
            return replace(
                state,
                chat_session=session,
                generation_status=GenerationStatus.ERROR,
                generation_error="Generation failed",
                current_response="",
                active_generation_id=None,
            )

        self._commit_cleanup(transform)

    def _commit_cleanup(
        self, transform: Callable[[ApplicationState], ApplicationState]
    ) -> None:
        state = self._state_store.update(transform)
        try:
            self._on_state_change(state)
        except Exception as error:
            logger.error("Chat cleanup callback failed error=%s", type(error).__name__)

    def clear_conversation(self) -> None:
        """Clear only chat history when no generation is active."""

        def transform(state: ApplicationState) -> ApplicationState:
            if state.active_generation_id is not None:
                raise ChatOperationError(
                    "Cancel the active generation before clearing the conversation."
                )
            return replace(
                state,
                chat_session=state.chat_session.clear(),
                generation_status=GenerationStatus.IDLE,
                generation_metrics=None,
                generation_error=None,
                generation_notice=None,
                current_response="",
            )

        self._publish(transform)

    def update_preferences(
        self,
        settings: GenerationSettings,
        system_prompt: str,
        show_system_prompt: bool,
    ) -> None:
        """Apply validated in-memory preferences without touching other lanes."""

        def transform(state: ApplicationState) -> ApplicationState:
            if state.active_generation_id is not None:
                raise ChatOperationError(
                    "Generation settings cannot change while generating."
                )
            return replace(
                state,
                chat_session=replace(
                    state.chat_session,
                    settings=settings,
                    system_prompt=system_prompt,
                    show_system_prompt=show_system_prompt,
                ),
            )

        self._publish(transform)
