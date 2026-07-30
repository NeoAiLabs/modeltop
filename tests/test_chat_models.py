"""Immutable chat value and session ordering tests."""

from dataclasses import FrozenInstanceError, replace

import pytest

from modeltop.chat.models import ChatMessage, GenerationSettings, GenerationStatus
from modeltop.chat.session import DEFAULT_SYSTEM_PROMPT, ChatSession
from modeltop.state import initial_application_state


def test_generation_settings_defaults_and_boundaries() -> None:
    settings = GenerationSettings()
    assert settings == GenerationSettings(0.7, 0.95, 1024, None, True)
    assert GenerationSettings(0.0, 0.0001, 1, -4)
    assert GenerationSettings(2.0, 1.0, 1, 0)
    assert GenerationSettings(enable_thinking=False).enable_thinking is False
    assert GenerationSettings(enable_thinking=True).enable_thinking is True

    for temperature in (-0.1, 2.1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            GenerationSettings(temperature=temperature)
    for top_p in (0.0, -0.1, 1.1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            GenerationSettings(top_p=top_p)
    with pytest.raises(ValueError):
        GenerationSettings(max_tokens=0)
    with pytest.raises(TypeError):
        GenerationSettings(seed=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        GenerationSettings(max_tokens=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        GenerationSettings(enable_thinking="false")  # type: ignore[arg-type]


def test_message_validation_and_immutability() -> None:
    message = ChatMessage("user", "hello")
    with pytest.raises(FrozenInstanceError):
        message.content = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError):
        ChatMessage("invalid", "text")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ChatMessage("user", 4)  # type: ignore[arg-type]


def test_session_builds_exact_order_and_system_once() -> None:
    session = ChatSession().append(
        ChatMessage("user", "first"),
        ChatMessage("assistant", "answer"),
    )
    context = session.request_context("follow up")
    assert context == (
        ChatMessage("system", DEFAULT_SYSTEM_PROMPT),
        ChatMessage("user", "first"),
        ChatMessage("assistant", "answer"),
        ChatMessage("user", "follow up"),
    )
    assert session.messages == context[1:-1]

    without_system = replace(session, system_prompt="  ")
    assert without_system.request_context("next") == (
        *session.messages,
        ChatMessage("user", "next"),
    )
    with pytest.raises(ValueError, match="blank"):
        session.request_context(" \n ")


def test_clear_preserves_preferences_and_capture() -> None:
    settings = GenerationSettings(temperature=1.2, seed=9)
    session = ChatSession(
        messages=(ChatMessage("user", "secret"),),
        system_prompt="system",
        show_system_prompt=True,
        settings=settings,
        server_id="server",
        model_id="model",
    )
    cleared = session.clear()
    assert cleared.messages == ()
    assert replace(cleared, messages=session.messages) == session


def test_initial_state_has_independent_idle_chat_lane() -> None:
    state = initial_application_state("server", hardware_enabled=False)
    assert state.active_view == "overview"
    assert state.chat_session == ChatSession()
    assert state.generation_status is GenerationStatus.IDLE
    assert state.generation_metrics is None
    assert state.generation_error is None
    assert state.generation_notice is None
    assert state.current_response == ""
    assert state.generation_id == 0
    assert state.active_generation_id is None
