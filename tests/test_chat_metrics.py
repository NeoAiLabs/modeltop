"""Generation metric precedence and boundary tests."""

from collections.abc import Iterator

import pytest

from modeltop.chat.metrics import CharacterTokenCounter, MetricCollector
from modeltop.chat.models import ChatMessage


class _Clock:
    def __init__(self, *values: float) -> None:
        self._values: Iterator[float] = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class _ExactCounter:
    name = "exact-test"
    exact = True

    def count(self, text: str) -> int:
        return len(text.split())


def test_character_counter_uses_nonblank_characters() -> None:
    counter = CharacterTokenCounter()
    assert not counter.exact
    assert counter.count("") == 0
    assert counter.count("   \n") == 0
    assert counter.count("abcd") == 1
    assert counter.count("abcde fgh") == 2


def test_stream_metric_formulas_and_estimate_labels() -> None:
    clock = _Clock(10.0, 10.2, 10.7)
    collector = MetricCollector((ChatMessage("user", "12345678"),), clock=clock)
    collector.add_content("12345678")
    metrics = collector.finish(finish_reason="stop")
    assert metrics.first_token_at == pytest.approx(10.2)
    assert metrics.completed_at == pytest.approx(10.7)
    assert metrics.prompt_tokens == 9
    assert metrics.completion_tokens == 2
    assert metrics.total_tokens == 11
    assert metrics.prompt_tokens_estimated
    assert metrics.completion_tokens_estimated
    assert metrics.total_tokens_estimated
    assert metrics.ttft_ms == pytest.approx(200)
    assert metrics.active_generation_duration_s == pytest.approx(0.5)
    assert metrics.total_duration_s == pytest.approx(0.7)
    assert metrics.output_tokens_per_second == pytest.approx(4)
    assert metrics.inter_token_latency_ms == pytest.approx(250)
    assert metrics.finish_reason == "stop"


def test_server_usage_wins_field_by_field_over_exact_counter() -> None:
    clock = _Clock(1.0, 1.1, 2.1)
    collector = MetricCollector(
        (ChatMessage("user", "three prompt words"),),
        clock=clock,
        token_counter=_ExactCounter(),
    )
    collector.add_content("two words")
    collector.update_usage(prompt_tokens=20, completion_tokens=None, total_tokens=30)
    metrics = collector.finish()
    assert metrics.prompt_tokens == 20
    assert metrics.completion_tokens == 2
    assert metrics.total_tokens == 30
    assert not metrics.prompt_tokens_estimated
    assert not metrics.completion_tokens_estimated
    assert not metrics.total_tokens_estimated


def test_non_stream_empty_zero_duration_and_cancelled_boundaries() -> None:
    fallback = MetricCollector((ChatMessage("user", "prompt"),), clock=_Clock(3.0, 4.0))
    fallback.mark_fallback()
    fallback.add_content("answer")
    non_stream = fallback.finish(finish_reason="stop")
    assert not non_stream.streamed
    assert non_stream.first_token_at is None
    assert non_stream.ttft_ms is None
    assert non_stream.active_generation_duration_s is None
    assert non_stream.output_tokens_per_second is None
    assert non_stream.inter_token_latency_ms is None

    empty = MetricCollector((), clock=_Clock(8.0, 8.0, 8.0))
    empty.add_content("")
    zero = empty.finish(cancelled=True)
    assert zero.completion_tokens == 0
    assert zero.output_tokens_per_second is None
    assert zero.inter_token_latency_ms is None
    assert zero.cancelled


def test_speculative_usage_merge_and_rate_derivation() -> None:
    clock = _Clock(1.0, 1.1, 2.1)
    collector = MetricCollector(
        (ChatMessage("user", "prompt"),),
        clock=clock,
        token_counter=_ExactCounter(),
    )
    collector.add_content("words")
    collector.update_usage(
        prompt_tokens=5,
        completion_tokens=2,
        total_tokens=7,
        draft_tokens=6,
        accepted_tokens=4,
    )
    metrics = collector.finish()
    assert metrics.draft_tokens == 6
    assert metrics.accepted_tokens == 4
    assert metrics.acceptance_rate == pytest.approx(4 / 6)

    partial = MetricCollector((ChatMessage("user", "prompt"),), clock=_Clock(1.0, 2.0))
    partial.update_usage(1, 1, 2, draft_tokens=3)
    partial.update_usage(None, None, None, accepted_tokens=1, acceptance_rate=0.5)
    merged = partial.finish()
    assert merged.draft_tokens == 3
    assert merged.accepted_tokens == 1
    assert merged.acceptance_rate == pytest.approx(0.5)

    zero_draft = MetricCollector((ChatMessage("user", "p"),), clock=_Clock(1.0, 2.0))
    zero_draft.update_usage(1, 1, 2, draft_tokens=0, accepted_tokens=0)
    zero = zero_draft.finish()
    assert zero.draft_tokens == 0
    assert zero.accepted_tokens == 0
    assert zero.acceptance_rate is None

    missing = MetricCollector((ChatMessage("user", "p"),), clock=_Clock(1.0, 2.0))
    missing.update_usage(1, 1, 2)
    bare = missing.finish()
    assert bare.draft_tokens is None
    assert bare.accepted_tokens is None
    assert bare.acceptance_rate is None
