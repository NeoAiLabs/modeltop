"""Drafter aggregate filtering and acceptance-rate statistics."""

import pytest

from modeltop.benchmarks.models import DrafterRunResult
from modeltop.benchmarks.statistics import build_drafter_aggregates
from modeltop.services.drafter_benchmark import build_drafter_observations


def _run(
    run_number: int,
    *,
    warmup: bool = False,
    success: bool = True,
    ttft_ms: float | None = None,
    speed: float | None = None,
    draft_tokens: int | None = None,
    accepted_tokens: int | None = None,
    acceptance_rate: float | None = None,
) -> DrafterRunResult:
    return DrafterRunResult(
        run_number=run_number,
        warmup=warmup,
        success=success,
        cancelled=False,
        error=None if success else "failed",
        prompt_tokens=10 if success else None,
        completion_tokens=20 if success else None,
        total_tokens=30 if success else None,
        prompt_tokens_estimated=False,
        completion_tokens_estimated=False,
        total_tokens_estimated=False,
        draft_tokens=draft_tokens,
        accepted_tokens=accepted_tokens,
        acceptance_rate=acceptance_rate,
        ttft_ms=ttft_ms,
        generation_duration_s=1.0 if success else None,
        total_duration_s=1.2 if success else None,
        output_tokens_per_second=speed,
        finish_reason="stop" if success else None,
        streamed=True,
    )


def test_aggregates_filter_warmup_and_failures() -> None:
    aggregates = build_drafter_aggregates(
        (
            _run(
                1,
                warmup=True,
                ttft_ms=1,
                speed=100,
                draft_tokens=9,
                accepted_tokens=8,
                acceptance_rate=0.9,
            ),
            _run(
                1,
                ttft_ms=10,
                speed=None,
                draft_tokens=6,
                accepted_tokens=4,
                acceptance_rate=4 / 6,
            ),
            _run(
                2,
                ttft_ms=None,
                speed=20,
                draft_tokens=8,
                accepted_tokens=4,
                acceptance_rate=0.5,
            ),
            _run(
                3,
                success=False,
                ttft_ms=999,
                speed=999,
                draft_tokens=999,
                accepted_tokens=999,
                acceptance_rate=1.0,
            ),
        )
    )
    assert aggregates.ttft_ms.count == 1
    assert aggregates.ttft_ms.mean == 10
    assert aggregates.output_tokens_per_second.count == 1
    assert aggregates.output_tokens_per_second.mean == 20
    assert aggregates.draft_tokens.count == 2
    assert aggregates.draft_tokens.mean == 7
    assert aggregates.accepted_tokens.count == 2
    assert aggregates.accepted_tokens.mean == 4
    assert aggregates.acceptance_rate.count == 2
    assert aggregates.acceptance_rate.mean == pytest.approx((4 / 6 + 0.5) / 2)


def test_empty_aggregates_and_observations() -> None:
    aggregates = build_drafter_aggregates(())
    assert aggregates.acceptance_rate.count == 0
    assert aggregates.acceptance_rate.mean is None
    assert build_drafter_observations((), None) == ()

    unavailable = build_drafter_observations(
        (_run(1, draft_tokens=None, accepted_tokens=None, acceptance_rate=None),),
        None,
    )
    assert len(unavailable) == 1
    assert unavailable[0].code == "speculative_telemetry_unavailable"

    partial = build_drafter_observations(
        (
            _run(1, draft_tokens=6, accepted_tokens=4, acceptance_rate=0.66),
            _run(2, draft_tokens=None, accepted_tokens=None, acceptance_rate=None),
        ),
        0.66,
    )
    assert partial[0].code == "partial_speculative_telemetry"

    low = build_drafter_observations(
        (
            _run(1, draft_tokens=10, accepted_tokens=2, acceptance_rate=0.2),
            _run(2, draft_tokens=10, accepted_tokens=2, acceptance_rate=0.2),
        ),
        0.2,
    )
    assert low[0].code == "low_mean_acceptance_rate"
    assert "0.20" in low[0].message
