"""Resolution-aligned Context probe planning contracts."""

from modeltop.benchmarks.context_probe import ContextProbePlanner


def test_probe_exponential_then_binary_is_unique_and_bounded() -> None:
    planner = ContextProbePlanner(
        start_tokens=4096,
        maximum_tokens=16384,
        resolution_tokens=1024,
        safety_maximum_tokens=16384,
    )
    while (candidate := planner.next_candidate()) is not None:
        accepted = candidate <= 10_000
        planner.record(candidate, (accepted, accepted, accepted))
    bounds = planner.bounds
    assert bounds.attempted_targets[:3] == (4096, 8192, 16384)
    assert len(bounds.attempted_targets) == len(set(bounds.attempted_targets))
    assert max(bounds.attempted_targets) <= 16384
    assert bounds.highest_confirmed_success == 9216
    assert bounds.first_confirmed_rejection == 10240
    assert bounds.stage == "complete"


def test_probe_mixed_or_unknown_outcome_does_not_move_bounds() -> None:
    planner = ContextProbePlanner(
        start_tokens=4096,
        maximum_tokens=8192,
        resolution_tokens=1024,
        safety_maximum_tokens=8192,
    )
    candidate = planner.next_candidate()
    assert candidate == 4096
    planner.record(candidate, (True, None))
    assert planner.next_candidate() is None
    assert planner.bounds.highest_confirmed_success is None
    assert planner.bounds.first_confirmed_rejection is None
    assert planner.bounds.stage == "inconclusive"


def test_start_rejection_has_no_fabricated_lower_bound() -> None:
    planner = ContextProbePlanner(
        start_tokens=4096,
        maximum_tokens=8192,
        resolution_tokens=1024,
        safety_maximum_tokens=8192,
    )
    planner.record(4096, (False, False))
    assert planner.bounds.highest_confirmed_success is None
    assert planner.bounds.first_confirmed_rejection == 4096
    assert planner.next_candidate() is None
