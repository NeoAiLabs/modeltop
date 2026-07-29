"""Deterministic retrieval placement, matching, and truncation contracts."""

from modeltop.benchmarks.context_retrieval import (
    RETRIEVAL_INSTRUCTION,
    RetrievalMarkerSpec,
    RetrievalPromptSpec,
    detect_possible_truncation,
    generate_retrieval_key,
    insert_retrieval_markers,
    normalise_retrieval_output,
    score_single_retrieval,
    score_tri_marker_retrieval,
)


def test_seeded_keys_are_stable_and_manual_collisions_fail() -> None:
    def key(*, absolute_attempt: int = 2, manual_key: str | None = None) -> str:
        return generate_retrieval_key(
            random_seed=42,
            target=4096,
            position="middle",
            absolute_attempt=absolute_attempt,
            marker_index=0,
            filler="fictional records",
            manual_key=manual_key,
        )

    first = key()
    assert first == key()
    assert first != key(absolute_attempt=3)
    try:
        key(manual_key="records")
    except ValueError:
        pass
    else:
        raise AssertionError("manual key collision was accepted")


def test_every_named_placement_uses_a_complete_boundary() -> None:
    sections = ("a" * 10, "b" * 10, "c" * 10, "d" * 10)
    for position in (
        "beginning",
        "quarter",
        "middle",
        "three_quarters",
        "end",
        "random",
    ):
        marker = RetrievalMarkerSpec(
            "MODELTOP_RETRIEVAL_KEY", "amber-cloud-1234", position
        )
        body, placements = insert_retrieval_markers(
            sections,
            RetrievalPromptSpec((marker,), RETRIEVAL_INSTRUCTION),
            random_seed=42,
            target=4096,
            absolute_attempt=1,
        )
        assert body.count(marker.line) == 1
        assert 0 <= placements[0][1] <= 100


def test_normalization_exact_containment_and_ambiguity() -> None:
    assert normalise_retrieval_output("  `amber-cloud-1234`\r\n") == "amber-cloud-1234"
    exact = score_single_retrieval(
        "amber-cloud-1234",
        expected="amber-cloud-1234",
        position="middle",
        realised_placement_percent=50,
        case_insensitive=False,
        containment=False,
    )
    contained = score_single_retrieval(
        "answer amber-cloud-1234",
        expected="amber-cloud-1234",
        position="middle",
        realised_placement_percent=50,
        case_insensitive=False,
        containment=True,
    )
    ambiguous = score_single_retrieval(
        "amber-cloud-1234 and cobalt-river-9999",
        expected="amber-cloud-1234",
        position="middle",
        realised_placement_percent=50,
        case_insensitive=False,
        containment=True,
    )
    assert (exact.status, contained.status, ambiguous.status) == (
        "pass",
        "pass",
        "ambiguous",
    )


def test_tri_marker_signals_only_adjacent_cautious_patterns() -> None:
    expected = ("amber-cloud-1000", "cobalt-river-2000", "jade-lake-3000")
    left = score_tri_marker_retrieval(
        "cobalt-river-2000\njade-lake-3000",
        expected=expected,
        realised_placements=(0, 50, 100),
    )
    right = score_tri_marker_retrieval(
        "amber-cloud-1000\ncobalt-river-2000",
        expected=expected,
        realised_placements=(0, 50, 100),
    )
    all_fail = score_tri_marker_retrieval(
        "none",
        expected=expected,
        realised_placements=(0, 50, 100),
    )
    assert detect_possible_truncation(left) == "possible_left_truncation"
    assert detect_possible_truncation(right) == "possible_right_truncation"
    assert detect_possible_truncation(all_fail) is None
