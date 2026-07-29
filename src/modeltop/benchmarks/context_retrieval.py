"""Pure deterministic retrieval markers, placement, and bounded scoring."""

import hashlib
import random
import re
from dataclasses import dataclass

from modeltop.benchmarks.models import (
    ContextRetrievalResult,
    RetrievalPosition,
    RetrievalStatus,
)

RETRIEVAL_INSTRUCTION = (
    "What is the exact value of MODELTOP_RETRIEVAL_KEY?\n\n"
    "Return only the value. Do not explain."
)
TRI_MARKER_INSTRUCTION = (
    "Return the BEGIN_MARKER, MIDDLE_MARKER, and END_MARKER values on three "
    "separate lines, in that order. Do not explain."
)
ADJECTIVES = (
    "amber",
    "cobalt",
    "copper",
    "crimson",
    "ivory",
    "jade",
    "silver",
    "violet",
)
NOUNS = (
    "cloud",
    "comet",
    "field",
    "harbor",
    "lake",
    "orchid",
    "pine",
    "river",
)
KEY_PATTERN = re.compile(r"[A-Za-z]+-[A-Za-z]+-[0-9]{4}")
PREVIEW_CHARACTERS = 512


@dataclass(frozen=True, slots=True)
class RetrievalMarkerSpec:
    """One marker assignment to insert at a complete-fragment boundary."""

    marker: str
    key: str
    position: RetrievalPosition

    @property
    def line(self) -> str:
        return f"{self.marker}: {self.key}"


@dataclass(frozen=True, slots=True)
class RetrievalPromptSpec:
    """Ephemeral marker set and exact final instruction for one request."""

    markers: tuple[RetrievalMarkerSpec, ...]
    instruction: str


def _seed_bytes(
    random_seed: int,
    target: int,
    position: RetrievalPosition,
    absolute_attempt: int,
    marker_index: int,
) -> bytes:
    value = f"{random_seed}:{target}:{position}:{absolute_attempt}:{marker_index}"
    return hashlib.sha256(value.encode("utf-8")).digest()[:8]


def generate_retrieval_key(
    *,
    random_seed: int,
    target: int,
    position: RetrievalPosition,
    absolute_attempt: int,
    marker_index: int,
    filler: str,
    manual_key: str | None = None,
    regenerate_per_run: bool = True,
) -> str:
    """Generate a deterministic absent key, or validate a fixed manual key."""
    if manual_key is not None:
        key = manual_key.strip()
        if key in filler:
            raise ValueError("The configured retrieval key already occurs in filler.")
        return key
    stable_attempt = absolute_attempt if regenerate_per_run else 0
    rng = random.Random(
        int.from_bytes(
            _seed_bytes(
                random_seed,
                target,
                position,
                stable_attempt,
                marker_index,
            ),
            "big",
        )
    )
    for _ in range(32):
        key = (
            f"{ADJECTIVES[rng.randrange(len(ADJECTIVES))]}-"
            f"{NOUNS[rng.randrange(len(NOUNS))]}-{rng.randrange(1000, 10000)}"
        )
        if key not in filler:
            return key
    raise ValueError("Unable to generate a retrieval key absent from prompt filler.")


def boundary_index_for_position(
    section_lengths: tuple[int, ...],
    position: RetrievalPosition,
    *,
    random_seed: int,
    target: int,
    absolute_attempt: int,
    marker_index: int,
) -> int:
    """Choose a deterministic complete-fragment boundary index."""
    count = len(section_lengths)
    if position == "beginning":
        return 0
    if position == "end":
        return count
    if position == "random":
        digest = _seed_bytes(
            random_seed,
            target,
            position,
            absolute_attempt,
            marker_index,
        )
        return int.from_bytes(digest, "big") % (count + 1)
    fraction = {
        "quarter": 0.25,
        "middle": 0.5,
        "three_quarters": 0.75,
    }[position]
    total = sum(section_lengths)
    candidates = [0]
    running = 0
    for length in section_lengths:
        running += length
        candidates.append(running)
    desired = total * fraction
    return min(
        range(len(candidates)), key=lambda index: abs(candidates[index] - desired)
    )


def insert_retrieval_markers(
    sections: tuple[str, ...],
    prompt_spec: RetrievalPromptSpec,
    *,
    random_seed: int,
    target: int,
    absolute_attempt: int,
) -> tuple[str, tuple[tuple[str, float], ...]]:
    """Insert each marker at its chosen boundary and report realised placement."""
    section_lengths = tuple(len(section) for section in sections)
    total_filler = sum(section_lengths)
    placements: list[tuple[int, int, RetrievalMarkerSpec]] = []
    for marker_index, marker in enumerate(prompt_spec.markers):
        boundary = boundary_index_for_position(
            section_lengths,
            marker.position,
            random_seed=random_seed,
            target=target,
            absolute_attempt=absolute_attempt,
            marker_index=marker_index,
        )
        placements.append((boundary, marker_index, marker))
    placements.sort(key=lambda item: (item[0], item[1]))
    by_boundary: dict[int, list[RetrievalMarkerSpec]] = {}
    for boundary, _, marker in placements:
        by_boundary.setdefault(boundary, []).append(marker)

    parts: list[str] = []
    realised: list[tuple[str, float]] = []
    filler_before = 0
    for boundary in range(len(sections) + 1):
        for marker in by_boundary.get(boundary, []):
            if parts and not parts[-1].endswith("\n"):
                parts.append("\n")
            parts.append(marker.line)
            parts.append("\n")
            percentage = (
                0.0 if total_filler == 0 else filler_before / total_filler * 100
            )
            realised.append((marker.marker, percentage))
        if boundary < len(sections):
            parts.append(sections[boundary])
            filler_before += len(sections[boundary])
    return "".join(parts), tuple(realised)


def measure_retrieval_placements(
    user_content: str,
    prompt_spec: RetrievalPromptSpec,
) -> tuple[float, ...]:
    """Measure marker placement over filler only from an ephemeral built prompt."""
    body, separator, _ = user_content.rpartition("\n\n" + prompt_spec.instruction)
    if not separator:
        body = user_content
    marker_lines = {
        marker.line: index for index, marker in enumerate(prompt_spec.markers)
    }
    filler_before = [0] * len(prompt_spec.markers)
    found = [False] * len(prompt_spec.markers)
    filler_count = 0
    for line in body.splitlines(keepends=True):
        assignment = line.strip()
        marker_index = marker_lines.get(assignment)
        if marker_index is None:
            filler_count += len(line)
        else:
            filler_before[marker_index] = filler_count
            found[marker_index] = True
    if not all(found):
        raise ValueError("Built retrieval prompt is missing a configured marker.")
    if filler_count == 0:
        return tuple(0.0 for _ in filler_before)
    return tuple(value / filler_count * 100 for value in filler_before)


def normalise_retrieval_output(value: str) -> str:
    """Normalize line endings, one quote pair, and all whitespace runs."""
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if (
        len(normalized) >= 2
        and normalized[0] == normalized[-1]
        and normalized[0] in {"'", '"', "`"}
    ):
        normalized = normalized[1:-1].strip()
    return " ".join(normalized.split())


def _previews(output: str) -> tuple[str, str, bool]:
    normalized = normalise_retrieval_output(output)
    truncated = len(output) > PREVIEW_CHARACTERS or len(normalized) > PREVIEW_CHARACTERS
    return (
        output[:PREVIEW_CHARACTERS],
        normalized[:PREVIEW_CHARACTERS],
        truncated,
    )


def score_single_retrieval(
    output: str | None,
    *,
    expected: str,
    position: RetrievalPosition,
    realised_placement_percent: float,
    case_insensitive: bool,
    containment: bool,
) -> ContextRetrievalResult:
    """Score exact or cautiously unambiguous containment retrieval."""
    if output is None:
        return ContextRetrievalResult(
            marker="MODELTOP_RETRIEVAL_KEY",
            position=position,
            realised_placement_percent=realised_placement_percent,
            expected_value=expected,
            raw_preview="",
            normalized_preview="",
            preview_truncated=False,
            status="error",
        )
    raw_preview, normalized_preview, preview_truncated = _previews(output)
    normalized = normalise_retrieval_output(output)
    comparable = normalized.casefold() if case_insensitive else normalized
    expected_comparable = expected.casefold() if case_insensitive else expected
    candidates = KEY_PATTERN.findall(normalized)
    candidate_comparables = (
        [candidate.casefold() for candidate in candidates]
        if case_insensitive
        else candidates
    )
    if comparable == expected_comparable:
        status: RetrievalStatus = "pass"
    elif containment and expected_comparable in comparable:
        other_candidates = [
            candidate
            for candidate in candidate_comparables
            if candidate != expected_comparable
        ]
        status = "ambiguous" if other_candidates else "pass"
    elif expected_comparable in comparable and any(
        candidate != expected_comparable for candidate in candidate_comparables
    ):
        status = "ambiguous"
    else:
        status = "fail"
    return ContextRetrievalResult(
        marker="MODELTOP_RETRIEVAL_KEY",
        position=position,
        realised_placement_percent=realised_placement_percent,
        expected_value=expected,
        raw_preview=raw_preview,
        normalized_preview=normalized_preview,
        preview_truncated=preview_truncated,
        status=status,
    )


def score_tri_marker_retrieval(
    output: str | None,
    *,
    expected: tuple[str, str, str],
    realised_placements: tuple[float, float, float],
) -> tuple[ContextRetrievalResult, ...]:
    """Score ordered tri-marker output without inferring a truncation direction."""
    positions: tuple[RetrievalPosition, ...] = ("beginning", "middle", "end")
    markers = ("BEGIN_MARKER", "MIDDLE_MARKER", "END_MARKER")
    if output is None:
        statuses: tuple[RetrievalStatus, ...] = ("error", "error", "error")
        raw_preview = normalized_preview = ""
        preview_truncated = False
    else:
        raw_preview, normalized_preview, preview_truncated = _previews(output)
        candidates = KEY_PATTERN.findall(normalise_retrieval_output(output))
        unexpected = any(candidate not in expected for candidate in candidates)
        present_expected = [
            candidate for candidate in candidates if candidate in expected
        ]
        expected_subset = [key for key in expected if key in present_expected]
        order_valid = present_expected == expected_subset
        status_values: list[RetrievalStatus] = []
        for key in expected:
            count = candidates.count(key)
            if count == 1 and not unexpected and order_valid:
                status_values.append("pass")
            elif count > 1 or unexpected:
                status_values.append("ambiguous")
            else:
                status_values.append("fail")
        statuses = tuple(status_values)  # type: ignore[assignment]
    return tuple(
        ContextRetrievalResult(
            marker=marker,
            position=position,
            realised_placement_percent=placement,
            expected_value=key,
            raw_preview=raw_preview,
            normalized_preview=normalized_preview,
            preview_truncated=preview_truncated,
            status=status,
        )
        for marker, position, key, placement, status in zip(
            markers, positions, expected, realised_placements, statuses, strict=True
        )
    )


def detect_possible_truncation(
    results: tuple[ContextRetrievalResult, ...],
) -> str | None:
    """Return only cautious adjacent tri-marker truncation observations."""
    if len(results) != 3:
        return None
    statuses = tuple(result.status for result in results)
    if statuses == ("fail", "pass", "pass"):
        return "possible_left_truncation"
    if statuses == ("pass", "pass", "fail"):
        return "possible_right_truncation"
    return None
