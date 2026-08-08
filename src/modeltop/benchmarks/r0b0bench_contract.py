"""Dependency-free r0b0bench profile and lane selection contract."""

from typing import Literal, get_args

R0b0benchProfile = Literal["core-subset", "core", "systems"]
R0b0benchLaneId = Literal[
    "canary",
    "bfcl_mt",
    "bfcl_ast",
    "latency",
    "concurrency",
    "throughput",
    "niah",
    "qa",
    "ifeval",
    "humaneval",
    "gsm8k",
    "perf",
]

R0B0BENCH_PROFILE_ORDER: tuple[R0b0benchProfile, ...] = (
    "core-subset",
    "core",
    "systems",
)
R0B0BENCH_SYSTEMS_ORDER: tuple[R0b0benchLaneId, ...] = (
    "canary",
    "bfcl_mt",
    "bfcl_ast",
    "latency",
    "concurrency",
    "throughput",
    "niah",
)
R0B0BENCH_QUALITY_ORDER: tuple[R0b0benchLaneId, ...] = (
    "qa",
    "ifeval",
    "humaneval",
    "gsm8k",
)
R0B0BENCH_PERF_COMPONENTS: tuple[R0b0benchLaneId, ...] = (
    "latency",
    "concurrency",
    "throughput",
)
R0B0BENCH_CANONICAL_ORDER: tuple[R0b0benchLaneId, ...] = (
    *R0B0BENCH_SYSTEMS_ORDER,
    *R0B0BENCH_QUALITY_ORDER,
)
R0B0BENCH_ALL_LANES: tuple[R0b0benchLaneId, ...] = get_args(R0b0benchLaneId)


def r0b0bench_profile_lanes(
    profile: R0b0benchProfile,
) -> tuple[R0b0benchLaneId, ...]:
    """Return the upstream canonical lane order for a profile."""
    if profile == "systems":
        return R0B0BENCH_SYSTEMS_ORDER
    if profile in {"core-subset", "core"}:
        return R0B0BENCH_CANONICAL_ORDER
    raise ValueError("unknown r0b0bench profile")


def r0b0bench_ordered_selection(
    profile: R0b0benchProfile,
    lanes: tuple[R0b0benchLaneId, ...],
) -> tuple[R0b0benchLaneId, ...]:
    """Return selected lanes in the exact order emitted by upstream rc2."""
    validate_r0b0bench_selection(profile, lanes)
    selected: set[R0b0benchLaneId] = set(lanes)
    ordered: tuple[R0b0benchLaneId, ...] = tuple(
        lane for lane in r0b0bench_profile_lanes(profile) if lane in selected
    )
    if "perf" in selected:
        return (*ordered, "perf")
    return ordered


def validate_r0b0bench_selection(
    profile: R0b0benchProfile,
    lanes: tuple[R0b0benchLaneId, ...],
) -> None:
    """Validate a non-empty, profile-compatible rc2 lane selection."""
    if not lanes:
        raise ValueError("select at least one r0b0bench test")
    if len(set(lanes)) != len(lanes):
        raise ValueError("r0b0bench tests must be unique")
    known = set(R0B0BENCH_ALL_LANES)
    if any(lane not in known for lane in lanes):
        raise ValueError("unknown r0b0bench test")
    allowed = {*r0b0bench_profile_lanes(profile), "perf"}
    if any(lane not in allowed for lane in lanes):
        raise ValueError("selected test is unavailable for the profile")
    selected = set(lanes)
    if "perf" in selected and selected.intersection(R0B0BENCH_PERF_COMPONENTS):
        raise ValueError("perf cannot be combined with its component tests")
