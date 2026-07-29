"""Pure duplicate-free exponential and binary Context probe planning."""

from collections.abc import Iterable

from modeltop.benchmarks.models import ContextProbeBounds


class ContextProbePlanner:
    """Plan resolution-aligned candidates without treating unknowns as evidence."""

    def __init__(
        self,
        *,
        start_tokens: int,
        maximum_tokens: int,
        resolution_tokens: int,
        safety_maximum_tokens: int,
    ) -> None:
        maximum = min(maximum_tokens, safety_maximum_tokens)
        if min(start_tokens, resolution_tokens, maximum) <= 0:
            raise ValueError("probe values must be positive")
        if start_tokens % resolution_tokens or maximum % resolution_tokens:
            raise ValueError("probe bounds must be resolution multiples")
        if maximum <= start_tokens:
            raise ValueError("probe maximum must exceed start")
        self._start = start_tokens
        self._maximum = maximum
        self._resolution = resolution_tokens
        self._success: int | None = None
        self._rejection: int | None = None
        self._last: int | None = None
        self._attempted: list[int] = []
        self._stage: str = "exponential"
        self._inconclusive_reason: str | None = None
        self._next: int | None = start_tokens

    @property
    def bounds(self) -> ContextProbeBounds:
        return ContextProbeBounds(
            highest_confirmed_success=self._success,
            first_confirmed_rejection=self._rejection,
            last_candidate=self._last,
            resolution_tokens=self._resolution,
            attempted_targets=tuple(self._attempted),
            stage=self._stage,  # type: ignore[arg-type]
            inconclusive_reason=self._inconclusive_reason,
        )

    def next_candidate(self) -> int | None:
        """Return the current unique candidate, or None when planning is terminal."""
        candidate = self._next
        if candidate is not None and candidate in self._attempted:
            raise RuntimeError("probe planner attempted to duplicate a candidate")
        return candidate

    def record(self, candidate: int, outcomes: Iterable[bool | None]) -> None:
        """Record all-accepted, all-rejected, or inconclusive repeated outcomes."""
        if candidate != self._next:
            raise ValueError("candidate does not match the planned probe target")
        values = tuple(outcomes)
        if not values:
            raise ValueError("at least one probe outcome is required")
        self._attempted.append(candidate)
        self._last = candidate
        if all(value is True for value in values):
            self._success = candidate
            self._advance_after_success(candidate)
        elif all(value is False for value in values):
            self._rejection = candidate
            self._advance_after_rejection()
        else:
            self.mark_inconclusive("Probe outcome was mixed or acceptance was unknown.")

    def _advance_after_success(self, candidate: int) -> None:
        if candidate >= self._maximum:
            self._stage = "complete"
            self._next = None
            return
        if self._rejection is None:
            next_candidate = min(candidate * 2, self._maximum)
            next_candidate = next_candidate // self._resolution * self._resolution
            if next_candidate <= candidate:
                next_candidate = min(candidate + self._resolution, self._maximum)
            self._next = next_candidate
            return
        self._choose_binary_candidate()

    def _advance_after_rejection(self) -> None:
        if self._success is None:
            self._stage = "complete"
            self._next = None
            return
        self._choose_binary_candidate()

    def _choose_binary_candidate(self) -> None:
        assert self._success is not None and self._rejection is not None
        if self._rejection - self._success <= self._resolution:
            self._stage = "complete"
            self._next = None
            return
        midpoint = (self._success + self._rejection) // 2
        candidate = midpoint // self._resolution * self._resolution
        if candidate <= self._success:
            candidate = self._success + self._resolution
        if candidate >= self._rejection or candidate in self._attempted:
            self._stage = "complete"
            self._next = None
            return
        self._stage = "binary"
        self._next = candidate

    def mark_inconclusive(self, reason: str) -> None:
        """Stop without moving either confirmed bound."""
        self._stage = "inconclusive"
        self._inconclusive_reason = reason
        self._next = None
