"""Narrow parsing for vLLM speculative-decoding Prometheus counters."""

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

_DRAFT_COUNTER = "vllm:spec_decode_num_draft_tokens_total"
_ACCEPTED_COUNTER = "vllm:spec_decode_num_accepted_tokens_total"
_COUNTER_NAMES = frozenset({_DRAFT_COUNTER, _ACCEPTED_COUNTER})
_SAMPLE_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?"
    r"[ \t]+(?P<value>[^ \t]+)(?:[ \t]+\d+)?[ \t]*$"
)
_LABEL_RE = re.compile(
    r'(?:^|,)(?P<key>[A-Za-z_][A-Za-z0-9_]*)="(?P<value>(?:\\.|[^"\\])*)"'
)


@dataclass(frozen=True, slots=True)
class SpeculativeCounters:
    """Cumulative vLLM speculative-decoding token counters for one model."""

    draft_tokens: int
    accepted_tokens: int


def _parse_labels(raw_labels: str) -> dict[str, str] | None:
    labels: dict[str, str] = {}
    position = 0
    while position < len(raw_labels):
        match = _LABEL_RE.match(raw_labels, position)
        if match is None:
            return None
        value = match.group("value")
        try:
            labels[match.group("key")] = bytes(value, "utf-8").decode("unicode_escape")
        except UnicodeDecodeError:
            return None
        position = match.end()
    return labels


def _parse_counter_value(raw_value: str) -> int | None:
    try:
        value = Decimal(raw_value)
    except InvalidOperation:
        return None
    if not value.is_finite() or value < 0 or value != value.to_integral_value():
        return None
    return int(value)


def parse_vllm_speculative_counters(
    payload: str, model_id: str
) -> SpeculativeCounters | None:
    """Return selected-model speculative counters, or ``None`` when unavailable."""
    totals: dict[str, int] = {}
    for line in payload.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        metric_name = stripped.split("{", 1)[0].split(None, 1)[0]
        if metric_name not in _COUNTER_NAMES:
            continue
        sample = _SAMPLE_RE.fullmatch(stripped)
        if sample is None:
            return None
        labels = _parse_labels(sample.group("labels") or "")
        if labels is None:
            return None
        if labels.get("model_name") != model_id:
            continue
        value = _parse_counter_value(sample.group("value"))
        if value is None:
            return None
        totals[sample.group("name")] = totals.get(sample.group("name"), 0) + value
    if _DRAFT_COUNTER not in totals or _ACCEPTED_COUNTER not in totals:
        return None
    return SpeculativeCounters(
        draft_tokens=totals[_DRAFT_COUNTER],
        accepted_tokens=totals[_ACCEPTED_COUNTER],
    )
