"""Read-only comparison for two redacted archived benchmark documents."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeGuard, cast

from textual.widgets import Static

from modeltop.services.result_archive import ArchivedResultDocument


class ArchivedComparisonPanel(Static):
    """Render family-safe archive comparisons without reconstructing configs."""

    def update_documents(
        self, baseline: ArchivedResultDocument, candidate: ArchivedResultDocument
    ) -> None:
        differing = (
            baseline.entry.configuration_fingerprint
            != candidate.entry.configuration_fingerprint
        )
        lines = [
            f"{baseline.entry.kind.upper()} COMPARISON",
            f"BASELINE  {baseline.entry.model_id} · {baseline.entry.result_id[-12:]}",
            f"CANDIDATE {candidate.entry.model_id} · {candidate.entry.result_id[-12:]}",
        ]
        if differing:
            lines.append("CONFIGURATION DIFFERS")
        lines.extend(["", "METRIC | BASELINE | CANDIDATE | Δ = CANDIDATE - BASELINE"])
        for label, before, after in _rows(baseline, candidate):
            before_text = _display(before)
            after_text = _display(after)
            lines.append(
                f"{label} | {before_text} | {after_text} | {_delta(before, after)}"
            )
        self.update("\n".join(lines))


def _rows(
    baseline: ArchivedResultDocument, candidate: ArchivedResultDocument
) -> list[tuple[str, object, object]]:
    kind = baseline.entry.kind
    if kind in {"speed-test", "drafter"}:
        rows = [
            (
                "Success count",
                baseline.entry.summary.get("successful_runs"),
                candidate.entry.summary.get("successful_runs"),
            ),
            (
                "TTFT p95 ms",
                _nested(baseline.details, "aggregates", "ttft_ms", "p95"),
                _nested(candidate.details, "aggregates", "ttft_ms", "p95"),
            ),
            (
                "Mean output tok/s",
                baseline.entry.summary.get("mean_output_tokens_per_second"),
                candidate.entry.summary.get("mean_output_tokens_per_second"),
            ),
            (
                "Mean total duration s",
                _nested(baseline.details, "aggregates", "total_duration_s", "mean"),
                _nested(candidate.details, "aggregates", "total_duration_s", "mean"),
            ),
        ]
        if kind == "drafter":
            rows.append(
                (
                    "Mean acceptance rate",
                    _nested(baseline.details, "aggregates", "acceptance_rate", "mean"),
                    _nested(candidate.details, "aggregates", "acceptance_rate", "mean"),
                )
            )
        return rows
    if kind == "concurrency":
        return _series_rows(
            baseline,
            candidate,
            "levels",
            "concurrency",
            (
                "success_rate_percent",
                "requests_per_second",
                "aggregate_output_tokens_per_second",
            ),
            ("ttft_ms", "p95"),
            ("latency_seconds", "p95"),
        )
    if kind == "context":
        return _series_rows(
            baseline,
            candidate,
            "lengths",
            "target_length",
            ("success_rate_percent", "context_rejected_requests"),
            ("ttft_ms", "p95"),
            ("output_tokens_per_second", "mean"),
            ("estimated_input_tokens_per_second", "mean"),
        )
    if kind == "r0b0bench":
        return _r0b0bench_rows(baseline, candidate)
    return [
        (
            "Completion rate",
            _nested(baseline.details, "coverage", "completion_rate_percent"),
            _nested(candidate.details, "coverage", "completion_rate_percent"),
        ),
        (
            "Final score",
            baseline.entry.summary.get("final_score"),
            candidate.entry.summary.get("final_score"),
        ),
        (
            "Points",
            baseline.entry.summary.get("total_points"),
            candidate.entry.summary.get("total_points"),
        ),
        (
            "Max points",
            baseline.entry.summary.get("max_points"),
            candidate.entry.summary.get("max_points"),
        ),
        (
            "Safety gate",
            _nested(baseline.details, "scoring", "safety_gate_passed"),
            _nested(candidate.details, "scoring", "safety_gate_passed"),
        ),
        (
            "Median turn ms",
            _nested(baseline.details, "scoring", "median_turn_ms"),
            _nested(candidate.details, "scoring", "median_turn_ms"),
        ),
    ]


def _r0b0bench_rows(
    baseline: ArchivedResultDocument,
    candidate: ArchivedResultDocument,
) -> list[tuple[str, object, object]]:
    summary_fields = (
        ("Selected", "selected_count"),
        ("Completed", "completed_count"),
        ("Pass", "pass_count"),
        ("Fail", "fail_count"),
        ("Error", "error_count"),
        ("Infrastructure errors", "infra_errors_total"),
        ("Invalid for publish", "invalid_for_publish"),
    )
    rows: list[tuple[str, object, object]] = [
        (
            label,
            baseline.entry.summary.get(field),
            candidate.entry.summary.get(field),
        )
        for label, field in summary_fields
    ]

    def metrics(
        document: ArchivedResultDocument,
    ) -> tuple[dict[tuple[str, str], object], tuple[tuple[str, str], ...]]:
        values: dict[tuple[str, str], object] = {}
        order: list[tuple[str, str]] = []
        lanes = document.details.get("lanes")
        if not isinstance(lanes, list):
            return values, ()
        for lane_value in cast(list[object], lanes):
            if not isinstance(lane_value, dict):
                continue
            lane = cast(dict[str, object], lane_value)
            lane_id = lane.get("lane_id")
            metric_values = lane.get("metrics")
            if not isinstance(lane_id, str) or not isinstance(metric_values, list):
                continue
            for metric_value in cast(list[object], metric_values):
                if not isinstance(metric_value, dict):
                    continue
                metric = cast(dict[str, object], metric_value)
                name = metric.get("name")
                value = metric.get("value")
                if not isinstance(name, str) or not isinstance(
                    value, (bool, int, float)
                ):
                    continue
                key = (lane_id, name)
                if key not in values:
                    order.append(key)
                values[key] = value
        return values, tuple(order)

    before, before_order = metrics(baseline)
    after, after_order = metrics(candidate)
    metric_order = (*before_order, *(key for key in after_order if key not in before))
    rows.extend(
        (f"{lane} · {metric}", before.get((lane, metric)), after.get((lane, metric)))
        for lane, metric in metric_order
    )
    return rows


def _series_rows(
    baseline: ArchivedResultDocument,
    candidate: ArchivedResultDocument,
    series: str,
    identifier: str,
    *metrics: tuple[str, ...],
) -> list[tuple[str, object, object]]:
    before = _indexed_series(baseline.details.get(series), identifier)
    after = _indexed_series(candidate.details.get(series), identifier)
    rows: list[tuple[str, object, object]] = []
    for key in sorted(set(before) | set(after)):
        for metric in metrics:
            label = f"{identifier} {key} · {' '.join(metric)}"
            rows.append(
                (
                    label,
                    _row_value(before.get(key), metric),
                    _row_value(after.get(key), metric),
                )
            )
    return rows


def _indexed_series(
    value: object, identifier: str
) -> dict[int | float | str, Mapping[str, object]]:
    indexed: dict[int | float | str, Mapping[str, object]] = {}
    if not isinstance(value, list):
        return indexed
    for row_value in cast(list[object], value):
        if not isinstance(row_value, dict):
            continue
        row = cast(dict[str, object], row_value)
        key = row.get(identifier)
        if isinstance(key, (int, float, str)) and not isinstance(key, bool):
            indexed[key] = row
    return indexed


def _is_object_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
    return isinstance(value, Mapping)


def _row_value(row: Mapping[str, object] | None, path: tuple[str, ...]) -> object:
    value: object = row
    for item in path:
        if not _is_object_mapping(value):
            return None
        value = value.get(item)
    return value


def _nested(value: Mapping[str, object], *path: str) -> object:
    return _row_value(value, path)


def _display(value: object) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _delta(before: object, after: object) -> str:
    if (
        not isinstance(before, (int, float))
        or not isinstance(after, (int, float))
        or isinstance(before, bool)
        or isinstance(after, bool)
    ):
        return "--"
    return f"{after - before:+.2f}"
