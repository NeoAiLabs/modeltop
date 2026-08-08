"""Durable redacted archive contracts."""
# pyright: reportPrivateUsage=false

import json
import runpy
from dataclasses import replace
from pathlib import Path
from typing import cast

from modeltop.benchmarks.models import R0b0benchMetric
from modeltop.services.result_archive import ResultArchive
from modeltop.widgets.archived_comparison import _rows


def test_speed_result_round_trips_without_sensitive_content(tmp_path: Path) -> None:
    result = runpy.run_path("tests/test_result_export.py")["_result"]()
    archive = ResultArchive(tmp_path / "history")

    first = archive.archive_result(result)
    second = archive.archive_result(result)
    restarted = archive.load_archive()

    assert len(first.entries) == len(second.entries) == len(restarted.entries) == 1
    entry = restarted.entries[0]
    assert entry.kind == "speed-test"
    document_path = archive.directory / entry.document_name
    payload = json.loads(document_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    rendered = json.dumps(payload, allow_nan=False)
    assert "exported benchmark prompt" not in rendered
    assert "generated-secret-text" not in rendered
    assert "GPU-secret-uuid" not in rendered


def test_r0b0bench_archive_redacts_evidence_and_compares_allowlisted_metrics(
    tmp_path: Path,
) -> None:
    factory = runpy.run_path("tests/test_r0b0bench_widgets.py")["_result"]
    baseline = factory(tmp_path)
    candidate_row = replace(
        baseline.lanes[0],
        metrics=(
            R0b0benchMetric("passed", True, None),
            R0b0benchMetric("cases", 2, "count"),
        ),
    )
    candidate = replace(
        baseline,
        benchmark_id="r0b0bench-20260804T120001Z-cafebabe",
        upstream_run_id="r0b0bench-20260804T120001Z-cafebabe",
        lanes=(candidate_row,),
    )
    archive = ResultArchive(tmp_path / "history")
    snapshot = archive.archive_result(baseline)
    snapshot = archive.archive_result(candidate)

    assert [entry.kind for entry in snapshot.entries] == [
        "r0b0bench",
        "r0b0bench",
    ]
    documents = snapshot.documents
    first = documents[baseline.benchmark_id]
    second = documents[candidate.benchmark_id]
    rendered = json.dumps(
        {
            "baseline": dict(first.details),
            "candidate": dict(second.details),
        },
        allow_nan=False,
    )
    assert "PRIVATE_EVIDENCE" not in rendered
    assert "127.0.0.1" not in rendered
    assert str(tmp_path) not in rendered
    assert "server_endpoint" not in rendered
    upstream = cast(dict[str, object], first.details["upstream"])
    assert upstream["commit"] == "d5ed83d8499a952546cf458e090be42ee4a48eef"
    comparison = _rows(first, second)
    assert ("canary · cases", 1, 2) in comparison
    assert any(label == "Invalid for publish" for label, _, _ in comparison)
