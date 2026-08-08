"""Integrity, CLI, and prerequisite tests for managed r0b0bench assets."""
# pyright: reportPrivateUsage=false

import hashlib
import json
from pathlib import Path
from typing import cast

import httpx
import pytest

import modeltop.cli as cli_module
import modeltop.services.r0b0bench_datasets as datasets_module
from modeltop.benchmarks.models import R0b0benchBenchmarkConfig
from modeltop.benchmarks.r0b0bench import resolve_r0b0bench_prerequisites
from modeltop.services.r0b0bench_datasets import (
    R0b0benchAssetId,
    R0b0benchAssetSpec,
    R0b0benchDatasetError,
    install_r0b0bench_assets,
    r0b0bench_asset_status,
    r0b0bench_installed_paths,
)

_ASSET_IDS: tuple[R0b0benchAssetId, ...] = (
    "qa",
    "ifeval",
    "humaneval",
    "gsm8k",
    "bfcl_run",
    "bfcl_ast",
)


def _fixture_specs() -> tuple[tuple[R0b0benchAssetSpec, ...], dict[str, bytes]]:
    payloads = {asset_id: f"fixture:{asset_id}\n".encode() for asset_id in _ASSET_IDS}
    paths = {
        "qa": Path("qa/arc_easy_test.jsonl"),
        "ifeval": Path("ifeval/input_data.jsonl"),
        "humaneval": Path("humaneval/HumanEval.jsonl"),
        "gsm8k": Path("gsm8k/test.jsonl"),
        "bfcl_run": Path("bfcl/bfcl_run.py"),
        "bfcl_ast": Path("bfcl/bfcl_ast_run.py"),
    }
    specs = tuple(
        R0b0benchAssetSpec(
            asset_id=asset_id,
            label=asset_id,
            relative_path=paths[asset_id],
            source_url=f"https://fixtures.invalid/{asset_id}",
            source_sha256=hashlib.sha256(payloads[asset_id]).hexdigest(),
            source_revision="fixture",
            output_sha256=hashlib.sha256(payloads[asset_id]).hexdigest(),
            output_size=len(payloads[asset_id]),
            row_count=None,
            transform="direct",
            license_name="fixture",
            license_url="https://fixtures.invalid/license",
        )
        for asset_id in _ASSET_IDS
    )
    return specs, payloads


def test_installer_writes_validated_assets_manifest_and_registry_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs, payloads = _fixture_specs()
    monkeypatch.setattr(datasets_module, "_ASSETS", specs)

    def respond(request: httpx.Request) -> httpx.Response:
        asset_id = cast(R0b0benchAssetId, request.url.path.removeprefix("/"))
        return httpx.Response(200, content=payloads[asset_id])

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        statuses = install_r0b0bench_assets(tmp_path, client=client)

    assert all(row.state == "installed" for row in statuses)
    assert all((row.path.stat().st_mode & 0o777) == 0o600 for row in statuses)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert [row["id"] for row in manifest["assets"]] == list(_ASSET_IDS)
    installed = r0b0bench_installed_paths(tmp_path)
    assert installed["qa_data_path"].endswith("qa/arc_easy_test.jsonl")
    assert installed["bfcl_scripts_directory"].endswith("bfcl")

    statuses[0].path.write_text("corrupted")
    states = {row.asset_id: row.state for row in r0b0bench_asset_status(tmp_path)}
    assert states["qa"] == "invalid"
    assert "qa_data_path" not in r0b0bench_installed_paths(tmp_path)


def test_installer_rejects_source_checksum_mismatch_without_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs, _ = _fixture_specs()
    monkeypatch.setattr(datasets_module, "_ASSETS", specs)

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"wrong")

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as client,
        pytest.raises(R0b0benchDatasetError, match="checksum mismatch"),
    ):
        install_r0b0bench_assets(tmp_path, client=client)
    assert not (tmp_path / specs[0].relative_path).exists()
    assert not (tmp_path / "manifest.json").exists()


def test_installed_quality_data_resolves_without_manual_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qa_path = tmp_path / "qa.jsonl"
    qa_path.write_text("{}\n")
    monkeypatch.setattr(
        datasets_module,
        "r0b0bench_installed_paths",
        lambda: {"qa_data_path": str(qa_path)},
    )
    config = R0b0benchBenchmarkConfig(
        profile="core-subset",
        selected_lanes=("qa",),
    )

    check = resolve_r0b0bench_prerequisites(config)

    assert check.issues == ()
    assert check.child_environment["R0B0BENCH_QA_DATA"] == str(qa_path)


def test_cli_status_reports_missing_assets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli_module.main(
        ["datasets", "status", "r0b0bench", "--root", str(tmp_path)]
    )

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "MISSING" in output
    assert "QA / ARC-Easy" in output
