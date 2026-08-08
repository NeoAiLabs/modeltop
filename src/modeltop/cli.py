"""Command-line entry point for the dashboard and local data management."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from modeltop.services.r0b0bench_datasets import (
    DEFAULT_R0B0BENCH_DATASET_ROOT,
    R0b0benchDatasetError,
    install_r0b0bench_assets,
    r0b0bench_asset_status,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mtop")
    commands = parser.add_subparsers(dest="command")
    datasets = commands.add_parser(
        "datasets", help="install or inspect benchmark datasets"
    )
    actions = datasets.add_subparsers(dest="datasets_action", required=True)
    for action in ("install", "status"):
        action_parser = actions.add_parser(action)
        action_parser.add_argument("suite", choices=("r0b0bench",))
        action_parser.add_argument(
            "--root",
            type=Path,
            default=DEFAULT_R0B0BENCH_DATASET_ROOT,
            help="dataset installation root",
        )
    return parser


def _print_status(root: Path) -> bool:
    statuses = r0b0bench_asset_status(root)
    for row in statuses:
        print(f"{row.state.upper():9} {row.label}: {row.path}")
    return all(row.state == "installed" for row in statuses)


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the TUI by default or execute one explicit management command."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        from modeltop.app import run

        run()
        return 0
    parsed = _parser().parse_args(arguments)
    if parsed.command != "datasets":
        return 2
    root = Path(parsed.root).expanduser()
    if parsed.datasets_action == "status":
        return 0 if _print_status(root) else 1
    try:
        install_r0b0bench_assets(root)
    except R0b0benchDatasetError as error:
        print(f"r0b0bench asset installation failed: {error}", file=sys.stderr)
        return 1
    _print_status(root)
    print(f"Manifest: {root / 'manifest.json'}")
    return 0
