"""Deterministic YAML configuration loading."""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import yaml
from pydantic import ValidationError

from modeltop.config import (
    BUILTIN_CONFIG,
    MODELTOP_CONFIG,
    repository_config_path,
    user_config_path,
)
from modeltop.models import ModelTopConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LoadedConfiguration:
    """A validated configuration and its source, if file-backed."""

    config: ModelTopConfig
    source_path: Path | None


class ConfigurationLoadError(Exception):
    """A configuration failure with separate user and logging text."""

    def __init__(self, user_message: str, detail: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


def _format_validation_error(error: ValidationError) -> str:
    issues: list[str] = []
    for issue in error.errors(
        include_input=False,
        include_url=False,
        include_context=False,
    ):
        location = ".".join(str(part) for part in issue["loc"])
        message = str(issue["msg"])
        issues.append(f"{location}: {message}" if location else message)
    return "; ".join(issues)


def _source_label(source_path: Path | None) -> str:
    return "<built-in defaults>" if source_path is None else str(source_path)


def _raise_load_error(source_path: Path, reason: str) -> NoReturn:
    message = f"Could not load configuration from {source_path}: {reason}"
    raise ConfigurationLoadError(message, message)


def _parse_configuration(
    payload: object,
    source_path: Path | None,
) -> LoadedConfiguration:
    label = _source_label(source_path)
    if not isinstance(payload, Mapping):
        if source_path is None:
            raise ConfigurationLoadError(
                "Built-in configuration has an invalid root",
                "Built-in configuration root is not a mapping",
            )
        _raise_load_error(source_path, "YAML root must be a mapping")

    try:
        config = ModelTopConfig.model_validate(payload)
    except ValidationError as error:
        reason = _format_validation_error(error)
        if source_path is None:
            raise ConfigurationLoadError(
                "Built-in configuration is invalid",
                f"Built-in configuration validation failed: {reason}",
            ) from error
        _raise_load_error(source_path, f"validation failed: {reason}")

    logger.info("Loaded configuration from %s", label)
    return LoadedConfiguration(config=config, source_path=source_path)


def _load_file(source_path: Path) -> LoadedConfiguration:
    try:
        text = source_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        _raise_load_error(source_path, f"unable to read file ({type(error).__name__})")

    try:
        payload: object = yaml.safe_load(text)
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        location = (
            f" at line {mark.line + 1}, column {mark.column + 1}"
            if mark is not None
            else ""
        )
        _raise_load_error(source_path, f"invalid YAML{location}")

    return _parse_configuration(payload, source_path)


def load_configuration(
    *,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    cwd: Path | None = None,
) -> LoadedConfiguration:
    """Load the first authoritative source in environment/user/repository order."""
    environment = os.environ if environ is None else environ
    current_directory = Path.cwd() if cwd is None else cwd
    configured_path = environment.get(MODELTOP_CONFIG, "").strip()

    if configured_path:
        source_path = Path(configured_path).expanduser()
        if not source_path.is_absolute():
            source_path = current_directory / source_path
        source_path = source_path.resolve()
        if not source_path.exists():
            _raise_load_error(source_path, "file does not exist")
        return _load_file(source_path)

    candidates = (
        user_config_path(home),
        repository_config_path(current_directory),
    )
    for source_path in candidates:
        if source_path.exists():
            return _load_file(source_path)

    return _parse_configuration(BUILTIN_CONFIG, None)
