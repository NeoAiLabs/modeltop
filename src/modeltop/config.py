"""Configuration paths, defaults, and application logging."""

import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TextIO

MODELTOP_CONFIG = "MODELTOP_CONFIG"

BUILTIN_CONFIG: Mapping[str, object] = {
    "application": {
        "refresh_interval_seconds": 5,
        "request_timeout_seconds": 5,
        "default_server": "local-vllm",
        "theme": "catppuccin-mocha",
    },
    "hardware": {
        "enabled": True,
        "refresh_interval_seconds": 2,
        "preferred_provider": "auto",
    },
    "benchmarks": {
        "concurrency": {
            "default_levels": [1, 2, 4, 8],
            "requests_per_level": 16,
            "warmup_requests": 2,
            "max_tokens": 256,
            "temperature": 0.0,
            "top_p": 1.0,
            "request_timeout_seconds": 120.0,
            "delay_between_levels_seconds": 3.0,
            "maximum_concurrency": 128,
            "unique_prompt_suffix_per_request": True,
        },
        "context": {
            "default_mode": "sweep",
            "default_lengths": [1024, 4096, 8192, 16384, 32768],
            "context_unit": "tokens",
            "repetitions_per_length": 3,
            "warmup_requests": 1,
            "content_source": "synthetic",
            "base_text": None,
            "random_seed": 42,
            "maximum_output_tokens": 128,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "request_timeout_seconds": 300.0,
            "delay_between_lengths_seconds": 3.0,
            "maximum_context_test_tokens": 262144,
            "warning_threshold_tokens": 65536,
            "prompt_target_tolerance_percent": 1.0,
            "hardware_sample_interval_seconds": 0.5,
            "estimated_input_rate_enabled": True,
            "reuse_prompt": True,
            "unique_prompt_suffix_per_run": False,
            "early_stop_enabled": True,
            "continue_after_timeout": True,
            "probe": {
                "start_tokens": 4096,
                "maximum_tokens": 131072,
                "resolution_tokens": 1024,
            },
            "retrieval": {
                "enabled": False,
                "positions": ["beginning", "middle", "end"],
                "key": None,
                "maximum_output_tokens": 32,
                "case_insensitive_match": False,
                "containment_match": False,
                "truncation_detection": True,
                "regenerate_per_run": True,
            },
        },
        "tool_calling": {
            "default_suite": "full",
            "request_timeout_seconds": 120.0,
        },
        "r0b0bench": {
            "default_profile": "core-subset",
            "default_tests": [
                "canary",
                "latency",
                "concurrency",
                "throughput",
            ],
            "request_timeout_seconds": 600.0,
            "tokenizer_path": None,
            "bfcl_python": None,
            "bfcl_scripts_directory": None,
            "qa_data_path": None,
            "ifeval_data_path": None,
            "humaneval_data_path": None,
            "gsm8k_data_path": None,
        },
        "drafter": {
            "warmup_runs": 1,
            "measured_runs": 5,
            "max_tokens": 256,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "request_timeout_seconds": 300.0,
            "continue_on_error": False,
        },
    },
    "servers": [
        {
            "id": "local-vllm",
            "name": "Local vLLM",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "EMPTY",
            "backend_hint": "vllm",
            "default_model": None,
        },
        {
            "id": "local-8888",
            "name": "Local 8888",
            "base_url": "http://127.0.0.1:8888/v1",
            "api_key": "EMPTY",
            "backend_hint": None,
            "default_model": None,
        },
    ],
}

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


class _ModelTopFileHandler(logging.FileHandler):
    """Marker type for the one handler owned by ModelTop."""


class _ModelTopStderrHandler(logging.StreamHandler[TextIO]):
    """Marker type for the logging fallback owned by ModelTop."""


def user_config_path(home: Path | None = None) -> Path:
    """Return the per-user configuration path."""
    root = Path.home() if home is None else home
    return root / ".config" / "modeltop" / "config.yaml"


def repository_config_path(cwd: Path | None = None) -> Path:
    """Return the source-checkout configuration path."""
    root = Path.cwd() if cwd is None else cwd
    return root / "config" / "modeltop.yaml"


def application_log_path(home: Path | None = None) -> Path:
    """Return the per-user application log path."""
    root = Path.home() if home is None else home
    return root / ".local" / "state" / "modeltop" / "modeltop.log"


def configure_logging(log_path: Path | None = None) -> Path | None:
    """Configure one sanitized file logger, falling back to standard error."""
    logger = logging.getLogger("modeltop")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    target = (application_log_path() if log_path is None else log_path).expanduser()

    owned_handlers = [
        handler
        for handler in logger.handlers
        if isinstance(handler, (_ModelTopFileHandler, _ModelTopStderrHandler))
    ]
    for handler in owned_handlers:
        if (
            isinstance(handler, _ModelTopFileHandler)
            and Path(handler.baseFilename) == target.resolve()
        ):
            return target
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_LOG_FORMAT)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        handler = _ModelTopFileHandler(target, encoding="utf-8")
    except OSError:
        fallback = _ModelTopStderrHandler(sys.stderr)
        fallback.setFormatter(formatter)
        logger.addHandler(fallback)
        logger.exception("Unable to configure file logging at %s", target)
        return None

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return target
