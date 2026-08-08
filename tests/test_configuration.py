"""Configuration validation, precedence, and logging tests."""

import logging
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from modeltop.config import BUILTIN_CONFIG, MODELTOP_CONFIG, configure_logging
from modeltop.models import (
    ConcurrencyBenchmarkDefaultsConfig,
    ModelTopConfig,
    ServerConfig,
)
from modeltop.services.configuration import (
    ConfigurationLoadError,
    load_configuration,
)


def _yaml(server_id: str, *, api_key: str = "EMPTY") -> str:
    return (
        "application:\n"
        "  refresh_interval_seconds: 5\n"
        "  request_timeout_seconds: 5\n"
        f"  default_server: {server_id}\n"
        "servers:\n"
        f"  - id: {server_id}\n"
        f"    name: {server_id}\n"
        "    base_url: http://127.0.0.1:8000/v1\n"
        f"    api_key: {api_key}\n"
        "    backend_hint: vllm\n"
        "    default_model: null\n"
    )


@pytest.mark.parametrize(
    "theme",
    (
        "catppuccin-latte",
        "catppuccin-frappe",
        "catppuccin-macchiato",
        "catppuccin-mocha",
    ),
)
def test_application_theme_accepts_supported_catppuccin_flavours(theme: str) -> None:
    """Every registered Textual Catppuccin theme is accepted verbatim."""
    config = ModelTopConfig.model_validate(
        {
            "application": {"theme": theme},
            "servers": [
                {"id": "server", "name": "Server", "base_url": "http://server"}
            ],
        }
    )
    assert config.application.theme == theme


def test_application_theme_defaults_to_mocha_and_rejects_other_values() -> None:
    """Theme is optional but constrained to ModelTop's four Catppuccin flavours."""
    config = ModelTopConfig.model_validate(
        {
            "application": {},
            "servers": [
                {"id": "server", "name": "Server", "base_url": "http://server"}
            ],
        }
    )
    assert config.application.theme == "catppuccin-mocha"
    with pytest.raises(ValidationError) as caught:
        ModelTopConfig.model_validate(
            {
                "application": {"theme": "dracula"},
                "servers": [
                    {"id": "server", "name": "Server", "base_url": "http://server"}
                ],
            }
        )
    assert caught.value.errors()[0]["loc"] == ("application", "theme")


def test_configuration_precedence_and_source_reporting(tmp_path: Path) -> None:
    """Environment, user, and repository sources are chosen in exact order."""
    home = tmp_path / "home"
    cwd = tmp_path / "checkout"
    user_path = home / ".config/modeltop/config.yaml"
    repository_path = cwd / "config/modeltop.yaml"
    environment_path = cwd / "environment.yaml"
    for path, server_id in (
        (user_path, "user"),
        (repository_path, "repository"),
        (environment_path, "environment"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_yaml(server_id), encoding="utf-8")

    environment = load_configuration(
        environ={MODELTOP_CONFIG: "environment.yaml"}, home=home, cwd=cwd
    )
    assert environment.source_path == environment_path.resolve()
    assert environment.config.servers[0].id == "environment"

    user = load_configuration(environ={}, home=home, cwd=cwd)
    assert user.source_path == user_path
    assert user.config.servers[0].id == "user"

    user_path.unlink()
    repository = load_configuration(environ={}, home=home, cwd=cwd)
    assert repository.source_path == repository_path
    assert repository.config.servers[0].id == "repository"

    repository_path.unlink()
    built_in = load_configuration(environ={}, home=home, cwd=cwd)
    assert built_in.source_path is None
    assert built_in.config == ModelTopConfig.model_validate(BUILTIN_CONFIG)
    assert built_in.config.servers[0].id == "local-vllm"
    assert [server.id for server in built_in.config.servers] == [
        "local-vllm",
        "local-8888",
    ]
    assert built_in.config.servers[1].base_url == "http://127.0.0.1:8888/v1"
    assert built_in.config.hardware.enabled
    assert built_in.config.hardware.refresh_interval_seconds == 2
    assert built_in.config.hardware.preferred_provider == "auto"


def test_authoritative_environment_path_must_load(tmp_path: Path) -> None:
    """An explicit missing path never falls through to a valid implicit file."""
    home = tmp_path / "home"
    user_path = home / ".config/modeltop/config.yaml"
    user_path.parent.mkdir(parents=True)
    user_path.write_text(_yaml("user"), encoding="utf-8")
    missing = tmp_path / "missing.yaml"

    with pytest.raises(ConfigurationLoadError) as caught:
        load_configuration(
            environ={MODELTOP_CONFIG: str(missing)},
            home=home,
            cwd=tmp_path,
        )

    assert str(missing) in caught.value.user_message
    assert "does not exist" in caught.value.user_message


def test_first_existing_implicit_file_is_authoritative(tmp_path: Path) -> None:
    """An invalid user file does not fall through to repository configuration."""
    home = tmp_path / "home"
    cwd = tmp_path / "checkout"
    user_path = home / ".config/modeltop/config.yaml"
    repository_path = cwd / "config/modeltop.yaml"
    user_path.parent.mkdir(parents=True)
    repository_path.parent.mkdir(parents=True)
    user_path.write_text("not: [valid", encoding="utf-8")
    repository_path.write_text(_yaml("repository"), encoding="utf-8")

    with pytest.raises(ConfigurationLoadError) as caught:
        load_configuration(environ={}, home=home, cwd=cwd)

    assert str(user_path) in caught.value.user_message
    assert "invalid YAML" in caught.value.user_message


@pytest.mark.parametrize("document", ["- item\n", "null\n", "text\n"])
def test_configuration_requires_mapping_root(tmp_path: Path, document: str) -> None:
    """Safe YAML roots must be mappings before Pydantic validation."""
    path = tmp_path / "config.yaml"
    path.write_text(document, encoding="utf-8")
    with pytest.raises(ConfigurationLoadError, match="root must be a mapping"):
        load_configuration(environ={MODELTOP_CONFIG: str(path)}, cwd=tmp_path)


def test_validation_failure_does_not_expose_api_key(tmp_path: Path) -> None:
    """Pydantic diagnostics identify fields without copying secret inputs."""
    path = tmp_path / "config.yaml"
    secret = "never-show-this-key"
    path.write_text(
        _yaml("bad", api_key=secret).replace(
            "http://127.0.0.1:8000/v1", "ftp://invalid"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationLoadError) as caught:
        load_configuration(environ={MODELTOP_CONFIG: str(path)}, cwd=tmp_path)

    assert "base_url" in caught.value.user_message
    assert secret not in caught.value.user_message
    assert secret not in caught.value.detail


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("-inf"), float("nan")])
@pytest.mark.parametrize(
    "field", ["refresh_interval_seconds", "request_timeout_seconds"]
)
def test_intervals_must_be_positive_and_finite(field: str, value: float) -> None:
    """Both configured intervals reject zero, negatives, and non-finite values."""
    payload = {
        "application": {field: value},
        "servers": [{"id": "s", "name": "S", "base_url": "http://server"}],
    }
    with pytest.raises(ValidationError):
        ModelTopConfig.model_validate(payload)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("-inf"), float("nan")])
def test_hardware_interval_must_be_positive_and_finite(value: float) -> None:
    payload = {
        "application": {},
        "hardware": {"refresh_interval_seconds": value},
        "servers": [{"id": "s", "name": "S", "base_url": "http://server"}],
    }
    with pytest.raises(ValidationError):
        ModelTopConfig.model_validate(payload)


def test_hardware_defaults_valid_modes_and_effective_disable_inputs() -> None:
    base: dict[str, object] = {
        "application": {},
        "servers": [{"id": "s", "name": "S", "base_url": "http://server"}],
    }
    defaults = ModelTopConfig.model_validate(base)
    assert defaults.hardware.enabled
    assert defaults.hardware.refresh_interval_seconds == 2.0
    assert defaults.hardware.preferred_provider == "auto"
    for mode in ("auto", "nvml", "nvidia-smi", "disabled"):
        config = ModelTopConfig.model_validate(
            {**base, "hardware": {"preferred_provider": mode}}
        )
        assert config.hardware.preferred_provider == mode
    disabled = ModelTopConfig.model_validate(
        {**base, "hardware": {"enabled": False, "preferred_provider": "auto"}}
    )
    assert not disabled.hardware.enabled


def test_invalid_hardware_provider_is_readable_and_rejected() -> None:
    payload = {
        "application": {},
        "hardware": {"preferred_provider": "amd-secret"},
        "servers": [{"id": "s", "name": "S", "base_url": "http://server"}],
    }
    with pytest.raises(ValidationError) as caught:
        ModelTopConfig.model_validate(payload)
    assert "preferred_provider" in str(caught.value)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://server/v1",
        "localhost:8000/v1",
        "http:///v1",
        "http://user@server/v1",
        "http://user:password@server/v1",
        "http://server/v1?key=value",
        "http://server/v1#fragment",
    ],
)
def test_server_url_is_http_absolute_and_credential_free(url: str) -> None:
    """Authentication and request details cannot be embedded in base URLs."""
    with pytest.raises(ValidationError):
        ServerConfig(id="s", name="S", base_url=url)


@pytest.mark.parametrize(
    "payload",
    [
        {"application": {}, "servers": []},
        {
            "application": {},
            "servers": [
                {"id": "same", "name": "A", "base_url": "http://a"},
                {"id": " same ", "name": "B", "base_url": "http://b"},
            ],
        },
        {
            "application": {"default_server": "missing"},
            "servers": [{"id": "one", "name": "A", "base_url": "http://a"}],
        },
    ],
)
def test_server_collection_invariants(payload: object) -> None:
    """At least one unique server and a valid default reference are required."""
    with pytest.raises(ValidationError):
        ModelTopConfig.model_validate(payload)


def test_display_labels_preserve_prefix_and_normalize_only_vllm() -> None:
    """Display helpers remove terminal v1 without backend adapter inference."""
    server = ServerConfig(
        id="s",
        name="S",
        base_url="https://server.example/prefix/v1/",
        backend_hint=" VLLM ",
    )
    assert server.endpoint_label == "server.example/prefix"
    assert server.backend_label == "vLLM"
    assert ServerConfig(id="x", name="X", base_url="http://x").backend_label == "--"


def test_configure_logging_creates_parent_and_is_idempotent(tmp_path: Path) -> None:
    """File logging creates its directory and owns one UTF-8 handler."""
    path = tmp_path / "nested" / "modeltop.log"
    assert configure_logging(path) == path
    assert configure_logging(path) == path
    handlers = [
        handler
        for handler in logging.getLogger("modeltop").handlers
        if isinstance(handler, logging.FileHandler)
    ]
    assert len(handlers) == 1
    assert Path(handlers[0].baseFilename) == path.resolve()
    logging.getLogger("modeltop.test").info("safe message")
    handlers[0].flush()
    assert "safe message" in path.read_text(encoding="utf-8")


def test_concurrency_defaults_are_backward_compatible_when_omitted() -> None:
    payload: dict[str, object] = {
        "application": {},
        "servers": [{"id": "s", "name": "S", "base_url": "http://server"}],
    }
    concurrency = ModelTopConfig.model_validate(payload).benchmarks.concurrency
    assert concurrency == ConcurrencyBenchmarkDefaultsConfig()
    assert concurrency.default_levels == (1, 2, 4, 8)
    assert concurrency.requests_per_level == 16
    assert concurrency.warmup_requests == 2
    assert concurrency.max_tokens == 256
    assert concurrency.temperature == 0.0
    assert concurrency.top_p == 1.0
    assert concurrency.request_timeout_seconds == 120.0
    assert concurrency.delay_between_levels_seconds == 3.0
    assert concurrency.maximum_concurrency == 128
    assert concurrency.unique_prompt_suffix_per_request is True


def test_builtin_and_repository_yaml_have_the_exact_concurrency_shape() -> None:
    expected = {
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
    }
    assert BUILTIN_CONFIG["benchmarks"]["concurrency"] == expected  # type: ignore[index]
    builtin = ModelTopConfig.model_validate(BUILTIN_CONFIG)
    assert builtin.benchmarks.concurrency.model_dump(mode="json") == expected

    repository_root = Path(__file__).parents[1]
    for relative_path in ("config/modeltop.yaml", "config/modeltop.example.yaml"):
        document = yaml.safe_load(
            (repository_root / relative_path).read_text(encoding="utf-8")
        )
        assert document["benchmarks"]["concurrency"] == expected
        validated = ModelTopConfig.model_validate(document)
        assert validated.benchmarks.concurrency == builtin.benchmarks.concurrency


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("default_levels", []),
        ("default_levels", [0, 1]),
        ("default_levels", [-1, 1]),
        ("default_levels", [1]),
        ("default_levels", [2, 1]),
        ("default_levels", [1, 1]),
        ("default_levels", [1, "2"]),
        ("default_levels", [True, 2]),
        ("requests_per_level", 0),
        ("requests_per_level", 1001),
        ("requests_per_level", 1.0),
        ("warmup_requests", -1),
        ("warmup_requests", 1001),
        ("max_tokens", 0),
        ("temperature", True),
        ("temperature", "0.5"),
        ("temperature", -0.01),
        ("temperature", 2.01),
        ("temperature", float("nan")),
        ("temperature", float("-inf")),
        ("temperature", float("inf")),
        ("top_p", 0.0),
        ("top_p", 1.01),
        ("top_p", float("nan")),
        ("top_p", float("inf")),
        ("top_p", float("-inf")),
        ("request_timeout_seconds", 0.0),
        ("request_timeout_seconds", float("nan")),
        ("request_timeout_seconds", float("inf")),
        ("request_timeout_seconds", float("-inf")),
        ("delay_between_levels_seconds", -0.01),
        ("delay_between_levels_seconds", float("nan")),
        ("delay_between_levels_seconds", float("inf")),
        ("delay_between_levels_seconds", float("-inf")),
        ("maximum_concurrency", 0),
        ("maximum_concurrency", 1.0),
    ],
)
def test_concurrency_yaml_defaults_reject_invalid_boundaries(
    field: str, value: object
) -> None:
    with pytest.raises(ValidationError):
        ConcurrencyBenchmarkDefaultsConfig.model_validate({field: value})


def test_concurrency_yaml_safety_maximum_and_request_caps() -> None:
    boundaries = ConcurrencyBenchmarkDefaultsConfig(
        default_levels=(1, 256),
        requests_per_level=1000,
        warmup_requests=1000,
        max_tokens=1,
        temperature=2.0,
        top_p=0.000001,
        request_timeout_seconds=0.000001,
        delay_between_levels_seconds=0.0,
        maximum_concurrency=256,
    )
    assert boundaries.default_levels == (1, 256)
    assert boundaries.maximum_concurrency == 256
    minimums = ConcurrencyBenchmarkDefaultsConfig(
        default_levels=(1, 2),
        requests_per_level=1,
        warmup_requests=0,
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        request_timeout_seconds=0.000001,
        delay_between_levels_seconds=0.0,
        maximum_concurrency=2,
    )
    assert minimums.requests_per_level == 1
    assert minimums.warmup_requests == 0

    with pytest.raises(ValidationError, match="maximum_concurrency"):
        ConcurrencyBenchmarkDefaultsConfig(
            default_levels=(1, 129), maximum_concurrency=128
        )


def test_concurrency_validation_error_remains_secret_free(tmp_path: Path) -> None:
    secret = "benchmark-must-not-leak-this-key"
    document = _yaml("bad", api_key=secret).replace(
        "servers:\n",
        "benchmarks:\n  concurrency:\n    default_levels: [1, 1]\nservers:\n",
    )
    path = tmp_path / "invalid-benchmark.yaml"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(ConfigurationLoadError) as caught:
        load_configuration(environ={MODELTOP_CONFIG: str(path)}, cwd=tmp_path)

    assert "default_levels" in caught.value.user_message
    assert secret not in caught.value.user_message
    assert secret not in caught.value.detail
