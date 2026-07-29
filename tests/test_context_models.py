"""Context configuration defaults, strict boundaries, and lane isolation."""

from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from modeltop.benchmarks.models import (
    ContextBenchmarkConfig,
    ContextBenchmarkStatus,
    context_benchmark_config_from_defaults,
)
from modeltop.models import (
    ContextBenchmarkDefaultsConfig,
    ModelTopConfig,
)
from modeltop.state import initial_application_state


def test_omitted_context_defaults_are_backward_compatible() -> None:
    config = ModelTopConfig.model_validate(
        {
            "application": {},
            "servers": [{"id": "s", "name": "S", "base_url": "http://s"}],
        }
    )
    assert config.benchmarks.context == ContextBenchmarkDefaultsConfig()
    runtime = context_benchmark_config_from_defaults(config.benchmarks.context)
    assert runtime.mode == "sweep"
    assert runtime.target_lengths == (1024, 4096, 8192, 16384, 32768)
    assert not runtime.retrieval_enabled


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repetitions_per_length", True),
        ("warmup_requests", 1.0),
        ("maximum_output_tokens", False),
        ("temperature", float("nan")),
        ("top_p", float("inf")),
        ("default_lengths", [4096, 1024]),
        ("default_lengths", [1024, 1024]),
    ],
)
def test_yaml_defaults_reject_non_strict_or_nonfinite_values(
    field: str, value: object
) -> None:
    with pytest.raises(ValidationError):
        ContextBenchmarkDefaultsConfig.model_validate({field: value})


def test_runtime_mode_and_private_value_contracts() -> None:
    with pytest.raises(ValidationError, match="fixed mode"):
        ContextBenchmarkConfig(mode="fixed", target_lengths=(1024, 2048))
    with pytest.raises(ValidationError, match="retrieval_enabled"):
        ContextBenchmarkConfig(mode="retrieval", target_lengths=(1024,))
    with pytest.raises(ValidationError, match="base_text"):
        ContextBenchmarkConfig(base_text="secret")
    retrieval = ContextBenchmarkConfig(
        mode="retrieval",
        target_lengths=(1024,),
        retrieval_enabled=True,
        retrieval_positions=("end", "beginning"),
        retrieval_key="  manual-key-1  ",
    )
    assert retrieval.retrieval_positions == ("end", "beginning")
    assert retrieval.retrieval_key == "manual-key-1"
    with pytest.raises(ValidationError):
        ContextBenchmarkConfig(
            mode="retrieval",
            target_lengths=(1024,),
            retrieval_enabled=True,
            retrieval_positions=("end", "end"),
        )


def test_context_lane_is_independent_and_immutable() -> None:
    state = initial_application_state("s", hardware_enabled=False)
    assert state.context_benchmark.status is ContextBenchmarkStatus.IDLE
    assert not state.benchmark_is_active
    with pytest.raises(FrozenInstanceError):
        state.context_benchmark.status = ContextBenchmarkStatus.RUNNING  # type: ignore[misc]
