"""Context measured-only aggregation and rate provenance tests."""

import pytest

from modeltop.benchmarks.context_statistics import build_context_length_result
from modeltop.benchmarks.models import (
    ContextBenchmarkConfig,
    ContextPromptMeasurement,
    ContextRequestResult,
)


def _request(sequence: int, *, success: bool = True) -> ContextRequestResult:
    measurement = ContextPromptMeasurement(
        requested_length=1024,
        requested_unit="tokens",
        visible_content_characters=4000,
        body_tokens=990,
        system_tokens=12,
        instruction_tokens=11,
        template_overhead_tokens=11,
        local_prompt_tokens=1024,
        server_prompt_tokens=1000,
        builder_difference=0,
        server_token_difference=-24,
        server_token_difference_percent=-2.34375,
        counter_name="test",
        estimated=True,
    )
    return ContextRequestResult(
        request_id=f"r{sequence}",
        target_length=1024,
        run_number=sequence,
        sequence_number=sequence,
        measurement=measurement,
        requested_at=float(sequence),
        first_token_at=float(sequence) + 0.1,
        completed_at=float(sequence) + 0.5,
        ttft_ms=100,
        total_latency_seconds=0.5,
        generation_duration_seconds=0.4,
        completion_tokens=20,
        completion_tokens_estimated=False,
        output_tokens_per_second=50,
        estimated_input_tokens_per_second=10_000,
        finish_reason="stop" if success else None,
        status_code=200 if success else None,
        streamed=True,
        success=success,
        state="done" if success else "error",
        accepted=True if success else None,
        context_rejected=False,
        timed_out=False,
        cancelled=False,
        retrieval_results=(),
        response_character_count=10 if success else 0,
        error_type=None if success else "ProtocolError",
        error_message=None if success else "Invalid response",
    )


def test_aggregates_success_metrics_and_server_prompt_usage() -> None:
    config = ContextBenchmarkConfig(
        mode="fixed", target_lengths=(1024,), warmup_requests=1
    )
    result = build_context_length_result(
        target_length=1024,
        config=config,
        configured_requests=3,
        requests=(_request(1), _request(2), _request(3, success=False)),
        hardware_samples=(),
    )
    assert result.attempted_requests == 3
    assert result.successful_requests == 2
    assert result.failed_requests == 1
    assert result.success_rate_percent == pytest.approx(200 / 3)
    assert result.prompt_tokens.median == 1000
    assert result.estimated_input_tokens_per_second.median == 10_000
    assert result.completion_tokens.count == 2
    assert result.hardware_summary is None
