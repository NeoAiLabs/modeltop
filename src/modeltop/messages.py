"""Typed widget-to-application dashboard commands."""

from textual.message import Message

from modeltop.benchmarks.models import (
    ConcurrencyBenchmarkConfig,
    ContextBenchmarkConfig,
    DrafterBenchmarkConfig,
    SpeedTestConfig,
    ToolCallingBenchmarkConfig,
)
from modeltop.chat.models import GenerationSettings


class PromptSubmitted(Message):
    """Request submission of the current composer text."""

    def __init__(self, prompt: str) -> None:
        super().__init__()
        self.prompt = prompt


class ClearConversationRequested(Message):
    """Request clearing the current in-memory conversation."""


class SettingsSubmitted(Message):
    """Submit already validated generation preferences."""

    def __init__(
        self,
        settings: GenerationSettings,
        system_prompt: str,
        show_system_prompt: bool,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.system_prompt = system_prompt
        self.show_system_prompt = show_system_prompt


class ConcurrencyBenchmarkStartRequested(Message):
    """Start one validated Concurrency benchmark configuration."""

    def __init__(self, config: ConcurrencyBenchmarkConfig) -> None:
        super().__init__()
        self.config = config


class ConcurrencyBenchmarkRunAgainRequested(Message):
    """Rerun the latest Concurrency benchmark configuration."""


class ConcurrencyBenchmarkEditRequested(Message):
    """Restore the latest Concurrency benchmark configuration for editing."""


class ContextBenchmarkStartRequested(Message):
    """Start one validated Context benchmark configuration."""

    def __init__(self, config: ContextBenchmarkConfig) -> None:
        super().__init__()
        self.config = config


class ContextBenchmarkRunAgainRequested(Message):
    """Rerun the latest immutable Context configuration."""


class ContextBenchmarkEditRequested(Message):
    """Restore the latest Context configuration for editing."""


class ToolCallingBenchmarkStartRequested(Message):
    """Start one validated Tool Calling benchmark configuration."""

    def __init__(self, config: ToolCallingBenchmarkConfig) -> None:
        super().__init__()
        self.config = config


class ToolCallingBenchmarkRunAgainRequested(Message):
    """Rerun the latest immutable Tool Calling configuration."""


class ToolCallingBenchmarkEditRequested(Message):
    """Restore the latest Tool Calling configuration for editing."""


class DrafterBenchmarkStartRequested(Message):
    """Start one validated Drafter benchmark configuration."""

    def __init__(self, config: DrafterBenchmarkConfig) -> None:
        super().__init__()
        self.config = config


class DrafterBenchmarkRunAgainRequested(Message):
    """Rerun the latest immutable Drafter configuration."""


class DrafterBenchmarkEditRequested(Message):
    """Restore the latest Drafter configuration for editing."""


class SpeedTestStartRequested(Message):
    """Start one validated Speed Test configuration."""

    def __init__(self, config: SpeedTestConfig) -> None:
        super().__init__()
        self.config = config


class SpeedTestCancelRequested(Message):
    """Cancel the currently active Speed Test."""


class SpeedTestRunAgainRequested(Message):
    """Restore and rerun a terminal result configuration."""

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id


class SpeedTestExportRequested(Message):
    """Export one terminal result by immutable run ID."""

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id


class SpeedTestCopySummaryRequested(Message):
    """Copy one terminal result summary by immutable run ID."""

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id


class SpeedTestResultSelected(Message):
    """Open one session result by immutable run ID."""

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id
