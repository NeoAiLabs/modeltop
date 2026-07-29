"""Model parsing, sorting, and selection policy."""

from dataclasses import dataclass

from pydantic import ValidationError

from modeltop.api.client import OpenAICompatibleClient
from modeltop.api.errors import ProtocolError
from modeltop.models import DiscoveredModel


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """A complete successful model discovery result."""

    models: tuple[DiscoveredModel, ...]
    selected_model_id: str | None
    latency_ms: float


def _validation_detail(index: int, error: ValidationError) -> str:
    issues: list[str] = []
    for issue in error.errors(
        include_input=False,
        include_url=False,
        include_context=False,
    ):
        location = ".".join(str(part) for part in issue["loc"])
        field = f".{location}" if location else ""
        issues.append(f"data[{index}]{field}: {issue['msg']}")
    return "; ".join(issues)


class ModelDiscoveryService:
    """Discover and deterministically select models from one client."""

    def __init__(self, client: OpenAICompatibleClient) -> None:
        self._client = client

    async def discover(
        self,
        previous_selection: str | None,
        configured_default: str | None,
    ) -> DiscoveryResult:
        """Fetch all models and apply the selection precedence policy."""
        response = await self._client.list_models()
        parsed: list[DiscoveredModel] = []
        for index, item in enumerate(response.data):
            try:
                parsed.append(DiscoveredModel.model_validate(item))
            except ValidationError as error:
                raise ProtocolError(
                    "Invalid response from server",
                    _validation_detail(index, error),
                ) from error

        models = tuple(
            sorted(parsed, key=lambda model: (model.id.casefold(), model.id))
        )
        available_ids = {model.id for model in models}
        if previous_selection in available_ids:
            selected_model_id = previous_selection
        elif configured_default in available_ids:
            selected_model_id = configured_default
        elif models:
            selected_model_id = models[0].id
        else:
            selected_model_id = None
        return DiscoveryResult(
            models=models,
            selected_model_id=selected_model_id,
            latency_ms=response.latency_ms,
        )
