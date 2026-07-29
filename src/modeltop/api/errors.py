"""Typed, display-safe API client failures."""


class APIClientError(Exception):
    """Base API failure with separate user and logging text."""

    def __init__(
        self,
        user_message: str,
        detail: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail
        self.status_code = status_code


class ServerConnectionError(APIClientError):
    """The configured server could not be reached."""


class RequestTimeoutError(APIClientError):
    """The server request exceeded its configured timeout."""


class AuthenticationError(APIClientError):
    """The server rejected the configured credentials."""


class ModelsEndpointNotFoundError(APIClientError):
    """The generic models endpoint does not exist."""


class ModelNotFoundError(APIClientError):
    """The selected model is not available on the chat endpoint."""


class ChatEndpointNotFoundError(APIClientError):
    """The generic chat-completions endpoint does not exist."""


class RateLimitError(APIClientError):
    """The server is currently rate limiting chat requests."""


class ContextLimitError(APIClientError):
    """The submitted conversation exceeds the server context limit."""


class RequestRejectedError(APIClientError):
    """The server rejected an otherwise valid chat request."""


class UnsupportedStreamingError(APIClientError):
    """The server explicitly does not support streamed chat completions."""


class HTTPResponseError(APIClientError):
    """The server returned another unsuccessful HTTP status."""


class ProtocolError(APIClientError):
    """The server response did not match the required protocol."""
