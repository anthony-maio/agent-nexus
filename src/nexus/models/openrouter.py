"""Async OpenRouter API client for Agent Nexus.

OpenRouter provides a unified API compatible with the OpenAI chat completions
format, routing requests to many different model providers transparently.

This module implements the primary LLM backend used by the nexus swarm.  All
cloud-hosted model calls (chat completions and embeddings) flow through this
client, which handles authentication, rate-limit retries, cost tracking, and
structured response parsing.

Usage::

    from nexus.models.openrouter import OpenRouterClient

    async with OpenRouterClient(api_key="sk-or-...") as client:
        response = await client.chat(
            model="minimax/minimax-m2.5",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        print(response.content)
        print(f"Cost so far: ${client.session_cost:.6f}")

The client implements :meth:`__aenter__` / :meth:`__aexit__` so it can be used
as an async context manager, which ensures the underlying ``aiohttp`` session
is properly closed on exit.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator

import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL: str = "https://openrouter.ai/api/v1"
"""Default OpenRouter API base URL."""

_MAX_RETRIES: int = 3
"""Maximum number of retry attempts on rate-limit (HTTP 429) responses."""

_RETRY_BACKOFF_BASE: float = 1.0
"""Base delay in seconds for exponential backoff (1s, 2s, 4s, ...)."""

_HTTP_REFERER: str = "https://github.com/agent-nexus"
"""Value sent in the ``HTTP-Referer`` header for OpenRouter analytics."""

_X_TITLE: str = "Agent Nexus"
"""Value sent in the ``X-Title`` header for OpenRouter analytics."""


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class OpenRouterError(Exception):
    """Raised when the OpenRouter API returns an error response.

    Attributes:
        message: Human-readable error description from the API.
        status_code: HTTP status code of the failed response.
        model: The model identifier that was requested, if available.
    """

    def __init__(
        self,
        message: str,
        status_code: int,
        model: str | None = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.model = model
        super().__init__(
            f"OpenRouter error {status_code}"
            f"{f' (model={model})' if model else ''}: {message}"
        )


# ---------------------------------------------------------------------------
# Response dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """Structured result from a chat completion request.

    Attributes:
        content: The assistant's reply text.
        model: Actual model that served the request (may differ from the
            requested model when OpenRouter performs automatic fallback).
        input_tokens: Number of prompt tokens consumed.
        output_tokens: Number of completion tokens generated.
        cost: Estimated cost in USD for this single request.
        finish_reason: Why the model stopped generating (e.g. ``"stop"``,
            ``"length"``, ``"tool_calls"``).
    """

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    finish_reason: str


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """A single chunk from a streaming chat completion.

    Attributes:
        delta: Incremental text content in this chunk (may be empty).
        model: The model serving the stream, if reported.
        finish_reason: Set on the final chunk; ``None`` otherwise.
    """

    delta: str
    model: str | None = None
    finish_reason: str | None = None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OpenRouterClient:
    """Async client for the OpenRouter unified LLM API.

    The HTTP session is created lazily on first use and reused for the
    lifetime of the client.  Call :meth:`close` (or use the client as an
    async context manager) to release the underlying connection pool.

    Args:
        api_key: OpenRouter API key (``sk-or-...``).
        base_url: API base URL.  Override for testing or proxying.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        self._api_key: str = api_key
        self._base_url: str = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None
        self.session_cost: float = 0.0
        """Cumulative USD cost across all requests made through this client."""

    # -- Async context manager ----------------------------------------------

    async def __aenter__(self) -> OpenRouterClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.close()

    # -- Internal helpers ---------------------------------------------------

    def _get_headers(self) -> dict[str, str]:
        """Build the default HTTP headers for all API requests."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": _HTTP_REFERER,
            "X-Title": _X_TITLE,
        }

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Return the shared session, creating it lazily if needed.

        The session is created outside ``__init__`` to avoid requiring an
        active event loop at construction time.
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=120),
            )
        return self._session

    async def _request_with_retries(
        self,
        method: str,
        endpoint: str,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Execute an HTTP request with exponential-backoff retry on 429.

        Args:
            method: HTTP method (``"POST"``).
            endpoint: Path appended to the base URL (e.g. ``"/chat/completions"``).
            payload: JSON body to send.

        Returns:
            A tuple of ``(response_json, response_headers)``.

        Raises:
            OpenRouterError: On non-retryable API errors or after exhausting
                all retry attempts.
        """
        session = await self._ensure_session()
        url = f"{self._base_url}{endpoint}"
        last_error: OpenRouterError | None = None

        for attempt in range(_MAX_RETRIES):
            async with session.request(method, url, json=payload) as resp:
                # Collect headers for cost tracking regardless of status.
                headers = {k: v for k, v in resp.headers.items()}

                if resp.status == 429:
                    # Rate-limited -- back off and retry.
                    delay = _RETRY_BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        "OpenRouter rate-limited (429). Retrying in %.1fs "
                        "(attempt %d/%d).",
                        delay,
                        attempt + 1,
                        _MAX_RETRIES,
                    )
                    last_error = OpenRouterError(
                        message="Rate limited (429)",
                        status_code=429,
                        model=payload.get("model"),
                    )
                    await asyncio.sleep(delay)
                    continue

                body: dict[str, Any] = await resp.json(content_type=None)

                # OpenRouter may embed error details inside the JSON body even
                # when the HTTP status indicates success.
                if "error" in body:
                    err = body["error"]
                    message = (
                        err.get("message", str(err))
                        if isinstance(err, dict)
                        else str(err)
                    )
                    raise OpenRouterError(
                        message=message,
                        status_code=resp.status,
                        model=payload.get("model"),
                    )

                if resp.status >= 400:
                    raise OpenRouterError(
                        message=f"HTTP {resp.status}: {body}",
                        status_code=resp.status,
                        model=payload.get("model"),
                    )

                return body, headers

        # All retries exhausted.
        raise last_error or OpenRouterError(
            message="Request failed after all retries.",
            status_code=429,
            model=payload.get("model"),
        )

    @staticmethod
    def _estimate_cost(
        usage: dict[str, Any],
        body: dict[str, Any],
    ) -> float:
        """Extract or estimate the USD cost for a single request.

        OpenRouter may include a ``usage.total_cost`` field or expose
        per-token pricing.  We fall back to zero when neither is available
        rather than raising.
        """
        # Some responses include a top-level 'usage' with 'total_cost'.
        if "total_cost" in usage:
            try:
                return float(usage["total_cost"])
            except (TypeError, ValueError):
                pass

        # Fallback: check for a top-level cost field added by some
        # OpenRouter response wrappers.
        if "cost" in body:
            try:
                return float(body["cost"])
            except (TypeError, ValueError):
                pass

        return 0.0

    # -- Public API ---------------------------------------------------------

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> ChatResponse | AsyncIterator[StreamChunk]:
        """Send a chat completion request to OpenRouter.

        Args:
            model: OpenRouter model identifier (e.g.
                ``"minimax/minimax-m2.5"``).
            messages: Conversation in OpenAI message format --
                ``[{"role": "user", "content": "..."}]``.
            temperature: Sampling temperature (0.0 -- 2.0).
            max_tokens: Maximum tokens to generate.
            stream: When ``True``, return an async iterator of
                :class:`StreamChunk` objects instead of a complete
                :class:`ChatResponse`.

        Returns:
            A :class:`ChatResponse` for non-streaming calls, or an
            :class:`AsyncIterator` of :class:`StreamChunk` for streaming.

        Raises:
            OpenRouterError: On API errors or rate-limit exhaustion.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        if stream:
            return self._stream_chat(model, payload)

        body, _headers = await self._request_with_retries(
            "POST", "/chat/completions", payload,
        )

        # Parse the response into a structured object.
        choices: list[dict[str, Any]] = body.get("choices", [])
        if not choices:
            raise OpenRouterError(
                message="No choices returned in chat completion response.",
                status_code=200,
                model=model,
            )

        first_choice = choices[0]
        message = first_choice.get("message", {})
        content: str = message.get("content", "") or ""
        finish_reason: str = first_choice.get("finish_reason", "unknown") or "unknown"
        actual_model: str = body.get("model", model)

        usage: dict[str, Any] = body.get("usage", {})
        input_tokens: int = int(usage.get("prompt_tokens", 0))
        output_tokens: int = int(usage.get("completion_tokens", 0))
        cost = self._estimate_cost(usage, body)

        # Accumulate session cost.
        self.session_cost += cost

        response = ChatResponse(
            content=content,
            model=actual_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            finish_reason=finish_reason,
        )

        logger.debug(
            "Chat completion: model=%s, tokens=%d+%d, cost=$%.6f, finish=%s",
            actual_model,
            input_tokens,
            output_tokens,
            cost,
            finish_reason,
        )

        return response

    async def _stream_chat(
        self,
        model: str,
        payload: dict[str, Any],
    ) -> AsyncIterator[StreamChunk]:
        """Internal streaming implementation for chat completions.

        Yields:
            :class:`StreamChunk` instances as they arrive from the API.
        """
        import json as _json

        session = await self._ensure_session()
        url = f"{self._base_url}/chat/completions"
        last_error: OpenRouterError | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 429:
                        delay = _RETRY_BACKOFF_BASE * (2 ** attempt)
                        logger.warning(
                            "OpenRouter stream rate-limited (429). Retrying in %.1fs "
                            "(attempt %d/%d).",
                            delay, attempt + 1, _MAX_RETRIES,
                        )
                        last_error = OpenRouterError(
                            message="Rate limited (429)",
                            status_code=429,
                            model=model,
                        )
                        await asyncio.sleep(delay)
                        continue

                    if resp.status >= 400:
                        body = await resp.json(content_type=None)
                        if "error" in body:
                            err = body["error"]
                            message = (
                                err.get("message", str(err))
                                if isinstance(err, dict)
                                else str(err)
                            )
                        else:
                            message = f"HTTP {resp.status}: {body}"
                        raise OpenRouterError(
                            message=message,
                            status_code=resp.status,
                            model=model,
                        )

                    # Stream chunks
                    async for raw_line in resp.content:
                        line = raw_line.decode("utf-8", errors="replace").strip()

                        if not line or line.startswith(":"):
                            continue

                        if line == "data: [DONE]":
                            break

                        if line.startswith("data: "):
                            line = line[6:]

                        try:
                            data: dict[str, Any] = _json.loads(line)
                        except _json.JSONDecodeError:
                            logger.debug("Skipping unparseable SSE line: %s", line[:120])
                            continue

                        choices = data.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})
                        finish = choices[0].get("finish_reason")

                        yield StreamChunk(
                            delta=delta.get("content", "") or "",
                            model=data.get("model"),
                            finish_reason=finish,
                        )
                    return  # Success - exit retry loop

            except OpenRouterError:
                raise

        if last_error:
            raise last_error

    async def embed(
        self,
        model: str,
        texts: list[str],
    ) -> list[list[float]]:
        """Get embeddings for a batch of texts.

        The request is sent to the OpenAI-compatible ``/embeddings`` endpoint
        on OpenRouter, which routes to the actual embedding provider.

        Args:
            model: Embedding model identifier (e.g.
                ``"qwen/qwen3-embedding-8b"``).
            texts: List of text strings to embed.

        Returns:
            A list of float vectors, one per input text, in the same order
            as the input.

        Raises:
            OpenRouterError: On API errors or rate-limit exhaustion.
        """
        if not texts:
            return []

        payload: dict[str, Any] = {
            "model": model,
            "input": texts,
        }

        body, _headers = await self._request_with_retries(
            "POST", "/embeddings", payload,
        )

        # Parse embedding vectors from the response.
        data_entries: list[dict[str, Any]] = body.get("data", [])
        if len(data_entries) != len(texts):
            logger.warning(
                "Embedding response returned %d vectors for %d inputs.",
                len(data_entries),
                len(texts),
            )

        # Sort by index to guarantee order matches the input.
        data_entries.sort(key=lambda d: d.get("index", 0))
        embeddings: list[list[float]] = [
            entry["embedding"] for entry in data_entries
        ]

        # Track cost if usage information is available.
        usage: dict[str, Any] = body.get("usage", {})
        cost = self._estimate_cost(usage, body)
        self.session_cost += cost

        total_tokens = int(usage.get("total_tokens", 0))
        logger.debug(
            "Embedding: model=%s, texts=%d, tokens=%d, cost=$%.6f",
            model,
            len(texts),
            total_tokens,
            cost,
        )

        return embeddings

    async def close(self) -> None:
        """Close the underlying HTTP session and release connections.

        Safe to call multiple times.  After closing, further requests will
        lazily create a new session.
        """
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
            logger.debug(
                "OpenRouter session closed. Total session cost: $%.6f",
                self.session_cost,
            )
