"""Async client for a local Ollama instance.

This module provides :class:`OllamaClient`, the Tier 2 model backend for
Agent Nexus.  It communicates with Ollama's OpenAI-compatible chat endpoint
(``/v1/chat/completions``) as well as the native embed and model-management
APIs.

Primary use-cases:

* **Task agents** -- LiquidAI 1.2B models dispatched for routing,
  tool-calling, RAG, and extraction tasks.
* **Local embeddings** -- When the user opts for the ``mxbai-embed-large-v1``
  embedding model instead of a cloud provider.

All public methods handle connection failures gracefully.  When Ollama is
unreachable an :class:`OllamaUnavailableError` is raised (or ``None`` /
``False`` is returned for probe methods) so the caller can fall back to a
cloud provider.
"""

from __future__ import annotations

import json as _json
import logging
from dataclasses import dataclass
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OllamaError(Exception):
    """Base exception for all Ollama client errors."""


class OllamaUnavailableError(OllamaError):
    """Raised when the Ollama server cannot be reached."""


class OllamaModelError(OllamaError):
    """Raised when a requested model is not available and cannot be pulled."""


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ChatResponse:
    """Normalised chat-completion response from Ollama.

    Attributes:
        content: The assistant's reply text.
        model: Model identifier that generated the response.
        input_tokens: Number of prompt tokens consumed.
        output_tokens: Number of completion tokens generated.
        cost: Always ``0.0`` for local inference.
        finish_reason: Why generation stopped (e.g. ``"stop"``, ``"length"``).
    """

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    finish_reason: str


# ---------------------------------------------------------------------------
# Timeout presets (seconds)
# ---------------------------------------------------------------------------

_CHAT_TIMEOUT = aiohttp.ClientTimeout(total=120)
_EMBED_TIMEOUT = aiohttp.ClientTimeout(total=60)
_PULL_TIMEOUT = aiohttp.ClientTimeout(total=300)
_PROBE_TIMEOUT = aiohttp.ClientTimeout(total=10)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OllamaClient:
    """Async HTTP client for a local Ollama instance.

    The client lazily creates an :class:`aiohttp.ClientSession` on first use
    so that instantiation is cheap and safe outside an async context.

    Args:
        base_url: Root URL of the Ollama server.  When running inside
            Docker, ``http://host.docker.internal:11434`` reaches the host
            machine.  For bare-metal, use ``http://localhost:11434``.
    """

    def __init__(self, base_url: str = "http://host.docker.internal:11434") -> None:
        self._base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    # -- Internal helpers ---------------------------------------------------

    def _get_session(self, timeout: aiohttp.ClientTimeout | None = None) -> aiohttp.ClientSession:
        """Return the shared session, creating it lazily if needed.

        A custom *timeout* is only applied at the **request** level (via the
        individual method calls), not baked into the session, so we create a
        single session with no default timeout and pass per-request timeouts
        where needed.
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Content-Type": "application/json"},
            )
        return self._session

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        timeout: aiohttp.ClientTimeout = _PROBE_TIMEOUT,
    ) -> dict[str, Any]:
        """Issue an HTTP request and return the parsed JSON body.

        Raises:
            OllamaUnavailableError: On any connection / timeout error.
            OllamaError: If the server returns a non-2xx status.
        """
        url = f"{self._base_url}{path}"
        session = self._get_session()
        try:
            async with session.request(method, url, json=json, timeout=timeout) as resp:
                body = await resp.json(content_type=None)
                if resp.status >= 400:
                    detail = body.get("error", resp.reason) if isinstance(body, dict) else resp.reason
                    raise OllamaError(
                        f"Ollama returned HTTP {resp.status} for {method} {path}: {detail}"
                    )
                return body  # type: ignore[return-value]
        except OllamaError:
            raise
        except (aiohttp.ClientError, OSError, TimeoutError) as exc:
            raise OllamaUnavailableError(
                f"Cannot reach Ollama at {self._base_url}: {exc}"
            ) from exc

    # -- Public API ---------------------------------------------------------

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        """Send a chat-completion request to Ollama.

        Uses the OpenAI-compatible ``/v1/chat/completions`` endpoint so that
        the request/response schema is identical to other providers.

        Args:
            model: Ollama model tag, e.g.
                ``"hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF"``.
            messages: Conversation history in OpenAI message format
                (``[{"role": "user", "content": "..."}]``).
            temperature: Sampling temperature.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            A :class:`ChatResponse` with the assistant's reply and token
            usage.

        Raises:
            OllamaUnavailableError: If the server is unreachable.
            OllamaError: If the server returns an error status.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        data = await self._request("POST", "/v1/chat/completions", json=payload, timeout=_CHAT_TIMEOUT)

        # Parse the OpenAI-compatible response envelope.
        choice = data["choices"][0]
        usage = data.get("usage", {})

        return ChatResponse(
            content=choice["message"]["content"],
            model=data.get("model", model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            cost=0.0,
            finish_reason=choice.get("finish_reason", "stop"),
        )

    async def embed(
        self,
        model: str,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for one or more texts.

        Uses Ollama's native ``/api/embed`` endpoint which accepts a batch
        of inputs in a single call.

        Args:
            model: Embedding model tag, e.g. ``"mxbai-embed-large-v1"``.
            texts: A list of strings to embed.

        Returns:
            A list of embedding vectors (one per input text), each a list of
            floats whose length matches the model's output dimensionality.

        Raises:
            OllamaUnavailableError: If the server is unreachable.
            OllamaError: If the server returns an error status.
        """
        payload: dict[str, Any] = {
            "model": model,
            "input": texts,
        }

        data = await self._request("POST", "/api/embed", json=payload, timeout=_EMBED_TIMEOUT)

        # The native embed endpoint returns {"embeddings": [[...], ...]}.
        embeddings: list[list[float]] = data["embeddings"]
        return embeddings

    async def is_available(self) -> bool:
        """Probe whether the Ollama server is running and reachable.

        Returns:
            ``True`` if a successful response is received from
            ``/api/tags``, ``False`` otherwise.  Never raises.
        """
        try:
            await self._request("GET", "/api/tags", timeout=_PROBE_TIMEOUT)
            return True
        except (OllamaError, Exception):  # noqa: BLE001
            return False

    async def list_models(self) -> list[str]:
        """Return the names of all locally available models.

        Returns:
            A sorted list of model name strings (e.g.
            ``["llama3:latest", "mxbai-embed-large-v1:latest"]``).

        Raises:
            OllamaUnavailableError: If the server is unreachable.
        """
        data = await self._request("GET", "/api/tags", timeout=_PROBE_TIMEOUT)
        models: list[dict[str, Any]] = data.get("models", [])
        return sorted(m["name"] for m in models)

    async def ensure_model(self, model: str) -> bool:
        """Ensure that *model* is available locally, pulling it if necessary.

        On first run the user may not have the required LiquidAI models.
        This method checks for the model and triggers a pull when it is
        missing.  Progress is logged at ``INFO`` level so the user can see
        download status.

        Args:
            model: The Ollama model tag to ensure.

        Returns:
            ``True`` if the model is now available, ``False`` if the pull
            failed or Ollama is unreachable.
        """
        # Step 1 -- check if the model is already present.
        try:
            local_models = await self.list_models()
        except OllamaUnavailableError:
            logger.warning("Ollama is not reachable -- cannot ensure model '%s'.", model)
            return False

        # Ollama stores tags with a ``:latest`` suffix when none is given.
        # Normalise both sides for comparison.
        def _normalise(name: str) -> str:
            return name if ":" in name else f"{name}:latest"

        target = _normalise(model)
        if any(_normalise(m) == target for m in local_models):
            logger.debug("Model '%s' is already available locally.", model)
            return True

        # Step 2 -- pull the model.
        logger.info("Model '%s' not found locally.  Pulling -- this may take a while ...", model)

        try:
            # The /api/pull endpoint streams JSON objects line-by-line when
            # stream=true (default).  We consume the stream to log progress
            # but use the non-streaming variant for simplicity.
            url = f"{self._base_url}/api/pull"
            session = self._get_session()
            payload = {"name": model, "stream": True}

            async with session.post(url, json=payload, timeout=_PULL_TIMEOUT) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.error("Failed to pull model '%s': HTTP %s -- %s", model, resp.status, body)
                    return False

                # Read streamed progress lines.
                last_status = ""
                async for line in resp.content:
                    decoded = line.decode("utf-8", errors="replace").strip()
                    if not decoded:
                        continue
                    try:
                        chunk = _json.loads(decoded)
                        status = chunk.get("status", "")
                        if status != last_status:
                            # Avoid spamming the log with identical status lines
                            # (e.g. repeated "downloading" with different %).
                            total = chunk.get("total", 0)
                            completed = chunk.get("completed", 0)
                            if total:
                                pct = completed / total * 100
                                logger.info(
                                    "Pulling '%s': %s (%.1f%%)", model, status, pct
                                )
                            else:
                                logger.info("Pulling '%s': %s", model, status)
                            last_status = status
                    except Exception:  # noqa: BLE001
                        # Non-JSON lines are ignored.
                        pass

            logger.info("Model '%s' pulled successfully.", model)
            return True

        except (aiohttp.ClientError, OSError, TimeoutError) as exc:
            logger.error("Failed to pull model '%s': %s", model, exc)
            return False

    # -- Lifecycle ----------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP session.

        Safe to call multiple times or when no session was ever created.
        """
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> OllamaClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.close()

    def __repr__(self) -> str:
        return f"OllamaClient(base_url={self._base_url!r})"
