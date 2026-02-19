"""Unified embedding interface for Agent Nexus.

This module provides a single :class:`EmbeddingProvider` that wraps either
OpenRouter (cloud) or Ollama (local) embedding backends, selected transparently
based on the model's :attr:`~nexus.models.registry.ModelProvider`.

**CRITICAL**: The embedding model is selected once during onboarding and
**CANNOT** be changed after the first run without losing all existing vectors
in Qdrant.  The vector dimensionality is baked into the Qdrant collection
schema and is immutable once created.

Usage::

    from nexus.models.embeddings import EmbeddingProvider

    provider = EmbeddingProvider(
        model_id="qwen/qwen3-embedding-8b",
        openrouter_client=openrouter,
    )
    vectors = await provider.embed(["hello world", "agent nexus"])
    single  = await provider.embed_one("hello world")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from nexus.models.registry import EMBEDDING_MODELS, EmbeddingSpec, ModelProvider

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants -- derived from the canonical registry
# ---------------------------------------------------------------------------

EMBEDDING_DIMENSIONS: dict[str, int] = {
    spec.id: spec.dimensions for spec in EMBEDDING_MODELS.values()
}
"""Mapping of model ID to output vector dimensionality.

These values MUST match the Qdrant collection schema.  Changing the embedding
model after initial setup will cause a dimension mismatch and corrupt queries.
"""

LOCAL_EMBEDDING_MODELS: frozenset[str] = frozenset(
    spec.id
    for spec in EMBEDDING_MODELS.values()
    if spec.provider is ModelProvider.OLLAMA
)
"""Set of model IDs that run locally through Ollama rather than via OpenRouter."""


# ---------------------------------------------------------------------------
# Protocols -- structural typing for embedding clients
# ---------------------------------------------------------------------------

@runtime_checkable
class SupportsEmbed(Protocol):
    """Structural type for any client that can produce embeddings.

    Both the OpenRouter and Ollama client implementations must satisfy this
    protocol by exposing an async ``embed`` method.
    """

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts and return their vector representations.

        Args:
            model: The model identifier to use for embedding.
            texts: A list of text strings to embed.

        Returns:
            A list of float vectors, one per input text, each with
            dimensionality matching the model specification.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class EmbeddingError(Exception):
    """Raised when an embedding operation fails.

    Common causes include:

    * The required client (OpenRouter or Ollama) was not provided.
    * The upstream API returned an error or unexpected payload.
    * The model ID is not recognised by the backend.
    * A dimension mismatch was detected after embedding.
    """


# ---------------------------------------------------------------------------
# Embedding provider
# ---------------------------------------------------------------------------

class EmbeddingProvider:
    """Unified embedding interface.

    Wraps OpenRouter or Ollama depending on whether the configured model is
    local or cloud-hosted.  The backend is resolved once at construction time
    and cannot change for the lifetime of the instance.

    Args:
        model_id: The embedding model ID from config
            (e.g. ``"qwen/qwen3-embedding-8b"``).
        openrouter_client: An OpenRouter client instance that satisfies
            :class:`SupportsEmbed`.  Required when *model_id* is a cloud model.
        ollama_client: An Ollama client instance that satisfies
            :class:`SupportsEmbed`.  Required when *model_id* is a local model.

    Raises:
        EmbeddingError: If *model_id* is not found in the embedding registry.

    Example::

        provider = EmbeddingProvider(
            model_id="qwen/qwen3-embedding-8b",
            openrouter_client=my_openrouter,
        )
        vecs = await provider.embed(["hello", "world"])
    """

    __slots__ = (
        "model_id",
        "spec",
        "dimensions",
        "_openrouter",
        "_ollama",
        "_is_local",
    )

    def __init__(
        self,
        model_id: str,
        openrouter_client: Any = None,
        ollama_client: Any = None,
    ) -> None:
        self.model_id: str = model_id

        # Resolve the spec from the canonical registry.
        if model_id not in EMBEDDING_MODELS:
            raise EmbeddingError(
                f"Unknown embedding model '{model_id}'. "
                f"Available: {sorted(EMBEDDING_MODELS.keys())}"
            )

        self.spec: EmbeddingSpec = EMBEDDING_MODELS[model_id]
        self._is_local: bool = model_id in LOCAL_EMBEDDING_MODELS
        self.dimensions: int = self.spec.dimensions
        self._openrouter: Any = openrouter_client
        self._ollama: Any = ollama_client

        # Eagerly validate that the required client is present so callers
        # get a clear error at construction time rather than mid-request.
        if self._is_local and self._ollama is None:
            logger.warning(
                "EmbeddingProvider created for local model '%s' without an "
                "Ollama client -- embed() will raise until one is set.",
                model_id,
            )
        if not self._is_local and self._openrouter is None:
            logger.warning(
                "EmbeddingProvider created for cloud model '%s' without an "
                "OpenRouter client -- embed() will raise until one is set.",
                model_id,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            A list of float vectors with length equal to ``len(texts)``.
            Each vector has :attr:`dimensions` components.

        Raises:
            EmbeddingError: If the required backend client is unavailable,
                the input is empty, or the backend returns an unexpected
                number of vectors.
        """
        if not texts:
            raise EmbeddingError("Cannot embed an empty list of texts.")

        vectors = await self._dispatch(texts)

        # Sanity-check: the backend must return exactly one vector per input.
        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"Dimension mismatch: sent {len(texts)} texts but received "
                f"{len(vectors)} vectors from model '{self.model_id}'."
            )

        return vectors

    async def embed_one(self, text: str) -> list[float]:
        """Embed a single text string.

        Convenience wrapper around :meth:`embed` for the common case of
        embedding a single query or document.

        Args:
            text: The text to embed.

        Returns:
            A single float vector with :attr:`dimensions` components.

        Raises:
            EmbeddingError: Propagated from :meth:`embed`.
        """
        results = await self.embed([text])
        return results[0]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _dispatch(self, texts: list[str]) -> list[list[float]]:
        """Route the embedding request to the correct backend.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            Raw vector list from the backend.

        Raises:
            EmbeddingError: If the backend client is ``None`` or the API
                call fails.
        """
        if self._is_local:
            if self._ollama is None:
                raise EmbeddingError(
                    "Ollama client is not available.  Cannot produce "
                    f"embeddings for local model '{self.model_id}'.  "
                    "Ensure Ollama is running and the client was provided "
                    "during EmbeddingProvider construction."
                )
            try:
                return await self._ollama.embed(self.model_id, texts)
            except EmbeddingError:
                raise
            except Exception as exc:
                raise EmbeddingError(
                    f"Ollama embedding failed for model '{self.model_id}': {exc}"
                ) from exc
        else:
            if self._openrouter is None:
                raise EmbeddingError(
                    "OpenRouter client is not available.  Cannot produce "
                    f"embeddings for cloud model '{self.model_id}'.  "
                    "Ensure OPENROUTER_API_KEY is set and the client was "
                    "provided during EmbeddingProvider construction."
                )
            try:
                return await self._openrouter.embed(self.model_id, texts)
            except EmbeddingError:
                raise
            except Exception as exc:
                raise EmbeddingError(
                    f"OpenRouter embedding failed for model '{self.model_id}': {exc}"
                ) from exc

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        backend = "ollama" if self._is_local else "openrouter"
        return (
            f"EmbeddingProvider(model_id={self.model_id!r}, "
            f"dimensions={self.dimensions}, backend={backend!r})"
        )
