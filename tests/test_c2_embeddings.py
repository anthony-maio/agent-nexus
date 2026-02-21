"""Tests for C2 embedding backends."""

from unittest.mock import MagicMock, patch

import pytest


def test_openrouter_embedder_returns_vector():
    """OpenRouterEmbedder calls the API and returns a float list."""
    from continuity_core.memory.embeddings import OpenRouterEmbedder

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [{"embedding": [0.1, 0.2, 0.3]}],
    }
    mock_response.raise_for_status = MagicMock()

    with patch("continuity_core.memory.embeddings.requests.post", return_value=mock_response) as mock_post:
        embedder = OpenRouterEmbedder(api_key="test-key", model="test/model")
        result = embedder.embed("hello world")

    assert result == [0.1, 0.2, 0.3]
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert "Authorization" in call_kwargs[1]["headers"]
    assert call_kwargs[1]["headers"]["Authorization"] == "Bearer test-key"


def test_build_embedder_openrouter():
    """build_embedder returns OpenRouterEmbedder for 'openrouter' backend."""
    from continuity_core.config import C2Config
    from continuity_core.memory.embeddings import OpenRouterEmbedder, build_embedder

    config = C2Config(
        embedding_backend="openrouter",
        openrouter_api_key="test-key",
        openrouter_embed_model="test/model",
    )
    embedder = build_embedder(config)
    assert isinstance(embedder, OpenRouterEmbedder)


def test_build_embedder_hash_fallback():
    """build_embedder returns HashEmbedder for unknown backend."""
    from continuity_core.config import C2Config
    from continuity_core.memory.embeddings import HashEmbedder, build_embedder

    config = C2Config(embedding_backend="unknown")
    embedder = build_embedder(config)
    assert isinstance(embedder, HashEmbedder)
