"""Shared fixtures for Agent Nexus test suite."""

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_openrouter():
    """Mock OpenRouter client."""
    client = AsyncMock()
    client.session_cost = 0.0
    client.chat = AsyncMock()
    client.embed = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_ollama():
    """Mock Ollama client."""
    client = AsyncMock()
    client.chat = AsyncMock()
    client.embed = AsyncMock()
    client.is_available = AsyncMock(return_value=True)
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_qdrant():
    """Mock Qdrant client."""
    client = MagicMock()
    client.get_collections = MagicMock()
    client.upsert = MagicMock()
    client.query_points = MagicMock()
    client.delete = MagicMock()
    client.get_collection = MagicMock()
    return client
