import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_settings(monkeypatch):
    monkeypatch.setenv("DISCORD_TOKEN", "test-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from nexus.config import NexusSettings
    return NexusSettings()


def test_c2_engine_import():
    """C2Engine class should be importable."""
    from nexus.integrations.c2_engine import C2Engine
    assert C2Engine is not None


def test_c2_engine_creates_config(mock_settings):
    """C2Engine should build a C2Config from NexusSettings."""
    from nexus.integrations.c2_engine import C2Engine
    engine = C2Engine.__new__(C2Engine)
    config = engine._build_c2_config(mock_settings)
    assert config.redis_url == mock_settings.REDIS_URL
    assert config.qdrant_url == mock_settings.QDRANT_URL
    assert config.postgres_url == mock_settings.POSTGRES_URL
    assert config.neo4j_uri == ""  # disabled
    assert config.embedding_backend == "openrouter"
    assert config.openrouter_api_key == mock_settings.OPENROUTER_API_KEY
    assert config.token_budget == mock_settings.C2_TOKEN_BUDGET


@pytest.mark.asyncio
async def test_c2_engine_start_stop(mock_settings):
    """C2Engine start/stop should work without real backends."""
    from nexus.integrations.c2_engine import C2Engine
    engine = C2Engine(mock_settings)
    assert not engine.is_running
    started = await engine.start()
    # start() succeeds even if backends are unavailable (graceful degradation)
    assert isinstance(started, bool)
    await engine.stop()
    assert not engine.is_running
