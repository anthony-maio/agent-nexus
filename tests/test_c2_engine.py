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


@pytest.mark.asyncio
async def test_write_event(mock_settings):
    """write_event should log to the C2 event store."""
    from nexus.integrations.c2_engine import C2Engine

    engine = C2Engine(mock_settings)
    await engine.start()
    result = await engine.write_event(
        actor="test",
        intent="test_intent",
        inp="hello",
        out="world",
    )
    assert result is not None
    assert result.get("actor") == "test"
    await engine.stop()


@pytest.mark.asyncio
async def test_events(mock_settings):
    """events() should return recently logged events."""
    from nexus.integrations.c2_engine import C2Engine

    engine = C2Engine(mock_settings)
    await engine.start()
    await engine.write_event(actor="test", intent="ping")
    result = await engine.events(limit=5)
    assert result is not None
    assert result.get("count", 0) >= 1
    await engine.stop()


@pytest.mark.asyncio
async def test_status(mock_settings):
    """status() should return backend health info."""
    from nexus.integrations.c2_engine import C2Engine

    engine = C2Engine(mock_settings)
    await engine.start()
    result = await engine.status()
    assert result is not None
    assert "event_log" in result
    await engine.stop()


@pytest.mark.asyncio
async def test_curiosity_returns_dict_or_none(mock_settings):
    """curiosity() should return a dict or None."""
    from nexus.integrations.c2_engine import C2Engine

    engine = C2Engine(mock_settings)
    await engine.start()
    result = await engine.curiosity()
    assert result is None or isinstance(result, dict)
    await engine.stop()
