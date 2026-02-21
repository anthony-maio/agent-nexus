"""Tests for the C2 status tool."""

from unittest.mock import MagicMock, patch


def test_status_returns_backend_health():
    """c2.status returns a dict with backend connection status."""
    mock_mem = MagicMock()
    mock_mem.neo4j = None
    mock_mem.qdrant = MagicMock()
    mock_mem.redis = None
    mock_mem._fallback = None
    mock_mem.event_log = MagicMock()
    mock_mem.event_log._store = MagicMock()
    mock_mem.event_log.tail.return_value = [1, 2, 3]
    mock_mem.embedder = MagicMock()
    mock_mem.embedder.__class__.__name__ = "HashEmbedder"
    type(mock_mem.embedder).__name__ = "HashEmbedder"
    mock_mem.get_mra_signals.return_value = None

    with patch(
        "continuity_core.mcp.tools.status.get_memory_system",
        return_value=mock_mem,
    ):
        from continuity_core.mcp.tools.status import status

        result = status({})

    assert result["neo4j"] == "offline"
    assert result["neo4j_nodes"] == 0
    assert result["qdrant"] == "connected"
    assert result["redis"] == "offline"
    assert result["embedding_backend"] == "HashEmbedder"
    assert result["stress_level"] == 0.0
    assert result["event_count"] == 3
    assert result["fallback_memory_count"] == 0
    assert result["event_backend"] == "in-memory"


def test_status_with_neo4j_connected():
    """c2.status reports neo4j as connected when available."""
    mock_mem = MagicMock()
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.single.return_value = {"cnt": 42}
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.run.return_value = mock_result
    mock_mem.neo4j._driver.session.return_value = mock_session

    mock_mem.qdrant = None
    mock_mem.redis = MagicMock()
    mock_mem._fallback = MagicMock()
    mock_mem._fallback._items = [1, 2, 3, 4, 5]
    mock_mem.event_log = MagicMock()
    mock_mem.event_log._store = MagicMock()
    mock_mem.event_log.tail.return_value = []
    mock_mem.embedder = MagicMock()
    type(mock_mem.embedder).__name__ = "SentenceTransformerEmbedder"
    mock_mem.get_mra_signals.return_value = None

    with patch(
        "continuity_core.mcp.tools.status.get_memory_system",
        return_value=mock_mem,
    ):
        from continuity_core.mcp.tools.status import status

        result = status({})

    assert result["neo4j"] == "connected"
    assert result["neo4j_nodes"] == 42
    assert result["qdrant"] == "offline"
    assert result["redis"] == "connected"
    assert result["fallback_memory_count"] == 5
    assert result["embedding_backend"] == "SentenceTransformerEmbedder"


def test_status_with_mra_stress():
    """c2.status reports stress level when MRA cache is available."""
    mock_mem = MagicMock()
    mock_mem.neo4j = None
    mock_mem.qdrant = None
    mock_mem.redis = None
    mock_mem._fallback = None
    mock_mem.event_log = MagicMock()
    mock_mem.event_log._store = MagicMock()
    mock_mem.event_log.tail.return_value = []
    mock_mem.embedder = MagicMock()
    type(mock_mem.embedder).__name__ = "HashEmbedder"

    mock_mra = MagicMock()
    mock_mra.last_stress.s_omega = 0.73
    mock_mem.get_mra_signals.return_value = mock_mra

    with patch(
        "continuity_core.mcp.tools.status.get_memory_system",
        return_value=mock_mem,
    ):
        from continuity_core.mcp.tools.status import status

        result = status({})

    assert result["stress_level"] == 0.73


def test_status_neo4j_error_handling():
    """c2.status reports neo4j as error when query fails."""
    mock_mem = MagicMock()
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.run.side_effect = RuntimeError("connection lost")
    mock_mem.neo4j._driver.session.return_value = mock_session

    mock_mem.qdrant = None
    mock_mem.redis = None
    mock_mem._fallback = None
    mock_mem.event_log = MagicMock()
    mock_mem.event_log._store = MagicMock()
    mock_mem.event_log.tail.return_value = []
    mock_mem.embedder = MagicMock()
    type(mock_mem.embedder).__name__ = "HashEmbedder"
    mock_mem.get_mra_signals.return_value = None

    with patch(
        "continuity_core.mcp.tools.status.get_memory_system",
        return_value=mock_mem,
    ):
        from continuity_core.mcp.tools.status import status

        result = status({})

    assert result["neo4j"] == "error"
    assert result["neo4j_nodes"] == 0
