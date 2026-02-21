"""Tests for the C2 events read tool."""

from unittest.mock import MagicMock, patch

from continuity_core.event_log import Event


def test_events_returns_recent_events():
    """c2.events returns recent events as a list of dicts."""
    mock_events = [
        Event(timestamp=1000.0, actor="human", intent="message", input="hello", output="", tags=["human"]),
        Event(timestamp=1001.0, actor="model", intent="response", input="", output="hi there", tags=["swarm"]),
    ]
    mock_mem = MagicMock()
    mock_mem.event_log.tail.return_value = mock_events

    with patch("continuity_core.mcp.tools.events_read.get_memory_system", return_value=mock_mem):
        from continuity_core.mcp.tools.events_read import read_events
        result = read_events({"limit": 10})

    assert len(result["events"]) == 2
    assert result["events"][0]["actor"] == "human"
    assert result["events"][1]["output"] == "hi there"
    mock_mem.event_log.tail.assert_called_once_with(n=10)


def test_events_default_limit():
    """c2.events defaults to 10 events when no limit specified."""
    mock_mem = MagicMock()
    mock_mem.event_log.tail.return_value = []

    with patch("continuity_core.mcp.tools.events_read.get_memory_system", return_value=mock_mem):
        from continuity_core.mcp.tools.events_read import read_events
        read_events({})

    mock_mem.event_log.tail.assert_called_once_with(n=10)


def test_events_clamps_limit_to_valid_range():
    """c2.events clamps limit between 1 and 50."""
    mock_mem = MagicMock()
    mock_mem.event_log.tail.return_value = []

    with patch("continuity_core.mcp.tools.events_read.get_memory_system", return_value=mock_mem):
        from continuity_core.mcp.tools.events_read import read_events

        read_events({"limit": 0})
        mock_mem.event_log.tail.assert_called_with(n=1)

        read_events({"limit": 100})
        mock_mem.event_log.tail.assert_called_with(n=50)
