"""Tests for C2Client tool wrappers."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.integrations.c2_client import C2Client


@pytest.mark.asyncio
async def test_status_calls_correct_tool():
    """C2Client.status() calls c2.status tool."""
    client = C2Client()
    client._initialized = True
    client._process = MagicMock()
    client._process.returncode = None

    with pytest.MonkeyPatch.context() as mp:
        mock_call = AsyncMock(return_value={"neo4j": "connected"})
        mp.setattr(client, "_call_tool", mock_call)
        result = await client.status()

    mock_call.assert_called_once_with("c2.status")
    assert result == {"neo4j": "connected"}


@pytest.mark.asyncio
async def test_events_calls_correct_tool():
    """C2Client.events() calls c2.events tool with limit."""
    client = C2Client()
    client._initialized = True
    client._process = MagicMock()
    client._process.returncode = None

    with pytest.MonkeyPatch.context() as mp:
        mock_call = AsyncMock(return_value={"events": [], "count": 0})
        mp.setattr(client, "_call_tool", mock_call)
        result = await client.events(limit=5)

    mock_call.assert_called_once_with("c2.events", {"limit": 5})
    assert result == {"events": [], "count": 0}
