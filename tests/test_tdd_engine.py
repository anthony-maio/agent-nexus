import pytest
from unittest.mock import AsyncMock, MagicMock


def test_nexus_llm_adapter_import():
    from nexus.synthesis.tdd_engine import NexusLLMAdapter
    assert NexusLLMAdapter is not None


def test_tdd_engine_import():
    from nexus.synthesis.tdd_engine import TDDEngine
    assert TDDEngine is not None


@pytest.mark.asyncio
async def test_nexus_llm_adapter_wraps_openrouter():
    """NexusLLMAdapter should wrap OpenRouterClient.chat()."""
    from nexus.synthesis.tdd_engine import NexusLLMAdapter

    mock_or = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = "Hello world"
    mock_response.finish_reason = "stop"
    mock_response.input_tokens = 10
    mock_response.output_tokens = 5
    mock_or.chat = AsyncMock(return_value=mock_response)

    adapter = NexusLLMAdapter(mock_or, model="test/model")
    result = await adapter.complete("Say hello")
    assert result.content == "Hello world"
    assert result.finish_reason == "stop"
    assert result.tokens_used == 15
    mock_or.chat.assert_called_once()
