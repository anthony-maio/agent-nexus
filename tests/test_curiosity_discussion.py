"""Tests for curiosity-driven swarm discussion."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_bot_mock():
    """Build a mock bot with the minimum attributes needed."""
    bot = MagicMock()
    bot.swarm_models = {"model/a": MagicMock(), "model/b": MagicMock()}
    bot.openrouter = MagicMock()
    bot.conversation = MagicMock()
    bot.conversation.add_message = AsyncMock()
    bot.conversation.build_messages_for_model = MagicMock(
        return_value=[{"role": "user", "content": "test"}]
    )
    bot.memory_store = MagicMock()
    bot.memory_store.is_connected = True
    bot.embeddings = MagicMock()
    bot.embeddings.embed_one = AsyncMock(return_value=[0.1] * 10)
    bot.memory_store.store = AsyncMock(return_value="mem-id")
    bot.crosstalk = MagicMock()
    bot.crosstalk.is_enabled = False
    bot.router = MagicMock()
    bot.router.nexus = MagicMock()
    bot.router.nexus.send = AsyncMock(return_value=MagicMock())
    bot.router.memory = MagicMock()
    bot.router.memory.send = AsyncMock()
    bot.c2 = MagicMock()
    bot.c2.is_running = True
    bot.c2.write_event = AsyncMock()
    bot._system_prompts = {"model/a": "You are A.", "model/b": "You are B."}
    bot.get_system_prompt = MagicMock(return_value="You are a model.")
    bot._spawn = MagicMock(side_effect=lambda coro: MagicMock())
    return bot


@pytest.mark.asyncio
async def test_trigger_curiosity_discussion_posts_to_nexus():
    """_trigger_curiosity_discussion posts model response to #nexus."""
    from nexus.orchestrator.loop import OrchestratorLoop

    bot = _make_bot_mock()
    response_mock = MagicMock()
    response_mock.content = "Interesting tension between X and Y."
    bot.openrouter.chat = AsyncMock(return_value=response_mock)

    loop = OrchestratorLoop(bot, interval=3600)

    curiosity_result = {
        "stress_level": 0.45,
        "contradictions": [{"s1": "A is true", "s2": "A is false", "score": 0.8}],
        "deep_tensions": [],
        "bridging_questions": ["What connects A and B?"],
        "suggested_action": "resolve_contradiction",
    }

    await loop._trigger_curiosity_discussion(curiosity_result)

    # Verify a model was called
    bot.openrouter.chat.assert_called_once()
    # Verify response posted to #nexus
    bot.router.nexus.send.assert_called()
    # Verify summary posted to #memory
    bot.router.memory.send.assert_called()


@pytest.mark.asyncio
async def test_night_cycle_triggers_discussion_on_high_stress():
    """_run_night_cycle triggers discussion when contradictions found."""
    from nexus.orchestrator.loop import OrchestratorLoop

    bot = _make_bot_mock()
    bot.c2.maintenance = AsyncMock(return_value={
        "stress_after": 0.35,
        "contradictions_found": 2,
        "deep_tensions_found": 0,
        "voids_found": 0,
    })
    bot.c2.curiosity = AsyncMock(return_value={
        "stress_level": 0.35,
        "contradictions": [{"s1": "X", "s2": "Y", "score": 0.7}],
        "deep_tensions": [],
        "bridging_questions": [],
        "suggested_action": "resolve_contradiction",
    })

    loop = OrchestratorLoop(bot, interval=3600)

    with patch.object(
        loop, "_trigger_curiosity_discussion", new_callable=AsyncMock,
    ) as mock_discuss:
        with patch.object(
            loop, "_post_curiosity_findings", new_callable=AsyncMock,
        ):
            await loop._run_night_cycle()

    mock_discuss.assert_called_once()


@pytest.mark.asyncio
async def test_night_cycle_skips_discussion_when_no_signals():
    """_run_night_cycle does NOT trigger discussion when nothing found."""
    from nexus.orchestrator.loop import OrchestratorLoop

    bot = _make_bot_mock()
    bot.c2.maintenance = AsyncMock(return_value={
        "stress_after": 0.05,
        "contradictions_found": 0,
        "deep_tensions_found": 0,
        "voids_found": 0,
    })

    loop = OrchestratorLoop(bot, interval=3600)

    with patch.object(
        loop, "_trigger_curiosity_discussion", new_callable=AsyncMock,
    ) as mock_discuss:
        with patch.object(
            loop, "_post_curiosity_findings", new_callable=AsyncMock,
        ):
            await loop._run_night_cycle()

    mock_discuss.assert_not_called()
