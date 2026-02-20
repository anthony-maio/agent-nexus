"""Tests for the conversation manager."""

import pytest

from nexus.swarm.conversation import ConversationManager


@pytest.fixture
def conversation():
    return ConversationManager()


@pytest.mark.asyncio
async def test_add_and_retrieve_messages(conversation):
    await conversation.add_message("model-a", "Hello", is_human=False)
    await conversation.add_message("human", "Hi there", is_human=True)
    history = conversation.get_history(limit=10)
    assert len(history) == 2
    assert history[0].model_id == "model-a"
    assert history[1].is_human is True


@pytest.mark.asyncio
async def test_history_limit_enforced(conversation):
    for i in range(60):
        await conversation.add_message(f"model-{i}", f"msg {i}")
    assert conversation.message_count == 50  # _max_history = 50


@pytest.mark.asyncio
async def test_clear_empties_history(conversation):
    await conversation.add_message("model-a", "Hello")
    conversation.clear()
    assert conversation.message_count == 0
    assert conversation.get_history() == []


@pytest.mark.asyncio
async def test_build_messages_for_model(conversation):
    await conversation.add_message("human", "What do you think?", is_human=True)
    await conversation.add_message("model-a", "I think X")
    await conversation.add_message("model-b", "I think Y")

    messages = conversation.build_messages_for_model("model-a", "You are model A", limit=10)
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are model A"
    # Human message should be role=user
    assert messages[1]["role"] == "user"
    # Own message should be role=assistant
    assert messages[2]["role"] == "assistant"
    # Other model's message should be role=user with attribution
    assert messages[3]["role"] == "user"
    assert "[" in messages[3]["content"]  # Has model name prefix
