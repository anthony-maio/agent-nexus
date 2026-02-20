"""Tests for MemoryStore initialization guards."""

import pytest

from nexus.memory.store import MemoryStore


@pytest.fixture
def uninitialized_store():
    return MemoryStore(url="http://localhost:6333", collection="test", dimensions=4096)


@pytest.mark.asyncio
async def test_store_without_init_raises(uninitialized_store):
    with pytest.raises(RuntimeError, match="not initialized"):
        await uninitialized_store.store(
            content="test", vector=[0.0] * 4096, source="test", channel="test"
        )


@pytest.mark.asyncio
async def test_search_without_init_raises(uninitialized_store):
    with pytest.raises(RuntimeError, match="not initialized"):
        await uninitialized_store.search(query_vector=[0.0] * 4096)


@pytest.mark.asyncio
async def test_count_without_init_raises(uninitialized_store):
    with pytest.raises(RuntimeError, match="not initialized"):
        await uninitialized_store.count()


@pytest.mark.asyncio
async def test_delete_without_init_raises(uninitialized_store):
    with pytest.raises(RuntimeError, match="not initialized"):
        await uninitialized_store.delete("some-id")


def test_is_connected_false_before_init(uninitialized_store):
    assert uninitialized_store.is_connected is False
