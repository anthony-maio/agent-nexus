"""Tests for channel router pre-initialization guards."""

import pytest

from nexus.channels.router import ChannelRouter


def test_human_before_init_raises():
    router = ChannelRouter()
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = router.human


def test_nexus_before_init_raises():
    router = ChannelRouter()
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = router.nexus


def test_memory_before_init_raises():
    router = ChannelRouter()
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = router.memory


def test_is_bot_channel_before_init_returns_false():
    router = ChannelRouter()
    assert router.is_bot_channel(12345) is False


def test_ready_flag_initially_false():
    router = ChannelRouter()
    assert router._ready is False
