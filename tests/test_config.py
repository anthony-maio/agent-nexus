"""Tests for config validation and repr redaction."""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from nexus.config import NexusSettings


def test_swarm_models_from_json_list():
    with patch.dict(os.environ, {
        "DISCORD_TOKEN": "test-token",
        "OPENROUTER_API_KEY": "sk-test",
        "SWARM_MODELS": '["model-a","model-b"]',
    }, clear=True):
        settings = NexusSettings()
        assert settings.SWARM_MODELS == ["model-a", "model-b"]


def test_missing_required_fields_raises():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValidationError):
            NexusSettings()


def test_sensitive_fields_redacted_in_repr():
    with patch.dict(os.environ, {
        "DISCORD_TOKEN": "secret-token-value",
        "OPENROUTER_API_KEY": "sk-secret-key",
    }, clear=True):
        settings = NexusSettings()
        r = repr(settings)
        assert "secret-token-value" not in r
        assert "sk-secret-key" not in r
        assert "***" in r


def test_optional_api_keys_show_none_in_repr():
    with patch.dict(os.environ, {
        "DISCORD_TOKEN": "test-token",
        "OPENROUTER_API_KEY": "sk-test",
    }, clear=True):
        settings = NexusSettings()
        r = repr(settings)
        # Optional keys not set should show None, not ***
        assert "ANTHROPIC_API_KEY=None" in r


def test_postgres_url_default(monkeypatch):
    """NexusSettings should include POSTGRES_URL with a sensible default."""
    monkeypatch.setenv("DISCORD_TOKEN", "test-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from nexus.config import NexusSettings
    settings = NexusSettings()
    assert hasattr(settings, "POSTGRES_URL")
    assert "postgresql" in settings.POSTGRES_URL


def test_c2_tuning_fields_exist(monkeypatch):
    """NexusSettings should include C2 algorithm tuning fields."""
    monkeypatch.setenv("DISCORD_TOKEN", "test-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from nexus.config import NexusSettings
    settings = NexusSettings()
    assert settings.C2_TOKEN_BUDGET == 2048
    assert settings.C2_EPSILON == 0.05
    assert settings.C2_LAMBDA == 0.001
    assert settings.C2_EDGE_HALF_LIFE_DAYS == 7.0
    assert settings.C2_DECAY_RATE == 0.95
    assert settings.C2_RECENCY_HALF_LIFE_DAYS == 14.0
