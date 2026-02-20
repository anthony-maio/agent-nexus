"""Central configuration for Agent Nexus.

All settings are loaded from environment variables (with ``.env`` file support
via *python-dotenv*).  Validation and type coercion are handled by
``pydantic-settings``.

Usage::

    from nexus.config import get_settings

    settings = get_settings()
    print(settings.DISCORD_TOKEN)

The :func:`get_settings` helper creates the :class:`NexusSettings` singleton
lazily so that importing this module never triggers validation before the
caller has had a chance to load a ``.env`` file or populate the environment.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Canonical .env locations (checked in order of priority).
ENV_PATHS: list[str] = ["config/.env", ".env"]

# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------


class NexusSettings(BaseSettings):
    """Validated configuration for the entire Agent Nexus stack.

    Required fields (no defaults):
        ``DISCORD_TOKEN``, ``OPENROUTER_API_KEY``

    Every other setting carries a sensible default so the system can start
    with just those two values.
    """

    model_config = SettingsConfigDict(
        # .env loading is handled by load_dotenv() in __main__.py so that
        # we can preprocess comma-separated list values before pydantic-
        # settings tries to JSON-parse them.  Do NOT set env_file here.
        env_file_encoding="utf-8",
        # Allow extra env vars without raising a validation error.
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Required -- no defaults
    # ------------------------------------------------------------------
    DISCORD_TOKEN: str = Field(
        ...,
        description="Discord bot token from the Developer Portal.",
    )
    OPENROUTER_API_KEY: str = Field(
        ...,
        description="API key for OpenRouter (https://openrouter.ai).",
    )

    # ------------------------------------------------------------------
    # Discord
    # ------------------------------------------------------------------
    DISCORD_GUILD_ID: int | None = Field(
        default=None,
        description=(
            "Restrict the bot to a single guild.  When ``None`` the bot "
            "auto-detects the first guild it joins."
        ),
    )

    # ------------------------------------------------------------------
    # Models -- main swarm (OpenRouter model IDs, comma-separated in env)
    # ------------------------------------------------------------------
    SWARM_MODELS: list[str] = Field(
        default=[
            "minimax/minimax-m2.5",
            "z-ai/glm-5",
            "moonshotai/kimi-k2.5",
            "qwen/qwen3-coder-next",
        ],
        description=(
            "Comma-separated list of OpenRouter model identifiers that "
            "form the core swarm."
        ),
    )

    # ------------------------------------------------------------------
    # Embedding model
    # ------------------------------------------------------------------
    EMBEDDING_MODEL: str = Field(
        default="qwen/qwen3-embedding-8b",
        description=(
            "Model used for vector embeddings.  CANNOT be changed after "
            "the first run without invalidating all stored vectors."
        ),
    )
    EMBEDDING_DIMENSIONS: int = Field(
        default=4096,
        description="Dimensionality of the embedding model output.",
    )

    # ------------------------------------------------------------------
    # Optional premium API keys
    # ------------------------------------------------------------------
    ANTHROPIC_API_KEY: str | None = Field(
        default=None,
        description="Anthropic API key (Claude models).",
    )
    OPENAI_API_KEY: str | None = Field(
        default=None,
        description="OpenAI API key.",
    )
    GOOGLE_API_KEY: str | None = Field(
        default=None,
        description="Google AI API key (Gemini models).",
    )

    # ------------------------------------------------------------------
    # Local models (Ollama)
    # ------------------------------------------------------------------
    OLLAMA_BASE_URL: str = Field(
        default="http://host.docker.internal:11434",
        description="Base URL for a local Ollama instance.",
    )

    # ------------------------------------------------------------------
    # PiecesOS
    # ------------------------------------------------------------------
    PIECES_MCP_ENABLED: bool = Field(
        default=False,
        description="Enable PiecesOS activity-tracking MCP integration.",
    )
    PIECES_MCP_URL: str = Field(
        default="http://localhost:39300",
        description="URL for the PiecesOS MCP server. Use http://host.docker.internal:39300 in Docker.",
    )

    # ------------------------------------------------------------------
    # Infrastructure
    # ------------------------------------------------------------------
    QDRANT_URL: str = Field(
        default="http://nexus-qdrant:6333",
        description="Qdrant vector-database gRPC/HTTP endpoint.",
    )
    QDRANT_COLLECTION: str = Field(
        default="nexus_memory",
        description="Name of the Qdrant collection used for shared memory.",
    )
    REDIS_URL: str = Field(
        default="redis://nexus-redis:6379/0",
        description="Redis connection URL for caching and pub/sub.",
    )

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
    ORCHESTRATOR_INTERVAL: int = Field(
        default=3600,
        ge=10,
        description="Seconds between autonomous orchestrator cycles.",
    )
    AUTONOMY_MODE: str = Field(
        default="escalate",
        description=(
            "Autonomy mode: 'observe' (always ask), 'escalate' (auto for "
            "low-risk, ask for high-risk), or 'autopilot' (auto-execute all)."
        ),
    )
    ACTIVITY_POLL_INTERVAL: int = Field(
        default=60,
        ge=10,
        description="Seconds between PiecesOS activity polls.",
    )
    CROSSTALK_PROBABILITY: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Probability that a swarm model spontaneously responds to "
            "another model's message."
        ),
    )
    CONSENSUS_THRESHOLD: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Agreement ratio required for multi-model consensus decisions."
        ),
    )

    # ------------------------------------------------------------------
    # Cost tracking
    # ------------------------------------------------------------------
    SESSION_COST_LIMIT: float = Field(
        default=10.0,
        ge=0.0,
        description="USD spending limit per session before a warning is emitted.",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("SWARM_MODELS", mode="before")
    @classmethod
    def _split_comma_separated_models(cls, value: Any) -> list[str]:
        """Accept a comma-separated string from the environment and split it
        into a proper list of model identifiers.

        If the value is already a list (e.g. when constructed from Python code)
        it is returned unchanged.
        """
        if isinstance(value, str):
            models = [m.strip() for m in value.split(",") if m.strip()]
            if not models:
                raise ValueError(
                    "SWARM_MODELS must contain at least one model identifier."
                )
            return models
        if isinstance(value, list):
            return value
        raise TypeError(
            f"SWARM_MODELS must be a comma-separated string or list, got {type(value).__name__}"
        )

    # ------------------------------------------------------------------
    # Repr safety -- redact secrets in logs / debug output
    # ------------------------------------------------------------------

    _SENSITIVE_FIELDS: ClassVar[set[str]] = {
        "DISCORD_TOKEN", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY", "GOOGLE_API_KEY",
    }

    def __repr__(self) -> str:
        fields = []
        for name in self.model_fields:
            val = getattr(self, name)
            if name in self._SENSITIVE_FIELDS:
                val = "***" if val else None
            fields.append(f"{name}={val!r}")
        return f"NexusSettings({', '.join(fields)})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_env_file() -> Path | None:
    """Return the first existing ``.env`` file from :data:`ENV_PATHS`."""
    for candidate in ENV_PATHS:
        p = Path(candidate)
        if p.is_file():
            return p
    return None


def has_config() -> bool:
    """Return ``True`` if a loadable ``.env`` file exists with required keys."""
    import os

    if os.environ.get("DISCORD_TOKEN") and os.environ.get("OPENROUTER_API_KEY"):
        return True
    env_file = find_env_file()
    if env_file is None:
        return False
    text = env_file.read_text(encoding="utf-8", errors="replace")
    has_token = any(
        line.strip().startswith("DISCORD_TOKEN=") and len(line.split("=", 1)[1].strip()) > 0
        for line in text.splitlines()
    )
    has_key = any(
        line.strip().startswith("OPENROUTER_API_KEY=") and len(line.split("=", 1)[1].strip()) > 0
        for line in text.splitlines()
    )
    return has_token and has_key


# ---------------------------------------------------------------------------
# Lazy singleton accessor
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def get_settings() -> NexusSettings:
    """Return the global :class:`NexusSettings` singleton.

    The instance is created on first call so that the module can be imported
    safely before any ``.env`` file has been loaded or environment variables
    have been set.  Subsequent calls return the cached instance.

    Raises:
        pydantic.ValidationError: If required settings (``DISCORD_TOKEN``,
            ``OPENROUTER_API_KEY``) are missing or any value fails validation.
    """
    logger.debug("Initialising NexusSettings from environment.")
    return NexusSettings()  # type: ignore[call-arg]
