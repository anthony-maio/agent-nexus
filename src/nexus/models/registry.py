"""Model registry for Agent Nexus.

Defines all available models, their metadata, and selection helpers.
The registry is organized into two tiers:

- **Tier 1 (Swarm):** General-intelligence models that converse in Discord,
  plan, review, and make collaborative decisions.  All models in the swarm
  see each other's messages.
- **Tier 2 (Task):** Small, fast LiquidAI models dispatched for specific
  tasks (routing, reasoning, tool-calling, RAG, extraction).  They execute
  and return results without participating in the conversation.

Embedding models are listed separately and used by the Qdrant memory layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ModelTier(Enum):
    """Classification of a model's role inside the nexus."""

    SWARM = "swarm"
    """Tier 1 -- participates in multi-model Discord conversation."""

    TASK = "task"
    """Tier 2 -- dispatched for a specific job, returns result only."""


class ModelProvider(Enum):
    """Where the model is hosted / how it is reached."""

    OPENROUTER = "openrouter"
    """Accessed via the OpenRouter unified API."""

    OLLAMA = "ollama"
    """Running locally through Ollama."""

    DIRECT_API = "direct_api"
    """Called directly through the vendor's own API."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Metadata for a language model available to the nexus.

    Attributes:
        id: Provider-qualified model identifier (e.g. ``minimax/minimax-m2.5``).
        name: Human-friendly display name.
        provider: Where the model is hosted.
        tier: Whether the model participates in the swarm or runs tasks.
        context_window: Maximum context length in tokens.
        cost_input_per_m: Cost in USD per 1 million input tokens.
        cost_output_per_m: Cost in USD per 1 million output tokens.
        strengths: Short tags describing what the model excels at.
        requires_api_key: Environment variable name for the API key, or
            ``None`` when the model is freely available or local.
    """

    id: str
    name: str
    provider: ModelProvider
    tier: ModelTier
    context_window: int
    cost_input_per_m: float = 0.0
    cost_output_per_m: float = 0.0
    strengths: list[str] = field(default_factory=list)
    requires_api_key: str | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingSpec:
    """Metadata for an embedding model.

    Attributes:
        id: Provider-qualified model identifier.
        name: Human-friendly display name.
        provider: Where the model is hosted.
        dimensions: Output vector dimensionality.
        context_window: Maximum input length in tokens.
        cost_per_m: Cost in USD per 1 million tokens embedded.
    """

    id: str
    name: str
    provider: ModelProvider
    dimensions: int
    context_window: int
    cost_per_m: float = 0.0


# ---------------------------------------------------------------------------
# Tier 1 -- Swarm Models (general intelligence, converse in Discord)
# ---------------------------------------------------------------------------

SWARM_MODELS: dict[str, ModelSpec] = {
    # -- Default free / cheap models (ship with these) ---------------------
    "minimax/minimax-m2.5": ModelSpec(
        id="minimax/minimax-m2.5",
        name="MiniMax M2.5",
        provider=ModelProvider.OPENROUTER,
        tier=ModelTier.SWARM,
        context_window=1_000_000,
        cost_input_per_m=0.0,
        cost_output_per_m=0.0,
        strengths=["programming", "long-context", "reasoning"],
        requires_api_key="OPENROUTER_API_KEY",
    ),
    "z-ai/glm-5": ModelSpec(
        id="z-ai/glm-5",
        name="Z.ai GLM-5",
        provider=ModelProvider.OPENROUTER,
        tier=ModelTier.SWARM,
        context_window=204_000,
        cost_input_per_m=0.30,
        cost_output_per_m=2.55,
        strengths=["agentic-planning", "reasoning", "multilingual"],
        requires_api_key="OPENROUTER_API_KEY",
    ),
    "moonshotai/kimi-k2.5": ModelSpec(
        id="moonshotai/kimi-k2.5",
        name="Kimi K2.5",
        provider=ModelProvider.OPENROUTER,
        tier=ModelTier.SWARM,
        context_window=262_000,
        cost_input_per_m=0.23,
        cost_output_per_m=3.00,
        strengths=["multimodal", "tool-calling", "long-context"],
        requires_api_key="OPENROUTER_API_KEY",
    ),
    "qwen/qwen3-coder-next": ModelSpec(
        id="qwen/qwen3-coder-next",
        name="Qwen3 Coder Next",
        provider=ModelProvider.OPENROUTER,
        tier=ModelTier.SWARM,
        context_window=262_000,
        cost_input_per_m=0.12,
        cost_output_per_m=0.75,
        strengths=["coding", "moe", "efficient"],
        requires_api_key="OPENROUTER_API_KEY",
    ),
    # -- Optional premium models -------------------------------------------
    "anthropic/claude-sonnet-4-6": ModelSpec(
        id="anthropic/claude-sonnet-4-6",
        name="Claude Sonnet 4.6",
        provider=ModelProvider.OPENROUTER,
        tier=ModelTier.SWARM,
        context_window=200_000,
        cost_input_per_m=3.00,
        cost_output_per_m=15.00,
        strengths=["reasoning", "safety", "coding", "analysis"],
        requires_api_key="OPENROUTER_API_KEY",
    ),
    "google/gemini-3-flash": ModelSpec(
        id="google/gemini-3-flash",
        name="Gemini 3 Flash",
        provider=ModelProvider.OPENROUTER,
        tier=ModelTier.SWARM,
        context_window=1_000_000,
        cost_input_per_m=0.10,
        cost_output_per_m=0.40,
        strengths=["speed", "multimodal", "long-context"],
        requires_api_key="OPENROUTER_API_KEY",
    ),
    "openai/chatgpt-5.2": ModelSpec(
        id="openai/chatgpt-5.2",
        name="ChatGPT 5.2",
        provider=ModelProvider.OPENROUTER,
        tier=ModelTier.SWARM,
        context_window=128_000,
        cost_input_per_m=2.50,
        cost_output_per_m=10.00,
        strengths=["general-intelligence", "tool-use", "creativity"],
        requires_api_key="OPENROUTER_API_KEY",
    ),
}

# The four default swarm model IDs that ship out-of-the-box.
DEFAULT_SWARM_IDS: list[str] = [
    "minimax/minimax-m2.5",
    "z-ai/glm-5",
    "moonshotai/kimi-k2.5",
    "qwen/qwen3-coder-next",
]

# ---------------------------------------------------------------------------
# Tier 2 -- Task Agent Models (LiquidAI, dispatched for specific tasks)
# ---------------------------------------------------------------------------

TASK_MODELS: dict[str, ModelSpec] = {
    "router": ModelSpec(
        id="liquid/lfm-2.5-1.2b-instruct:free",
        name="LFM 2.5 1.2B Instruct (Router/Classifier)",
        provider=ModelProvider.OPENROUTER,
        tier=ModelTier.TASK,
        context_window=32_000,
        cost_input_per_m=0.0,
        cost_output_per_m=0.0,
        strengths=["routing", "classification", "fast"],
    ),
    "reasoning": ModelSpec(
        id="liquid/lfm-2.5-1.2b-thinking:free",
        name="LFM 2.5 1.2B Thinking (Reasoning)",
        provider=ModelProvider.OPENROUTER,
        tier=ModelTier.TASK,
        context_window=32_000,
        cost_input_per_m=0.0,
        cost_output_per_m=0.0,
        strengths=["reasoning", "chain-of-thought", "fast"],
    ),
    "tool_calling": ModelSpec(
        id="LiquidAI/LFM2-1.2B-Tool",
        name="LFM2 1.2B Tool",
        provider=ModelProvider.OLLAMA,
        tier=ModelTier.TASK,
        context_window=32_000,
        cost_input_per_m=0.0,
        cost_output_per_m=0.0,
        strengths=["tool-calling", "function-execution", "local"],
    ),
    "rag": ModelSpec(
        id="LiquidAI/LFM2-1.2B-RAG",
        name="LFM2 1.2B RAG",
        provider=ModelProvider.OLLAMA,
        tier=ModelTier.TASK,
        context_window=32_000,
        cost_input_per_m=0.0,
        cost_output_per_m=0.0,
        strengths=["retrieval", "context-grounding", "local"],
    ),
    "extraction": ModelSpec(
        id="LiquidAI/LFM2-1.2B-Extract",
        name="LFM2 1.2B Extract",
        provider=ModelProvider.OLLAMA,
        tier=ModelTier.TASK,
        context_window=32_000,
        cost_input_per_m=0.0,
        cost_output_per_m=0.0,
        strengths=["data-extraction", "structured-output", "local"],
    ),
}

# ---------------------------------------------------------------------------
# Embedding Models
# ---------------------------------------------------------------------------

EMBEDDING_MODELS: dict[str, EmbeddingSpec] = {
    "qwen/qwen3-embedding-8b": EmbeddingSpec(
        id="qwen/qwen3-embedding-8b",
        name="Qwen3 Embedding 8B",
        provider=ModelProvider.OPENROUTER,
        dimensions=4096,
        context_window=32_000,
        cost_per_m=0.01,
    ),
    "openai/text-embedding-3-small": EmbeddingSpec(
        id="openai/text-embedding-3-small",
        name="OpenAI Embedding 3 Small",
        provider=ModelProvider.OPENROUTER,
        dimensions=1536,
        context_window=8_000,
        cost_per_m=0.02,
    ),
    "openai/text-embedding-3-large": EmbeddingSpec(
        id="openai/text-embedding-3-large",
        name="OpenAI Embedding 3 Large",
        provider=ModelProvider.OPENROUTER,
        dimensions=3072,
        context_window=8_000,
        cost_per_m=0.13,
    ),
    "google/gemini-embedding-001": EmbeddingSpec(
        id="google/gemini-embedding-001",
        name="Gemini Embedding 001",
        provider=ModelProvider.OPENROUTER,
        dimensions=3072,
        context_window=20_000,
        cost_per_m=0.15,
    ),
    "mxbai-embed-large-v1": EmbeddingSpec(
        id="mxbai-embed-large-v1",
        name="mxbai Embed Large v1 (Local)",
        provider=ModelProvider.OLLAMA,
        dimensions=1024,
        context_window=512,
        cost_per_m=0.0,
    ),
}

DEFAULT_EMBEDDING_ID: str = "qwen/qwen3-embedding-8b"


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------

def get_active_swarm(model_ids: list[str] | None = None) -> list[ModelSpec]:
    """Return the swarm models that the user has selected.

    Args:
        model_ids: Explicit list of model IDs to activate.  When ``None``
            or empty, the four default swarm models are returned.

    Returns:
        Ordered list of :class:`ModelSpec` instances for the active swarm.

    Raises:
        KeyError: If any *model_id* is not found in :data:`SWARM_MODELS`.
    """
    if not model_ids:
        model_ids = DEFAULT_SWARM_IDS

    active: list[ModelSpec] = []
    for mid in model_ids:
        if mid not in SWARM_MODELS:
            raise KeyError(
                f"Unknown swarm model '{mid}'. "
                f"Available: {sorted(SWARM_MODELS.keys())}"
            )
        active.append(SWARM_MODELS[mid])
    return active


def get_embedding(model_id: str | None = None) -> EmbeddingSpec:
    """Return the embedding model spec for the given ID.

    Args:
        model_id: The embedding model identifier.  When ``None``, the
            default embedding model is returned.

    Returns:
        The corresponding :class:`EmbeddingSpec`.

    Raises:
        KeyError: If *model_id* is not found in :data:`EMBEDDING_MODELS`.
    """
    if model_id is None:
        model_id = DEFAULT_EMBEDDING_ID

    if model_id not in EMBEDDING_MODELS:
        raise KeyError(
            f"Unknown embedding model '{model_id}'. "
            f"Available: {sorted(EMBEDDING_MODELS.keys())}"
        )
    return EMBEDDING_MODELS[model_id]
