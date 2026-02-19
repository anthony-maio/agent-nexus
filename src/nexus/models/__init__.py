"""Model registry and specifications for Agent Nexus."""

from nexus.models.embeddings import (
    EMBEDDING_DIMENSIONS,
    LOCAL_EMBEDDING_MODELS,
    EmbeddingError,
    EmbeddingProvider,
)
from nexus.models.ollama import (
    ChatResponse,
    OllamaClient,
    OllamaError,
    OllamaModelError,
    OllamaUnavailableError,
)
from nexus.models.openrouter import (
    OpenRouterClient,
    OpenRouterError,
    StreamChunk,
)
from nexus.models.registry import (
    EMBEDDING_MODELS,
    SWARM_MODELS,
    TASK_MODELS,
    EmbeddingSpec,
    ModelProvider,
    ModelSpec,
    ModelTier,
    get_active_swarm,
    get_embedding,
)

__all__ = [
    "ChatResponse",
    "EMBEDDING_DIMENSIONS",
    "EMBEDDING_MODELS",
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingSpec",
    "LOCAL_EMBEDDING_MODELS",
    "ModelProvider",
    "ModelSpec",
    "ModelTier",
    "OllamaClient",
    "OllamaError",
    "OllamaModelError",
    "OllamaUnavailableError",
    "OpenRouterClient",
    "OpenRouterError",
    "SWARM_MODELS",
    "StreamChunk",
    "TASK_MODELS",
    "get_active_swarm",
    "get_embedding",
]
