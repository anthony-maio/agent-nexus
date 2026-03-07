"""Memory subsystem for Continuity Core."""

from .consolidation import ConsolidationPolicy, RecallGatedConsolidator
from .decay import DecayPolicy
from .embeddings import (
    Embedder,
    HashEmbedder,
    OllamaEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
)
from .gates import AdaptiveGate, BandpassGate, Gate, SmoothGate, ThresholdGate
from .stores import InMemoryStore, MemoryItem, MemoryStore
from .system import ScoredMemory, TieredMemorySystem

__all__ = [
    "MemoryItem",
    "MemoryStore",
    "InMemoryStore",
    "Gate",
    "ThresholdGate",
    "BandpassGate",
    "SmoothGate",
    "AdaptiveGate",
    "ConsolidationPolicy",
    "RecallGatedConsolidator",
    "DecayPolicy",
    "Embedder",
    "HashEmbedder",
    "OllamaEmbedder",
    "SentenceTransformerEmbedder",
    "build_embedder",
    "TieredMemorySystem",
    "ScoredMemory",
]
