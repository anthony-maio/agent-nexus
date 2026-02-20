"""Memory subsystem for Continuity Core."""

from .stores import MemoryItem, MemoryStore, InMemoryStore
from .gates import Gate, ThresholdGate, BandpassGate, SmoothGate, AdaptiveGate
from .consolidation import ConsolidationPolicy, RecallGatedConsolidator
from .decay import DecayPolicy
from .embeddings import Embedder, HashEmbedder, OllamaEmbedder, SentenceTransformerEmbedder, build_embedder
from .system import TieredMemorySystem, ScoredMemory

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
