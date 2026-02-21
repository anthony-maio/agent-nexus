from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class C2Config:
    # Core stores
    redis_url: str = os.getenv("C2_REDIS_URL", "redis://localhost:6379/0")
    qdrant_url: str = os.getenv("C2_QDRANT_URL", "http://localhost:6333")
    postgres_url: str = os.getenv("C2_POSTGRES_URL", "postgresql://c2:c2@localhost:5432/continuity_core")
    neo4j_uri: str = os.getenv("C2_NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("C2_NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("C2_NEO4J_PASSWORD", "letmein")
    embedding_backend: str = os.getenv("C2_EMBEDDING_BACKEND", "hash")
    embedding_model: str = os.getenv("C2_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    ollama_base_url: str = os.getenv("C2_OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_embed_model: str = os.getenv("C2_OLLAMA_EMBED_MODEL", "nomic-embed-text")
    openrouter_api_key: str = os.getenv("C2_OPENROUTER_API_KEY", "")
    openrouter_embed_model: str = os.getenv("C2_OPENROUTER_EMBED_MODEL", "qwen/qwen3-embedding-8b")

    # Context composer
    token_budget: int = int(os.getenv("C2_TOKEN_BUDGET", "2048"))
    epsilon: float = float(os.getenv("C2_EPSILON", "0.05"))
    lambda_penalty: float = float(os.getenv("C2_LAMBDA", "0.001"))

    # Decay and consolidation
    edge_half_life_days: float = float(os.getenv("C2_EDGE_HALF_LIFE_DAYS", "7"))
    decay_rate: float = float(os.getenv("C2_DECAY_RATE", "0.95"))
    decay_time_unit_sec: float = float(os.getenv("C2_DECAY_TIME_UNIT_SEC", "86400"))
    recency_half_life_days: float = float(os.getenv("C2_RECENCY_HALF_LIFE_DAYS", "14"))


def load_config() -> C2Config:
    return C2Config()
