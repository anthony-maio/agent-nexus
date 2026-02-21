from __future__ import annotations

import hashlib
import math
from typing import List, Protocol

import requests

from continuity_core.config import C2Config


class Embedder(Protocol):
    def embed(self, text: str) -> List[float]:
        ...


class HashEmbedder:
    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    def embed(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [b / 255.0 for b in digest]
        if len(values) < self._dim:
            values = (values * ((self._dim // len(values)) + 1))[:self._dim]
        else:
            values = values[:self._dim]
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    def embed(self, text: str) -> List[float]:
        url = f"{self._base_url}/api/embeddings"
        resp = requests.post(url, json={"model": self._model, "prompt": text}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["embedding"]


class OpenRouterEmbedder:
    """Embedding via OpenRouter's /api/v1/embeddings endpoint."""

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._url = "https://openrouter.ai/api/v1/embeddings"

    def embed(self, text: str) -> List[float]:
        resp = requests.post(
            self._url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"input": text, "model": self._model},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("sentence-transformers is required for this embedder") from exc
        self._model = SentenceTransformer(model_name, device=device)

    def embed(self, text: str) -> List[float]:
        vec = self._model.encode([text], normalize_embeddings=True)
        return vec[0].tolist()


def build_embedder(config: C2Config) -> Embedder:
    backend = config.embedding_backend.lower()
    if backend == "openrouter":
        return OpenRouterEmbedder(config.openrouter_api_key, config.openrouter_embed_model)
    if backend == "ollama":
        return OllamaEmbedder(config.ollama_base_url, config.ollama_embed_model)
    if backend in {"sbert", "sentence-transformers"}:
        return SentenceTransformerEmbedder(config.embedding_model)
    return HashEmbedder()
