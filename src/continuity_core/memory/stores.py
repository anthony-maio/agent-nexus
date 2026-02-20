from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Protocol, Tuple


EmbedFn = Callable[[str], List[float]]


@dataclass
class MemoryItem:
    content: Any
    embedding: List[float]
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    salience: float = 1.0
    metadata: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def touch(self) -> None:
        self.access_count += 1
        self.last_access = time.time()


class MemoryStore(Protocol):
    def add(self, content: Any, embedding: Optional[List[float]] = None,
            salience: float = 1.0, metadata: Optional[dict] = None) -> MemoryItem:
        ...

    def query(self, query: str, embedding: Optional[List[float]] = None,
              top_k: int = 5) -> List[Tuple[MemoryItem, float]]:
        ...

    def apply_decay(self, decay_rate: float, time_unit_sec: float) -> None:
        ...

    def prune(self, min_salience: float) -> int:
        ...


class InMemoryStore:
    def __init__(self, capacity: int, embed_fn: Optional[EmbedFn] = None) -> None:
        self.capacity = capacity
        self.embed_fn = embed_fn
        self._items: List[MemoryItem] = []

    def add(self, content: Any, embedding: Optional[List[float]] = None,
            salience: float = 1.0, metadata: Optional[dict] = None) -> MemoryItem:
        if embedding is None:
            if not self.embed_fn:
                raise ValueError("embed_fn required when embedding is not provided")
            embedding = self.embed_fn(str(content))
        item = MemoryItem(content=content, embedding=embedding, salience=salience, metadata=metadata or {})
        if len(self._items) >= self.capacity:
            self._items.pop(0)
        self._items.append(item)
        return item

    def query(self, query: str, embedding: Optional[List[float]] = None,
              top_k: int = 5) -> List[Tuple[MemoryItem, float]]:
        if not self._items:
            return []
        if embedding is None:
            if not self.embed_fn:
                raise ValueError("embed_fn required when embedding is not provided")
            embedding = self.embed_fn(query)
        scored: List[Tuple[MemoryItem, float]] = []
        for item in self._items:
            sim = _cosine_similarity(embedding, item.embedding)
            scored.append((item, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        for item, _ in scored[:top_k]:
            item.touch()
        return scored[:top_k]

    def apply_decay(self, decay_rate: float, time_unit_sec: float) -> None:
        now = time.time()
        for item in self._items:
            elapsed = max(0.0, now - item.last_access)
            periods = elapsed / max(1.0, time_unit_sec)
            item.salience *= (decay_rate ** periods)

    def prune(self, min_salience: float) -> int:
        before = len(self._items)
        self._items = [item for item in self._items if item.salience >= min_salience]
        return before - len(self._items)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
