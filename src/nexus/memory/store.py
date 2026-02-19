from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, Filter, FieldCondition, MatchValue

log = logging.getLogger(__name__)


@dataclass
class Memory:
    """A single memory entry."""
    id: str
    content: str
    source: str          # Who created it: model_id or "human"
    channel: str         # Which channel context: "human", "nexus", "memory"
    timestamp: datetime
    metadata: dict = field(default_factory=dict)
    score: float = 0.0   # Relevance score from search


class MemoryStore:
    """Qdrant-backed vector memory for the swarm."""

    def __init__(self, url: str, collection: str, dimensions: int):
        self.url = url
        self.collection = collection
        self.dimensions = dimensions
        self._client: QdrantClient | None = None

    async def initialize(self) -> None:
        """Connect to Qdrant and ensure collection exists."""
        from qdrant_client import QdrantClient
        self._client = QdrantClient(url=self.url)
        collections = [c.name for c in self._client.get_collections().collections]
        if self.collection not in collections:
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.dimensions,
                    distance=Distance.COSINE,
                ),
            )
            log.info(f"Created Qdrant collection: {self.collection} (dims={self.dimensions})")
        else:
            log.info(f"Using existing Qdrant collection: {self.collection}")

    async def store(
        self,
        content: str,
        vector: list[float],
        source: str,
        channel: str,
        metadata: dict | None = None,
    ) -> str:
        """Store a memory. Returns the memory ID."""
        memory_id = str(uuid.uuid4())
        point = PointStruct(
            id=memory_id,
            vector=vector,
            payload={
                "content": content,
                "source": source,
                "channel": channel,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **(metadata or {}),
            },
        )
        self._client.upsert(
            collection_name=self.collection,
            points=[point],
        )
        return memory_id

    async def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        source_filter: str | None = None,
    ) -> list[Memory]:
        """Search memories by vector similarity."""
        search_filter = None
        if source_filter:
            search_filter = Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
            )
        results = self._client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=limit,
            query_filter=search_filter,
        ).points
        memories = []
        for point in results:
            payload = point.payload
            memories.append(Memory(
                id=str(point.id),
                content=payload.get("content", ""),
                source=payload.get("source", "unknown"),
                channel=payload.get("channel", "unknown"),
                timestamp=datetime.fromisoformat(payload.get("timestamp", datetime.now(timezone.utc).isoformat())),
                metadata={k: v for k, v in payload.items() if k not in ("content", "source", "channel", "timestamp")},
                score=point.score if hasattr(point, "score") else 0.0,
            ))
        return memories

    async def delete(self, memory_id: str) -> None:
        """Delete a memory by ID."""
        self._client.delete(
            collection_name=self.collection,
            points_selector=[memory_id],
        )

    async def count(self) -> int:
        """Get total number of memories."""
        info = self._client.get_collection(self.collection)
        return info.points_count

    @property
    def is_connected(self) -> bool:
        return self._client is not None
