from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels


EmbedFn = Callable[[str], List[float]]


@dataclass
class QdrantResult:
    id: str
    score: float
    payload: Dict[str, Any]


class QdrantMemoryStore:
    def __init__(self, url: str, collection: str, embed_fn: EmbedFn, vector_size: int) -> None:
        self._client = QdrantClient(url=url)
        self._collection = collection
        self._embed_fn = embed_fn
        self._vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            self._client.get_collection(self._collection)
        except Exception:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qmodels.VectorParams(size=self._vector_size, distance=qmodels.Distance.COSINE),
            )

    def remember(self, content: str, memory_type: str, importance: int = 5,
                 metadata: Optional[Dict[str, Any]] = None) -> str:
        vector = self._embed_fn(content)
        mem_id = str(uuid.uuid4())
        now = time.time()
        payload = {
            "content": content,
            "type": memory_type,
            "importance": importance,
            "created_at": now,
            "last_accessed": now,
        }
        if metadata:
            payload.update(metadata)
        self._client.upsert(
            collection_name=self._collection,
            points=[qmodels.PointStruct(id=mem_id, vector=vector, payload=payload)],
        )
        return mem_id

    def recall(self, query: str, top_k: int = 5,
               type_filter: Optional[str] = None) -> List[QdrantResult]:
        qvec = self._embed_fn(query)
        qfilter = None
        if type_filter:
            qfilter = qmodels.Filter(must=[qmodels.FieldCondition(key="type", match=qmodels.MatchValue(value=type_filter))])
        results = self._client.search(
            collection_name=self._collection,
            query_vector=qvec,
            query_filter=qfilter,
            limit=top_k,
        )
        out: List[QdrantResult] = []
        for r in results:
            payload = dict(r.payload or {})
            out.append(QdrantResult(id=str(r.id), score=float(r.score), payload=payload))
        return out

    def update_access(self, ids: List[str]) -> None:
        if not ids:
            return
        now = time.time()
        self._client.set_payload(
            collection_name=self._collection,
            payload={"last_accessed": now},
            points=ids,
        )
