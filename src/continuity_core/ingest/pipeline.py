from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Protocol, Set

from continuity_core.event_log import EventLog
from continuity_core.graph.canonical_schema import GraphNode, NodeType
from continuity_core.ingest.chunker import chunk_text
from continuity_core.ingest.loaders import (
    DEFAULT_EXCLUDE_DIRS,
    DEFAULT_TEXT_EXTS,
    Document,
    discover_files,
    load_text_file,
    normalize_exts,
)


@dataclass
class IngestResult:
    files_seen: int = 0
    docs_ingested: int = 0
    chunks_ingested: int = 0
    skipped: int = 0
    errors: int = 0
    error_details: list[dict[str, str]] = field(default_factory=list)
    duration_sec: float = 0.0


class MemorySystem(Protocol):
    event_log: EventLog
    neo4j: Any | None

    def remember(
        self,
        content: str,
        memory_type: str,
        importance: int = 5,
        metadata: Optional[dict] = None,
    ) -> str:
        ...


class IngestPipeline:
    def __init__(
        self,
        memory_system: Optional[MemorySystem] = None,
        allowed_exts: Optional[Set[str]] = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        max_bytes: int = 2_000_000,
        include_hidden: bool = False,
        exclude_dirs: Optional[Set[str]] = None,
        memory_type: str = "document_chunk",
        importance: int = 5,
        enable_graph: bool = True,
    ) -> None:
        if memory_system is None:
            from continuity_core.memory.system import TieredMemorySystem

            self._memory = TieredMemorySystem()
        else:
            self._memory = memory_system
        self.allowed_exts = normalize_exts(allowed_exts or DEFAULT_TEXT_EXTS)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_bytes = max_bytes
        self.include_hidden = include_hidden
        self.exclude_dirs = {d.lower() for d in (exclude_dirs or DEFAULT_EXCLUDE_DIRS)}
        self.memory_type = memory_type
        self.importance = importance
        self.enable_graph = enable_graph

    def ingest_paths(self, paths: Iterable[str]) -> IngestResult:
        start = time.time()
        result = IngestResult()
        files = discover_files(
            paths,
            allowed_exts=self.allowed_exts,
            exclude_dirs=self.exclude_dirs,
            include_hidden=self.include_hidden,
        )
        result.files_seen = len(files)

        for path in files:
            try:
                doc = load_text_file(path, max_bytes=self.max_bytes if self.max_bytes > 0 else None)
                if doc is None:
                    result.skipped += 1
                    continue
                chunks = chunk_text(doc.text, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
                if not chunks:
                    result.skipped += 1
                    continue
                self._store_document(doc, chunks)
                result.docs_ingested += 1
                result.chunks_ingested += len(chunks)
            except Exception as exc:
                result.errors += 1
                result.error_details.append({"path": str(path), "error": str(exc)})
                logging.getLogger(__name__).warning("Ingest failed for %s: %s", path, exc)
        result.duration_sec = time.time() - start
        return result

    def _store_document(self, doc: Document, chunks: list[str]) -> None:
        if self.enable_graph and self._memory.neo4j is not None:
            self._upsert_source_node(doc)

        for idx, chunk in enumerate(chunks):
            metadata = dict(doc.metadata)
            metadata.update(
                {
                    "chunk_index": idx,
                    "chunk_total": len(chunks),
                    "origin": "ingest",
                }
            )
            self._memory.remember(
                chunk,
                memory_type=self.memory_type,
                importance=self.importance,
                metadata=metadata,
            )

        _log_ingest_event(
            self._memory.event_log,
            doc,
            len(chunks),
            memory_type=self.memory_type,
        )

    def _upsert_source_node(self, doc: Document) -> None:
        graph = self._memory.neo4j
        if graph is None:
            return
        excerpt = _excerpt(doc.text, limit=200)
        node = GraphNode(
            type=NodeType.SOURCE,
            id=doc.source_id,
            name=doc.metadata.get("source_name", doc.metadata.get("source_path", "source")),
            description=excerpt,
            origin="ingest",
            metadata={
                "source_path": doc.metadata.get("source_path"),
                "source_ext": doc.metadata.get("source_ext"),
                "source_bytes": doc.metadata.get("source_bytes"),
                "source_mtime": doc.metadata.get("source_mtime"),
                "content_hash": doc.metadata.get("content_hash"),
            },
        )
        graph.upsert_nodes([node])


def _excerpt(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    return compact[:limit]


def _log_ingest_event(event_log: EventLog, doc: Document, chunk_count: int, memory_type: str) -> None:
    metadata = {
        "source_id": doc.source_id,
        "source_name": str(doc.metadata.get("source_name", "")),
        "chunks": str(chunk_count),
        "memory_type": memory_type,
    }
    event_log.log(
        actor="ingest",
        intent="ingest_document",
        inp=str(doc.metadata.get("source_path", "")),
        out=f"chunks={chunk_count}",
        tags=["ingest", "document"],
        metadata=metadata,
    )
