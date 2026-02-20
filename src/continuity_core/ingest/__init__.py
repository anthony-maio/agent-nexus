"""Ingestion pipeline for Continuity Core."""

from continuity_core.ingest.chunker import chunk_text
from continuity_core.ingest.loaders import (
    DEFAULT_EXCLUDE_DIRS,
    DEFAULT_TEXT_EXTS,
    Document,
    discover_files,
    load_text_file,
)
from continuity_core.ingest.pipeline import IngestPipeline, IngestResult

__all__ = [
    "chunk_text",
    "DEFAULT_EXCLUDE_DIRS",
    "DEFAULT_TEXT_EXTS",
    "Document",
    "discover_files",
    "load_text_file",
    "IngestPipeline",
    "IngestResult",
]
