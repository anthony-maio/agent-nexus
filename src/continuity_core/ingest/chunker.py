from __future__ import annotations

from typing import List


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200, min_chunk_size: int = 200) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not text:
        return []
    overlap = max(0, min(overlap, chunk_size - 1))
    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: List[str] = []
    start = 0
    total = len(cleaned)
    while start < total:
        end = min(total, start + chunk_size)
        if end < total:
            window = cleaned[start:end]
            end = _find_break(window, start, min_chunk_size, end)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= total:
            break
        start = max(0, end - overlap)
        if start >= total:
            break
    return chunks


def _find_break(window: str, start: int, min_chunk_size: int, end: int) -> int:
    for sep in ("\n\n", "\n", ". ", " "):
        idx = window.rfind(sep)
        if idx >= min_chunk_size:
            return start + idx + len(sep)
    return end
