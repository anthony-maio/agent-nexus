from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set


DEFAULT_TEXT_EXTS = {
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".log",
    ".md",
    ".py",
    ".rst",
    ".sql",
    ".toml",
    ".ts",
    ".txt",
    ".yaml",
    ".yml",
}

DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    "node_modules",
    "private",
}


@dataclass
class Document:
    source_id: str
    text: str
    metadata: dict


def normalize_exts(exts: Iterable[str]) -> Set[str]:
    normalized: Set[str] = set()
    for ext in exts:
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.add(ext)
    return normalized


def discover_files(
    paths: Iterable[str | Path],
    allowed_exts: Optional[Set[str]] = None,
    exclude_dirs: Optional[Set[str]] = None,
    include_hidden: bool = False,
    follow_symlinks: bool = False,
) -> List[Path]:
    exts = normalize_exts(allowed_exts or [])
    excludes = {d.lower() for d in (exclude_dirs or set())}
    files: List[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.exists():
            continue
        if path.is_file():
            if _should_skip(path, exts, excludes, include_hidden):
                continue
            files.append(path)
            continue
        for file_path in path.rglob("*"):
            if file_path.is_dir():
                continue
            if not follow_symlinks and file_path.is_symlink():
                continue
            if _should_skip(file_path, exts, excludes, include_hidden):
                continue
            files.append(file_path)
    files.sort(key=lambda p: str(p))
    return files


def load_text_file(path: Path, max_bytes: Optional[int] = None) -> Optional[Document]:
    try:
        stat = path.stat()
    except OSError:
        return None
    if max_bytes and stat.st_size > max_bytes:
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data:
        return None
    text = _decode_bytes(data).strip()
    if not text:
        return None
    source_id = make_source_id(path)
    metadata = {
        "source_id": source_id,
        "source_path": str(path),
        "source_name": path.name,
        "source_ext": path.suffix.lower(),
        "source_bytes": stat.st_size,
        "source_mtime": stat.st_mtime,
        "content_hash": hashlib.sha1(data).hexdigest(),
    }
    return Document(source_id=source_id, text=text, metadata=metadata)


def make_source_id(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()
    return f"source:{digest}"


def _should_skip(path: Path, exts: Set[str], excludes: Set[str], include_hidden: bool) -> bool:
    if exts and path.suffix.lower() not in exts:
        return True
    for part in path.parts:
        part_lower = part.lower()
        if part_lower in excludes:
            return True
        if not include_hidden and part.startswith("."):
            return True
    return False


def _decode_bytes(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")
