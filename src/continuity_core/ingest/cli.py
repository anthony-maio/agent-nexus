from __future__ import annotations

import argparse
from typing import List, Set

from continuity_core.ingest.loaders import DEFAULT_EXCLUDE_DIRS, DEFAULT_TEXT_EXTS, discover_files, normalize_exts
from continuity_core.ingest.pipeline import IngestPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Continuity Core ingestion pipeline")
    parser.add_argument("paths", nargs="*", default=["."], help="Files or directories to ingest")
    parser.add_argument("--ext", nargs="*", default=None, help="Allowed extensions, e.g. .md .txt .py")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap in characters")
    parser.add_argument("--max-bytes", type=int, default=2_000_000, help="Skip files larger than this size")
    parser.add_argument("--include-hidden", action="store_true", help="Include hidden files and folders")
    parser.add_argument("--include-private", action="store_true", help="Include the private/ directory")
    parser.add_argument("--exclude", nargs="*", default=None, help="Additional directory names to exclude")
    parser.add_argument("--memory-type", default="document_chunk", help="Memory type label")
    parser.add_argument("--importance", type=int, default=5, help="Importance score (1-10)")
    parser.add_argument("--no-graph", action="store_true", help="Skip Neo4j source node upserts")
    parser.add_argument("--dry-run", action="store_true", help="List files discovered and exit")
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    allowed_exts = normalize_exts(args.ext) if args.ext is not None else normalize_exts(DEFAULT_TEXT_EXTS)
    excludes: Set[str] = {d.lower() for d in DEFAULT_EXCLUDE_DIRS}
    if args.include_private:
        excludes.discard("private")
    if args.exclude:
        excludes.update(d.lower() for d in args.exclude)

    if args.dry_run:
        files = discover_files(
            args.paths,
            allowed_exts=allowed_exts,
            exclude_dirs=excludes,
            include_hidden=args.include_hidden,
        )
        print(f"discovered={len(files)}")
        for path in files:
            print(path)
        return 0

    pipeline = IngestPipeline(
        allowed_exts=allowed_exts,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_bytes=args.max_bytes,
        include_hidden=args.include_hidden,
        exclude_dirs=excludes,
        memory_type=args.memory_type,
        importance=args.importance,
        enable_graph=not args.no_graph,
    )
    result = pipeline.ingest_paths(args.paths)
    print(
        "files={files} docs={docs} chunks={chunks} skipped={skipped} errors={errors} duration={duration:.2f}s".format(
            files=result.files_seen,
            docs=result.docs_ingested,
            chunks=result.chunks_ingested,
            skipped=result.skipped,
            errors=result.errors,
            duration=result.duration_sec,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
