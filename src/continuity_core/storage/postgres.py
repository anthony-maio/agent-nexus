from __future__ import annotations

import json
from typing import List, Optional

import psycopg
from psycopg.rows import dict_row

from continuity_core.event_log import Event


class PostgresEventStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn = psycopg.connect(dsn, row_factory=dict_row, connect_timeout=3)
        self._ensure_schema()

    def close(self) -> None:
        self._conn.close()

    def _ensure_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS event_log (
                    id BIGSERIAL PRIMARY KEY,
                    ts DOUBLE PRECISION NOT NULL,
                    actor TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    input TEXT NOT NULL,
                    output TEXT NOT NULL,
                    tags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                )
                """
            )
        self._conn.commit()

    def append(self, event: Event) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO event_log (ts, actor, intent, input, output, tags, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    event.timestamp,
                    event.actor,
                    event.intent,
                    event.input,
                    event.output,
                    event.tags,
                    json.dumps(event.metadata),
                ),
            )
        self._conn.commit()

    def tail(self, n: int = 10) -> List[Event]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT ts, actor, intent, input, output, tags, metadata
                FROM event_log
                ORDER BY id DESC
                LIMIT %s
                """,
                (n,),
            )
            rows = cur.fetchall()
        rows = list(reversed(rows))
        return [self._row_to_event(r) for r in rows]

    def query(self, tag: Optional[str] = None, limit: int = 50) -> List[Event]:
        with self._conn.cursor() as cur:
            if tag:
                cur.execute(
                    """
                    SELECT ts, actor, intent, input, output, tags, metadata
                    FROM event_log
                    WHERE %s = ANY(tags)
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    (tag.lower(), limit),
                )
            else:
                cur.execute(
                    """
                    SELECT ts, actor, intent, input, output, tags, metadata
                    FROM event_log
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
            rows = cur.fetchall()
        rows = list(reversed(rows))
        return [self._row_to_event(r) for r in rows]

    @staticmethod
    def _row_to_event(row: dict) -> Event:
        return Event(
            timestamp=float(row["ts"]),
            actor=row["actor"],
            intent=row["intent"],
            input=row["input"],
            output=row["output"],
            tags=[t.lower() for t in (row.get("tags") or [])],
            metadata=row.get("metadata") or {},
        )
