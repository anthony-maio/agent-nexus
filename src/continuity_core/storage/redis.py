from __future__ import annotations

import json
from typing import Any, Dict, List

import redis


class RedisWorkingContext:
    def __init__(self, url: str, max_len: int = 20, ttl_sec: int = 86400) -> None:
        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._max_len = max_len
        self._ttl_sec = ttl_sec

    def append(self, thread_id: str, message: Dict[str, Any]) -> None:
        key = self._key(thread_id)
        self._client.rpush(key, json.dumps(message))
        self._client.ltrim(key, -self._max_len, -1)
        self._client.expire(key, self._ttl_sec)

    def get_recent(self, thread_id: str, limit: int | None = None) -> List[Dict[str, Any]]:
        key = self._key(thread_id)
        if limit is None:
            limit = self._max_len
        raw = self._client.lrange(key, -limit, -1)
        return [json.loads(item) for item in raw]

    def clear(self, thread_id: str) -> None:
        self._client.delete(self._key(thread_id))

    @staticmethod
    def _key(thread_id: str) -> str:
        return f"c2:context:{thread_id}"
