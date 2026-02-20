from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Protocol


@dataclass
class Event:
    timestamp: float
    actor: str
    intent: str
    input: str
    output: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


class EventStore(Protocol):
    def append(self, event: Event) -> None:
        ...

    def tail(self, n: int = 10) -> List[Event]:
        ...

    def query(self, tag: Optional[str] = None, limit: int = 50) -> List[Event]:
        ...


class InMemoryEventStore:
    def __init__(self) -> None:
        self._events: List[Event] = []

    def append(self, event: Event) -> None:
        self._events.append(event)

    def tail(self, n: int = 10) -> List[Event]:
        return self._events[-n:]

    def query(self, tag: Optional[str] = None, limit: int = 50) -> List[Event]:
        if tag is None:
            return self._events[-limit:]
        tag_l = tag.lower()
        matches = [e for e in self._events if tag_l in [t.lower() for t in e.tags]]
        return matches[-limit:]


class EventLog:
    def __init__(self, store: Optional[EventStore] = None) -> None:
        self._store = store or InMemoryEventStore()

    def log(self, actor: str, intent: str, inp: str, out: str,
            tags: Optional[Iterable[str]] = None,
            metadata: Optional[Dict[str, str]] = None) -> Event:
        event = Event(
            timestamp=time.time(),
            actor=actor,
            intent=intent,
            input=inp,
            output=out,
            tags=[t.lower() for t in (tags or [])],
            metadata=metadata or {},
        )
        self._store.append(event)
        return event

    def tail(self, n: int = 10) -> List[Event]:
        return self._store.tail(n)

    def query(self, tag: Optional[str] = None, limit: int = 50) -> List[Event]:
        return self._store.query(tag=tag, limit=limit)
