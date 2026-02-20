from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4


class NodeType(str, Enum):
    CONCEPT = "Concept"
    FACT = "Fact"
    ENTITY = "Entity"
    SOURCE = "Source"
    TASK = "Task"
    PERSON = "Person"
    EVENT = "Event"
    PREFERENCE = "Preference"
    NOTE = "Note"
    IDEA = "Idea"
    GOAL = "Goal"


class EdgeType(str, Enum):
    RELATED_TO = "RELATED_TO"
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    VALIDATES = "VALIDATES"
    DERIVED_FROM = "DERIVED_FROM"
    CONTRADICTS = "CONTRADICTS"
    ASSIGNED_TO = "ASSIGNED_TO"
    SCHEDULED_AT = "SCHEDULED_AT"
    TAGGED_WITH = "TAGGED_WITH"
    ADVANCES = "ADVANCES"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class GraphNode:
    type: NodeType
    name: str
    id: str = field(default_factory=lambda: f"node:{uuid4().hex}")
    description: Optional[str] = None
    confidence: Optional[float] = None
    origin: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "props": {
                "name": self.name,
                "description": self.description,
                "node_type": self.type.value,
                "confidence": self.confidence,
                "origin": self.origin,
                "user_id": self.user_id,
                "metadata": self.metadata,
                "updated_at": _utc_now(),
            },
        }


@dataclass
class GraphEdge:
    type: EdgeType
    start_id: str
    end_id: str
    confidence: Optional[float] = None
    origin: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_node_type: Optional[NodeType] = None
    end_node_type: Optional[NodeType] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_id": self.start_id,
            "end_id": self.end_id,
            "rel_type": self.type.value,
            "start_node_type": self.start_node_type.value if self.start_node_type else None,
            "end_node_type": self.end_node_type.value if self.end_node_type else None,
            "props": {
                "confidence": self.confidence,
                "origin": self.origin,
                "user_id": self.user_id,
                "metadata": self.metadata,
                "updated_at": _utc_now(),
            },
        }
