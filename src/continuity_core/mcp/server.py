from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from continuity_core import __version__
from continuity_core.mcp.tools import (
    build_context,
    curiosity,
    introspect,
    read_events,
    status,
    write_event,
)
from continuity_core.mcp.tools.maintenance import maintenance


@dataclass
class Tool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def list(self) -> List[Dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "inputSchema": t.input_schema}
            for t in self._tools.values()
        ]

    def call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name].handler(arguments)


def _build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="c2.write_event",
            description="Append to the Event Log and return the stored event.",
            input_schema={
                "type": "object",
                "properties": {
                    "actor": {"type": "string"},
                    "intent": {"type": "string"},
                    "input": {"type": "string"},
                    "output": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "metadata": {"type": "object"},
                },
            },
            handler=write_event,
        )
    )
    registry.register(
        Tool(
            name="c2.context",
            description="Compose a prompt pack from candidate memory items.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "thread_id": {"type": "string"},
                    "use_pipeline": {"type": "boolean"},
                    "token_budget": {"type": "integer"},
                    "candidates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "text": {"type": "string"},
                                "store": {"type": "string"},
                                "token_cost": {"type": "integer"},
                                "relevance": {"type": "number"},
                                "recency_sec": {"type": "number"},
                                "centrality": {"type": "number"},
                                "confidence": {"type": "number"},
                                "task_match": {"type": "number"},
                                "salience": {"type": "number"},
                            },
                        },
                    },
                },
            },
            handler=build_context,
        )
    )
    registry.register(
        Tool(
            name="c2.introspect",
            description="Run MRA stress and void detection.",
            input_schema={
                "type": "object",
                "properties": {
                    "statements": {"type": "array", "items": {"type": "string"}},
                    "concept_contexts": {"type": "object"},
                    "graph": {"type": "object"},
                },
            },
            handler=introspect,
        )
    )
    registry.register(
        Tool(
            name="c2.curiosity",
            description=(
                "Return prioritized epistemic tensions, contradictions, and "
                "bridging questions. Call this to discover what the agent should "
                "be curious about — unresolved tensions in its own knowledge."
            ),
            input_schema={
                "type": "object",
                "properties": {},
            },
            handler=curiosity,
        )
    )
    registry.register(
        Tool(
            name="c2.maintenance",
            description=(
                "Run a Night Cycle maintenance pass: decay, pruning, stress "
                "recomputation, void scanning, and harmonic integration."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "graph": {"type": "object"},
                    "prune_threshold": {"type": "number"},
                    "statement_window": {"type": "integer"},
                },
            },
            handler=maintenance,
        )
    )
    registry.register(
        Tool(
            name="c2.events",
            description="Return the most recent events from the C2 event log.",
            input_schema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of events (1-50, default 10)",
                    },
                },
            },
            handler=read_events,
        )
    )
    registry.register(
        Tool(
            name="c2.status",
            description="Return C2 backend health, item counts, and MRA stress level.",
            input_schema={
                "type": "object",
                "properties": {},
            },
            handler=status,
        )
    )
    return registry


def _handle_request(registry: ToolRegistry, req: Dict[str, Any]) -> Dict[str, Any] | None:
    req_id = req.get("id")
    method = req.get("method")
    params = req.get("params") or {}

    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "continuity-core", "version": __version__},
            "capabilities": {"tools": {}},
        }
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    if method == "tools/list":
        result = {"tools": registry.list()}
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        try:
            payload = registry.call(name, arguments)
            result = {"content": [{"type": "text", "text": json.dumps(payload)}]}
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(exc)},
            }

    if req_id is None:
        return None

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Unknown method: {method}"},
    }


def main() -> None:
    registry = _build_registry()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            err = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()
            continue

        resp = _handle_request(registry, req)
        if resp is None:
            continue
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
