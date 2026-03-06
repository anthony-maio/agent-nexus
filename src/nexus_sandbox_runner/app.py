"""Sandbox runner FastAPI service."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

from fastapi import FastAPI
from pydantic import BaseModel, Field


class ExecuteStepRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=64)
    step_id: str = Field(min_length=1, max_length=64)
    action_type: str = Field(min_length=1, max_length=64)
    instruction: str = Field(min_length=1, max_length=2000)


class ExecuteStepResponse(BaseModel):
    output_text: str
    citations: list[dict[str, str]]
    artifacts: list[dict[str, str]]
    metadata: dict[str, str]


def create_app() -> FastAPI:
    app = FastAPI(title="Agent Nexus Sandbox Runner", version="0.1.0")
    sandbox_root = Path("data/sandbox").resolve()
    sandbox_root.mkdir(parents=True, exist_ok=True)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/execute-step", response_model=ExecuteStepResponse)
    def execute_step(request: ExecuteStepRequest) -> ExecuteStepResponse:
        run_dir = sandbox_root / request.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        action = request.action_type.strip().lower()

        ts = datetime.now(timezone.utc).isoformat()
        output = _step_output(action, request.instruction)
        citations = _citations(action, request.instruction)
        artifacts: list[dict[str, str]] = []

        if action in {"extract", "write", "export"}:
            name = f"{request.step_id}-{action}.txt"
            rel_path = f"{request.run_id}/{name}"
            full_path = run_dir / name
            full_path.write_text(output, encoding="utf-8")
            artifacts.append(
                {
                    "kind": "text",
                    "name": name,
                    "rel_path": rel_path,
                    "sandbox_path": str(full_path),
                    "sha256": _sha256(full_path),
                }
            )

        # Keep per-step metadata log for auditability in sandbox space.
        meta_path = run_dir / f"{request.step_id}.json"
        meta_path.write_text(
            json.dumps(
                {
                    "step_id": request.step_id,
                    "run_id": request.run_id,
                    "action_type": action,
                    "instruction": request.instruction,
                    "timestamp": ts,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return ExecuteStepResponse(
            output_text=output,
            citations=citations,
            artifacts=artifacts,
            metadata={"timestamp": ts, "sandbox_root": str(sandbox_root)},
        )

    return app


def _step_output(action: str, instruction: str) -> str:
    if action == "navigate":
        return f"[sandbox-browser] Navigated and collected candidate pages for: {instruction}"
    if action == "extract":
        return (
            "[sandbox-browser] Evidence summary with citations prepared. "
            f"Focus: {instruction}"
        )
    if action == "export":
        return (
            "[sandbox-browser] Export artifact prepared for promotion into canonical workspace. "
            f"Request: {instruction}"
        )
    return f"[sandbox-browser] Executed action `{action}`: {instruction}"


def _citations(action: str, instruction: str) -> list[dict[str, str]]:
    if action not in {"navigate", "extract"}:
        return []
    query = quote_plus(instruction[:120])
    return [
        {
            "url": f"https://example.com/search?q={query}",
            "title": "Sandbox Search Result",
            "snippet": f"Evidence candidate for: {instruction[:120]}",
        }
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
