"""Sandbox runner FastAPI service."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from nexus_sandbox_runner.executors import StepRequest, build_executor_from_env

_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
_ALLOWED_ACTIONS: frozenset[str] = frozenset(
    {
        "navigate",
        "extract",
        "read",
        "write",
        "export",
        "submit",
    }
)


class ExecuteStepRequest(BaseModel):
    run_id: str = Field(pattern=_ID_PATTERN)
    step_id: str = Field(pattern=_ID_PATTERN)
    action_type: str = Field(min_length=1, max_length=64)
    instruction: str = Field(min_length=1, max_length=2000)


class ExecuteStepResponse(BaseModel):
    output_text: str
    citations: list[dict[str, str]]
    artifacts: list[dict[str, str]]
    metadata: dict[str, str]


def create_app() -> FastAPI:
    app = FastAPI(title="Agent Nexus Sandbox Runner", version="0.2.0")
    sandbox_root = Path("data/sandbox").resolve()
    sandbox_token = os.environ.get("SANDBOX_RUNNER_TOKEN", "").strip()
    executor = build_executor_from_env(dict(os.environ))
    sandbox_root.mkdir(parents=True, exist_ok=True)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "executor_backend": getattr(executor, "backend_name", "unknown"),
        }

    @app.post("/execute-step", response_model=ExecuteStepResponse)
    def execute_step(
        request: ExecuteStepRequest,
        x_sandbox_token: str | None = Header(default=None, alias="X-Sandbox-Token"),
    ) -> ExecuteStepResponse:
        if sandbox_token and x_sandbox_token != sandbox_token:
            raise HTTPException(status_code=401, detail="Invalid sandbox token")
        action = request.action_type.strip().lower()
        if action not in _ALLOWED_ACTIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported action_type: {action}",
            )
        try:
            step_result = executor.execute(
                StepRequest(
                    run_id=request.run_id,
                    step_id=request.step_id,
                    action_type=action,
                    instruction=request.instruction,
                ),
                sandbox_root,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return ExecuteStepResponse(
            output_text=step_result.output_text,
            citations=step_result.citations,
            artifacts=step_result.artifacts,
            metadata=step_result.metadata,
        )

    return app
