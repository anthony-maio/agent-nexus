"""Service context wiring for Nexus API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from nexus_api.adapters import SandboxExecutionAdapter, WebInteractionAdapter
from nexus_api.config import ApiSettings
from nexus_api.db import build_engine, build_session_factory
from nexus_api.migrator import run_migrations
from nexus_core.events import RunEventBus
from nexus_core.models import StepDefinition
from nexus_core.planner import plan_steps_for_objective


def default_steps_for_objective(objective: str) -> list[StepDefinition]:
    """Generate browser-first baseline steps for app-first execution."""
    return plan_steps_for_objective(objective)


@dataclass
class ApiContext:
    """Runtime context shared by route handlers."""

    settings: ApiSettings
    execution_adapter: Any
    interaction_adapter: Any
    db_engine: Any
    session_factory: sessionmaker[Session]
    events: RunEventBus


def build_context(settings: ApiSettings | None = None) -> ApiContext:
    settings = settings or ApiSettings()
    Path("data/app").mkdir(parents=True, exist_ok=True)
    settings.canonical_workspace_path.mkdir(parents=True, exist_ok=True)
    settings.sandbox_artifact_root_path.mkdir(parents=True, exist_ok=True)

    run_migrations(settings.APP_DATABASE_URL)
    db_engine = build_engine(settings.APP_DATABASE_URL)
    session_factory = build_session_factory(db_engine)

    events = RunEventBus()
    execution_adapter = SandboxExecutionAdapter(
        base_url=settings.SANDBOX_RUNNER_URL,
        auth_token=settings.SANDBOX_RUNNER_TOKEN,
    )
    interaction_adapter = WebInteractionAdapter()
    return ApiContext(
        settings=settings,
        execution_adapter=execution_adapter,
        interaction_adapter=interaction_adapter,
        db_engine=db_engine,
        session_factory=session_factory,
        events=events,
    )
