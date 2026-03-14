"""Service context wiring for Nexus API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from nexus_api.adapters import SandboxExecutionAdapter, WebInteractionAdapter
from nexus_api.adaptive_planner import ChatCompletionsAdaptivePlanner, OpenRouterAdaptivePlanner
from nexus_api.config import ApiSettings
from nexus_api.db import build_engine, build_session_factory
from nexus_api.migrator import run_migrations
from nexus_core.events import RunEventBus
from nexus_core.planner import (
    CompositeAdaptivePlanner,
    RuleAdaptivePlanner,
)


@dataclass
class ApiContext:
    """Runtime context shared by route handlers."""

    settings: ApiSettings
    execution_adapter: Any
    interaction_adapter: Any
    db_engine: Any
    session_factory: sessionmaker[Session]
    events: RunEventBus
    adaptive_planner: Any


def build_model_adaptive_planner(settings: ApiSettings) -> Any | None:
    provider = settings.APP_MODEL_REPLANNER_PROVIDER.strip().lower() or "openrouter"

    if provider == "openrouter":
        api_key = settings.APP_MODEL_REPLANNER_API_KEY.strip() or settings.OPENROUTER_API_KEY.strip()
        base_url = settings.APP_MODEL_REPLANNER_BASE_URL.strip() or settings.OPENROUTER_BASE_URL
        model = settings.APP_MODEL_REPLANNER_MODEL.strip() or settings.OPENROUTER_MODEL
        if not api_key or not model.strip():
            return None
        return OpenRouterAdaptivePlanner(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_sec=settings.APP_REPLANNER_TIMEOUT_SEC,
            max_steps=settings.APP_REPLANNER_MAX_STEPS,
        )

    if provider in {"openai_compatible", "openai-compatible", "chat_completions", "chat-completions", "local"}:
        base_url = settings.APP_MODEL_REPLANNER_BASE_URL.strip()
        model = settings.APP_MODEL_REPLANNER_MODEL.strip()
        if not base_url or not model:
            return None
        return ChatCompletionsAdaptivePlanner(
            api_key=settings.APP_MODEL_REPLANNER_API_KEY.strip(),
            model=model,
            base_url=base_url,
            timeout_sec=settings.APP_REPLANNER_TIMEOUT_SEC,
            max_steps=settings.APP_REPLANNER_MAX_STEPS,
            provider_name=provider,
        )

    return None


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
    rule_planner = RuleAdaptivePlanner()
    adaptive_planner: Any = rule_planner
    if settings.APP_ENABLE_MODEL_REPLANNER:
        model_planner = build_model_adaptive_planner(settings)
        if model_planner is not None:
            adaptive_planner = CompositeAdaptivePlanner([model_planner, rule_planner])
    return ApiContext(
        settings=settings,
        execution_adapter=execution_adapter,
        interaction_adapter=interaction_adapter,
        db_engine=db_engine,
        session_factory=session_factory,
        events=events,
        adaptive_planner=adaptive_planner,
    )
