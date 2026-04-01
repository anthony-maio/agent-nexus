"""Service context wiring for Nexus API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from nexus_api.adapters import (
    ExternalToolDispatchExecutionAdapter,
    SandboxExecutionAdapter,
    ToolAugmentedInteractionAdapter,
    WebInteractionAdapter,
)
from nexus_api.adaptive_planner import ChatCompletionsAdaptivePlanner, OpenRouterAdaptivePlanner
from nexus_api.config import ApiSettings
from nexus_api.db import build_engine, build_session_factory
from nexus_api.external_tools import (
    ExternalToolRegistry,
    StdioExternalToolInvoker,
    parse_external_tool_config,
)
from nexus_api.migrator import run_migrations
from nexus_api.model_router import ModelProfile, parse_model_router_config
from nexus_api.synthesis_bridge import SynthesisBridge, synthesis_skill_paths
from nexus_core.events import RunEventBus
from nexus_core.planner import (
    CompositeAdaptivePlanner,
    RuleAdaptivePlanner,
)
from nexus_core.skills import CapabilityResolver, SkillRegistry


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
    skill_registry: SkillRegistry
    capability_resolver: CapabilityResolver | None
    external_tool_registry: ExternalToolRegistry
    synthesis_bridge: SynthesisBridge | None


def build_model_adaptive_planner(settings: ApiSettings) -> Any | None:
    routed_profiles = parse_model_router_config(settings.APP_MODEL_ROUTER_CONFIG).profiles_for_role(
        "planning"
    )
    if routed_profiles:
        model_planners = [
            planner
            for planner in (
                _build_planner_from_profile(profile, settings) for profile in routed_profiles
            )
            if planner is not None
        ]
        if not model_planners:
            return None
        if len(model_planners) == 1:
            return model_planners[0]
        return CompositeAdaptivePlanner(model_planners)

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


def _build_planner_from_profile(profile: ModelProfile, settings: ApiSettings) -> Any | None:
    provider = profile.provider.strip().lower()
    model = profile.model.strip()
    if not provider or not model:
        return None

    if provider == "openrouter":
        api_key = profile.api_key or settings.APP_MODEL_REPLANNER_API_KEY.strip() or settings.OPENROUTER_API_KEY.strip()
        base_url = profile.base_url or settings.APP_MODEL_REPLANNER_BASE_URL.strip() or settings.OPENROUTER_BASE_URL
        if not api_key:
            return None
        return OpenRouterAdaptivePlanner(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_sec=settings.APP_REPLANNER_TIMEOUT_SEC,
            max_steps=settings.APP_REPLANNER_MAX_STEPS,
            route_label=profile.name,
        )

    if provider in {"openai_compatible", "openai-compatible", "chat_completions", "chat-completions", "local"}:
        base_url = profile.base_url or settings.APP_MODEL_REPLANNER_BASE_URL.strip()
        if not base_url:
            return None
        return ChatCompletionsAdaptivePlanner(
            api_key=profile.api_key or settings.APP_MODEL_REPLANNER_API_KEY.strip(),
            model=model,
            base_url=base_url,
            timeout_sec=settings.APP_REPLANNER_TIMEOUT_SEC,
            max_steps=settings.APP_REPLANNER_MAX_STEPS,
            provider_name=provider,
            route_label=profile.name,
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
    external_tool_registry = parse_external_tool_config(settings.APP_EXTERNAL_TOOL_CONFIG)
    external_tool_invoker = (
        StdioExternalToolInvoker()
        if any(
            (tool.transport or {}).get("kind") == "stdio"
            for tool in external_tool_registry.list_tools()
        )
        else None
    )
    sandbox_execution_adapter = SandboxExecutionAdapter(
        base_url=settings.SANDBOX_RUNNER_URL,
        auth_token=settings.SANDBOX_RUNNER_TOKEN,
    )
    execution_adapter = ExternalToolDispatchExecutionAdapter(
        base_adapter=sandbox_execution_adapter,
        tool_registry=external_tool_registry,
        tool_invoker=external_tool_invoker,
    )
    interaction_adapter: Any = WebInteractionAdapter()
    if external_tool_invoker is not None:
        interaction_adapter = ToolAugmentedInteractionAdapter(
            base_adapter=interaction_adapter,
            tool_registry=external_tool_registry,
            tool_invoker=external_tool_invoker,
        )
    rule_planner = RuleAdaptivePlanner()
    adaptive_planner: Any = rule_planner
    if settings.APP_ENABLE_MODEL_REPLANNER:
        model_planner = build_model_adaptive_planner(settings)
        if model_planner is not None:
            if isinstance(model_planner, CompositeAdaptivePlanner):
                adaptive_planner = CompositeAdaptivePlanner([*model_planner.planners, rule_planner])
            else:
                adaptive_planner = CompositeAdaptivePlanner([model_planner, rule_planner])
    extra_skill_paths = (
        synthesis_skill_paths(
            host_root=settings.APP_SYNTHESIS_HOST_ROOT,
            canonical_repo_path=settings.APP_SYNTHESIS_CANONICAL_REPO_PATH,
        )
        if settings.APP_ENABLE_SYNTHESIS
        else []
    )
    skill_registry = SkillRegistry([*settings.skill_paths, *extra_skill_paths])
    capability_resolver = (
        CapabilityResolver(skill_registry, max_matches=settings.APP_SKILL_MAX_MATCHES)
        if settings.APP_ENABLE_SKILL_RESOLVER
        else None
    )
    synthesis_bridge = (
        SynthesisBridge.from_settings(
            synthesis_root=settings.APP_SYNTHESIS_ROOT,
            host_root=settings.APP_SYNTHESIS_HOST_ROOT,
            canonical_repo_path=settings.APP_SYNTHESIS_CANONICAL_REPO_PATH,
            provider_type=settings.APP_SYNTHESIS_PROVIDER_TYPE,
            api_key=settings.APP_SYNTHESIS_API_KEY,
            model=settings.APP_SYNTHESIS_MODEL,
            base_url=settings.APP_SYNTHESIS_BASE_URL,
        )
        if settings.APP_ENABLE_SYNTHESIS
        else None
    )
    return ApiContext(
        settings=settings,
        execution_adapter=execution_adapter,
        interaction_adapter=interaction_adapter,
        db_engine=db_engine,
        session_factory=session_factory,
        events=events,
        adaptive_planner=adaptive_planner,
        skill_registry=skill_registry,
        capability_resolver=capability_resolver,
        external_tool_registry=external_tool_registry,
        synthesis_bridge=synthesis_bridge,
    )
