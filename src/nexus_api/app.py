"""FastAPI app for app-first Agent Nexus control plane."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Iterator, TypeVar

from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from nexus_api.auth import (
    authenticate_user,
    create_session_token,
    ensure_admin_user,
    validate_bearer_token,
)
from nexus_api.bootstrap import bootstrap_status, write_bootstrap_config
from nexus_api.repository import SqlRunRepository
from nexus_api.schemas import (
    ApprovalRequest,
    BootstrapConfigureRequest,
    BootstrapConfigureResponse,
    BootstrapStatusResponse,
    PendingApprovalItem,
    PromotionRequest,
    RunCreateRequest,
    SessionCreateRequest,
    SessionCreateResponse,
    SkillAcquireRequest,
)
from nexus_api.service import ApiContext, build_context
from nexus_core.engine import RunEngine
from nexus_core.models import RunMode, RunStatus
from nexus_core.planner import annotate_planner_steps

log = logging.getLogger(__name__)

_EnumT = TypeVar("_EnumT", bound=Enum)


def create_app(context: ApiContext | None = None) -> FastAPI:
    """Create configured FastAPI app."""
    ctx = context or build_context()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        with ctx.session_factory() as session:
            ensure_admin_user(
                session,
                username=ctx.settings.APP_ADMIN_USERNAME,
                password=ctx.settings.APP_ADMIN_PASSWORD,
            )
            session.commit()
        log.info("Nexus API started with single-admin auth enabled.")
        yield

    app = FastAPI(title="Agent Nexus API", version="0.1.0", lifespan=lifespan)
    app.state.ctx = ctx

    def get_session() -> Iterator[Session]:
        session = ctx.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def current_user(
        authorization: str = Header(default=""),
        session: Session = Depends(get_session),
    ) -> dict[str, Any]:
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer token")
        token = authorization.split(" ", 1)[1].strip()
        user = validate_bearer_token(session, token)
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return {"id": user.id, "username": user.username}

    def parse_optional_datetime(raw: str, field_name: str) -> datetime | None:
        value = raw.strip()
        if not value:
            return None
        normalized = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(f"Invalid {field_name} datetime: {raw}") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def parse_optional_enum(raw: str, enum_type: type[_EnumT], field_name: str) -> str:
        value = raw.strip()
        if not value:
            return ""
        try:
            parsed = enum_type(value)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in enum_type)
            raise ValueError(
                f"Invalid {field_name} value: {raw}. Allowed values: {allowed}"
            ) from exc
        return str(parsed.value)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/skills")
    def list_skills(
        user: dict[str, Any] = Depends(current_user),
    ) -> dict[str, Any]:
        _ = user
        items = [manifest.to_dict() for manifest in ctx.skill_registry.list_manifests()]
        return {"items": items, "total": len(items)}

    @app.get("/skills/resolve")
    def resolve_skills(
        objective: str = Query(default="", min_length=1, max_length=2000),
        user: dict[str, Any] = Depends(current_user),
    ) -> dict[str, Any]:
        _ = user
        matches = [match.to_dict() for match in ctx.capability_resolver.resolve_matches(objective)]
        return {"objective": objective, "items": matches, "total": len(matches)}

    @app.post("/skills/acquire")
    async def acquire_skill(
        request: SkillAcquireRequest,
        user: dict[str, Any] = Depends(current_user),
    ) -> dict[str, Any]:
        _ = user
        if ctx.synthesis_bridge is None:
            raise HTTPException(status_code=503, detail="Synthesis integration is not configured")
        payload = await ctx.synthesis_bridge.acquire_skill(
            intent=request.intent,
            requirements=request.requirements,
        )
        ctx.skill_registry.refresh()
        return payload

    @app.get("/bootstrap/status", response_model=BootstrapStatusResponse)
    def get_bootstrap_status() -> BootstrapStatusResponse:
        return bootstrap_status(ctx.settings)

    @app.post("/bootstrap/configure", response_model=BootstrapConfigureResponse)
    def configure_bootstrap(
        request: BootstrapConfigureRequest,
    ) -> BootstrapConfigureResponse:
        status = bootstrap_status(ctx.settings)
        if not status.setup_required:
            raise HTTPException(status_code=409, detail="App is already configured")
        return write_bootstrap_config(ctx.settings, request)

    @app.post("/sessions", response_model=SessionCreateResponse)
    def create_session(
        request: SessionCreateRequest,
        session: Session = Depends(get_session),
    ) -> SessionCreateResponse:
        # Ensure configured admin exists even when startup hooks are not invoked.
        ensure_admin_user(
            session,
            username=ctx.settings.APP_ADMIN_USERNAME,
            password=ctx.settings.APP_ADMIN_PASSWORD,
        )
        user = authenticate_user(session, request.username, request.password)
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        tok, raw_token = create_session_token(
            session,
            user,
            ttl_hours=ctx.settings.APP_SESSION_TTL_HOURS,
        )
        expires = tok.expires_at.astimezone(timezone.utc).isoformat()
        return SessionCreateResponse(
            session_id=tok.id,
            token=raw_token,
            username=user.username,
            expires_at=expires,
        )

    @app.post("/runs")
    async def create_run(
        request: RunCreateRequest,
        user: dict[str, Any] = Depends(current_user),
        session: Session = Depends(get_session),
    ) -> dict[str, Any]:
        if request.steps:
            steps = annotate_planner_steps(
                request.steps,
                planner_source="user",
                planner_phase="initial",
            )
        else:
            steps = []
        repo = SqlRunRepository(session)
        engine = RunEngine(
            repository=repo,
            execution=ctx.execution_adapter,
            interaction=ctx.interaction_adapter,
            events=ctx.events,
            canonical_workspace=ctx.settings.canonical_workspace_path,
            sandbox_artifacts_root=ctx.settings.sandbox_artifact_root_path,
            adaptive_planner=ctx.adaptive_planner,
            capability_resolver=ctx.capability_resolver,
            max_autonomous_steps=ctx.settings.APP_KERNEL_MAX_AUTONOMOUS_STEPS,
            max_step_retries=ctx.settings.APP_KERNEL_MAX_STEP_RETRIES,
            max_identical_step_streak=ctx.settings.APP_KERNEL_MAX_IDENTICAL_STEP_STREAK,
        )
        run = await engine.create_run(
            objective=request.objective,
            mode=RunMode(request.mode),
            steps=steps,
            parent_run_id=request.parent_run_id.strip() or None,
            delegation=(
                request.delegation.model_dump()
                if request.delegation is not None
                else None
            ),
        )
        run["created_by"] = user["username"]
        return run

    @app.get("/runs")
    def list_runs(
        limit: int = Query(default=25, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
        status: str = Query(default=""),
        mode: str = Query(default=""),
        search: str = Query(default=""),
        created_after: str = Query(default=""),
        created_before: str = Query(default=""),
        include_children: bool = Query(default=False),
        user: dict[str, Any] = Depends(current_user),
        session: Session = Depends(get_session),
    ) -> dict[str, Any]:
        _ = user
        normalized_status = parse_optional_enum(status, RunStatus, "status")
        normalized_mode = parse_optional_enum(mode, RunMode, "mode")
        parsed_after = parse_optional_datetime(created_after, "created_after")
        parsed_before = parse_optional_datetime(created_before, "created_before")
        repo = SqlRunRepository(session)
        items, total = repo.list_runs(
            limit=limit,
            offset=offset,
            status=normalized_status,
            mode=normalized_mode,
            search=search.strip(),
            created_after=parsed_after,
            created_before=parsed_before,
            include_children=include_children,
        )
        return {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    @app.get("/runs/{run_id}")
    def get_run(
        run_id: str,
        user: dict[str, Any] = Depends(current_user),
        session: Session = Depends(get_session),
    ) -> dict[str, Any]:
        _ = user
        repo = SqlRunRepository(session)
        run = repo.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        run["citations"] = repo.list_citations(run_id)
        run["artifacts"] = repo.list_artifacts(run_id)
        run["approvals"] = repo.list_approvals(run_id)
        return run

    @app.get("/runs/{run_id}/timeline")
    def get_run_timeline(
        run_id: str,
        user: dict[str, Any] = Depends(current_user),
        session: Session = Depends(get_session),
    ) -> dict[str, Any]:
        _ = user
        repo = SqlRunRepository(session)
        run = repo.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return {"run_id": run_id, "timeline": repo.timeline(run_id)}

    @app.post("/runs/{run_id}/approvals/{step_id}")
    async def decide_approval(
        run_id: str,
        step_id: str,
        request: ApprovalRequest,
        user: dict[str, Any] = Depends(current_user),
        session: Session = Depends(get_session),
    ) -> dict[str, Any]:
        repo = SqlRunRepository(session)
        engine = RunEngine(
            repository=repo,
            execution=ctx.execution_adapter,
            interaction=ctx.interaction_adapter,
            events=ctx.events,
            canonical_workspace=ctx.settings.canonical_workspace_path,
            sandbox_artifacts_root=ctx.settings.sandbox_artifact_root_path,
            adaptive_planner=ctx.adaptive_planner,
            capability_resolver=ctx.capability_resolver,
            max_autonomous_steps=ctx.settings.APP_KERNEL_MAX_AUTONOMOUS_STEPS,
            max_step_retries=ctx.settings.APP_KERNEL_MAX_STEP_RETRIES,
            max_identical_step_streak=ctx.settings.APP_KERNEL_MAX_IDENTICAL_STEP_STREAK,
        )
        run = await engine.decide_approval(
            run_id=run_id,
            step_id=step_id,
            decision=request.decision,
            decided_by=user["username"],
            reason=request.reason,
        )
        return run

    @app.post("/runs/{run_id}/resume")
    async def resume_run(
        run_id: str,
        user: dict[str, Any] = Depends(current_user),
        session: Session = Depends(get_session),
    ) -> dict[str, Any]:
        _ = user
        repo = SqlRunRepository(session)
        engine = RunEngine(
            repository=repo,
            execution=ctx.execution_adapter,
            interaction=ctx.interaction_adapter,
            events=ctx.events,
            canonical_workspace=ctx.settings.canonical_workspace_path,
            sandbox_artifacts_root=ctx.settings.sandbox_artifact_root_path,
            adaptive_planner=ctx.adaptive_planner,
            capability_resolver=ctx.capability_resolver,
            max_autonomous_steps=ctx.settings.APP_KERNEL_MAX_AUTONOMOUS_STEPS,
            max_step_retries=ctx.settings.APP_KERNEL_MAX_STEP_RETRIES,
            max_identical_step_streak=ctx.settings.APP_KERNEL_MAX_IDENTICAL_STEP_STREAK,
        )
        return await engine.resume_run(run_id=run_id)

    @app.post("/runs/{run_id}/retry")
    async def retry_run(
        run_id: str,
        user: dict[str, Any] = Depends(current_user),
        session: Session = Depends(get_session),
    ) -> dict[str, Any]:
        _ = user
        repo = SqlRunRepository(session)
        engine = RunEngine(
            repository=repo,
            execution=ctx.execution_adapter,
            interaction=ctx.interaction_adapter,
            events=ctx.events,
            canonical_workspace=ctx.settings.canonical_workspace_path,
            sandbox_artifacts_root=ctx.settings.sandbox_artifact_root_path,
            adaptive_planner=ctx.adaptive_planner,
            capability_resolver=ctx.capability_resolver,
            max_autonomous_steps=ctx.settings.APP_KERNEL_MAX_AUTONOMOUS_STEPS,
            max_step_retries=ctx.settings.APP_KERNEL_MAX_STEP_RETRIES,
            max_identical_step_streak=ctx.settings.APP_KERNEL_MAX_IDENTICAL_STEP_STREAK,
        )
        return await engine.retry_run(run_id=run_id)

    @app.post("/runs/{run_id}/artifacts/{artifact_id}/promote")
    async def promote_artifact(
        run_id: str,
        artifact_id: str,
        request: PromotionRequest,
        user: dict[str, Any] = Depends(current_user),
        session: Session = Depends(get_session),
    ) -> dict[str, Any]:
        promoted_by = user["username"]
        if request.promoted_by and request.promoted_by != promoted_by:
            raise HTTPException(
                status_code=403,
                detail="promoted_by must match authenticated user",
            )
        repo = SqlRunRepository(session)
        engine = RunEngine(
            repository=repo,
            execution=ctx.execution_adapter,
            interaction=ctx.interaction_adapter,
            events=ctx.events,
            canonical_workspace=ctx.settings.canonical_workspace_path,
            sandbox_artifacts_root=ctx.settings.sandbox_artifact_root_path,
            adaptive_planner=ctx.adaptive_planner,
            capability_resolver=ctx.capability_resolver,
            max_autonomous_steps=ctx.settings.APP_KERNEL_MAX_AUTONOMOUS_STEPS,
            max_step_retries=ctx.settings.APP_KERNEL_MAX_STEP_RETRIES,
            max_identical_step_streak=ctx.settings.APP_KERNEL_MAX_IDENTICAL_STEP_STREAK,
        )
        return await engine.promote_artifact(
            run_id=run_id,
            artifact_id=artifact_id,
            promoted_by=promoted_by,
        )

    @app.get("/runs/{run_id}/citations")
    def get_citations(
        run_id: str,
        user: dict[str, Any] = Depends(current_user),
        session: Session = Depends(get_session),
    ) -> dict[str, Any]:
        _ = user
        repo = SqlRunRepository(session)
        if repo.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return {"run_id": run_id, "citations": repo.list_citations(run_id)}

    @app.get("/approvals/pending")
    def pending_approvals(
        user: dict[str, Any] = Depends(current_user),
        session: Session = Depends(get_session),
    ) -> dict[str, list[PendingApprovalItem]]:
        _ = user
        repo = SqlRunRepository(session)
        data = [
            PendingApprovalItem(
                run_id=step["run_id"],
                step_id=step["id"],
                action_type=step["action_type"],
                instruction=step["instruction"],
                risk_tier=step["risk_tier"],
            )
            for step in repo.list_pending_approval_steps()
        ]
        return {"items": data}

    @app.websocket("/runs/{run_id}/stream")
    async def stream_run(websocket: WebSocket, run_id: str, token: str = "") -> None:
        session = ctx.session_factory()
        try:
            if not token:
                await websocket.close(code=4401)
                return
            user = validate_bearer_token(session, token)
            if user is None:
                await websocket.close(code=4401)
                return
            repo = SqlRunRepository(session)
            run = repo.get_run(run_id)
            if run is None:
                await websocket.close(code=4404)
                return
            await websocket.accept()
            await websocket.send_json(
                {
                    "run_id": run_id,
                    "event_type": "stream.ready",
                    "payload": {"status": run["status"], "user": user.username},
                }
            )
            queue = ctx.events.subscribe(run_id)
            try:
                while True:
                    event = await queue.get()
                    await websocket.send_json(event.model_dump())
            except WebSocketDisconnect:
                return
            finally:
                ctx.events.unsubscribe(run_id, queue)
        finally:
            session.close()

    @app.exception_handler(ValueError)
    async def value_error_handler(_, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(_, exc: FileNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(PermissionError)
    async def permission_error_handler(_, exc: PermissionError) -> JSONResponse:
        return JSONResponse(status_code=403, content={"detail": str(exc)})

    return app
