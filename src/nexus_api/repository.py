"""Persistence adapter implementing nexus_core engine repository protocol."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from nexus_api.models import Approval, Artifact, Citation, Promotion, Run, RunDelegation, RunStep
from nexus_core.models import RiskTier, RunStatus, StepDefinition, StepStatus
from nexus_core.policy import risk_tier_for_action


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SqlRunRepository:
    """Run repository using SQLAlchemy sessions."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_run(
        self,
        objective: str,
        mode: str,
        steps: list[StepDefinition],
        parent_run_id: str | None = None,
        delegation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run = Run(
            objective=objective,
            mode=mode,
            status=RunStatus.RUNNING.value,
            parent_run_id=parent_run_id or None,
        )
        self.session.add(run)
        self.session.flush()

        if parent_run_id:
            delegation_payload = delegation or {}
            self.session.add(
                RunDelegation(
                    parent_run_id=parent_run_id,
                    child_run_id=run.id,
                    role=str(delegation_payload.get("role", "worker")),
                    objective=str(delegation_payload.get("objective", objective)),
                    status=str(delegation_payload.get("status", "pending")),
                    summary=str(delegation_payload.get("summary", "")),
                    context_json=self._serialize_context(delegation_payload.get("context")),
                )
            )

        for idx, step in enumerate(steps):
            risk = risk_tier_for_action(step.action_type, step.instruction)
            self.session.add(
                RunStep(
                    run_id=run.id,
                    step_index=idx,
                    action_type=step.action_type.strip().lower(),
                    instruction=step.instruction.strip(),
                    risk_tier=risk.value,
                    status=StepStatus.PENDING.value,
                )
            )
        self.session.flush()
        return self.get_run(run.id) or {"id": run.id}

    def list_runs(
        self,
        limit: int = 25,
        offset: int = 0,
        status: str = "",
        mode: str = "",
        search: str = "",
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        base_stmt = select(Run)
        count_stmt = select(func.count()).select_from(Run)
        filters = []
        normalized_status = status.strip()
        if normalized_status:
            filters.append(Run.status == normalized_status)
        normalized_mode = mode.strip()
        if normalized_mode:
            filters.append(Run.mode == normalized_mode)
        normalized_search = search.strip().lower()
        if normalized_search:
            filters.append(func.lower(Run.objective).like(f"%{normalized_search}%"))
        if created_after is not None:
            filters.append(Run.created_at >= created_after)
        if created_before is not None:
            filters.append(Run.created_at <= created_before)

        if filters:
            base_stmt = base_stmt.where(*filters)
            count_stmt = count_stmt.where(*filters)

        total = int(self.session.scalar(count_stmt) or 0)
        rows = self.session.scalars(
            base_stmt.order_by(Run.created_at.desc(), Run.id.desc())
            .limit(max(1, min(limit, 100)))
            .offset(max(0, offset))
        ).all()
        items: list[dict[str, Any]] = []
        for run in rows:
            steps = self.list_steps(run.id)
            items.append(
                {
                    "id": run.id,
                    "objective": run.objective,
                    "mode": run.mode,
                    "status": run.status,
                    "parent_run_id": run.parent_run_id,
                    "created_at": run.created_at.isoformat() if run.created_at else None,
                    "updated_at": run.updated_at.isoformat() if run.updated_at else None,
                    "step_count": len(steps),
                    "pending_approval_count": sum(
                        1 for step in steps if step["status"] == StepStatus.PENDING_APPROVAL.value
                    ),
                    "failed_count": sum(
                        1 for step in steps if step["status"] == StepStatus.FAILED.value
                    ),
                }
            )
        return items, total

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        run = self.session.get(Run, run_id)
        if run is None:
            return None
        steps = self.list_steps(run_id)
        return {
            "id": run.id,
            "objective": run.objective,
            "mode": run.mode,
            "status": run.status,
            "parent_run_id": run.parent_run_id,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "updated_at": run.updated_at.isoformat() if run.updated_at else None,
            "steps": steps,
            "delegation": self._delegation_for_child(run_id),
            "child_runs": self.list_child_runs(run_id),
        }

    def list_steps(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.session.scalars(
            select(RunStep).where(RunStep.run_id == run_id).order_by(RunStep.step_index.asc())
        ).all()
        return [self._step_to_dict(r) for r in rows]

    def insert_steps_after(self, step_id: str, steps: list[StepDefinition]) -> None:
        if not steps:
            return
        anchor = self.session.get(RunStep, step_id)
        if anchor is None:
            raise ValueError("Anchor step not found")

        shift = len(steps)
        self.session.execute(
            update(RunStep)
            .where(RunStep.run_id == anchor.run_id, RunStep.step_index > anchor.step_index)
            .values(step_index=RunStep.step_index + shift)
        )

        for offset, step in enumerate(steps, start=1):
            risk = risk_tier_for_action(step.action_type, step.instruction)
            self.session.add(
                RunStep(
                    run_id=anchor.run_id,
                    step_index=anchor.step_index + offset,
                    action_type=step.action_type.strip().lower(),
                    instruction=step.instruction.strip(),
                    risk_tier=risk.value,
                    status=StepStatus.PENDING.value,
                )
            )
        self.session.flush()

    def get_step(self, step_id: str) -> dict[str, Any] | None:
        step = self.session.get(RunStep, step_id)
        if step is None:
            return None
        return self._step_to_dict(step)

    def mark_run_status(self, run_id: str, status: str) -> None:
        run = self.session.get(Run, run_id)
        if run is None:
            return
        run.status = status
        run.updated_at = _utc_now()
        self.session.flush()

    def mark_step_status(
        self,
        step_id: str,
        status: str,
        output_text: str = "",
        error_text: str = "",
    ) -> None:
        step = self.session.get(RunStep, step_id)
        if step is None:
            return
        step.status = status
        if status == StepStatus.RUNNING.value:
            step.started_at = _utc_now()
        terminal_statuses = {
            StepStatus.COMPLETED.value,
            StepStatus.FAILED.value,
            StepStatus.REJECTED.value,
        }
        if status in terminal_statuses:
            step.ended_at = _utc_now()
        if output_text:
            step.output_text = output_text
        if error_text:
            step.error_text = error_text
        self.session.flush()

    def reset_step_for_retry(self, step_id: str) -> bool:
        step = self.session.get(RunStep, step_id)
        if step is None:
            return False
        step.status = StepStatus.PENDING.value
        step.output_text = ""
        step.error_text = ""
        step.started_at = None
        step.ended_at = None
        self.session.flush()
        return True

    def add_citations(self, run_id: str, step_id: str, citations: list[dict[str, str]]) -> None:
        for citation in citations:
            self.session.add(
                Citation(
                    run_id=run_id,
                    step_id=step_id,
                    url=citation.get("url", ""),
                    title=citation.get("title", ""),
                    snippet=citation.get("snippet", ""),
                )
            )
        self.session.flush()

    def add_artifacts(self, run_id: str, step_id: str, artifacts: list[dict[str, str]]) -> None:
        for artifact in artifacts:
            self.session.add(
                Artifact(
                    run_id=run_id,
                    step_id=step_id,
                    kind=artifact.get("kind", "text"),
                    name=artifact.get("name", "artifact.txt"),
                    rel_path=artifact.get("rel_path", ""),
                    sandbox_path=artifact.get("sandbox_path", ""),
                    sha256=artifact.get("sha256", ""),
                )
            )
        self.session.flush()

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        artifact = self.session.get(Artifact, artifact_id)
        if artifact is None:
            return None
        return {
            "id": artifact.id,
            "run_id": artifact.run_id,
            "step_id": artifact.step_id,
            "kind": artifact.kind,
            "name": artifact.name,
            "rel_path": artifact.rel_path,
            "sandbox_path": artifact.sandbox_path,
            "sha256": artifact.sha256,
            "promoted": artifact.promoted,
            "created_at": artifact.created_at.isoformat() if artifact.created_at else None,
        }

    def mark_artifact_promoted(self, artifact_id: str) -> None:
        artifact = self.session.get(Artifact, artifact_id)
        if artifact is None:
            return
        artifact.promoted = True
        self.session.flush()

    def add_approval(
        self,
        run_id: str,
        step_id: str,
        decision: str,
        decided_by: str,
        reason: str = "",
    ) -> None:
        self.session.add(
            Approval(
                run_id=run_id,
                step_id=step_id,
                decision=decision,
                decided_by=decided_by,
                reason=reason,
            )
        )
        self.session.flush()

    def add_promotion(
        self,
        run_id: str,
        artifact_id: str,
        source_path: str,
        target_path: str,
        promoted_by: str,
    ) -> None:
        self.session.add(
            Promotion(
                run_id=run_id,
                artifact_id=artifact_id,
                source_path=source_path,
                target_path=target_path,
                promoted_by=promoted_by,
            )
        )
        self.session.flush()

    def latest_approval_for_step(self, run_id: str, step_id: str) -> dict[str, Any] | None:
        row = self.session.scalar(
            select(Approval)
            .where(Approval.run_id == run_id, Approval.step_id == step_id)
            .order_by(Approval.decided_at.desc())
        )
        if row is None:
            return None
        return {
            "id": row.id,
            "run_id": row.run_id,
            "step_id": row.step_id,
            "decision": row.decision,
            "decided_by": row.decided_by,
            "reason": row.reason,
            "decided_at": row.decided_at.isoformat() if row.decided_at else None,
        }

    def list_citations(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.session.scalars(
            select(Citation).where(Citation.run_id == run_id).order_by(Citation.created_at.asc())
        ).all()
        return [
            {
                "id": r.id,
                "run_id": r.run_id,
                "step_id": r.step_id,
                "url": r.url,
                "title": r.title,
                "snippet": r.snippet,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]

    def list_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.session.scalars(
            select(Artifact).where(Artifact.run_id == run_id).order_by(Artifact.created_at.asc())
        ).all()
        return [
            {
                "id": r.id,
                "run_id": r.run_id,
                "step_id": r.step_id,
                "kind": r.kind,
                "name": r.name,
                "rel_path": r.rel_path,
                "sandbox_path": r.sandbox_path,
                "sha256": r.sha256,
                "promoted": r.promoted,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]

    def list_approvals(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.session.scalars(
            select(Approval).where(Approval.run_id == run_id).order_by(Approval.decided_at.asc())
        ).all()
        return [
            {
                "id": r.id,
                "run_id": r.run_id,
                "step_id": r.step_id,
                "decision": r.decision,
                "decided_by": r.decided_by,
                "reason": r.reason,
                "decided_at": r.decided_at.isoformat() if r.decided_at else None,
            }
            for r in rows
        ]

    def list_promotions(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.session.scalars(
            select(Promotion)
            .where(Promotion.run_id == run_id)
            .order_by(Promotion.promoted_at.asc())
        ).all()
        return [
            {
                "id": r.id,
                "run_id": r.run_id,
                "artifact_id": r.artifact_id,
                "source_path": r.source_path,
                "target_path": r.target_path,
                "promoted_by": r.promoted_by,
                "promoted_at": r.promoted_at.isoformat() if r.promoted_at else None,
            }
            for r in rows
        ]

    def list_pending_approval_steps(self) -> list[dict[str, Any]]:
        rows = self.session.scalars(
            select(RunStep).where(RunStep.status == StepStatus.PENDING_APPROVAL.value)
        ).all()
        return [self._step_to_dict(r) for r in rows]

    def timeline(self, run_id: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        run = self.get_run(run_id)
        if run is None:
            return events

        for step in run["steps"]:
            events.append(
                {
                    "type": f"step.{step['status']}",
                    "timestamp": step["ended_at"] or step["started_at"] or step["created_at"],
                    "step_id": step["id"],
                    "action_type": step["action_type"],
                    "instruction": step["instruction"],
                    "output_text": step["output_text"],
                    "error_text": step["error_text"],
                }
            )
        for approval in self.list_approvals(run_id):
            events.append(
                {
                    "type": "approval.recorded",
                    "timestamp": approval["decided_at"],
                    "step_id": approval["step_id"],
                    "decision": approval["decision"],
                    "reason": approval["reason"],
                }
            )
        for promotion in self.list_promotions(run_id):
            events.append(
                {
                    "type": "artifact.promoted",
                    "timestamp": promotion["promoted_at"],
                    "artifact_id": promotion["artifact_id"],
                    "target_path": promotion["target_path"],
                    "promoted_by": promotion["promoted_by"],
                }
            )
        for delegation in self.list_delegations(run_id):
            events.append(
                {
                    "type": "delegate.started",
                    "timestamp": delegation["created_at"],
                    "child_run_id": delegation["child_run_id"],
                    "role": delegation["role"],
                    "objective": delegation["objective"],
                }
            )
            if delegation["status"] in {"completed", "failed"}:
                events.append(
                    {
                        "type": f"delegate.{delegation['status']}",
                        "timestamp": delegation["updated_at"],
                        "child_run_id": delegation["child_run_id"],
                        "role": delegation["role"],
                        "summary": delegation["summary"],
                    }
                )
        events.sort(key=lambda e: e.get("timestamp") or "")
        return events

    def list_child_runs(self, parent_run_id: str) -> list[dict[str, Any]]:
        rows = self.session.execute(
            select(RunDelegation, Run)
            .join(Run, Run.id == RunDelegation.child_run_id)
            .where(RunDelegation.parent_run_id == parent_run_id)
            .order_by(Run.created_at.asc(), Run.id.asc())
        ).all()
        child_runs: list[dict[str, Any]] = []
        for delegation, run in rows:
            child_runs.append(
                {
                    "id": run.id,
                    "objective": run.objective,
                    "mode": run.mode,
                    "status": run.status,
                    "parent_run_id": run.parent_run_id,
                    "created_at": run.created_at.isoformat() if run.created_at else None,
                    "updated_at": run.updated_at.isoformat() if run.updated_at else None,
                    "steps": self.list_steps(run.id),
                    "delegation_role": delegation.role,
                    "delegation_objective": delegation.objective,
                    "delegation_status": delegation.status,
                    "delegation_summary": delegation.summary,
                    "delegation_context": self._deserialize_context(delegation.context_json),
                }
            )
        return child_runs

    def list_delegations(self, parent_run_id: str) -> list[dict[str, Any]]:
        rows = self.session.scalars(
            select(RunDelegation)
            .where(RunDelegation.parent_run_id == parent_run_id)
            .order_by(RunDelegation.created_at.asc(), RunDelegation.id.asc())
        ).all()
        return [
            {
                "id": delegation.id,
                "parent_run_id": delegation.parent_run_id,
                "child_run_id": delegation.child_run_id,
                "role": delegation.role,
                "objective": delegation.objective,
                "status": delegation.status,
                "summary": delegation.summary,
                "context": self._deserialize_context(delegation.context_json),
                "created_at": delegation.created_at.isoformat() if delegation.created_at else None,
                "updated_at": delegation.updated_at.isoformat() if delegation.updated_at else None,
            }
            for delegation in rows
        ]

    def update_delegation(self, child_run_id: str, status: str, summary: str = "") -> None:
        delegation = self.session.scalar(
            select(RunDelegation).where(RunDelegation.child_run_id == child_run_id)
        )
        if delegation is None:
            return
        delegation.status = status
        delegation.summary = summary
        delegation.updated_at = _utc_now()
        self.session.flush()

    def _delegation_for_child(self, child_run_id: str) -> dict[str, Any] | None:
        delegation = self.session.scalar(
            select(RunDelegation).where(RunDelegation.child_run_id == child_run_id)
        )
        if delegation is None:
            return None
        return {
            "id": delegation.id,
            "parent_run_id": delegation.parent_run_id,
            "child_run_id": delegation.child_run_id,
            "role": delegation.role,
            "objective": delegation.objective,
            "status": delegation.status,
            "summary": delegation.summary,
            "context": self._deserialize_context(delegation.context_json),
            "created_at": delegation.created_at.isoformat() if delegation.created_at else None,
            "updated_at": delegation.updated_at.isoformat() if delegation.updated_at else None,
        }

    @staticmethod
    def _step_to_dict(step: RunStep) -> dict[str, Any]:
        return {
            "id": step.id,
            "run_id": step.run_id,
            "step_index": step.step_index,
            "action_type": step.action_type,
            "instruction": step.instruction,
            "risk_tier": step.risk_tier,
            "status": step.status,
            "output_text": step.output_text,
            "error_text": step.error_text,
            "created_at": step.created_at.isoformat() if step.created_at else None,
            "started_at": step.started_at.isoformat() if step.started_at else None,
            "ended_at": step.ended_at.isoformat() if step.ended_at else None,
            "requires_approval": step.risk_tier == RiskTier.HIGH.value,
        }

    @staticmethod
    def _serialize_context(value: Any) -> str:
        if not isinstance(value, dict) or not value:
            return "{}"
        try:
            return json.dumps(value)
        except TypeError:
            return "{}"

    @staticmethod
    def _deserialize_context(raw: str | None) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
