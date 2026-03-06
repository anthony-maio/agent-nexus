"""Persistence adapter implementing nexus_core engine repository protocol."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from nexus_api.models import Approval, Artifact, Citation, Promotion, Run, RunStep
from nexus_core.models import RiskTier, RunStatus, StepDefinition, StepStatus
from nexus_core.policy import risk_tier_for_action


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SqlRunRepository:
    """Run repository using SQLAlchemy sessions."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_run(self, objective: str, mode: str, steps: list[StepDefinition]) -> dict[str, Any]:
        run = Run(objective=objective, mode=mode, status=RunStatus.RUNNING.value)
        self.session.add(run)
        self.session.flush()

        for idx, step in enumerate(steps):
            risk = risk_tier_for_action(step.action_type)
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
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "updated_at": run.updated_at.isoformat() if run.updated_at else None,
            "steps": steps,
        }

    def list_steps(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.session.scalars(
            select(RunStep).where(RunStep.run_id == run_id).order_by(RunStep.step_index.asc())
        ).all()
        return [self._step_to_dict(r) for r in rows]

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
        for artifact in self.list_artifacts(run_id):
            if artifact["promoted"]:
                events.append(
                    {
                        "type": "artifact.promoted",
                        "timestamp": artifact["created_at"],
                        "artifact_id": artifact["id"],
                        "name": artifact["name"],
                    }
                )
        events.sort(key=lambda e: e.get("timestamp") or "")
        return events

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
