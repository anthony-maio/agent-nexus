"""Run orchestration engine independent of transport adapters."""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Any, Protocol

from nexus_core.adapters import ExecutionAdapter, InteractionAdapter
from nexus_core.events import RunEventBus
from nexus_core.models import (
    ApprovalDecision,
    RunEvent,
    RunMode,
    RunStatus,
    StepDefinition,
    StepStatus,
)
from nexus_core.policy import is_high_risk_action

log = logging.getLogger(__name__)


class EngineRepository(Protocol):
    """Persistence contract used by :class:`RunEngine`."""

    def create_run(
        self,
        objective: str,
        mode: str,
        steps: list[StepDefinition],
    ) -> dict[str, Any]: ...

    def get_run(self, run_id: str) -> dict[str, Any] | None: ...

    def list_steps(self, run_id: str) -> list[dict[str, Any]]: ...

    def get_step(self, step_id: str) -> dict[str, Any] | None: ...

    def mark_run_status(self, run_id: str, status: str) -> None: ...

    def mark_step_status(
        self,
        step_id: str,
        status: str,
        output_text: str = "",
        error_text: str = "",
    ) -> None: ...

    def add_citations(self, run_id: str, step_id: str, citations: list[dict[str, str]]) -> None: ...

    def add_artifacts(self, run_id: str, step_id: str, artifacts: list[dict[str, str]]) -> None: ...

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None: ...

    def mark_artifact_promoted(self, artifact_id: str) -> None: ...

    def add_approval(
        self,
        run_id: str,
        step_id: str,
        decision: str,
        decided_by: str,
        reason: str = "",
    ) -> None: ...

    def add_promotion(
        self,
        run_id: str,
        artifact_id: str,
        source_path: str,
        target_path: str,
        promoted_by: str,
    ) -> None: ...


class RunEngine:
    """Executes run steps using risk-tier policy and adapter boundaries."""

    def __init__(
        self,
        repository: EngineRepository,
        execution: ExecutionAdapter,
        interaction: InteractionAdapter,
        events: RunEventBus,
        canonical_workspace: Path,
    ) -> None:
        self.repo = repository
        self.execution = execution
        self.interaction = interaction
        self.events = events
        self.canonical_workspace = canonical_workspace
        self.canonical_workspace.mkdir(parents=True, exist_ok=True)

    async def create_run(
        self,
        objective: str,
        mode: RunMode,
        steps: list[StepDefinition],
    ) -> dict[str, Any]:
        """Create a run and execute until completion or approval gate."""
        run = self.repo.create_run(objective=objective, mode=mode.value, steps=steps)
        await self._publish(run["id"], "run.created", {"objective": objective})
        await self._execute_until_gate(run["id"])
        return self.repo.get_run(run["id"]) or run

    async def decide_approval(
        self,
        run_id: str,
        step_id: str,
        decision: ApprovalDecision,
        decided_by: str,
        reason: str = "",
    ) -> dict[str, Any]:
        """Apply approval decision and continue run if approved."""
        step = self.repo.get_step(step_id)
        if step is None or step["run_id"] != run_id:
            raise ValueError("Step not found for run")
        if step["status"] != StepStatus.PENDING_APPROVAL.value:
            raise ValueError("Step is not pending approval")

        self.repo.add_approval(
            run_id=run_id,
            step_id=step_id,
            decision=decision.value,
            decided_by=decided_by,
            reason=reason,
        )
        await self._publish(
            run_id,
            "approval.recorded",
            {
                "step_id": step_id,
                "decision": decision.value,
                "decided_by": decided_by,
            },
        )

        if decision == ApprovalDecision.REJECT:
            self.repo.mark_step_status(step_id, StepStatus.REJECTED.value, error_text=reason)
            self.repo.mark_run_status(run_id, RunStatus.PAUSED.value)
            await self.interaction.deliver_status(
                run_id,
                RunStatus.PAUSED.value,
                "Approval rejected",
            )
            await self._publish(run_id, "run.paused", {"step_id": step_id, "reason": reason})
            return self.repo.get_run(run_id) or {}

        self.repo.mark_run_status(run_id, RunStatus.RUNNING.value)
        ok = await self._execute_step(step)
        if not ok:
            self.repo.mark_run_status(run_id, RunStatus.FAILED.value)
            await self.interaction.deliver_status(run_id, RunStatus.FAILED.value, "Run failed")
            await self._publish(run_id, "run.failed", {"step_id": step_id})
            return self.repo.get_run(run_id) or {}
        await self._execute_until_gate(run_id)
        return self.repo.get_run(run_id) or {}

    async def promote_artifact(
        self,
        run_id: str,
        artifact_id: str,
        promoted_by: str,
    ) -> dict[str, Any]:
        """Copy artifact from sandbox space into canonical app workspace."""
        artifact = self.repo.get_artifact(artifact_id)
        if artifact is None or artifact["run_id"] != run_id:
            raise ValueError("Artifact not found for run")

        source = Path(artifact["sandbox_path"])
        if not source.exists():
            raise FileNotFoundError(f"Sandbox artifact missing: {source}")

        run_dir = self.canonical_workspace / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        target = run_dir / artifact["name"]
        shutil.copy2(source, target)

        self.repo.mark_artifact_promoted(artifact_id)
        self.repo.add_promotion(
            run_id=run_id,
            artifact_id=artifact_id,
            source_path=str(source),
            target_path=str(target),
            promoted_by=promoted_by,
        )

        await self._publish(
            run_id,
            "artifact.promoted",
            {
                "artifact_id": artifact_id,
                "source": str(source),
                "target": str(target),
            },
        )
        return {
            "artifact_id": artifact_id,
            "target_path": str(target),
            "sha256": self._sha256(target),
        }

    async def _execute_until_gate(self, run_id: str) -> None:
        """Execute queued steps until run completes or approval is required."""
        run = self.repo.get_run(run_id)
        if run is None:
            return

        for step in self.repo.list_steps(run_id):
            if step["status"] in {
                StepStatus.COMPLETED.value,
                StepStatus.REJECTED.value,
            }:
                continue
            if step["status"] == StepStatus.FAILED.value:
                self.repo.mark_run_status(run_id, RunStatus.FAILED.value)
                await self.interaction.deliver_status(run_id, RunStatus.FAILED.value, "Run failed")
                await self._publish(run_id, "run.failed", {"step_id": step["id"]})
                return

            mode = RunMode(run["mode"])
            if (
                mode == RunMode.SUPERVISED
                and is_high_risk_action(step["action_type"])
                and step["status"] == StepStatus.PENDING.value
            ):
                self.repo.mark_step_status(step["id"], StepStatus.PENDING_APPROVAL.value)
                self.repo.mark_run_status(run_id, RunStatus.PENDING_APPROVAL.value)
                summary = f"{step['action_type']}: {step['instruction'][:120]}"
                await self.interaction.request_approval(
                    run_id=run_id,
                    step_id=step["id"],
                    summary=summary,
                    action_type=step["action_type"],
                )
                await self._publish(
                    run_id,
                    "step.pending_approval",
                    {
                        "step_id": step["id"],
                        "action_type": step["action_type"],
                        "instruction": step["instruction"],
                    },
                )
                return

            if step["status"] == StepStatus.PENDING_APPROVAL.value:
                self.repo.mark_run_status(run_id, RunStatus.PENDING_APPROVAL.value)
                return

            ok = await self._execute_step(step)
            if not ok:
                self.repo.mark_run_status(run_id, RunStatus.FAILED.value)
                await self.interaction.deliver_status(run_id, RunStatus.FAILED.value, "Run failed")
                await self._publish(run_id, "run.failed", {"step_id": step["id"]})
                return

        self.repo.mark_run_status(run_id, RunStatus.COMPLETED.value)
        await self.interaction.deliver_status(run_id, RunStatus.COMPLETED.value, "Run completed")
        await self._publish(run_id, "run.completed", {})

    async def _execute_step(self, step: dict[str, Any]) -> bool:
        run_id = step["run_id"]
        step_id = step["id"]
        self.repo.mark_step_status(step_id, StepStatus.RUNNING.value)
        await self._publish(
            run_id,
            "step.running",
            {
                "step_id": step_id,
                "action_type": step["action_type"],
                "instruction": step["instruction"],
            },
        )
        try:
            result = await self.execution.execute_step(
                run_id=run_id,
                step_id=step_id,
                action_type=step["action_type"],
                instruction=step["instruction"],
            )
            self.repo.mark_step_status(
                step_id,
                StepStatus.COMPLETED.value,
                output_text=result.output_text,
            )
            self.repo.add_citations(
                run_id=run_id,
                step_id=step_id,
                citations=[c.model_dump() for c in result.citations],
            )
            self.repo.add_artifacts(
                run_id=run_id,
                step_id=step_id,
                artifacts=[a.model_dump() for a in result.artifacts],
            )
            await self._publish(
                run_id,
                "step.completed",
                {
                    "step_id": step_id,
                    "output_text": result.output_text[:500],
                    "citations": len(result.citations),
                    "artifacts": len(result.artifacts),
                },
            )
            return True
        except Exception as exc:
            log.warning("Step execution failed for %s: %s", step_id, exc)
            self.repo.mark_step_status(
                step_id,
                StepStatus.FAILED.value,
                error_text=str(exc),
            )
            await self._publish(
                run_id,
                "step.failed",
                {"step_id": step_id, "error": str(exc)},
            )
            return False

    async def _publish(self, run_id: str, event_type: str, payload: dict[str, Any]) -> None:
        await self.events.publish(
            RunEvent(run_id=run_id, event_type=event_type, payload=payload)
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            while chunk := fh.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
