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
    StepExecutionResult,
    StepStatus,
)
from nexus_core.planner import AdaptivePlanner, RuleAdaptivePlanner, apply_follow_up_policy
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

    def insert_steps_after(self, step_id: str, steps: list[StepDefinition]) -> None: ...

    def get_step(self, step_id: str) -> dict[str, Any] | None: ...

    def mark_run_status(self, run_id: str, status: str) -> None: ...

    def mark_step_status(
        self,
        step_id: str,
        status: str,
        output_text: str = "",
        error_text: str = "",
    ) -> None: ...

    def reset_step_for_retry(self, step_id: str) -> bool: ...

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

    def latest_approval_for_step(self, run_id: str, step_id: str) -> dict[str, Any] | None: ...


class RunEngine:
    """Executes run steps using risk-tier policy and adapter boundaries."""

    def __init__(
        self,
        repository: EngineRepository,
        execution: ExecutionAdapter,
        interaction: InteractionAdapter,
        events: RunEventBus,
        canonical_workspace: Path,
        sandbox_artifacts_root: Path | None = None,
        adaptive_planner: AdaptivePlanner | None = None,
    ) -> None:
        self.repo = repository
        self.execution = execution
        self.interaction = interaction
        self.events = events
        self.adaptive_planner = adaptive_planner or RuleAdaptivePlanner()
        self.canonical_workspace = canonical_workspace
        self.sandbox_artifacts_root = (
            sandbox_artifacts_root.resolve() if sandbox_artifacts_root else None
        )
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
        run = self.repo.get_run(run_id)
        result = await self._execute_step(step)
        if result is None:
            await self._mark_run_failed(run_id, step_id)
            return self.repo.get_run(run_id) or {}
        if run is not None:
            await self._apply_follow_up_planning(run, step, result)
        await self._execute_until_gate(run_id)
        return self.repo.get_run(run_id) or {}

    async def promote_artifact(
        self,
        run_id: str,
        artifact_id: str,
        promoted_by: str,
    ) -> dict[str, Any]:
        """Copy artifact from sandbox space into canonical app workspace."""
        run = self.repo.get_run(run_id)
        if run is None:
            raise ValueError("Run not found")
        artifact = self.repo.get_artifact(artifact_id)
        if artifact is None or artifact["run_id"] != run_id:
            raise ValueError("Artifact not found for run")
        if artifact.get("promoted"):
            raise ValueError("Artifact already promoted")

        step = self.repo.get_step(artifact["step_id"])
        if step is None:
            raise ValueError("Artifact step not found")
        if step["status"] != StepStatus.COMPLETED.value:
            raise ValueError("Artifact step is not completed")
        if (
            is_high_risk_action(step["action_type"], step["instruction"])
            and run.get("mode") != RunMode.AUTOPILOT.value
        ):
            approval = self.repo.latest_approval_for_step(run_id, step["id"])
            if approval is None or approval.get("decision") != ApprovalDecision.APPROVE.value:
                raise PermissionError("High-risk artifact requires approved step before promotion")

        source = Path(artifact["sandbox_path"]).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Sandbox artifact missing: {source}")
        if self.sandbox_artifacts_root is not None:
            root = self.sandbox_artifacts_root
            if root != source and root not in source.parents:
                raise PermissionError("Artifact path is outside sandbox artifact root")
        expected_sha = str(artifact.get("sha256", "")).strip()
        if expected_sha and self._sha256(source) != expected_sha:
            raise PermissionError("Artifact integrity check failed")

        artifact_name = Path(str(artifact.get("name", ""))).name
        if not artifact_name or artifact_name != artifact.get("name", ""):
            raise ValueError("Artifact name is invalid")

        run_dir = self.canonical_workspace / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        target = (run_dir / artifact_name).resolve()
        run_dir_resolved = run_dir.resolve()
        if run_dir_resolved != target and run_dir_resolved not in target.parents:
            raise PermissionError("Promotion target escapes canonical workspace")
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

    async def resume_run(self, run_id: str) -> dict[str, Any]:
        """Resume a paused run and continue execution until next gate."""
        run = self.repo.get_run(run_id)
        if run is None:
            raise ValueError("Run not found")
        if run["status"] == RunStatus.COMPLETED.value:
            raise ValueError("Run already completed")
        if run["status"] == RunStatus.CANCELLED.value:
            raise ValueError("Cancelled runs cannot be resumed")
        if run["status"] == RunStatus.FAILED.value:
            raise ValueError("Failed runs require retry")

        self.repo.mark_run_status(run_id, RunStatus.RUNNING.value)
        await self._publish(run_id, "run.resumed", {})
        await self._execute_until_gate(run_id)
        return self.repo.get_run(run_id) or {}

    async def retry_run(self, run_id: str) -> dict[str, Any]:
        """Reset failed/rejected steps to pending and re-run."""
        run = self.repo.get_run(run_id)
        if run is None:
            raise ValueError("Run not found")

        retryable = [
            step
            for step in self.repo.list_steps(run_id)
            if step["status"] in {StepStatus.FAILED.value, StepStatus.REJECTED.value}
        ]
        if not retryable:
            raise ValueError("Run has no failed or rejected steps to retry")

        for step in retryable:
            self.repo.reset_step_for_retry(step["id"])

        self.repo.mark_run_status(run_id, RunStatus.RUNNING.value)
        await self._publish(
            run_id,
            "run.retry_requested",
            {"retry_step_count": len(retryable)},
        )
        await self._execute_until_gate(run_id)
        return self.repo.get_run(run_id) or {}

    async def _execute_until_gate(self, run_id: str) -> None:
        """Execute queued steps until run completes or approval is required."""
        while True:
            run = self.repo.get_run(run_id)
            if run is None:
                return
            mode = RunMode(run["mode"])
            next_step: dict[str, Any] | None = None

            for step in self.repo.list_steps(run_id):
                status = step["status"]
                if status in {StepStatus.COMPLETED.value, StepStatus.REJECTED.value}:
                    continue
                if status == StepStatus.FAILED.value:
                    await self._mark_run_failed(run_id, step["id"])
                    return
                if status == StepStatus.PENDING_APPROVAL.value:
                    self.repo.mark_run_status(run_id, RunStatus.PENDING_APPROVAL.value)
                    return
                if status == StepStatus.PENDING.value:
                    next_step = step
                    break

            if next_step is None:
                self.repo.mark_run_status(run_id, RunStatus.COMPLETED.value)
                await self.interaction.deliver_status(
                    run_id,
                    RunStatus.COMPLETED.value,
                    "Run completed",
                )
                await self._publish(run_id, "run.completed", {})
                return

            if mode == RunMode.SUPERVISED and is_high_risk_action(
                next_step["action_type"], next_step["instruction"]
            ):
                self.repo.mark_step_status(next_step["id"], StepStatus.PENDING_APPROVAL.value)
                self.repo.mark_run_status(run_id, RunStatus.PENDING_APPROVAL.value)
                summary = f"{next_step['action_type']}: {next_step['instruction'][:120]}"
                await self.interaction.request_approval(
                    run_id=run_id,
                    step_id=next_step["id"],
                    summary=summary,
                    action_type=next_step["action_type"],
                )
                await self._publish(
                    run_id,
                    "step.pending_approval",
                    {
                        "step_id": next_step["id"],
                        "action_type": next_step["action_type"],
                        "instruction": next_step["instruction"],
                    },
                )
                return

            result = await self._execute_step(next_step)
            if result is None:
                await self._mark_run_failed(run_id, next_step["id"])
                return

            await self._apply_follow_up_planning(run, next_step, result)

    async def _execute_step(self, step: dict[str, Any]) -> StepExecutionResult | None:
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
            return result
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
            return None

    async def _apply_follow_up_planning(
        self,
        run: dict[str, Any],
        completed_step: dict[str, Any],
        result: StepExecutionResult,
    ) -> None:
        completed_action = str(completed_step.get("action_type", "")).strip().lower()
        existing_steps = self.repo.list_steps(run["id"])
        if not self._should_plan_follow_up(
            completed_step=completed_step,
            completed_action=completed_action,
            result=result,
            existing_steps=existing_steps,
        ):
            return

        proposed_steps = await self.adaptive_planner.propose_follow_up(
            objective=run["objective"],
            completed_step=completed_step,
            result=result,
            existing_steps=existing_steps,
        )
        follow_up_steps = apply_follow_up_policy(
            proposed_steps,
            mode=RunMode(run["mode"]),
        )
        if not follow_up_steps:
            return

        self.repo.insert_steps_after(completed_step["id"], follow_up_steps)
        await self._publish(
            run["id"],
            "run.replanned",
            {
                "after_step_id": completed_step["id"],
                "inserted_steps": [s.model_dump() for s in follow_up_steps],
            },
        )

    async def _mark_run_failed(self, run_id: str, step_id: str) -> None:
        self.repo.mark_run_status(run_id, RunStatus.FAILED.value)
        await self.interaction.deliver_status(run_id, RunStatus.FAILED.value, "Run failed")
        await self._publish(run_id, "run.failed", {"step_id": step_id})

    async def _publish(self, run_id: str, event_type: str, payload: dict[str, Any]) -> None:
        await self.events.publish(RunEvent(run_id=run_id, event_type=event_type, payload=payload))

    @staticmethod
    def _should_plan_follow_up(
        completed_step: dict[str, Any],
        completed_action: str,
        result: StepExecutionResult,
        existing_steps: list[dict[str, Any]],
    ) -> bool:
        if RunEngine._has_pending_steps_after(completed_step, existing_steps):
            if result.metadata.get("next_steps"):
                return True
            if completed_action == "extract" and not result.citations:
                return True
            return (
                completed_action == "extract"
                and RunEngine._next_pending_action_after(completed_step, existing_steps) == "export"
            )
        return completed_action not in {"export", "write"}

    @staticmethod
    def _has_pending_steps_after(
        completed_step: dict[str, Any],
        existing_steps: list[dict[str, Any]],
    ) -> bool:
        return bool(RunEngine._next_pending_action_after(completed_step, existing_steps))

    @staticmethod
    def _next_pending_action_after(
        completed_step: dict[str, Any],
        existing_steps: list[dict[str, Any]],
    ) -> str:
        try:
            step_index = int(completed_step.get("step_index", -1))
        except (TypeError, ValueError):
            step_index = -1
        for step in existing_steps:
            try:
                candidate_index = int(step.get("step_index", -1))
            except (TypeError, ValueError):
                candidate_index = -1
            if candidate_index <= step_index:
                continue
            if str(step.get("status", "")).strip().lower() == StepStatus.PENDING.value:
                return str(step.get("action_type", "")).strip().lower()
        return ""

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            while chunk := fh.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
