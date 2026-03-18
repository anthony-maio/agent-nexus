"""Run orchestration engine independent of transport adapters."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from nexus_core.adapters import ExecutionAdapter, InteractionAdapter
from nexus_core.events import RunEventBus
from nexus_core.models import (
    ApprovalDecision,
    ArtifactRecord,
    CitationRecord,
    DelegationStepPayload,
    RunEvent,
    RunMode,
    RunStatus,
    StepDefinition,
    StepExecutionResult,
    StepStatus,
)
from nexus_core.planner import (
    AdaptivePlanner,
    RuleAdaptivePlanner,
    annotate_planner_fallback,
    annotate_planner_steps,
    apply_follow_up_policy,
    apply_initial_plan_policy,
    plan_follow_up_steps,
    plan_steps_for_objective,
    request_next_steps,
)
from nexus_core.policy import (
    delegated_output_contract_violations,
    delegated_role_allows_action,
    delegated_workspace_path_allowed,
    is_high_risk_action,
)
from nexus_core.skills import CapabilityResolver, SkillManifest, serialize_skill_context

log = logging.getLogger(__name__)


class EngineRepository(Protocol):
    """Persistence contract used by :class:`RunEngine`."""

    def create_run(
        self,
        objective: str,
        mode: str,
        steps: list[StepDefinition],
        parent_run_id: str | None = None,
        delegation: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def get_run(self, run_id: str) -> dict[str, Any] | None: ...

    def list_steps(self, run_id: str) -> list[dict[str, Any]]: ...

    def insert_steps_after(self, step_id: str, steps: list[StepDefinition]) -> None: ...

    def get_step(self, step_id: str) -> dict[str, Any] | None: ...

    def mark_run_status(self, run_id: str, status: str) -> None: ...

    def merge_run_metadata(self, run_id: str, metadata: dict[str, Any]) -> None: ...

    def mark_step_status(
        self,
        step_id: str,
        status: str,
        output_text: str = "",
        error_text: str = "",
    ) -> None: ...

    def reset_step_for_retry(self, step_id: str) -> bool: ...

    def merge_step_metadata(self, step_id: str, metadata: dict[str, Any]) -> None: ...

    def add_citations(self, run_id: str, step_id: str, citations: list[dict[str, str]]) -> None: ...

    def add_artifacts(self, run_id: str, step_id: str, artifacts: list[dict[str, str]]) -> None: ...

    def list_citations(self, run_id: str) -> list[dict[str, Any]]: ...

    def list_artifacts(self, run_id: str) -> list[dict[str, Any]]: ...

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

    def update_delegation(self, child_run_id: str, status: str, summary: str = "") -> None: ...


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
        capability_resolver: CapabilityResolver | None = None,
        max_autonomous_steps: int = 24,
        max_step_retries: int = 0,
        max_identical_step_streak: int = 0,
    ) -> None:
        self.repo = repository
        self.execution = execution
        self.interaction = interaction
        self.events = events
        self.adaptive_planner = adaptive_planner or RuleAdaptivePlanner()
        self.capability_resolver = capability_resolver
        self.canonical_workspace = canonical_workspace
        self.sandbox_artifacts_root = (
            sandbox_artifacts_root.resolve() if sandbox_artifacts_root else None
        )
        self.max_autonomous_steps = max(int(max_autonomous_steps), 1)
        self.max_step_retries = max(int(max_step_retries), 0)
        self.max_identical_step_streak = max(int(max_identical_step_streak), 0)
        self.canonical_workspace.mkdir(parents=True, exist_ok=True)

    async def create_run(
        self,
        objective: str,
        mode: RunMode,
        steps: list[StepDefinition],
        parent_run_id: str | None = None,
        delegation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a run and execute until completion or approval gate."""
        planned_steps = steps or await self._plan_next_steps(
            objective=objective,
            mode=mode,
            existing_steps=[],
        )
        run = self.repo.create_run(
            objective=objective,
            mode=mode.value,
            steps=planned_steps,
            parent_run_id=parent_run_id,
            delegation=delegation,
        )
        await self._update_run_capability_state(
            run["id"],
            self._resolved_skills_for_objective(objective),
        )
        await self._update_run_kernel_state(run["id"], phase="running")
        await self._publish(run["id"], "run.created", {"objective": objective})
        await self._execute_until_gate(run["id"])
        return self.repo.get_run(run["id"]) or run

    async def _plan_next_steps(
        self,
        *,
        objective: str,
        mode: RunMode,
        existing_steps: list[dict[str, Any]],
        completed_step: dict[str, Any] | None = None,
        result: StepExecutionResult | None = None,
    ) -> list[StepDefinition]:
        resolved_skills = self._resolved_skills_for_objective(objective)
        skill_context = serialize_skill_context(resolved_skills)
        proposed = await request_next_steps(
            self.adaptive_planner,
            objective=objective,
            mode=mode,
            existing_steps=existing_steps,
            completed_step=completed_step,
            result=result,
            skill_context=skill_context,
        )
        if completed_step is None or result is None:
            planned = apply_initial_plan_policy(proposed, mode=mode)
            if planned:
                return self._annotate_skill_context(planned, resolved_skills)
            fallback_reason = "policy_rejected" if proposed else "no_steps"
            return self._annotate_skill_context(
                annotate_planner_fallback(
                    annotate_planner_steps(
                        plan_steps_for_objective(objective),
                        planner_source="rule",
                        planner_phase="initial",
                    ),
                    fallback_reason=fallback_reason,
                ),
                resolved_skills,
            )
        planned = apply_follow_up_policy(
            proposed,
            mode=mode,
        )
        if planned:
            return self._annotate_skill_context(planned, resolved_skills)
        fallback_reason = "policy_rejected" if proposed else "no_steps"
        fallback_steps = apply_follow_up_policy(
            annotate_planner_fallback(
                annotate_planner_steps(
                    plan_follow_up_steps(
                        objective=objective,
                        completed_step=completed_step,
                        result=result,
                        existing_steps=existing_steps,
                        skill_context=skill_context,
                    ),
                    planner_source="rule",
                    planner_phase="follow_up",
                ),
                fallback_reason=fallback_reason,
            ),
            mode=mode,
        )
        return self._annotate_skill_context(fallback_steps, resolved_skills)

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
            await self._update_run_kernel_state(
                run_id,
                phase="paused",
                failure_reason=reason,
            )
            await self.interaction.deliver_status(
                run_id,
                RunStatus.PAUSED.value,
                "Approval rejected",
            )
            await self._publish(run_id, "run.paused", {"step_id": step_id, "reason": reason})
            child_run = self.repo.get_run(run_id)
            if child_run is not None:
                await self._pause_waiting_parent_for_child(child_run, reason)
            return self.repo.get_run(run_id) or {}

        self.repo.mark_run_status(run_id, RunStatus.RUNNING.value)
        await self._update_run_kernel_state(run_id, phase="running")
        run = self.repo.get_run(run_id)
        result = await self._execute_step(step)
        if result is None:
            await self._mark_run_failed(run_id, step_id)
            return self.repo.get_run(run_id) or {}
        if self._is_deferred_delegate_result(result):
            return self.repo.get_run(run_id) or {}
        if run is not None:
            await self._apply_follow_up_planning(run, step, result)
        await self._execute_until_gate(run_id)
        child_run = self.repo.get_run(run_id)
        if child_run is not None:
            await self._resume_waiting_parent_for_child(child_run)
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
        await self._update_run_kernel_state(run_id, phase="running")
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
        await self._update_run_kernel_state(run_id, phase="running")
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
                    await self._update_run_kernel_state(run_id, phase="awaiting_approval")
                    return
                if status == StepStatus.PENDING.value:
                    next_step = step
                    break

            if next_step is None:
                self.repo.mark_run_status(run_id, RunStatus.COMPLETED.value)
                await self._update_run_kernel_state(run_id, phase="completed")
                await self.interaction.deliver_status(
                    run_id,
                    RunStatus.COMPLETED.value,
                    "Run completed",
                )
                await self._publish(run_id, "run.completed", {})
                return

            if self._autonomous_step_budget_reached(run_id):
                message = (
                    "Run exceeded the autonomous step budget before converging."
                )
                await self._fail_pending_step(run_id, next_step, message, retryable=True)
                await self._mark_run_failed(run_id, next_step["id"])
                return

            if self._identical_step_streak_reached(run_id, next_step):
                message = (
                    "Run stopped because the same step repeated without progress."
                )
                await self._fail_pending_step(run_id, next_step, message, retryable=False)
                await self._mark_run_failed(run_id, next_step["id"])
                return

            try:
                self._assert_delegated_step_allowed(run, next_step)
            except PermissionError as exc:
                self.repo.mark_step_status(
                    next_step["id"],
                    StepStatus.FAILED.value,
                    error_text=str(exc),
                )
                await self._publish(
                    run_id,
                    "step.failed",
                    self._step_event_payload(
                        next_step,
                        step_id=next_step["id"],
                        error=str(exc),
                    ),
                )
                await self._mark_run_failed(run_id, next_step["id"])
                return

            if mode == RunMode.SUPERVISED and is_high_risk_action(
                next_step["action_type"], next_step["instruction"]
            ):
                self.repo.mark_step_status(next_step["id"], StepStatus.PENDING_APPROVAL.value)
                self.repo.mark_run_status(run_id, RunStatus.PENDING_APPROVAL.value)
                await self._update_run_kernel_state(run_id, phase="awaiting_approval")
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
                    self._step_event_payload(next_step, step_id=next_step["id"]),
                )
                return

            result = await self._execute_step(next_step)
            if result is None:
                if await self._retry_failed_step_if_allowed(run_id, next_step["id"]):
                    continue
                await self._mark_run_failed(run_id, next_step["id"])
                return
            if self._is_deferred_delegate_result(result):
                return

            await self._apply_follow_up_planning(run, next_step, result)

    async def _execute_step(self, step: dict[str, Any]) -> StepExecutionResult | None:
        run_id = step["run_id"]
        step_id = step["id"]
        try:
            run = self.repo.get_run(run_id)
            if run is not None:
                self._assert_delegated_step_allowed(run, step)
            self.repo.mark_step_status(step_id, StepStatus.RUNNING.value)
            await self._publish(
                run_id,
                "step.running",
                self._step_event_payload(step, step_id=step_id),
            )
            if str(step["action_type"]).strip().lower() == "delegate":
                result = await self._execute_delegate_step(step)
            else:
                result = await self.execution.execute_step(
                    run_id=run_id,
                    step_id=step_id,
                    action_type=step["action_type"],
                    instruction=step["instruction"],
                )
            if self._is_deferred_delegate_result(result):
                self.repo.mark_run_status(run_id, RunStatus.PENDING_APPROVAL.value)
                return result
            verification_result = self._verification_result(result)
            self.repo.mark_step_status(
                step_id,
                StepStatus.COMPLETED.value,
                output_text=result.output_text,
            )
            self.repo.merge_step_metadata(
                step_id,
                {
                    **result.metadata,
                    "verification_result": verification_result,
                },
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
                self._step_event_payload(
                    step,
                    step_id=step_id,
                    output_text=result.output_text[:500],
                    citations=len(result.citations),
                    artifacts=len(result.artifacts),
                    verification_result=verification_result,
                ),
            )
            return result
        except Exception as exc:
            log.warning("Step execution failed for %s: %s", step_id, exc)
            self.repo.merge_step_metadata(
                step_id,
                {
                    "verification_result": "failed",
                    "kernel_decision": "fail",
                    "retryable": True,
                },
            )
            self.repo.mark_step_status(
                step_id,
                StepStatus.FAILED.value,
                error_text=str(exc),
            )
            await self._publish(
                run_id,
                "step.failed",
                self._step_event_payload(step, step_id=step_id, error=str(exc)),
            )
            return None

    async def _execute_delegate_step(self, step: dict[str, Any]) -> StepExecutionResult:
        payload = self._parse_delegate_payload(step["instruction"])
        child_steps = payload.steps
        child_mode = self._delegated_child_mode(step["run_id"], payload, child_steps)
        child_run = await self.create_run(
            objective=payload.objective,
            mode=child_mode,
            steps=child_steps,
            parent_run_id=step["run_id"],
            delegation={
                "role": payload.role,
                "objective": payload.objective,
                "status": RunStatus.RUNNING.value,
                "summary": payload.summary,
                "context": self._build_delegation_context(step["run_id"], payload.context),
            },
        )
        child_run_id = child_run["id"]
        await self._publish(
            step["run_id"],
            "delegate.started",
            {
                "step_id": step["id"],
                "child_run_id": child_run_id,
                "role": payload.role,
                "objective": payload.objective,
            },
        )
        child_run = self.repo.get_run(child_run_id) or child_run
        child_status = str(child_run.get("status", RunStatus.FAILED.value))
        if child_status == RunStatus.PENDING_APPROVAL.value:
            waiting_summary = f"{payload.role} awaiting delegated approval: {payload.objective}"
            self.repo.update_delegation(child_run_id, status=child_status, summary=waiting_summary)
            await self._publish(
                step["run_id"],
                "delegate.pending_approval",
                {
                    "step_id": step["id"],
                    "child_run_id": child_run_id,
                    "role": payload.role,
                    "status": child_status,
                    "summary": waiting_summary,
                },
            )
            return StepExecutionResult(
                metadata={
                    "child_run_id": child_run_id,
                    "child_status": child_status,
                    "delegate_role": payload.role,
                    "defer_completion": True,
                }
            )
        return await self._finalize_delegate_step(step, payload, child_run)

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
            decision = (
                "continue"
                if self._has_pending_steps_after(completed_step, existing_steps)
                else "stop"
            )
            await self._record_kernel_decision(
                run["id"],
                completed_step,
                decision,
                reason="existing_pending_steps" if decision == "continue" else "no_follow_up_needed",
            )
            return

        follow_up_steps = await self._plan_next_steps(
            objective=run["objective"],
            mode=RunMode(run["mode"]),
            existing_steps=existing_steps,
            completed_step=completed_step,
            result=result,
        )
        if not follow_up_steps:
            decision = (
                "continue"
                if self._has_pending_steps_after(completed_step, self.repo.list_steps(run["id"]))
                else "stop"
            )
            await self._record_kernel_decision(
                run["id"],
                completed_step,
                decision,
                reason="no_follow_up_steps",
            )
            return

        self.repo.insert_steps_after(completed_step["id"], follow_up_steps)
        gated_follow_up_step = self._delegated_high_risk_follow_up_step(
            run,
            completed_step,
            follow_up_steps,
        )
        await self._publish(
            run["id"],
            "run.replanned",
            {
                "after_step_id": completed_step["id"],
                "inserted_steps": [s.model_dump() for s in follow_up_steps],
            },
        )
        if gated_follow_up_step is not None:
            await self._record_kernel_decision(
                run["id"],
                completed_step,
                "await_approval",
                reason="high_risk_follow_up",
            )
            self.repo.mark_step_status(gated_follow_up_step["id"], StepStatus.PENDING_APPROVAL.value)
            self.repo.mark_run_status(run["id"], RunStatus.PENDING_APPROVAL.value)
            summary = f"{gated_follow_up_step['action_type']}: {gated_follow_up_step['instruction'][:120]}"
            await self.interaction.request_approval(
                run_id=run["id"],
                step_id=gated_follow_up_step["id"],
                summary=summary,
                action_type=gated_follow_up_step["action_type"],
            )
            await self._publish(
                run["id"],
                "step.pending_approval",
                self._step_event_payload(gated_follow_up_step, step_id=gated_follow_up_step["id"]),
            )
            return
        await self._record_kernel_decision(
            run["id"],
            completed_step,
            "continue",
            reason="follow_up_inserted",
        )

    async def _mark_run_failed(self, run_id: str, step_id: str) -> None:
        self.repo.mark_run_status(run_id, RunStatus.FAILED.value)
        failed_step = self.repo.get_step(step_id)
        failure_reason = ""
        if failed_step is not None:
            failure_reason = str(failed_step.get("error_text", "")).strip()
        await self._update_run_kernel_state(
            run_id,
            phase="failed",
            failure_reason=failure_reason,
        )
        await self.interaction.deliver_status(run_id, RunStatus.FAILED.value, "Run failed")
        await self._publish(run_id, "run.failed", {"step_id": step_id})

    async def _publish(self, run_id: str, event_type: str, payload: dict[str, Any]) -> None:
        await self.events.publish(RunEvent(run_id=run_id, event_type=event_type, payload=payload))

    async def _update_run_kernel_state(
        self,
        run_id: str,
        *,
        phase: str,
        failure_reason: str = "",
    ) -> None:
        run = self.repo.get_run(run_id)
        if run is None:
            return
        steps = run.get("steps") if isinstance(run.get("steps"), list) else []
        metadata = run.get("metadata") if isinstance(run.get("metadata"), dict) else {}
        capability_state = (
            metadata.get("capability_state")
            if isinstance(metadata.get("capability_state"), dict)
            else {}
        )
        completed_step_count = sum(
            1 for step in steps if str(step.get("status", "")) == StepStatus.COMPLETED.value
        )
        pending_step_count = sum(
            1
            for step in steps
            if str(step.get("status", "")) in {
                StepStatus.PENDING.value,
                StepStatus.RUNNING.value,
                StepStatus.PENDING_APPROVAL.value,
            }
        )
        failed_step_count = sum(
            1 for step in steps if str(step.get("status", "")) == StepStatus.FAILED.value
        )
        rejected_step_count = sum(
            1 for step in steps if str(step.get("status", "")) == StepStatus.REJECTED.value
        )
        current_step_id = ""
        for step in steps:
            if str(step.get("status", "")) in {
                StepStatus.RUNNING.value,
                StepStatus.PENDING_APPROVAL.value,
                StepStatus.PENDING.value,
            }:
                current_step_id = str(step.get("id", ""))
                break
        kernel_state: dict[str, Any] = {
            "phase": phase,
            "completed_step_count": completed_step_count,
            "pending_step_count": pending_step_count,
            "failed_step_count": failed_step_count,
            "rejected_step_count": rejected_step_count,
        }
        if current_step_id:
            kernel_state["current_step_id"] = current_step_id
        kernel_state.update(
            self._kernel_strategy_snapshot(
                objective=str(run.get("objective", "")),
                phase=phase,
                steps=steps,
                capability_state=capability_state,
            )
        )
        normalized_failure_reason = failure_reason.strip()
        if normalized_failure_reason:
            kernel_state["failure_reason"] = normalized_failure_reason
        self.repo.merge_run_metadata(run_id, {"kernel_state": kernel_state})
        await self._publish(run_id, "run.kernel", kernel_state)

    async def _update_run_capability_state(
        self,
        run_id: str,
        resolved_skills: list[SkillManifest],
    ) -> None:
        capability_state = self._capability_state(resolved_skills)
        if not capability_state:
            return
        self.repo.merge_run_metadata(run_id, {"capability_state": capability_state})
        await self._publish(run_id, "run.capability", capability_state)

    @staticmethod
    def _step_event_payload(step: dict[str, Any], **extra: Any) -> dict[str, Any]:
        payload = {
            "step_id": str(step.get("id", "")),
            "action_type": str(step.get("action_type", "")),
            "instruction": str(step.get("instruction", "")),
        }
        metadata = step.get("metadata")
        if isinstance(metadata, dict):
            planner_source = str(metadata.get("planner_source", "")).strip()
            planner_phase = str(metadata.get("planner_phase", "")).strip()
            planner_fallback_reason = str(metadata.get("planner_fallback_reason", "")).strip()
            if planner_source:
                payload["planner_source"] = planner_source
            if planner_phase:
                payload["planner_phase"] = planner_phase
            if planner_fallback_reason:
                payload["planner_fallback_reason"] = planner_fallback_reason
            skill_source = str(metadata.get("skill_source", "")).strip()
            skill_names = metadata.get("skill_names")
            verification_result = str(metadata.get("verification_result", "")).strip()
            kernel_decision = str(metadata.get("kernel_decision", "")).strip()
            retryable = metadata.get("retryable")
            retry_count = metadata.get("retry_count")
            if skill_source:
                payload["skill_source"] = skill_source
            if isinstance(skill_names, list) and skill_names:
                payload["skill_names"] = [str(name) for name in skill_names]
            if verification_result:
                payload["verification_result"] = verification_result
            if kernel_decision:
                payload["kernel_decision"] = kernel_decision
            if isinstance(retryable, bool):
                payload["retryable"] = retryable
            if isinstance(retry_count, int):
                payload["retry_count"] = retry_count
        payload.update(extra)
        return payload

    @staticmethod
    def _verification_result(result: StepExecutionResult) -> str:
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        if metadata.get("command_failed") is True:
            return "failed"
        exit_code = metadata.get("exit_code")
        if isinstance(exit_code, int) and exit_code != 0:
            return "failed"
        if result.citations or result.artifacts or str(result.output_text or "").strip():
            return "passed"
        return "unknown"

    def _resolved_skills_for_objective(self, objective: str) -> list[SkillManifest]:
        if self.capability_resolver is None:
            return []
        return self.capability_resolver.resolve(objective)

    @staticmethod
    def _capability_state(resolved_skills: list[SkillManifest]) -> dict[str, Any]:
        if not resolved_skills:
            return {}
        resolved_payload = [
            {
                "name": skill.name,
                "description": skill.description,
                "path": skill.path,
                "preferred_initial_actions": list(skill.preferred_initial_actions),
                "preferred_follow_up_actions": list(skill.preferred_follow_up_actions),
            }
            for skill in resolved_skills
        ]
        return {
            "skill_source": "capability_resolver",
            "skill_names": [skill.name for skill in resolved_skills],
            "resolved_skill_count": len(resolved_skills),
            "resolved_skills": resolved_payload,
        }

    async def _record_kernel_decision(
        self,
        run_id: str,
        completed_step: dict[str, Any],
        decision: str,
        *,
        reason: str = "",
    ) -> None:
        metadata = {"kernel_decision": decision}
        if reason.strip():
            metadata["kernel_decision_reason"] = reason.strip()
        self.repo.merge_step_metadata(completed_step["id"], metadata)
        payload = self._step_event_payload(
            self.repo.get_step(completed_step["id"]) or completed_step,
            decision=decision,
        )
        if reason.strip():
            payload["reason"] = reason.strip()
        await self._publish(run_id, "kernel.decision", payload)

    @staticmethod
    def _kernel_strategy_snapshot(
        *,
        objective: str,
        phase: str,
        steps: list[dict[str, Any]],
        capability_state: dict[str, Any],
    ) -> dict[str, Any]:
        strategy = RunEngine._strategy_for_objective(objective)
        current_step = next(
            (
                step
                for step in steps
                if str(step.get("status", "")) in {
                    StepStatus.RUNNING.value,
                    StepStatus.PENDING_APPROVAL.value,
                    StepStatus.PENDING.value,
                }
            ),
            None,
        )
        last_completed_step = next(
            (
                step
                for step in reversed(steps)
                if str(step.get("status", "")) == StepStatus.COMPLETED.value
            ),
            None,
        )
        tactic = "idle"
        tactic_reason = ""
        if current_step is not None:
            tactic = RunEngine._tactic_for_action(str(current_step.get("action_type", "")))
            tactic_reason = (
                f"Current step is {str(current_step.get('action_type', '')).strip().lower()}."
            )
        elif last_completed_step is not None:
            last_action = str(last_completed_step.get("action_type", "")).strip().lower()
            last_metadata = (
                last_completed_step.get("metadata")
                if isinstance(last_completed_step.get("metadata"), dict)
                else {}
            )
            decision = str(last_metadata.get("kernel_decision", "")).strip().lower()
            tactic = RunEngine._tactic_after_decision(last_action, decision)
            if decision:
                tactic_reason = f"Last completed step decided to {decision.replace('_', ' ')}."
            elif last_action:
                tactic_reason = f"Last completed step was {last_action}."
        if phase == "awaiting_approval":
            tactic = "approval"
            tactic_reason = "Kernel is waiting for approval before continuing."
        elif phase == "completed":
            tactic = "done"
            tactic_reason = "Run has no remaining pending work."
        elif phase == "failed":
            tactic = "blocked"
            tactic_reason = "Run failed and needs intervention or retry."
        snapshot: dict[str, Any] = {
            "strategy": strategy,
            "tactic": tactic,
        }
        if tactic_reason:
            snapshot["tactic_reason"] = tactic_reason
        skill_names = capability_state.get("skill_names")
        if isinstance(skill_names, list) and skill_names:
            snapshot["skill_names"] = [str(name) for name in skill_names]
        return snapshot

    @staticmethod
    def _strategy_for_objective(objective: str) -> str:
        lowered = objective.lower()
        if any(token in lowered for token in ("fix", "implement", "refactor", "repo", "codebase", "test")):
            return "coding"
        if any(token in lowered for token in ("api", "endpoint", "webhook", "http request")):
            return "api"
        if any(token in lowered for token in ("image", "illustration", "graphic", "poster", "hero image")):
            return "image"
        if any(token in lowered for token in ("chart", "graph", "plot", "report", "brief", "memo")):
            return "artifact"
        if "http://" in lowered or "https://" in lowered or any(
            token in lowered
            for token in ("submit", "fill out", "apply", "register", "book", "checkout", "form")
        ):
            return "workflow"
        return "research"

    @staticmethod
    def _tactic_for_action(action_type: str) -> str:
        action = action_type.strip().lower()
        if action in {"search_web", "fetch_url", "navigate", "inspect", "list_files", "read_file", "call_api"}:
            return "observe"
        if action in {"extract", "read", "generate_report", "generate_chart", "generate_image", "export"}:
            return "synthesize"
        if action in {"execute_code", "write_file", "edit_file", "type", "click", "submit", "write"}:
            return "act"
        if action in {"wait", "scroll"}:
            return "stabilize"
        if action == "delegate":
            return "delegate"
        return "act"

    @staticmethod
    def _tactic_after_decision(action_type: str, decision: str) -> str:
        if decision == "await_approval":
            return "approval"
        if decision == "stop":
            return "done"
        if decision == "continue":
            return RunEngine._tactic_for_action(action_type)
        return "review"

    def _autonomous_step_budget_reached(self, run_id: str) -> bool:
        executed_steps = 0
        for step in self.repo.list_steps(run_id):
            if step.get("started_at"):
                executed_steps += 1
        return executed_steps >= self.max_autonomous_steps

    def _identical_step_streak_reached(self, run_id: str, next_step: dict[str, Any]) -> bool:
        if self.max_identical_step_streak <= 0:
            return False
        steps = self.repo.list_steps(run_id)
        next_step_id = str(next_step.get("id", ""))
        prefix: list[dict[str, Any]] = []
        for step in steps:
            if str(step.get("id", "")) == next_step_id:
                break
            prefix.append(step)
        streak = 0
        baseline_output = ""
        baseline_verification = ""
        for step in reversed(prefix):
            if str(step.get("status", "")) != StepStatus.COMPLETED.value:
                break
            if str(step.get("action_type", "")) != str(next_step.get("action_type", "")):
                break
            if str(step.get("instruction", "")) != str(next_step.get("instruction", "")):
                break
            metadata = step.get("metadata")
            verification = (
                str(metadata.get("verification_result", "")).strip()
                if isinstance(metadata, dict)
                else ""
            )
            output_text = str(step.get("output_text", "")).strip()
            if streak == 0:
                baseline_output = output_text
                baseline_verification = verification
            elif output_text != baseline_output or verification != baseline_verification:
                break
            streak += 1
        return streak >= self.max_identical_step_streak

    async def _retry_failed_step_if_allowed(self, run_id: str, step_id: str) -> bool:
        if self.max_step_retries <= 0:
            return False
        step = self.repo.get_step(step_id)
        if step is None or step.get("status") != StepStatus.FAILED.value:
            return False
        metadata = step.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("retryable") is not True:
            return False
        retry_count = int(metadata.get("retry_count", 0) or 0)
        if retry_count >= self.max_step_retries:
            return False
        if not self.repo.reset_step_for_retry(step_id):
            return False
        await self._publish(
            run_id,
            "step.retrying",
            self._step_event_payload(
                self.repo.get_step(step_id) or step,
                step_id=step_id,
                retry_count=retry_count + 1,
            ),
        )
        return True

    async def _fail_pending_step(
        self,
        run_id: str,
        step: dict[str, Any],
        message: str,
        *,
        retryable: bool,
    ) -> None:
        self.repo.merge_step_metadata(
            step["id"],
            {
                "verification_result": "failed",
                "kernel_decision": "fail",
                "retryable": retryable,
            },
        )
        self.repo.mark_step_status(
            step["id"],
            StepStatus.FAILED.value,
            error_text=message,
        )
        await self._publish(
            run_id,
            "step.failed",
            self._step_event_payload(
                step,
                step_id=step["id"],
                error=message,
            ),
        )


    @staticmethod
    def _annotate_skill_context(
        steps: list[StepDefinition],
        resolved_skills: list[SkillManifest],
    ) -> list[StepDefinition]:
        capability_state = RunEngine._capability_state(resolved_skills)
        if not capability_state:
            return steps
        annotated: list[StepDefinition] = []
        for step in steps:
            metadata = dict(step.metadata)
            metadata.update(capability_state)
            annotated.append(
                StepDefinition(
                    action_type=step.action_type,
                    instruction=step.instruction,
                    metadata=metadata,
                )
            )
        return annotated

    @staticmethod
    def _parse_delegate_payload(instruction: str) -> DelegationStepPayload:
        try:
            payload = json.loads(instruction)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Delegate steps require JSON instruction payloads.") from exc
        try:
            return DelegationStepPayload.model_validate(payload)
        except ValidationError as exc:
            raise RuntimeError(f"Invalid delegate payload: {exc}") from exc

    @staticmethod
    def _delegation_summary(payload: DelegationStepPayload, child_run: dict[str, Any]) -> str:
        child_status = str(child_run.get("status", "unknown"))
        if child_status == RunStatus.COMPLETED.value:
            return payload.summary or f"{payload.role} completed: {payload.objective}"
        failure_reason = RunEngine._delegation_failure_reason(child_run)
        if failure_reason:
            return f"{payload.role} failed: {failure_reason}"
        return f"{payload.role} failed: {payload.objective}"

    @staticmethod
    def _delegation_failure_reason(child_run: dict[str, Any]) -> str:
        explicit_reason = str(child_run.get("delegation_failure_reason", "")).strip()
        if explicit_reason:
            return explicit_reason

        steps = child_run.get("steps")
        if not isinstance(steps, list):
            return ""

        for step in steps:
            if not isinstance(step, dict):
                continue
            if str(step.get("status", "")).strip().lower() != StepStatus.FAILED.value:
                continue
            error_text = str(step.get("error_text", "")).strip()
            if error_text:
                return error_text
        return ""

    def _merged_child_citations(self, child_run_id: str) -> list[CitationRecord]:
        citations: list[CitationRecord] = []
        for item in self.repo.list_citations(child_run_id):
            citations.append(
                CitationRecord(
                    url=str(item.get("url", "")),
                    title=str(item.get("title", "")),
                    snippet=str(item.get("snippet", "")),
                )
            )
        return citations

    def _merged_child_artifacts(self, child_run_id: str) -> list[ArtifactRecord]:
        artifacts: list[ArtifactRecord] = []
        for item in self.repo.list_artifacts(child_run_id):
            artifacts.append(
                ArtifactRecord(
                    kind=str(item.get("kind", "text")),
                    name=str(item.get("name", "")),
                    rel_path=str(item.get("rel_path", "")),
                    sandbox_path=str(item.get("sandbox_path", "")),
                    sha256=str(item.get("sha256", "")),
                )
            )
        return artifacts

    def _build_delegation_context(
        self,
        parent_run_id: str,
        explicit_context: dict[str, Any],
    ) -> dict[str, Any]:
        run = self.repo.get_run(parent_run_id) or {}
        context = dict(explicit_context)
        context.setdefault("parent_run_id", parent_run_id)
        context.setdefault("parent_objective", str(run.get("objective", "")))
        context.setdefault("parent_mode", str(run.get("mode", "")))
        context.setdefault(
            "citations",
            [
                {
                    "url": str(item.get("url", "")),
                    "title": str(item.get("title", "")),
                    "snippet": str(item.get("snippet", "")),
                }
                for item in self.repo.list_citations(parent_run_id)
            ],
        )
        context.setdefault(
            "artifacts",
            [
                {
                    "kind": str(item.get("kind", "text")),
                    "name": str(item.get("name", "")),
                    "rel_path": str(item.get("rel_path", "")),
                }
                for item in self.repo.list_artifacts(parent_run_id)
            ],
        )
        return context

    def _delegated_output_contract_violations(self, child_run: dict[str, Any]) -> list[str]:
        delegation = child_run.get("delegation")
        context: dict[str, Any] = {}
        if isinstance(delegation, dict):
            raw_context = delegation.get("context")
            if isinstance(raw_context, dict):
                context = raw_context
        return delegated_output_contract_violations(
            context,
            citations=self.repo.list_citations(str(child_run.get("id", ""))),
            artifacts=self.repo.list_artifacts(str(child_run.get("id", ""))),
        )

    async def _finalize_delegate_step(
        self,
        parent_step: dict[str, Any],
        payload: DelegationStepPayload,
        child_run: dict[str, Any],
    ) -> StepExecutionResult:
        child_run_id = str(child_run.get("id", ""))
        child_status = str(child_run.get("status", RunStatus.FAILED.value))
        if child_status == RunStatus.COMPLETED.value:
            contract_violations = self._delegated_output_contract_violations(child_run)
            if contract_violations:
                child_status = RunStatus.FAILED.value
                self.repo.mark_run_status(child_run_id, child_status)
                child_run = dict(child_run)
                child_run["status"] = child_status
                child_run["delegation_failure_reason"] = "; ".join(contract_violations)
        output_text = self._delegation_summary(payload, child_run)
        self.repo.update_delegation(child_run_id, status=child_status, summary=output_text)
        event_type = (
            "delegate.completed"
            if child_status == RunStatus.COMPLETED.value
            else "delegate.failed"
        )
        await self._publish(
            parent_step["run_id"],
            event_type,
            {
                "step_id": parent_step["id"],
                "child_run_id": child_run_id,
                "role": payload.role,
                "status": child_status,
                "summary": output_text,
            },
        )
        return StepExecutionResult(
            output_text=output_text,
            citations=self._merged_child_citations(child_run_id),
            artifacts=self._merged_child_artifacts(child_run_id),
            metadata={
                "child_run_id": child_run_id,
                "child_status": child_status,
                "delegate_role": payload.role,
            },
        )

    @staticmethod
    def _is_deferred_delegate_result(result: StepExecutionResult) -> bool:
        return bool(result.metadata.get("defer_completion"))

    def _delegated_child_mode(
        self,
        parent_run_id: str,
        payload: DelegationStepPayload,
        child_steps: list[StepDefinition],
    ) -> RunMode:
        parent_run = self.repo.get_run(parent_run_id) or {}
        parent_mode = str(parent_run.get("mode", "")).strip().lower()
        if parent_mode != RunMode.SUPERVISED.value:
            return payload.mode
        if any(is_high_risk_action(step.action_type, step.instruction) for step in child_steps):
            return payload.mode
        return RunMode.SUPERVISED

    def _delegated_high_risk_follow_up_step(
        self,
        run: dict[str, Any],
        completed_step: dict[str, Any],
        follow_up_steps: list[StepDefinition],
    ) -> dict[str, Any] | None:
        delegation = run.get("delegation")
        if not isinstance(delegation, dict) or not delegation:
            return None

        context = delegation.get("context")
        if not isinstance(context, dict):
            return None
        if str(context.get("parent_mode", "")).strip().lower() != RunMode.SUPERVISED.value:
            return None

        candidate_defs = [
            step
            for step in follow_up_steps
            if is_high_risk_action(step.action_type, step.instruction)
        ]
        if not candidate_defs:
            return None

        completed_index = int(completed_step.get("step_index", -1))
        pending_steps = [
            step
            for step in self.repo.list_steps(run["id"])
            if int(step.get("step_index", -1)) > completed_index
            and str(step.get("status", "")).strip().lower() == StepStatus.PENDING.value
        ]
        for candidate in candidate_defs:
            for step in pending_steps:
                if step.get("action_type") != candidate.action_type:
                    continue
                if step.get("instruction") != candidate.instruction:
                    continue
                return step
        return None

    async def _resume_waiting_parent_for_child(self, child_run: dict[str, Any]) -> None:
        parent_run_id = str(child_run.get("parent_run_id", "")).strip()
        if not parent_run_id:
            return

        child_status = str(child_run.get("status", "")).strip().lower()
        if child_status in {
            RunStatus.PENDING_APPROVAL.value,
            RunStatus.PAUSED.value,
            RunStatus.RUNNING.value,
        }:
            return

        parent_run = self.repo.get_run(parent_run_id)
        if parent_run is None:
            return
        waiting_step = self._waiting_parent_delegate_step(parent_run)
        if waiting_step is None:
            return

        payload = self._parse_delegate_payload(waiting_step["instruction"])
        result = await self._finalize_delegate_step(waiting_step, payload, child_run)
        self.repo.mark_step_status(waiting_step["id"], StepStatus.COMPLETED.value, output_text=result.output_text)
        self.repo.merge_step_metadata(waiting_step["id"], result.metadata)
        self.repo.add_citations(
            run_id=parent_run_id,
            step_id=waiting_step["id"],
            citations=[c.model_dump() for c in result.citations],
        )
        self.repo.add_artifacts(
            run_id=parent_run_id,
            step_id=waiting_step["id"],
            artifacts=[a.model_dump() for a in result.artifacts],
        )
        await self._publish(
            parent_run_id,
            "step.completed",
            self._step_event_payload(
                waiting_step,
                step_id=waiting_step["id"],
                output_text=result.output_text[:500],
                citations=len(result.citations),
                artifacts=len(result.artifacts),
            ),
        )
        parent_run = self.repo.get_run(parent_run_id)
        if parent_run is None:
            return
        await self._apply_follow_up_planning(parent_run, waiting_step, result)
        await self._execute_until_gate(parent_run_id)

    async def _pause_waiting_parent_for_child(self, child_run: dict[str, Any], reason: str) -> None:
        parent_run_id = str(child_run.get("parent_run_id", "")).strip()
        if not parent_run_id:
            return

        parent_run = self.repo.get_run(parent_run_id)
        if parent_run is None:
            return
        waiting_step = self._waiting_parent_delegate_step(parent_run)
        if waiting_step is None:
            return

        message = reason or "Delegated child approval rejected."
        self.repo.mark_step_status(waiting_step["id"], StepStatus.REJECTED.value, error_text=message)
        self.repo.mark_run_status(parent_run_id, RunStatus.PAUSED.value)
        await self._publish(
            parent_run_id,
            "delegate.failed",
            {
                "step_id": waiting_step["id"],
                "child_run_id": child_run["id"],
                "role": str(child_run.get("delegation", {}).get("role", "worker")),
                "status": RunStatus.PAUSED.value,
                "summary": message,
            },
        )
        await self._publish(parent_run_id, "run.paused", {"step_id": waiting_step["id"], "reason": message})

    @staticmethod
    def _waiting_parent_delegate_step(parent_run: dict[str, Any]) -> dict[str, Any] | None:
        steps = parent_run.get("steps")
        if not isinstance(steps, list):
            return None
        for step in steps:
            if not isinstance(step, dict):
                continue
            if str(step.get("action_type", "")).strip().lower() != "delegate":
                continue
            if str(step.get("status", "")).strip().lower() == StepStatus.RUNNING.value:
                return step
        return None

    def _assert_delegated_step_allowed(
        self,
        run: dict[str, Any],
        step: dict[str, Any],
    ) -> None:
        delegation = run.get("delegation")
        if not isinstance(delegation, dict) or not delegation:
            return

        role = str(delegation.get("role", "")).strip().lower()
        action = str(step.get("action_type", "")).strip().lower()
        if action == "delegate":
            raise PermissionError("Nested delegation is not allowed for delegated runs.")
        if not delegated_role_allows_action(role, action):
            raise PermissionError(f"Delegated role `{role or 'unknown'}` does not allow action `{action}`.")

        if action in {"list_files", "read_file", "write_file", "edit_file"}:
            context = delegation.get("context")
            workspace_paths = []
            if isinstance(context, dict):
                raw_paths = context.get("workspace_paths")
                if isinstance(raw_paths, list):
                    workspace_paths = [str(path) for path in raw_paths]
            target_path = self._delegated_workspace_target(step.get("instruction", ""))
            if not delegated_workspace_path_allowed(workspace_paths, target_path):
                raise PermissionError(
                    f"Delegated workspace path `{target_path}` is outside delegated workspace scope."
                )

    @staticmethod
    def _delegated_workspace_target(instruction: str) -> str:
        try:
            payload = json.loads(instruction)
        except json.JSONDecodeError:
            return "."
        if not isinstance(payload, dict):
            return "."
        target_path = str(payload.get("path") or payload.get("file") or ".").strip()
        return target_path or "."

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
                completed_action in {"extract", "scroll"}
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
