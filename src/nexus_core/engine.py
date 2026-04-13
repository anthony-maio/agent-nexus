"""Run orchestration engine independent of transport adapters."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
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
    RunVerificationRecord,
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
    continue_tool_follow_up_sequence,
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
from nexus_core.skills import CapabilityResolver, SkillManifest, SkillRegistry, serialize_skill_context
from nexus_core.verification import evaluate_run_completion

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

    def list_citations(self, run_id: str) -> list[dict[str, Any]]: ...

    def list_artifacts(self, run_id: str) -> list[dict[str, Any]]: ...

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


class SkillAcquirer(Protocol):
    async def acquire_skill(self, intent: str, requirements: str = "") -> dict[str, Any]: ...


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
        skill_registry: SkillRegistry | None = None,
        external_tool_registry: Any | None = None,
        skill_acquirer: SkillAcquirer | None = None,
        auto_acquire_skills: bool = False,
        max_autonomous_steps: int = 24,
        max_step_retries: int = 0,
        max_identical_step_streak: int = 0,
        max_completion_recovery_attempts: int = 1,
    ) -> None:
        self.repo = repository
        self.execution = execution
        self.interaction = interaction
        self.events = events
        self.adaptive_planner = adaptive_planner or RuleAdaptivePlanner()
        self.capability_resolver = capability_resolver
        self.skill_registry = skill_registry
        self.external_tool_registry = external_tool_registry
        self.skill_acquirer = skill_acquirer
        self.auto_acquire_skills = bool(auto_acquire_skills)
        self.canonical_workspace = canonical_workspace
        self.sandbox_artifacts_root = (
            sandbox_artifacts_root.resolve() if sandbox_artifacts_root else None
        )
        self.max_autonomous_steps = max(int(max_autonomous_steps), 1)
        self.max_step_retries = max(int(max_step_retries), 0)
        self.max_identical_step_streak = max(int(max_identical_step_streak), 0)
        self.max_completion_recovery_attempts = max(int(max_completion_recovery_attempts), 0)
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
        resolved_skills, capability_state = await self._resolve_capability_context(
            objective,
            allow_acquire=True,
        )
        if steps:
            planned_steps = steps
        else:
            setup_steps = self._setup_steps_for_resolved_skills(
                objective=objective,
                resolved_skills=resolved_skills,
            )
            planned_steps = setup_steps or await self._plan_next_steps(
                objective=objective,
                mode=mode,
                existing_steps=[],
                resolved_skills=resolved_skills,
            )
        run = self.repo.create_run(
            objective=objective,
            mode=mode.value,
            steps=planned_steps,
            parent_run_id=parent_run_id,
            delegation=delegation,
        )
        await self._update_run_capability_state(run["id"], capability_state)
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
        kernel_context: dict[str, Any] | None = None,
        resolved_skills: list[SkillManifest] | None = None,
    ) -> list[StepDefinition]:
        resolved_skills = resolved_skills if resolved_skills is not None else self._resolved_skills_for_objective(objective)
        skill_context = serialize_skill_context(resolved_skills)
        external_tool_context = self._external_tool_context()
        proposed = await request_next_steps(
            self.adaptive_planner,
            objective=objective,
            mode=mode,
            existing_steps=existing_steps,
            completed_step=completed_step,
            result=result,
            skill_context=skill_context,
            kernel_context=kernel_context,
            external_tool_context=external_tool_context,
        )
        if completed_step is None or result is None:
            planned = apply_initial_plan_policy(proposed, mode=mode)
            if planned:
                return self._annotate_skill_context(planned, resolved_skills)
            fallback_reason = "policy_rejected" if proposed else "no_steps"
            return self._annotate_skill_context(
                annotate_planner_fallback(
                    annotate_planner_steps(
                        plan_steps_for_objective(
                            objective,
                            skill_context=skill_context,
                            external_tool_context=external_tool_context,
                        ),
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
                        external_tool_context=external_tool_context,
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
                await self._finalize_run_completion(run_id)
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
            if self._clear_run_kernel_focus(run_id):
                await self._update_run_kernel_state(run_id, phase="running")
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
            setup_failure_reason = self._setup_verification_failure_reason(
                step=step,
                result=result,
                verification_result=verification_result,
            )
            if setup_failure_reason:
                self.repo.merge_step_metadata(
                    step_id,
                    {
                        **result.metadata,
                        "verification_result": "failed",
                        "setup_verification_result": "failed",
                    },
                )
                self.repo.mark_step_status(
                    step_id,
                    StepStatus.FAILED.value,
                    error_text=setup_failure_reason,
                )
                await self._update_capability_setup_state(
                    run_id,
                    status="failed",
                    reason=setup_failure_reason,
                    step_id=step_id,
                )
                await self._publish(
                    run_id,
                    "step.failed",
                    self._step_event_payload(step, step_id=step_id, error=setup_failure_reason),
                )
                return None
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
                    **(
                        {"setup_verification_result": "passed"}
                        if self._is_skill_setup_step(step)
                        else {}
                    ),
                },
            )
            if self._is_skill_setup_step(step):
                await self._refresh_capability_setup_state(run_id)
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
            if self._is_skill_setup_step(step):
                await self._update_capability_setup_state(
                    run_id,
                    status="failed",
                    reason=str(exc),
                    step_id=step_id,
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
        kernel_focus = self._kernel_focus_for_result(completed_action, result)
        if kernel_focus is not None:
            await self._set_run_kernel_focus(
                run["id"],
                tactic=kernel_focus["tactic"],
                reason=kernel_focus["tactic_reason"],
                source_step_id=str(completed_step.get("id", "")),
            )
            run = self.repo.get_run(run["id"]) or run
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
            kernel_context=self._current_kernel_context(run),
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

    async def _finalize_run_completion(self, run_id: str) -> None:
        run = self.repo.get_run(run_id)
        if run is None:
            return
        verification = self._evaluate_run_completion(run)
        if verification.result in {"blocked", "provisional"}:
            if await self._attempt_completion_recovery(run, verification):
                return
        if verification.result == "blocked":
            self.repo.merge_run_metadata(run_id, {"run_verification": verification.model_dump()})
            await self._publish(run_id, "run.verification", verification.model_dump())
            self.repo.mark_run_status(run_id, RunStatus.FAILED.value)
            await self._update_run_kernel_state(
                run_id,
                phase="failed",
                failure_reason=verification.reason,
            )
            message = verification.reason or "Run blocked by completion verification."
            await self.interaction.deliver_status(run_id, RunStatus.FAILED.value, message)
            await self._publish(
                run_id,
                "run.failed",
                {
                    "reason": message,
                    "verification_result": verification.result,
                },
            )
            return

        self.repo.merge_run_metadata(run_id, {"run_verification": verification.model_dump()})
        await self._publish(run_id, "run.verification", verification.model_dump())
        self.repo.mark_run_status(run_id, RunStatus.COMPLETED.value)
        await self._update_run_kernel_state(run_id, phase="completed")
        message = (
            "Run completed with verified completion signals."
            if verification.result == "verified"
            else "Run completed provisionally; stronger completion verification is not yet available."
        )
        await self.interaction.deliver_status(run_id, RunStatus.COMPLETED.value, message)
        await self._publish(
            run_id,
            "run.completed",
            {"verification_result": verification.result},
        )

    def _evaluate_run_completion(self, run: dict[str, Any]) -> RunVerificationRecord:
        run_id = str(run.get("id", "")).strip()
        strategy = self._strategy_for_objective(str(run.get("objective", "")))
        citations = self.repo.list_citations(run_id) if run_id else []
        artifacts = self.repo.list_artifacts(run_id) if run_id else []
        return evaluate_run_completion(
            strategy=strategy,
            run=run,
            citations=citations,
            artifacts=artifacts,
        )

    async def _attempt_completion_recovery(
        self,
        run: dict[str, Any],
        verification: RunVerificationRecord,
    ) -> bool:
        if verification.result == "verified" or self.max_completion_recovery_attempts <= 0:
            return False
        metadata = run.get("metadata") if isinstance(run.get("metadata"), dict) else {}
        if run.get("parent_run_id") or run.get("delegation"):
            return False
        recovery_count = int(metadata.get("completion_recovery_attempt_count", 0) or 0)
        if recovery_count >= self.max_completion_recovery_attempts:
            return False
        recovery_steps = self._completion_recovery_steps(run, verification)
        if not recovery_steps:
            return False
        last_completed = self._latest_completed_step(run)
        if last_completed is None:
            return False
        resolved_skills = self._resolved_skills_for_objective(str(run.get("objective", "")))
        planned_steps = self._annotate_skill_context(
            recovery_steps,
            resolved_skills,
        )
        self.repo.insert_steps_after(str(last_completed.get("id", "")), planned_steps)
        self.repo.merge_run_metadata(
            str(run.get("id", "")),
            {"completion_recovery_attempt_count": recovery_count + 1},
        )
        await self._record_kernel_decision(
            str(run.get("id", "")),
            last_completed,
            "continue",
            reason="completion_recovery",
        )
        await self._publish(
            str(run.get("id", "")),
            "run.replanned",
            {
                "after_step_id": str(last_completed.get("id", "")),
                "inserted_steps": [step.model_dump() for step in planned_steps],
                "recovery": True,
                "verification_reason": verification.reason,
            },
        )
        self.repo.mark_run_status(str(run.get("id", "")), RunStatus.RUNNING.value)
        await self._update_run_kernel_state(str(run.get("id", "")), phase="running")
        await self._execute_until_gate(str(run.get("id", "")))
        return True

    async def _resolve_capability_context(
        self,
        objective: str,
        *,
        allow_acquire: bool,
    ) -> tuple[list[SkillManifest], dict[str, Any]]:
        resolved_skills = self._resolved_skills_for_objective(objective)
        acquisition: dict[str, Any] = {}
        source = "capability_resolver"
        if (
            self._should_attempt_skill_acquisition(resolved_skills)
            and allow_acquire
            and self.auto_acquire_skills
            and self.skill_acquirer is not None
            and self.skill_registry is not None
            and self.capability_resolver is not None
        ):
            acquisition = await self._acquire_skill_for_objective(objective)
            if acquisition.get("success") is True:
                self.skill_registry.refresh()
                resolved_skills = self.capability_resolver.resolve(objective)
                if resolved_skills:
                    source = "synthesis_acquisition"
        return resolved_skills, self._capability_state(
            resolved_skills,
            source=source,
            acquisition=acquisition,
        )

    @staticmethod
    def _should_attempt_skill_acquisition(resolved_skills: list[SkillManifest]) -> bool:
        if not resolved_skills:
            return True
        for skill in resolved_skills:
            trust_level = skill.trust_level.strip().lower()
            source_type = skill.source_type.strip().lower()
            lifecycle_stage = skill.lifecycle_stage.strip().lower()
            if trust_level in {"trusted", "probation"}:
                return False
            if source_type in {"canonical", "installed"}:
                return False
            if lifecycle_stage in {"stable", "challenger"}:
                return False
        return True

    async def _acquire_skill_for_objective(self, objective: str) -> dict[str, Any]:
        if self.skill_acquirer is None:
            return {}
        try:
            return await self.skill_acquirer.acquire_skill(
                intent=objective,
                requirements=(
                    "Install or retrieve a reusable capability package that helps "
                    "Agent Nexus complete this objective end-to-end."
                ),
            )
        except Exception:
            log.exception("Automatic skill acquisition failed for objective: %s", objective)
            return {}

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
        kernel_focus = (
            metadata.get("kernel_focus")
            if isinstance(metadata.get("kernel_focus"), dict)
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
                kernel_focus=kernel_focus,
            )
        )
        normalized_failure_reason = failure_reason.strip()
        if normalized_failure_reason:
            kernel_state["failure_reason"] = normalized_failure_reason
        self.repo.merge_run_metadata(run_id, {"kernel_state": kernel_state})
        await self._publish(run_id, "run.kernel", kernel_state)

    @staticmethod
    def _current_kernel_context(run: dict[str, Any]) -> dict[str, Any]:
        metadata = run.get("metadata")
        if not isinstance(metadata, dict):
            return {}
        kernel_state = metadata.get("kernel_state")
        return kernel_state if isinstance(kernel_state, dict) else {}

    async def _set_run_kernel_focus(
        self,
        run_id: str,
        *,
        tactic: str,
        reason: str,
        source_step_id: str = "",
    ) -> None:
        run = self.repo.get_run(run_id)
        if run is None:
            return
        focus = {
            "tactic": tactic.strip(),
            "tactic_reason": reason.strip(),
        }
        if source_step_id.strip():
            focus["source_step_id"] = source_step_id.strip()
        self.repo.merge_run_metadata(run_id, {"kernel_focus": focus})
        await self._update_run_kernel_state(
            run_id,
            phase=self._phase_for_run_status(str(run.get("status", ""))),
        )
        refreshed_run = self.repo.get_run(run_id)
        if refreshed_run is not None:
            refreshed_metadata = (
                refreshed_run.get("metadata")
                if isinstance(refreshed_run.get("metadata"), dict)
                else {}
            )
            kernel_state = (
                refreshed_metadata.get("kernel_state")
                if isinstance(refreshed_metadata.get("kernel_state"), dict)
                else {}
            )
            if kernel_state:
                self._append_kernel_event_history(run_id, kernel_state)

    def _clear_run_kernel_focus(self, run_id: str) -> bool:
        run = self.repo.get_run(run_id)
        if run is None:
            return False
        metadata = run.get("metadata")
        if not isinstance(metadata, dict):
            return False
        focus = metadata.get("kernel_focus")
        if not isinstance(focus, dict) or not focus:
            return False
        self.repo.merge_run_metadata(run_id, {"kernel_focus": {}})
        return True

    def _append_kernel_event_history(self, run_id: str, kernel_state: dict[str, Any]) -> None:
        run = self.repo.get_run(run_id)
        if run is None:
            return
        metadata = run.get("metadata")
        if not isinstance(metadata, dict):
            return
        raw_history = metadata.get("kernel_event_history")
        history = list(raw_history) if isinstance(raw_history, list) else []
        history.append(
            {
                **kernel_state,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.repo.merge_run_metadata(run_id, {"kernel_event_history": history[-24:]})

    async def _update_run_capability_state(
        self,
        run_id: str,
        capability_state: dict[str, Any],
    ) -> None:
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
            verification_signals = metadata.get("verification_signals")
            required_artifact_kinds = metadata.get("required_artifact_kinds")
            external_tools = metadata.get("external_tools")
            verification_result = str(metadata.get("verification_result", "")).strip()
            kernel_decision = str(metadata.get("kernel_decision", "")).strip()
            kernel_decision_reason = str(metadata.get("kernel_decision_reason", "")).strip()
            retryable = metadata.get("retryable")
            retry_count = metadata.get("retry_count")
            if skill_source:
                payload["skill_source"] = skill_source
            if isinstance(skill_names, list) and skill_names:
                payload["skill_names"] = [str(name) for name in skill_names]
            if isinstance(verification_signals, list) and verification_signals:
                payload["verification_signals"] = [str(item) for item in verification_signals]
            if isinstance(required_artifact_kinds, list) and required_artifact_kinds:
                payload["required_artifact_kinds"] = [str(item) for item in required_artifact_kinds]
            if isinstance(external_tools, list) and external_tools:
                payload["external_tools"] = [str(item) for item in external_tools]
            if verification_result:
                payload["verification_result"] = verification_result
            if kernel_decision:
                payload["kernel_decision"] = kernel_decision
            if kernel_decision_reason:
                payload["kernel_decision_reason"] = kernel_decision_reason
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

    @staticmethod
    def _latest_completed_step(run: dict[str, Any]) -> dict[str, Any] | None:
        steps = run.get("steps")
        if not isinstance(steps, list):
            return None
        for step in reversed(steps):
            if not isinstance(step, dict):
                continue
            if str(step.get("status", "")).strip().lower() != StepStatus.COMPLETED.value:
                continue
            return step
        return None

    def _completion_recovery_steps(
        self,
        run: dict[str, Any],
        verification: RunVerificationRecord,
    ) -> list[StepDefinition]:
        objective = str(run.get("objective", "")).strip()
        signals = verification.signals if isinstance(verification.signals, dict) else {}
        last_completed = self._latest_completed_step(run)
        sequence_step = self._completion_recovery_step_for_tool_sequence(
            run=run,
            completed_step=last_completed,
        )
        if sequence_step is not None:
            return annotate_planner_steps(
                [sequence_step],
                planner_source="kernel",
                planner_phase="completion_recovery",
            )
        completed_actions = {
            str(item).strip().lower()
            for item in (signals.get("completed_actions") or [])
            if str(item).strip()
        }
        if (
            self._looks_like_memory_resume_objective(objective)
            and not self._run_used_memory_tool(run)
        ):
            memory_tool = self._find_memory_tool_name()
            if memory_tool:
                return annotate_planner_steps(
                    [
                        StepDefinition(
                            action_type="external_tool",
                            instruction=json.dumps(
                                {
                                    "tool_name": memory_tool,
                                    "arguments": {
                                        "query": objective,
                                        "scope": "task",
                                        "reason": verification.reason,
                                        "strategy": verification.strategy,
                                    },
                                }
                            ),
                        )
                    ],
                    planner_source="kernel",
                    planner_phase="completion_recovery",
                )
        successful_execute_code_count = int(signals.get("successful_execute_code_count", 0) or 0)
        mutating_code_action_count = int(signals.get("mutating_code_action_count", 0) or 0)
        if (
            verification.strategy == "coding"
            and mutating_code_action_count <= 0
            and successful_execute_code_count <= 0
            and not self._run_used_repo_tool(run)
        ):
            repo_tool = self._find_repo_tool_name()
            if repo_tool:
                return annotate_planner_steps(
                    [
                        StepDefinition(
                            action_type="external_tool",
                            instruction=json.dumps(
                                {
                                    "tool_name": repo_tool,
                                    "arguments": {
                                        "objective": objective,
                                        "scope": "repo",
                                        "reason": verification.reason,
                                        "strategy": verification.strategy,
                                    },
                                }
                            ),
                        )
                    ],
                    planner_source="kernel",
                    planner_phase="completion_recovery",
                )
        citation_count = int(signals.get("citation_count", 0) or 0)
        terminal_artifact_actions = {"export", "generate_report", "generate_chart", "generate_image"}
        if (
            verification.strategy == "research"
            and citation_count > 0
            and not (completed_actions & terminal_artifact_actions)
            and "generate_report" not in completed_actions
        ):
            return annotate_planner_steps(
                [
                    StepDefinition(
                        action_type="generate_report",
                        instruction=f"Generate a grounded research report for: {objective}",
                    )
                ],
                planner_source="kernel",
                planner_phase="completion_recovery",
            )
        if (
            verification.strategy == "coding"
            and mutating_code_action_count > 0
            and successful_execute_code_count <= 0
            and "execute_code" not in completed_actions
        ):
            return annotate_planner_steps(
                [
                    StepDefinition(
                        action_type="execute_code",
                        instruction=(
                            "Run the most relevant verification or test command for: "
                            f"{objective}"
                        ),
                    )
                ],
                planner_source="kernel",
                planner_phase="completion_recovery",
            )
        return []

    def _completion_recovery_step_for_tool_sequence(
        self,
        *,
        run: dict[str, Any],
        completed_step: dict[str, Any] | None,
    ) -> StepDefinition | None:
        if completed_step is None:
            return None
        metadata = (
            completed_step.get("metadata")
            if isinstance(completed_step.get("metadata"), dict)
            else {}
        )
        raw_remaining = metadata.get("tool_follow_up_sequence_remaining")
        if not isinstance(raw_remaining, list):
            return None
        remaining = [str(item).strip().lower() for item in raw_remaining if str(item).strip()]
        if not remaining:
            return None
        existing_steps = run.get("steps") if isinstance(run.get("steps"), list) else []
        step_result = self._step_result_from_record(completed_step)
        return continue_tool_follow_up_sequence(
            completed_step=completed_step,
            objective=str(run.get("objective", "")),
            result=step_result,
            existing_steps=existing_steps,
        )

    @staticmethod
    def _step_result_from_record(step: dict[str, Any]) -> StepExecutionResult:
        metadata = step.get("metadata")
        return StepExecutionResult(
            output_text=str(step.get("output_text", "") or ""),
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
        )

    @staticmethod
    def _looks_like_memory_resume_objective(objective: str) -> bool:
        lowered = objective.lower()
        return any(
            hint in lowered
            for hint in (
                "resume",
                "prior work",
                "previous work",
                "history",
                "remember",
                "memory",
                "context from before",
            )
        )

    def _find_memory_tool_name(self) -> str:
        for tool in self._external_tool_context():
            if not isinstance(tool, dict):
                continue
            name = str(tool.get("name", "")).strip()
            lowered = name.lower()
            raw_tags = tool.get("tags")
            tags = (
                {
                    str(tag).strip().lower()
                    for tag in raw_tags
                    if str(tag).strip()
                }
                if isinstance(raw_tags, list)
                else set()
            )
            if lowered.startswith("mnemos.") or "memory" in tags or "retrieval" in tags:
                return name
        return ""

    def _find_repo_tool_name(self) -> str:
        for tool in self._external_tool_context():
            if not isinstance(tool, dict):
                continue
            name = str(tool.get("name", "")).strip()
            lowered = name.lower()
            raw_tags = tool.get("tags")
            tags = (
                {
                    str(tag).strip().lower()
                    for tag in raw_tags
                    if str(tag).strip()
                }
                if isinstance(raw_tags, list)
                else set()
            )
            if lowered.startswith("cartographer.") or "repo" in tags or "context" in tags:
                return name
        return ""

    @staticmethod
    def _run_used_memory_tool(run: dict[str, Any]) -> bool:
        steps = run.get("steps")
        if not isinstance(steps, list):
            return False
        for step in steps:
            if not isinstance(step, dict):
                continue
            metadata = step.get("metadata")
            if not isinstance(metadata, dict):
                continue
            external_tool = metadata.get("external_tool")
            if not isinstance(external_tool, dict):
                continue
            name = str(external_tool.get("name", "")).strip().lower()
            raw_tags = external_tool.get("tags")
            tags = (
                {
                    str(tag).strip().lower()
                    for tag in raw_tags
                    if str(tag).strip()
                }
                if isinstance(raw_tags, list)
                else set()
            )
            if name.startswith("mnemos.") or "memory" in tags or "retrieval" in tags:
                return True
        return False

    @staticmethod
    def _run_used_repo_tool(run: dict[str, Any]) -> bool:
        steps = run.get("steps")
        if not isinstance(steps, list):
            return False
        for step in steps:
            if not isinstance(step, dict):
                continue
            metadata = step.get("metadata")
            if not isinstance(metadata, dict):
                continue
            external_tool = metadata.get("external_tool")
            if not isinstance(external_tool, dict):
                continue
            name = str(external_tool.get("name", "")).strip().lower()
            raw_tags = external_tool.get("tags")
            tags = (
                {
                    str(tag).strip().lower()
                    for tag in raw_tags
                    if str(tag).strip()
                }
                if isinstance(raw_tags, list)
                else set()
            )
            if name.startswith("cartographer.") or "repo" in tags or "context" in tags:
                return True
        return False

    def _resolved_skills_for_objective(self, objective: str) -> list[SkillManifest]:
        if self.capability_resolver is None:
            return []
        return self.capability_resolver.resolve(objective)

    @staticmethod
    def _capability_state(
        resolved_skills: list[SkillManifest],
        *,
        source: str = "capability_resolver",
        acquisition: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not resolved_skills:
            return {}
        verification_signals = RunEngine._merged_skill_values(
            resolved_skills,
            attribute="verification_signals",
        )
        required_artifact_kinds = RunEngine._merged_skill_values(
            resolved_skills,
            attribute="required_artifact_kinds",
        )
        external_tools = RunEngine._merged_skill_values(
            resolved_skills,
            attribute="external_tools",
        )
        preferred_planning_roles = RunEngine._merged_skill_values(
            resolved_skills,
            attribute="preferred_planning_roles",
        )
        setup_steps = RunEngine._merged_skill_setup_steps(resolved_skills)
        resolved_payload = [
            {
                "name": skill.name,
                "description": skill.description,
                "path": skill.path,
                "trust_level": skill.trust_level,
                "source_type": skill.source_type,
                "lifecycle_stage": skill.lifecycle_stage,
                "capability_family": skill.capability_family,
                "source_repo": skill.source_repo,
                "preferred_planning_roles": list(skill.preferred_planning_roles),
                "preferred_initial_actions": list(skill.preferred_initial_actions),
                "preferred_follow_up_actions": list(skill.preferred_follow_up_actions),
                "external_tools": list(skill.external_tools),
                "external_tool_arguments": (
                    {key: dict(value) for key, value in skill.external_tool_arguments.items()}
                    if skill.external_tool_arguments
                    else {}
                ),
                "external_tool_follow_up_actions": (
                    {
                        key: list(value)
                        for key, value in skill.external_tool_follow_up_actions.items()
                    }
                    if skill.external_tool_follow_up_actions
                    else {}
                ),
                "external_tool_follow_up_sequences": (
                    {
                        key: list(value)
                        for key, value in skill.external_tool_follow_up_sequences.items()
                    }
                    if skill.external_tool_follow_up_sequences
                    else {}
                ),
                "setup_steps": [
                    {
                        **step,
                        "metadata": dict(step.get("metadata", {}))
                        if isinstance(step.get("metadata"), dict)
                        else {},
                    }
                    for step in skill.setup_steps
                ],
                "verification_signals": list(skill.verification_signals),
                "required_artifact_kinds": list(skill.required_artifact_kinds),
            }
            for skill in resolved_skills
        ]
        capability_state: dict[str, Any] = {
            "skill_source": source,
            "skill_names": [skill.name for skill in resolved_skills],
            "resolved_skill_count": len(resolved_skills),
            "resolved_skills": resolved_payload,
        }
        if verification_signals:
            capability_state["verification_signals"] = verification_signals
        if required_artifact_kinds:
            capability_state["required_artifact_kinds"] = required_artifact_kinds
        if external_tools:
            capability_state["external_tools"] = external_tools
        if preferred_planning_roles:
            capability_state["preferred_planning_roles"] = preferred_planning_roles
        if setup_steps:
            capability_state["setup_steps"] = setup_steps
            capability_state["setup_status"] = "pending"
        if isinstance(acquisition, dict) and acquisition:
            capability_state["skill_acquisition"] = {
                "success": bool(acquisition.get("success")),
                "method": str(acquisition.get("method", "")).strip(),
                "primary_skill_name": str(
                    (acquisition.get("primary_skill") or {}).get("name", "")
                ).strip(),
                "activation_message": str(acquisition.get("activation_message", "")).strip(),
            }
        return capability_state

    @staticmethod
    def _merged_skill_values(
        resolved_skills: list[SkillManifest],
        *,
        attribute: str,
    ) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for skill in resolved_skills:
            raw_values = getattr(skill, attribute, ())
            for raw_value in raw_values:
                value = str(raw_value).strip().lower()
                if not value or value in seen:
                    continue
                seen.add(value)
                merged.append(value)
        return merged

    @staticmethod
    def _merged_skill_setup_steps(resolved_skills: list[SkillManifest]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for skill in resolved_skills:
            for step in skill.setup_steps:
                fingerprint = json.dumps(step, sort_keys=True, default=str)
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                merged.append(
                    {
                        **step,
                        "metadata": dict(step.get("metadata", {}))
                        if isinstance(step.get("metadata"), dict)
                        else {},
                    }
                )
        return merged

    def _external_tool_context(self) -> list[dict[str, Any]]:
        registry = self.external_tool_registry
        if registry is None or not hasattr(registry, "list_tools"):
            return []
        try:
            tools = registry.list_tools()
        except Exception:
            return []

        serialized: list[dict[str, Any]] = []
        for tool in tools:
            if hasattr(tool, "to_dict"):
                payload = tool.to_dict()
                if isinstance(payload, dict):
                    serialized.append(payload)
        return serialized

    async def _update_capability_setup_state(
        self,
        run_id: str,
        *,
        status: str,
        reason: str = "",
        step_id: str = "",
    ) -> None:
        run = self.repo.get_run(run_id)
        if run is None:
            return
        metadata = run.get("metadata") if isinstance(run.get("metadata"), dict) else {}
        capability_state = (
            dict(metadata.get("capability_state"))
            if isinstance(metadata.get("capability_state"), dict)
            else {}
        )
        if not capability_state:
            return
        capability_state["setup_status"] = status.strip().lower()
        if reason.strip():
            capability_state["setup_failure_reason"] = reason.strip()
        else:
            capability_state.pop("setup_failure_reason", None)
        if step_id.strip():
            capability_state["setup_status_step_id"] = step_id.strip()
        self.repo.merge_run_metadata(run_id, {"capability_state": capability_state})
        await self._publish(run_id, "run.capability", capability_state)

    async def _refresh_capability_setup_state(self, run_id: str) -> None:
        run = self.repo.get_run(run_id)
        if run is None:
            return
        steps = run.get("steps") if isinstance(run.get("steps"), list) else []
        setup_steps = [step for step in steps if self._is_skill_setup_step(step)]
        if not setup_steps:
            return
        if any(str(step.get("status", "")).strip().lower() == StepStatus.FAILED.value for step in setup_steps):
            failed_step = next(
                (
                    step
                    for step in reversed(setup_steps)
                    if str(step.get("status", "")).strip().lower() == StepStatus.FAILED.value
                ),
                None,
            )
            await self._update_capability_setup_state(
                run_id,
                status="failed",
                reason=str((failed_step or {}).get("error_text", "")).strip(),
                step_id=str((failed_step or {}).get("id", "")),
            )
            return
        if any(
            str(step.get("status", "")).strip().lower()
            in {StepStatus.PENDING.value, StepStatus.RUNNING.value, StepStatus.PENDING_APPROVAL.value}
            for step in setup_steps
        ):
            await self._update_capability_setup_state(run_id, status="pending")
            return
        await self._update_capability_setup_state(run_id, status="ready")

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
    def _is_skill_setup_step(step: dict[str, Any]) -> bool:
        metadata = step.get("metadata")
        return isinstance(metadata, dict) and metadata.get("skill_setup") is True

    def _setup_verification_failure_reason(
        self,
        *,
        step: dict[str, Any],
        result: StepExecutionResult,
        verification_result: str,
    ) -> str:
        metadata = step.get("metadata")
        if not isinstance(metadata, dict):
            return ""
        raw_expectations = metadata.get("setup_expectations")
        if isinstance(raw_expectations, str):
            expectations = [item.strip().lower() for item in raw_expectations.split(",") if item.strip()]
        elif isinstance(raw_expectations, list):
            expectations = [str(item).strip().lower() for item in raw_expectations if str(item).strip()]
        else:
            expectations = []
        if not expectations:
            return ""
        result_metadata = result.metadata if isinstance(result.metadata, dict) else {}
        for expectation in expectations:
            if expectation == "output" and str(result.output_text or "").strip():
                continue
            if expectation in {"citation", "citations"} and result.citations:
                continue
            if expectation in {"artifact", "artifacts"} and result.artifacts:
                continue
            if expectation == "tool_result" and isinstance(result_metadata.get("tool_result"), dict):
                continue
            if expectation == "page_context" and any(
                result_metadata.get(key)
                for key in ("current_url", "page_title", "page_excerpt", "page_affordances")
            ):
                continue
            if expectation in {"command_success", "verification_passed"} and verification_result == "passed":
                continue
            return f"Skill setup step did not satisfy required readiness signal `{expectation}`."
        return ""

    @staticmethod
    def _kernel_strategy_snapshot(
        *,
        objective: str,
        phase: str,
        steps: list[dict[str, Any]],
        capability_state: dict[str, Any],
        kernel_focus: dict[str, Any],
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
            raw_remaining = last_metadata.get("tool_follow_up_sequence_remaining")
            remaining_actions = (
                [str(item).strip().lower() for item in raw_remaining if str(item).strip()]
                if isinstance(raw_remaining, list)
                else []
            )
            decision = str(last_metadata.get("kernel_decision", "")).strip().lower()
            if remaining_actions:
                tactic = "sequence_follow_up"
                tactic_reason = (
                    "Capability-declared tool follow-up sequence still has pending actions: "
                    + ", ".join(remaining_actions)
                    + "."
                )
            else:
                tactic = RunEngine._tactic_after_decision(last_action, decision)
            if decision and not remaining_actions:
                tactic_reason = f"Last completed step decided to {decision.replace('_', ' ')}."
            elif last_action and not remaining_actions:
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
        focus_tactic = str(kernel_focus.get("tactic", "")).strip()
        if focus_tactic:
            tactic = focus_tactic
            tactic_reason = str(kernel_focus.get("tactic_reason", "")).strip() or tactic_reason
        snapshot: dict[str, Any] = {
            "strategy": strategy,
            "tactic": tactic,
        }
        if tactic_reason:
            snapshot["tactic_reason"] = tactic_reason
        focus_source_step_id = str(kernel_focus.get("source_step_id", "")).strip()
        if focus_source_step_id:
            snapshot["focus_source_step_id"] = focus_source_step_id
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
    def _phase_for_run_status(status: str) -> str:
        normalized = status.strip().lower()
        mapping = {
            RunStatus.RUNNING.value: "running",
            RunStatus.PENDING_APPROVAL.value: "awaiting_approval",
            RunStatus.PAUSED.value: "paused",
            RunStatus.COMPLETED.value: "completed",
            RunStatus.FAILED.value: "failed",
        }
        return mapping.get(normalized, "running")

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

    def _kernel_focus_for_result(
        self,
        completed_action: str,
        result: StepExecutionResult,
    ) -> dict[str, str] | None:
        verification = self._verification_result(result)
        if verification == "failed":
            return {
                "tactic": "diagnose",
                "tactic_reason": (
                    f"{completed_action or 'step'} verification failed; inspect the grounded evidence before continuing."
                ),
            }
        if completed_action == "extract" and not result.citations:
            return {
                "tactic": "recover",
                "tactic_reason": (
                    "Extraction returned no citations; gather more grounded evidence before continuing."
                ),
            }
        return None

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
    def _setup_steps_for_resolved_skills(
        *,
        objective: str,
        resolved_skills: list[SkillManifest],
    ) -> list[StepDefinition]:
        if not resolved_skills:
            return []
        rendered: list[StepDefinition] = []
        seen: set[str] = set()
        for skill in resolved_skills:
            for raw_step in skill.setup_steps:
                action_type = str(raw_step.get("action_type", "")).strip().lower()
                if not action_type:
                    continue
                instruction_value = RunEngine._render_setup_value(
                    raw_step.get("instruction", ""),
                    objective=objective,
                    skill=skill,
                )
                instruction = (
                    instruction_value
                    if isinstance(instruction_value, str)
                    else json.dumps(instruction_value)
                )
                metadata = raw_step.get("metadata")
                rendered_metadata = RunEngine._render_setup_value(
                    metadata if isinstance(metadata, dict) else {},
                    objective=objective,
                    skill=skill,
                )
                step_metadata = dict(rendered_metadata) if isinstance(rendered_metadata, dict) else {}
                step_metadata.setdefault("planner_source", "skill")
                step_metadata.setdefault("planner_phase", "setup")
                step_metadata.setdefault("skill_setup", True)
                step_metadata.setdefault("skill_setup_source", skill.name)
                fingerprint = json.dumps(
                    {
                        "action_type": action_type,
                        "instruction": instruction,
                        "metadata": step_metadata,
                    },
                    sort_keys=True,
                )
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                rendered.append(
                    StepDefinition(
                        action_type=action_type,
                        instruction=instruction,
                        metadata=step_metadata,
                    )
                )
        return rendered

    @staticmethod
    def _render_setup_value(value: Any, *, objective: str, skill: SkillManifest) -> Any:
        replacements = {
            "objective": objective,
            "skill_name": skill.name,
            "skill_path": skill.path,
        }
        if isinstance(value, str):
            rendered = value
            for key, replacement in replacements.items():
                rendered = rendered.replace(f"{{{key}}}", replacement)
            return rendered
        if isinstance(value, list):
            return [
                RunEngine._render_setup_value(item, objective=objective, skill=skill)
                for item in value
            ]
        if isinstance(value, dict):
            return {
                str(key): RunEngine._render_setup_value(item, objective=objective, skill=skill)
                for key, item in value.items()
            }
        return value

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
