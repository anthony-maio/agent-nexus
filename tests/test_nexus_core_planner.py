from __future__ import annotations

import json
from typing import Any

import pytest

from nexus_core.models import CitationRecord, RunMode, StepDefinition, StepExecutionResult
from nexus_core.planner import (
    CompositeAdaptivePlanner,
    RuleAdaptivePlanner,
    plan_steps_for_objective,
    plan_follow_up_steps,
    request_next_steps,
)


def _step(
    step_index: int,
    action_type: str,
    *,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    step = {
        "step_index": step_index,
        "action_type": action_type,
        "instruction": f"{action_type} step",
        "status": "completed",
    }
    if metadata:
        step["metadata"] = metadata
    return step


def test_plan_follow_up_steps_workflow_inspect_returns_only_type_step() -> None:
    objective = "Fill out the signup form at https://example.com/register"

    steps = plan_follow_up_steps(
        objective=objective,
        completed_step=_step(1, "inspect"),
        result=StepExecutionResult(
            output_text="found the form fields",
            metadata={
                "page_affordances": {
                    "input_fields": [
                        {"tag": "input", "type": "email", "name": "email"},
                        {"tag": "textarea", "name": "message"},
                    ],
                    "buttons": [{"text": "Send message", "type": "submit"}],
                }
            },
        ),
        existing_steps=[_step(0, "navigate"), _step(1, "inspect")],
    )

    assert [step.action_type for step in steps] == ["type"]
    assert "email" in steps[0].instruction
    assert "message" in steps[0].instruction


def test_plan_steps_for_objective_uses_skill_preferred_initial_action() -> None:
    steps = plan_steps_for_objective(
        "Generate a chart from local sales data and summarize it",
        skill_context=[
            {
                "name": "chart-maker",
                "description": "Generate charts from local tabular data.",
                "path": "skills/chart-maker/SKILL.md",
                "preferred_initial_actions": ["list_files", "read_file"],
            }
        ],
    )

    assert [step.action_type for step in steps] == ["list_files"]


def test_plan_steps_for_objective_uses_skill_declared_external_tool() -> None:
    steps = plan_steps_for_objective(
        "Retrieve payment retry memory for the current objective",
        skill_context=[
            {
                "name": "memory-helper",
                "description": "Retrieve scoped memory and repo maps from external tools.",
                "path": "skills/memory-helper/SKILL.md",
                "external_tools": ["mnemos.retrieve"],
            }
        ],
        external_tool_context=[
            {
                "name": "mnemos.retrieve",
                "description": "Retrieve scoped memory from Mnemos.",
                "source": "mcp://mnemos",
            }
        ],
    )

    assert [step.action_type for step in steps] == ["external_tool"]
    assert '"tool_name": "mnemos.retrieve"' in steps[0].instruction


def test_plan_steps_for_objective_applies_skill_external_tool_arguments() -> None:
    steps = plan_steps_for_objective(
        "Retrieve payment retry memory for the current objective",
        skill_context=[
            {
                "name": "memory-helper",
                "description": "Retrieve scoped memory and repo maps from external tools.",
                "path": "skills/memory-helper/SKILL.md",
                "external_tools": ["mnemos.retrieve"],
                "external_tool_arguments": {
                    "mnemos.retrieve": {
                        "query": "{objective}",
                        "scope": "task",
                    }
                },
            }
        ],
        external_tool_context=[
            {
                "name": "mnemos.retrieve",
                "description": "Retrieve scoped memory from Mnemos.",
                "source": "mcp://mnemos",
            }
        ],
    )

    payload = json.loads(steps[0].instruction)
    assert payload["tool_name"] == "mnemos.retrieve"
    assert payload["arguments"]["query"] == "Retrieve payment retry memory for the current objective"
    assert payload["arguments"]["scope"] == "task"


def test_plan_follow_up_steps_uses_skill_preferred_follow_up_action() -> None:
    steps = plan_follow_up_steps(
        objective="Review payments API findings and package the result",
        completed_step=_step(2, "extract"),
        result=StepExecutionResult(
            output_text="grounded findings gathered",
            citations=[CitationRecord(url="https://example.com", title="Example", snippet="ok")],
        ),
        existing_steps=[_step(0, "search_web"), _step(1, "fetch_url"), _step(2, "extract")],
        skill_context=[
            {
                "name": "report-maker",
                "description": "Produce polished reports from gathered findings.",
                "path": "skills/report-maker/SKILL.md",
                "preferred_follow_up_actions": ["generate_report", "export"],
            }
        ],
    )

    assert [step.action_type for step in steps] == ["generate_report"]
    assert '"path": "reports/' in steps[0].instruction


def test_plan_follow_up_steps_external_tool_advances_to_extract() -> None:
    steps = plan_follow_up_steps(
        objective="Retrieve payment retry memory for the current objective",
        completed_step=_step(1, "external_tool"),
        result=StepExecutionResult(
            output_text="Retrieved scoped memory entries.",
            citations=[CitationRecord(url="mcp://mnemos", title="mnemos.retrieve", snippet="ok")],
        ),
        existing_steps=[_step(0, "external_tool"), _step(1, "external_tool")],
    )

    assert [step.action_type for step in steps] == ["extract"]
    assert "external tool result" in steps[0].instruction.lower()


def test_plan_follow_up_steps_uses_kernel_strategy_for_ambiguous_workflow_context() -> None:
    steps = plan_follow_up_steps(
        objective="Continue the operator task",
        completed_step=_step(1, "fetch_url"),
        result=StepExecutionResult(
            output_text="page loaded",
            citations=[CitationRecord(url="https://example.com", title="Example", snippet="ok")],
        ),
        existing_steps=[_step(0, "search_web"), _step(1, "fetch_url")],
        kernel_context={"strategy": "workflow", "tactic": "observe"},
    )

    assert [step.action_type for step in steps] == ["inspect"]


def test_plan_follow_up_steps_workflow_inspect_without_inputs_falls_back_to_extract() -> None:
    objective = "Fill out the signup form at https://example.com/register"

    steps = plan_follow_up_steps(
        objective=objective,
        completed_step=_step(1, "inspect"),
        result=StepExecutionResult(
            output_text="found only static content",
            metadata={
                "page_affordances": {
                    "input_fields": [],
                    "buttons": [{"text": "Learn more", "type": "button"}],
                }
            },
        ),
        existing_steps=[_step(0, "navigate"), _step(1, "inspect")],
    )

    assert [step.action_type for step in steps] == ["extract"]


def test_plan_follow_up_steps_workflow_advances_one_action_at_a_time() -> None:
    objective = "Fill out the signup form at https://example.com/register"
    existing_steps = [_step(0, "navigate"), _step(1, "inspect"), _step(2, "type")]

    click_steps = plan_follow_up_steps(
        objective=objective,
        completed_step=_step(2, "type"),
        result=StepExecutionResult(
            output_text="typed required values",
            metadata={
                "page_affordances": {
                    "buttons": [{"text": "Continue", "type": "submit"}]
                }
            },
        ),
        existing_steps=existing_steps,
    )
    wait_steps = plan_follow_up_steps(
        objective=objective,
        completed_step=_step(3, "click"),
        result=StepExecutionResult(output_text="clicked continue"),
        existing_steps=existing_steps + [_step(3, "click")],
    )
    extract_steps = plan_follow_up_steps(
        objective=objective,
        completed_step=_step(4, "wait"),
        result=StepExecutionResult(output_text="page settled"),
        existing_steps=existing_steps + [_step(3, "click"), _step(4, "wait")],
    )

    assert [step.action_type for step in click_steps] == ["click"]
    assert [step.action_type for step in wait_steps] == ["wait"]
    assert [step.action_type for step in extract_steps] == ["extract"]
    assert "Continue" in click_steps[0].instruction


def test_plan_follow_up_steps_extract_without_citations_scrolls_before_retrying() -> None:
    steps = plan_follow_up_steps(
        objective="Research grounded browser runtime docs",
        completed_step=_step(2, "extract"),
        result=StepExecutionResult(output_text="no citations yet"),
        existing_steps=[_step(0, "search_web"), _step(1, "fetch_url"), _step(2, "extract")],
    )

    assert [step.action_type for step in steps] == ["scroll"]


def test_plan_follow_up_steps_workflow_extract_grounds_submit_control() -> None:
    objective = "Fill out the signup form at https://example.com/register"
    existing_steps = [
        _step(0, "navigate"),
        _step(1, "inspect"),
        _step(2, "type"),
        _step(3, "click"),
        _step(4, "wait"),
        _step(5, "extract"),
    ]

    steps = plan_follow_up_steps(
        objective=objective,
        completed_step=_step(5, "extract"),
        result=StepExecutionResult(
            output_text="workflow ready for submission",
            citations=[
                CitationRecord(url="https://example.com/register", title="Signup form", snippet="Ready")
            ],
            metadata={
                "page_affordances": {
                    "buttons": [{"text": "Continue", "type": "submit"}]
                }
            },
        ),
        existing_steps=existing_steps,
    )

    assert [step.action_type for step in steps] == ["submit"]
    assert "Continue" in steps[0].instruction


def test_plan_follow_up_steps_scroll_retries_extraction_after_context_gathering() -> None:
    steps = plan_follow_up_steps(
        objective="Research grounded browser runtime docs",
        completed_step=_step(3, "scroll"),
        result=StepExecutionResult(
            output_text="additional context gathered",
            citations=[
                CitationRecord(
                    url="https://docs.example.org/runtime",
                    title="Runtime docs",
                    snippet="Grounded docs",
                )
            ],
        ),
        existing_steps=[
            _step(0, "search_web"),
            _step(1, "fetch_url"),
            _step(2, "extract"),
            _step(3, "scroll"),
        ],
    )

    assert [step.action_type for step in steps] == ["extract"]


def test_plan_follow_up_steps_extract_for_report_objective_prefers_generate_report() -> None:
    steps = plan_follow_up_steps(
        objective="Prepare a report on grounded browser runtime docs",
        completed_step=_step(2, "extract"),
        result=StepExecutionResult(
            output_text="grounded evidence gathered",
            citations=[
                CitationRecord(
                    url="https://docs.example.org/runtime",
                    title="Runtime docs",
                    snippet="Grounded evidence",
                )
            ],
            metadata={"current_url": "https://docs.example.org/runtime"},
        ),
        existing_steps=[_step(0, "search_web"), _step(1, "fetch_url"), _step(2, "extract")],
    )

    assert [step.action_type for step in steps] == ["generate_report"]
    assert "reports/" in steps[0].instruction


def test_plan_follow_up_steps_extract_for_chart_objective_prefers_generate_chart() -> None:
    steps = plan_follow_up_steps(
        objective="Research payment latency and generate a chart of the findings",
        completed_step=_step(2, "extract"),
        result=StepExecutionResult(
            output_text="grounded evidence gathered",
            citations=[
                CitationRecord(
                    url="https://docs.example.org/metrics",
                    title="Metrics",
                    snippet="Latency data",
                )
            ],
            metadata={
                "chart_data": [
                    {"provider": "A", "latency_ms": 110},
                    {"provider": "B", "latency_ms": 85},
                ],
                "chart_title": "Latency by provider",
            },
        ),
        existing_steps=[_step(0, "search_web"), _step(1, "fetch_url"), _step(2, "extract")],
    )

    assert [step.action_type for step in steps] == ["generate_chart"]
    assert "Latency by provider" in steps[0].instruction


def test_plan_follow_up_steps_extract_for_image_objective_prefers_generate_image() -> None:
    steps = plan_follow_up_steps(
        objective="Research stablecoin settlement and generate a hero image for the findings",
        completed_step=_step(2, "extract"),
        result=StepExecutionResult(
            output_text="grounded evidence gathered",
            citations=[
                CitationRecord(
                    url="https://docs.example.org/settlement",
                    title="Settlement docs",
                    snippet="Settlement workflow",
                )
            ],
            metadata={"current_url": "https://docs.example.org/settlement"},
        ),
        existing_steps=[_step(0, "search_web"), _step(1, "fetch_url"), _step(2, "extract")],
    )

    assert [step.action_type for step in steps] == ["generate_image"]
    assert "images/" in steps[0].instruction


@pytest.mark.asyncio
async def test_rule_adaptive_planner_initial_bootstrap_returns_single_seed_step() -> None:
    planner = RuleAdaptivePlanner()

    research_steps = await planner.plan_initial_steps(
        objective="Research grounded browser runtime docs",
        mode=RunMode.SUPERVISED,
    )
    workflow_steps = await planner.plan_initial_steps(
        objective="Fill out the signup form at https://example.com/register",
        mode=RunMode.SUPERVISED,
    )

    assert [step.action_type for step in research_steps] == ["search_web"]
    assert research_steps[0].metadata["planner_phase"] == "initial"
    assert [step.action_type for step in workflow_steps] == ["navigate"]
    assert workflow_steps[0].metadata["planner_phase"] == "initial"


def test_plan_steps_for_code_objective_bootstraps_workspace_discovery() -> None:
    steps = plan_steps_for_objective(
        "Implement the payment retry backoff fix in the repo and update tests"
    )

    assert [step.action_type for step in steps] == ["list_files"]
    assert steps[0].instruction == '{"path": "."}'


def test_plan_steps_for_api_objective_bootstraps_call_api() -> None:
    steps = plan_steps_for_objective(
        "Call the Circle sandbox API endpoint at https://api.example.org/v1/payments and inspect the JSON response"
    )

    assert [step.action_type for step in steps] == ["call_api"]
    assert "api.example.org/v1/payments" in steps[0].instruction


def test_plan_follow_up_steps_call_api_advances_to_extract() -> None:
    steps = plan_follow_up_steps(
        objective="Call the Circle sandbox API endpoint at https://api.example.org/v1/payments and inspect the JSON response",
        completed_step=_step(0, "call_api"),
        result=StepExecutionResult(
            output_text='{"id":"payment_123","status":"confirmed"}',
            citations=[
                CitationRecord(
                    url="https://api.example.org/v1/payments",
                    title="API GET 200",
                    snippet="payment_123 confirmed",
                )
            ],
            metadata={"response_kind": "json", "status_code": 200},
        ),
        existing_steps=[_step(0, "call_api")],
    )

    assert [step.action_type for step in steps] == ["extract"]


def test_plan_follow_up_steps_code_list_files_prefers_source_file_over_tests() -> None:
    steps = plan_follow_up_steps(
        objective="Implement the payment retry backoff fix in the repo and update tests",
        completed_step=_step(0, "list_files"),
        result=StepExecutionResult(
            output_text="done:list_files",
            metadata={"files": ["tests/test_retry.py", "src/payments/retry.py"]},
        ),
        existing_steps=[_step(0, "list_files")],
    )

    assert [step.action_type for step in steps] == ["read_file"]
    assert steps[0].instruction == '{"path": "src/payments/retry.py"}'


def test_plan_follow_up_steps_code_read_file_prefers_execute_code() -> None:
    steps = plan_follow_up_steps(
        objective="Implement the payment retry backoff fix in the repo and update tests",
        completed_step=_step(1, "read_file"),
        result=StepExecutionResult(
            output_text="def retry_backoff():\n    return 1",
            metadata={"file_path": "src/payments/retry.py"},
        ),
        existing_steps=[_step(0, "list_files"), _step(1, "read_file")],
    )

    assert [step.action_type for step in steps] == ["execute_code"]
    assert steps[0].instruction == '{"command": ["python", "-m", "pytest", "-q"]}'


def test_plan_follow_up_steps_failed_code_execution_reads_diagnostic_file() -> None:
    steps = plan_follow_up_steps(
        objective="Implement the payment retry backoff fix in the repo and update tests",
        completed_step=_step(2, "execute_code"),
        result=StepExecutionResult(
            output_text="tests failed",
            metadata={
                "command_failed": True,
                "exit_code": 1,
                "stderr": "AssertionError in src/payments/retry.py:12",
            },
        ),
        existing_steps=[_step(0, "list_files"), _step(1, "read_file"), _step(2, "execute_code")],
    )

    assert [step.action_type for step in steps] == ["read_file"]
    assert steps[0].instruction == '{"path": "src/payments/retry.py"}'
    assert steps[0].metadata["code_follow_up"] == "failed_test_diagnostic"


def test_plan_follow_up_steps_diagnostic_read_file_summarizes_instead_of_rerunning_tests() -> None:
    steps = plan_follow_up_steps(
        objective="Implement the payment retry backoff fix in the repo and update tests",
        completed_step=_step(
            3,
            "read_file",
            metadata={"code_follow_up": "failed_test_diagnostic"},
        ),
        result=StepExecutionResult(
            output_text="def retry_backoff():\n    return base_delay * 2",
            metadata={"file_path": "src/payments/retry.py"},
        ),
        existing_steps=[
            _step(0, "list_files"),
            _step(1, "read_file"),
            _step(2, "execute_code"),
            _step(3, "read_file", metadata={"code_follow_up": "failed_test_diagnostic"}),
        ],
    )

    assert [step.action_type for step in steps] == ["extract"]


@pytest.mark.asyncio
async def test_request_next_steps_supports_unified_planner_contract() -> None:
    class UnifiedOnlyPlanner:
        async def plan_next_steps(
            self,
            objective: str,
            mode: RunMode,
            existing_steps: list[dict[str, Any]],
            completed_step: dict[str, Any] | None = None,
            result: StepExecutionResult | None = None,
            kernel_context: dict[str, Any] | None = None,
        ) -> list[StepDefinition]:
            _ = mode, existing_steps
            if completed_step is None or result is None:
                return [
                    StepDefinition(
                        action_type="search_web",
                        instruction=f"bootstrap {objective} {kernel_context['strategy']}",
                    )
                ]
            return [
                StepDefinition(
                    action_type="extract",
                    instruction=f"follow up {objective} {kernel_context['tactic']}",
                )
            ]

    planner = UnifiedOnlyPlanner()

    initial_steps = await request_next_steps(
        planner,
        objective="Research grounded browser runtime docs",
        mode=RunMode.SUPERVISED,
        existing_steps=[],
        kernel_context={"strategy": "research"},
    )
    follow_up_steps = await request_next_steps(
        planner,
        objective="Research grounded browser runtime docs",
        mode=RunMode.SUPERVISED,
        existing_steps=[_step(0, "search_web")],
        completed_step=_step(0, "search_web"),
        result=StepExecutionResult(output_text="search complete"),
        kernel_context={"tactic": "observe"},
    )

    assert [step.action_type for step in initial_steps] == ["search_web"]
    assert "research" in initial_steps[0].instruction
    assert [step.action_type for step in follow_up_steps] == ["extract"]
    assert "observe" in follow_up_steps[0].instruction


@pytest.mark.asyncio
async def test_composite_adaptive_planner_marks_fallback_steps_when_primary_returns_no_steps() -> None:
    class EmptyPlanner:
        async def plan_next_steps(
            self,
            objective: str,
            mode: RunMode,
            existing_steps: list[dict[str, Any]],
            completed_step: dict[str, Any] | None = None,
            result: StepExecutionResult | None = None,
        ) -> list[StepDefinition]:
            _ = objective, mode, existing_steps, completed_step, result
            return []

    planner = CompositeAdaptivePlanner([EmptyPlanner(), RuleAdaptivePlanner()])

    steps = await planner.plan_next_steps(
        objective="Research grounded browser runtime docs",
        mode=RunMode.SUPERVISED,
        existing_steps=[],
    )

    assert [step.action_type for step in steps] == ["search_web"]
    assert steps[0].metadata["planner_source"] == "rule"
    assert steps[0].metadata["planner_fallback_reason"] == "no_steps"
