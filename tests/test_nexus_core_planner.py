from __future__ import annotations

from nexus_core.models import CitationRecord, StepExecutionResult
from nexus_core.planner import plan_follow_up_steps


def _step(step_index: int, action_type: str) -> dict[str, object]:
    return {
        "step_index": step_index,
        "action_type": action_type,
        "instruction": f"{action_type} step",
        "status": "completed",
    }


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
