"""Tests for prompt building and injection safety."""

from nexus.personality.prompts import build_task_prompt, build_system_prompt


def test_task_prompt_escapes_braces():
    """Review #14: curly braces in input should not cause format errors."""
    result = build_task_prompt("Parse this JSON: {key: value}")
    assert "{key: value}" in result


def test_task_prompt_contains_description():
    result = build_task_prompt("Summarize the document")
    assert "Summarize the document" in result


def test_task_prompt_with_nested_braces():
    result = build_task_prompt("Code: def foo(): return {a: {b: c}}")
    assert "{a: {b: c}}" in result


def test_system_prompt_contains_model_name():
    # Uses actual identity data from identities.py
    # minimax/minimax-m2.5 is one of the default swarm models
    result = build_system_prompt(
        "minimax/minimax-m2.5",
        ["minimax/minimax-m2.5", "z-ai/glm-5"],
    )
    assert "Agent Nexus" in result
    assert len(result) > 100  # Should be a substantial prompt
