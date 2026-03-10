"""Unit tests for model-assisted adaptive replanner wiring."""

from __future__ import annotations

import json as jsonlib
from typing import Any

import pytest

from nexus_api.adaptive_planner import OpenRouterAdaptivePlanner
from nexus_core.models import CitationRecord, RunMode, StepExecutionResult


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


@pytest.mark.asyncio
async def test_openrouter_follow_up_request_includes_actions_and_evidence_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _FakeAsyncClient:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        async def __aenter__(self) -> _FakeAsyncClient:
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

        async def post(
            self,
            url: str,
            *,
            headers: dict[str, str],
            json: dict[str, Any],
        ) -> _FakeResponse:
            captured["url"] = url
            captured["headers"] = headers
            captured["request_body"] = json
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": jsonlib.dumps(
                                    {
                                        "next_steps": [
                                            {
                                                "action_type": "fetch_url",
                                                "instruction": "Open the top source.",
                                            }
                                        ]
                                    }
                                )
                            }
                        }
                    ]
                }
            )

    monkeypatch.setattr("nexus_api.adaptive_planner.httpx.AsyncClient", _FakeAsyncClient)

    planner = OpenRouterAdaptivePlanner(api_key="test-key", model="test-model")
    steps = await planner.propose_follow_up(
        objective="Research grounded billing docs",
        completed_step={
            "action_type": "search_web",
            "instruction": "find grounded docs",
        },
        result=StepExecutionResult(
            output_text="search complete",
            citations=[
                CitationRecord(
                    url="https://docs.example.org/start",
                    title="Grounded docs",
                    snippet="Grounded snippet",
                )
            ],
            metadata={
                "current_url": "https://docs.example.org/start",
                "page_affordances": {
                    "forms_count": 1,
                    "input_fields": [
                        {"tag": "input", "type": "email", "name": "email"}
                    ],
                    "buttons": [{"text": "Continue", "type": "submit"}],
                },
                "search_results": [
                    {
                        "url": "https://docs.example.org/start",
                        "title": "Grounded docs",
                        "snippet": "Grounded snippet",
                    }
                ],
            },
        ),
        existing_steps=[{"action_type": "search_web", "status": "completed"}],
    )

    assert steps
    assert steps[0].action_type == "fetch_url"
    request_body = captured["request_body"]
    payload = jsonlib.loads(request_body["messages"][1]["content"])
    allowed_actions = payload["constraints"]["allowed_actions"]
    assert "search_web" in allowed_actions
    assert "fetch_url" in allowed_actions
    completed_payload = payload["completed_step"]
    assert completed_payload["citations"][0]["url"] == "https://docs.example.org/start"
    assert completed_payload["metadata"]["current_url"] == "https://docs.example.org/start"
    assert completed_payload["metadata"]["page_affordances"]["forms_count"] == 1
    assert completed_payload["metadata"]["page_affordances"]["buttons"][0]["text"] == "Continue"


@pytest.mark.asyncio
async def test_openrouter_plan_initial_steps_uses_single_step_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _FakeAsyncClient:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        async def __aenter__(self) -> _FakeAsyncClient:
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

        async def post(
            self,
            url: str,
            *,
            headers: dict[str, str],
            json: dict[str, Any],
        ) -> _FakeResponse:
            captured["request_body"] = json
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": jsonlib.dumps(
                                    {
                                        "next_steps": [
                                            {"action_type": "search_web", "instruction": "Find sources."},
                                            {"action_type": "fetch_url", "instruction": "Open source."},
                                        ]
                                    }
                                )
                            }
                        }
                    ]
                }
            )

    monkeypatch.setattr("nexus_api.adaptive_planner.httpx.AsyncClient", _FakeAsyncClient)

    planner = OpenRouterAdaptivePlanner(api_key="test-key", model="test-model", max_steps=4)
    steps = await planner.plan_initial_steps(
        objective="Research grounded sources",
        mode=RunMode.SUPERVISED,
    )

    assert len(steps) == 1
    assert steps[0].action_type == "search_web"
    payload = jsonlib.loads(captured["request_body"]["messages"][1]["content"])
    assert payload["constraints"]["max_steps"] == 1


@pytest.mark.asyncio
async def test_openrouter_follow_up_uses_single_step_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _FakeAsyncClient:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        async def __aenter__(self) -> _FakeAsyncClient:
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

        async def post(
            self,
            url: str,
            *,
            headers: dict[str, str],
            json: dict[str, Any],
        ) -> _FakeResponse:
            captured["request_body"] = json
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": jsonlib.dumps(
                                    {
                                        "next_steps": [
                                            {"action_type": "fetch_url", "instruction": "Open source."},
                                            {"action_type": "extract", "instruction": "Summarize source."},
                                        ]
                                    }
                                )
                            }
                        }
                    ]
                }
            )

    monkeypatch.setattr("nexus_api.adaptive_planner.httpx.AsyncClient", _FakeAsyncClient)

    planner = OpenRouterAdaptivePlanner(api_key="test-key", model="test-model", max_steps=4)
    steps = await planner.propose_follow_up(
        objective="Research grounded billing docs",
        completed_step={"action_type": "search_web", "instruction": "find docs"},
        result=StepExecutionResult(output_text="search complete"),
        existing_steps=[{"action_type": "search_web", "status": "completed"}],
    )

    assert len(steps) == 1
    assert steps[0].action_type == "fetch_url"
    payload = jsonlib.loads(captured["request_body"]["messages"][1]["content"])
    assert payload["constraints"]["max_steps"] == 1
