"""Unit tests for model-assisted adaptive replanner wiring."""

from __future__ import annotations

import json as jsonlib
from typing import Any

import pytest

from nexus_api.adaptive_planner import ChatCompletionsAdaptivePlanner, OpenRouterAdaptivePlanner
from nexus_api.config import ApiSettings
from nexus_api.service import build_model_adaptive_planner
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
                "command_failed": True,
                "exit_code": 1,
                "stderr": "AssertionError in src/payments/retry.py:12",
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
    assert "page_affordances" in request_body["messages"][0]["content"]
    assert "command_failed" in request_body["messages"][0]["content"]
    assert "For edit_file use instruction payload" in request_body["messages"][0]["content"]
    assert "For execute_code use instruction payload" in request_body["messages"][0]["content"]
    completed_payload = payload["completed_step"]
    assert completed_payload["citations"][0]["url"] == "https://docs.example.org/start"
    assert completed_payload["metadata"]["current_url"] == "https://docs.example.org/start"
    assert completed_payload["metadata"]["command_failed"] is True
    assert completed_payload["metadata"]["exit_code"] == 1
    assert completed_payload["metadata"]["stderr"] == "AssertionError in src/payments/retry.py:12"
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
    assert steps[0].metadata["planner_source"] == "model"
    assert steps[0].metadata["planner_phase"] == "initial"
    payload = jsonlib.loads(captured["request_body"]["messages"][1]["content"])
    assert payload["constraints"]["max_steps"] == 1
    assert payload["constraints"]["allowed_actions"] == [
        "search_web",
        "fetch_url",
        "navigate",
        "inspect",
        "read",
        "list_files",
        "read_file",
    ]


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
    assert steps[0].metadata["planner_source"] == "model"
    assert steps[0].metadata["planner_phase"] == "follow_up"
    payload = jsonlib.loads(captured["request_body"]["messages"][1]["content"])
    assert payload["constraints"]["max_steps"] == 1


@pytest.mark.asyncio
async def test_openrouter_follow_up_accepts_structured_instruction_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            _ = url, headers, json
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": jsonlib.dumps(
                                    {
                                        "next_steps": [
                                            {
                                                "action_type": "edit_file",
                                                "instruction": {
                                                    "path": "src/payments/retry.py",
                                                    "old": "return base_delay",
                                                    "new": "return base_delay * 2",
                                                },
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

    planner = OpenRouterAdaptivePlanner(api_key="test-key", model="test-model", max_steps=4)
    steps = await planner.propose_follow_up(
        objective="Implement the payment retry backoff fix in the repo and update tests",
        completed_step={"action_type": "read_file", "instruction": '{"path":"src/payments/retry.py"}'},
        result=StepExecutionResult(
            output_text="def retry_backoff(base_delay):\n    return base_delay",
            metadata={"file_path": "src/payments/retry.py"},
        ),
        existing_steps=[
            {"action_type": "list_files", "status": "completed"},
            {"action_type": "read_file", "status": "completed"},
        ],
    )

    assert len(steps) == 1
    assert steps[0].action_type == "edit_file"
    assert jsonlib.loads(steps[0].instruction) == {
        "path": "src/payments/retry.py",
        "old": "return base_delay",
        "new": "return base_delay * 2",
    }
    assert steps[0].metadata["planner_source"] == "model"
    assert steps[0].metadata["planner_phase"] == "follow_up"


@pytest.mark.asyncio
async def test_openrouter_follow_up_accepts_payload_alias_for_instruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            _ = url, headers, json
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": jsonlib.dumps(
                                    {
                                        "next_steps": [
                                            {
                                                "action_type": "execute_code",
                                                "payload": {
                                                    "command": ["python", "-m", "pytest", "-q"]
                                                },
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

    planner = OpenRouterAdaptivePlanner(api_key="test-key", model="test-model", max_steps=4)
    steps = await planner.propose_follow_up(
        objective="Implement the payment retry backoff fix in the repo and update tests",
        completed_step={"action_type": "read_file", "instruction": '{"path":"src/payments/retry.py"}'},
        result=StepExecutionResult(
            output_text="def retry_backoff(base_delay):\n    return base_delay",
            metadata={"file_path": "src/payments/retry.py"},
        ),
        existing_steps=[
            {"action_type": "list_files", "status": "completed"},
            {"action_type": "read_file", "status": "completed"},
        ],
    )

    assert len(steps) == 1
    assert steps[0].action_type == "execute_code"
    assert jsonlib.loads(steps[0].instruction) == {"command": ["python", "-m", "pytest", "-q"]}


@pytest.mark.asyncio
async def test_openrouter_follow_up_accepts_action_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            _ = url, headers, json
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": jsonlib.dumps(
                                    {
                                        "next_steps": [
                                            {
                                                "action": "write_file",
                                                "instruction": {
                                                    "path": "reports/summary.md",
                                                    "content": "updated summary",
                                                },
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

    planner = OpenRouterAdaptivePlanner(api_key="test-key", model="test-model", max_steps=4)
    steps = await planner.propose_follow_up(
        objective="Implement the payment retry backoff fix in the repo and update tests",
        completed_step={"action_type": "read_file", "instruction": '{"path":"src/payments/retry.py"}'},
        result=StepExecutionResult(
            output_text="def retry_backoff(base_delay):\n    return base_delay",
            metadata={"file_path": "src/payments/retry.py"},
        ),
        existing_steps=[
            {"action_type": "list_files", "status": "completed"},
            {"action_type": "read_file", "status": "completed"},
        ],
    )

    assert len(steps) == 1
    assert steps[0].action_type == "write_file"
    assert jsonlib.loads(steps[0].instruction) == {
        "path": "reports/summary.md",
        "content": "updated summary",
    }


@pytest.mark.asyncio
async def test_chat_completions_planner_supports_local_endpoint_without_auth(
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
                                                "action_type": "search_web",
                                                "instruction": "Start with grounded search.",
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

    planner = ChatCompletionsAdaptivePlanner(
        api_key="",
        model="local-qwen",
        base_url="http://localhost:11434/v1",
        provider_name="local",
    )
    steps = await planner.plan_initial_steps(
        objective="Research grounded sources",
        mode=RunMode.SUPERVISED,
    )

    assert len(steps) == 1
    assert steps[0].action_type == "search_web"
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
    assert "Authorization" not in captured["headers"]
    payload = jsonlib.loads(captured["request_body"]["messages"][1]["content"])
    assert payload["constraints"]["max_steps"] == 1


def test_build_model_adaptive_planner_supports_openai_compatible_local_settings() -> None:
    settings = ApiSettings(
        APP_DATABASE_URL="sqlite:///./data/app/test.db",
        APP_MODEL_REPLANNER_PROVIDER="openai_compatible",
        APP_MODEL_REPLANNER_BASE_URL="http://localhost:11434/v1",
        APP_MODEL_REPLANNER_MODEL="local-qwen",
        APP_MODEL_REPLANNER_API_KEY="",
        OPENROUTER_API_KEY="",
    )

    planner = build_model_adaptive_planner(settings)

    assert isinstance(planner, ChatCompletionsAdaptivePlanner)
    assert planner.base_url == "http://localhost:11434/v1"
    assert planner.model == "local-qwen"


def test_build_model_adaptive_planner_preserves_openrouter_defaults() -> None:
    settings = ApiSettings(
        APP_DATABASE_URL="sqlite:///./data/app/test.db",
        APP_MODEL_REPLANNER_PROVIDER="openrouter",
        APP_MODEL_REPLANNER_API_KEY="router-key",
        APP_MODEL_REPLANNER_MODEL="openai/gpt-4.1-mini",
        APP_MODEL_REPLANNER_BASE_URL="https://openrouter.ai/api/v1",
    )

    planner = build_model_adaptive_planner(settings)

    assert isinstance(planner, OpenRouterAdaptivePlanner)
    assert planner.base_url == "https://openrouter.ai/api/v1"
    assert planner.model == "openai/gpt-4.1-mini"
