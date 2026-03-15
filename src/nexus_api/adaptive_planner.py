"""Model-assisted adaptive replanner implementations."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from nexus_core.models import RunMode, StepDefinition, StepExecutionResult
from nexus_core.planner import AdaptivePlanner, annotate_planner_steps, request_next_steps

log = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


class ChatCompletionsAdaptivePlanner:
    """Adaptive planner that targets an OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout_sec: float = 12.0,
        max_steps: int = 4,
        provider_name: str = "chat_completions",
        route_label: str = "",
    ) -> None:
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.max_steps = max(1, min(max_steps, 8))
        self.provider_name = provider_name.strip() or "chat_completions"
        self.route_label = route_label.strip() or f"{self.provider_name}:{self.model}".strip(":")

    async def plan_next_steps(
        self,
        objective: str,
        mode: RunMode,
        existing_steps: list[dict[str, Any]],
        completed_step: dict[str, Any] | None = None,
        result: StepExecutionResult | None = None,
        skill_context: list[dict[str, str]] | None = None,
    ) -> list[StepDefinition]:
        if not self.model:
            return []

        step_budget = 1
        _ = existing_steps
        if completed_step is None or result is None:
            allowed_actions = [
                "search_web",
                "fetch_url",
                "navigate",
                "inspect",
                "read",
                "list_files",
                "read_file",
            ]
            payload = {
                "objective": objective,
                "mode": mode.value,
                "constraints": {
                    "allowed_actions": allowed_actions,
                    "max_steps": step_budget,
                },
            }
            if skill_context:
                payload["resolved_skills"] = skill_context
            parsed = await self._request_plan(
                prompt=(
                    "Return strict JSON with shape "
                    '{"next_steps":[{"action_type":"...","instruction":"..."}]}. '
                    "Plan exactly one grounded bootstrap tool call. "
                    "Use only low-risk starting actions that gather context or open the correct page/file. "
                    "Prefer workspace read tools when the objective references local files. "
                    "When resolved_skills are present, follow their guidance before improvising a toolchain. "
                    'For list_files/read_file use instruction payload {"path":"..."} as a JSON string or JSON object. '
                    "No prose."
                ),
                payload=payload,
            )
            return annotate_planner_steps(
                _annotate_model_route(
                    self._step_definitions_from_payload(parsed, limit=step_budget),
                    self.route_label,
                ),
                planner_source="model",
                planner_phase="initial",
            )

        payload = {
            "objective": objective,
            "completed_step": {
                "action_type": completed_step.get("action_type", ""),
                "instruction": completed_step.get("instruction", ""),
                "output_text": _truncate_text(result.output_text, 1200),
                "citations_count": len(result.citations),
                "artifacts_count": len(result.artifacts),
                "citations": _serialize_citations(result.citations, limit=3),
                "metadata": _compact_metadata(result.metadata, max_items=8),
            },
            "existing_steps": [
                {
                    "action_type": step.get("action_type", ""),
                    "status": step.get("status", ""),
                    "instruction": str(step.get("instruction", ""))[:220],
                }
                for step in existing_steps
            ],
            "constraints": {
                "allowed_actions": [
                    "search_web",
                    "fetch_url",
                    "list_files",
                    "read_file",
                    "write_file",
                    "edit_file",
                    "execute_code",
                    "delegate",
                    "navigate",
                    "inspect",
                    "scroll",
                    "extract",
                    "read",
                    "click",
                    "type",
                    "wait",
                    "write",
                    "submit",
                    "export",
                    "generate_report",
                    "generate_chart",
                    "generate_image",
                ],
                "max_steps": step_budget,
            },
        }
        if skill_context:
            payload["resolved_skills"] = skill_context
        parsed = await self._request_plan(
            prompt=(
                "Return strict JSON with shape "
                '{"next_steps":[{"action_type":"...","instruction":"..."}]}. '
                "No prose. Return exactly one next step. Prefer workspace and code tools when result metadata references files. "
                "When resolved_skills are present, use that guidance before inventing a new procedure. "
                'For list_files/read_file use instruction payload {"path":"..."}. '
                'For write_file use instruction payload {"path":"...","content":"..."}. '
                'For edit_file use instruction payload {"path":"...","old":"...","new":"..."} or {"path":"...","content":"..."}. '
                'For execute_code use instruction payload {"command":["cmd","arg"]}. '
                'For generate_report use instruction payload {"path":"reports/...md","title":"...","sources":[...]}. '
                'For generate_chart use instruction payload {"path":"charts/...html","chart_type":"bar","title":"...","x_key":"...","y_key":"...","data":[...]}. '
                'For generate_image use instruction payload {"path":"images/...svg","title":"...","prompt":"...","sources":[...]}. '
                "When metadata includes page_affordances, use those grounded inputs, buttons, and links instead of generic UI guesses. "
                "When metadata includes command_failed or a non-zero exit_code, treat that as diagnostic evidence and inspect or edit the referenced code path instead of repeating the same command blindly."
            ),
            payload=payload,
        )
        return annotate_planner_steps(
            _annotate_model_route(
                self._step_definitions_from_payload(parsed, limit=step_budget),
                self.route_label,
            ),
            planner_source="model",
            planner_phase="follow_up",
        )

    async def plan_initial_steps(
        self,
        objective: str,
        mode: RunMode,
        skill_context: list[dict[str, str]] | None = None,
    ) -> list[StepDefinition]:
        return await self.plan_next_steps(
            objective=objective,
            mode=mode,
            existing_steps=[],
            skill_context=skill_context,
        )

    async def propose_follow_up(
        self,
        objective: str,
        completed_step: dict[str, Any],
        result: StepExecutionResult,
        existing_steps: list[dict[str, Any]],
        skill_context: list[dict[str, str]] | None = None,
    ) -> list[StepDefinition]:
        return await self.plan_next_steps(
            objective=objective,
            mode=RunMode.MANUAL,
            existing_steps=existing_steps,
            completed_step=completed_step,
            result=result,
            skill_context=skill_context,
        )

    async def _request_plan(
        self,
        prompt: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request_body = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload)},
            ],
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=request_body,
                )
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            log.warning("%s adaptive replanner request failed: %s", self.provider_name, exc)
            return {}
        content = str(data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
        return _parse_model_json(content)

    def _step_definitions_from_payload(
        self,
        payload: dict[str, Any],
        limit: int,
    ) -> list[StepDefinition]:
        raw_steps = payload.get("next_steps")
        if not isinstance(raw_steps, list):
            return []

        steps: list[StepDefinition] = []
        for item in raw_steps[:limit]:
            if not isinstance(item, dict):
                continue
            action_type = _stringify_action_type(
                item.get("action_type", item.get("action", item.get("tool", item.get("tool_name", ""))))
            )
            instruction = _stringify_step_instruction(
                item.get("instruction", item.get("payload", item.get("arguments", "")))
            )
            if not action_type or not instruction:
                continue
            try:
                steps.append(StepDefinition(action_type=action_type, instruction=instruction))
            except Exception:
                continue
        return steps


class OpenRouterAdaptivePlanner(ChatCompletionsAdaptivePlanner):
    """Adaptive planner that proposes next steps from an OpenRouter model."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_sec: float = 12.0,
        max_steps: int = 4,
        route_label: str = "",
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_sec=timeout_sec,
            max_steps=max_steps,
            provider_name="openrouter",
            route_label=route_label,
        )


class RuleFallbackAdaptivePlanner:
    """Bridge wrapper around core rule planner for composition."""

    def __init__(self, planner: AdaptivePlanner) -> None:
        self.planner = planner

    async def plan_next_steps(
        self,
        objective: str,
        mode: RunMode,
        existing_steps: list[dict[str, Any]],
        completed_step: dict[str, Any] | None = None,
        result: StepExecutionResult | None = None,
        skill_context: list[dict[str, str]] | None = None,
    ) -> list[StepDefinition]:
        return await request_next_steps(
            self.planner,
            objective=objective,
            mode=mode,
            existing_steps=existing_steps,
            completed_step=completed_step,
            result=result,
            skill_context=skill_context,
        )

    async def plan_initial_steps(
        self,
        objective: str,
        mode: RunMode,
        skill_context: list[dict[str, str]] | None = None,
    ) -> list[StepDefinition]:
        return await self.plan_next_steps(
            objective=objective,
            mode=mode,
            existing_steps=[],
            skill_context=skill_context,
        )

    async def propose_follow_up(
        self,
        objective: str,
        completed_step: dict[str, Any],
        result: StepExecutionResult,
        existing_steps: list[dict[str, Any]],
        skill_context: list[dict[str, str]] | None = None,
    ) -> list[StepDefinition]:
        return await self.plan_next_steps(
            objective=objective,
            mode=RunMode.MANUAL,
            existing_steps=existing_steps,
            completed_step=completed_step,
            result=result,
            skill_context=skill_context,
        )


def _parse_model_json(content: str) -> dict[str, Any]:
    if not content:
        return {}
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    block_match = _JSON_BLOCK_RE.search(content)
    if not block_match:
        return {}
    try:
        parsed = json.loads(block_match.group(1))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _annotate_model_route(steps: list[StepDefinition], route_label: str) -> list[StepDefinition]:
    if not route_label:
        return steps
    annotated: list[StepDefinition] = []
    for step in steps:
        metadata = dict(step.metadata)
        metadata["model_route"] = route_label
        annotated.append(
            StepDefinition(
                action_type=step.action_type,
                instruction=step.instruction,
                metadata=metadata,
            )
        )
    return annotated


def _stringify_step_instruction(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if value is None:
        return ""
    return str(value).strip()


def _stringify_action_type(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _truncate_text(value: str, limit: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "…"


def _serialize_citations(citations: list[Any], *, limit: int) -> list[dict[str, str]]:
    serialized: list[dict[str, str]] = []
    for citation in citations[:limit]:
        if hasattr(citation, "model_dump"):
            raw = citation.model_dump()
        elif isinstance(citation, dict):
            raw = citation
        else:
            continue
        serialized.append(
            {
                "url": _truncate_text(str(raw.get("url", "")), 240),
                "title": _truncate_text(str(raw.get("title", "")), 160),
                "snippet": _truncate_text(str(raw.get("snippet", "")), 320),
            }
        )
    return serialized


def _compact_metadata(metadata: dict[str, Any], *, max_items: int) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for idx, (key, value) in enumerate(metadata.items()):
        if idx >= max_items:
            break
        compact[str(key)] = _compact_value(value)
    return compact


def _compact_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_text(value, 320)
    if isinstance(value, list):
        return [_compact_value(item) for item in value[:6]]
    if isinstance(value, dict):
        nested: dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= 6:
                break
            nested[str(key)] = _compact_value(item)
        return nested
    return _truncate_text(str(value), 320)
