"""Model-assisted adaptive replanner implementations."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from nexus_core.models import StepDefinition, StepExecutionResult
from nexus_core.planner import AdaptivePlanner

log = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


class OpenRouterAdaptivePlanner:
    """Adaptive planner that proposes next steps from an OpenRouter model."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_sec: float = 12.0,
        max_steps: int = 4,
    ) -> None:
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.max_steps = max(1, min(max_steps, 8))

    async def propose_follow_up(
        self,
        objective: str,
        completed_step: dict[str, Any],
        result: StepExecutionResult,
        existing_steps: list[dict[str, Any]],
    ) -> list[StepDefinition]:
        if not self.api_key or not self.model:
            return []

        payload = {
            "objective": objective,
            "completed_step": {
                "action_type": completed_step.get("action_type", ""),
                "instruction": completed_step.get("instruction", ""),
                "output_text": result.output_text[:1200],
                "citations_count": len(result.citations),
                "artifacts_count": len(result.artifacts),
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
                ],
                "max_steps": self.max_steps,
            },
        }
        prompt = (
            "Return strict JSON with shape "
            '{"next_steps":[{"action_type":"...","instruction":"..."}]}. '
            "No prose. Use at most max_steps items."
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
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
            log.warning("OpenRouter adaptive replanner request failed: %s", exc)
            return []

        content = str(
            data.get("choices", [{}])[0].get("message", {}).get("content", "")
        ).strip()
        parsed = _parse_model_json(content)
        if not parsed:
            return []
        raw_steps = parsed.get("next_steps")
        if not isinstance(raw_steps, list):
            return []

        steps: list[StepDefinition] = []
        for item in raw_steps[: self.max_steps]:
            if not isinstance(item, dict):
                continue
            action_type = str(item.get("action_type", "")).strip()
            instruction = str(item.get("instruction", "")).strip()
            if not action_type or not instruction:
                continue
            try:
                steps.append(
                    StepDefinition(
                        action_type=action_type,
                        instruction=instruction,
                    )
                )
            except Exception:
                continue
        return steps


class RuleFallbackAdaptivePlanner:
    """Bridge wrapper around core rule planner for composition."""

    def __init__(self, planner: AdaptivePlanner) -> None:
        self.planner = planner

    async def propose_follow_up(
        self,
        objective: str,
        completed_step: dict[str, Any],
        result: StepExecutionResult,
        existing_steps: list[dict[str, Any]],
    ) -> list[StepDefinition]:
        return await self.planner.propose_follow_up(
            objective=objective,
            completed_step=completed_step,
            result=result,
            existing_steps=existing_steps,
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
