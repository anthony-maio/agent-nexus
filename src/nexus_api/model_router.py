"""Model router configuration for role-based provider selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """One configured model/provider route."""

    name: str
    role: str
    provider: str
    model: str
    base_url: str = ""
    api_key: str = ""


@dataclass(frozen=True, slots=True)
class ModelRouterConfig:
    """Ordered model profiles grouped by task role."""

    profiles: list[ModelProfile]

    def profiles_for_role(self, role: str) -> list[ModelProfile]:
        normalized_role = role.strip().lower()
        return [
            profile
            for profile in self.profiles
            if profile.role.strip().lower() == normalized_role
        ]


def parse_model_router_config(raw: str) -> ModelRouterConfig:
    """Parse a JSON router config into normalized model profiles."""

    if not raw.strip():
        return ModelRouterConfig(profiles=[])

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return ModelRouterConfig(profiles=[])

    if isinstance(parsed, dict):
        raw_profiles = parsed.get("profiles", [])
    elif isinstance(parsed, list):
        raw_profiles = parsed
    else:
        raw_profiles = []

    profiles: list[ModelProfile] = []
    for item in raw_profiles:
        if not isinstance(item, dict):
            continue
        provider = str(item.get("provider", "")).strip().lower()
        model = str(item.get("model", "")).strip()
        if not provider or not model:
            continue
        role = str(item.get("role", "planning")).strip().lower() or "planning"
        name = str(item.get("name", "")).strip() or f"{role}-{provider}-{model}"
        profiles.append(
            ModelProfile(
                name=name,
                role=role,
                provider=provider,
                model=model,
                base_url=str(item.get("base_url", "")).strip(),
                api_key=str(item.get("api_key", "")).strip(),
            )
        )
    return ModelRouterConfig(profiles=profiles)
