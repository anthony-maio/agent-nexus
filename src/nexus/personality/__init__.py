"""Personality layer for Agent Nexus swarm members."""

from nexus.personality.identities import (
    IDENTITIES,
    MAIN_MODEL_IDS,
    PREMIUM_MODEL_IDS,
    TASK_MODEL_IDS,
    ModelIdentity,
    format_name,
    get_identity,
)
from nexus.personality.prompts import (
    build_system_prompt,
    build_task_prompt,
)

__all__ = [
    "IDENTITIES",
    "MAIN_MODEL_IDS",
    "PREMIUM_MODEL_IDS",
    "TASK_MODEL_IDS",
    "ModelIdentity",
    "build_system_prompt",
    "build_task_prompt",
    "format_name",
    "get_identity",
]
