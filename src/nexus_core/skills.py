"""Runtime skill discovery and lightweight capability resolution."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_FRONTMATTER_BOUNDARY = "---"
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "as",
    "at",
    "be",
    "before",
    "by",
    "can",
    "do",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "use",
    "when",
    "with",
}


@dataclass(frozen=True, slots=True)
class SkillManifest:
    """Normalized metadata for one installed skill."""

    name: str
    description: str
    path: str
    guidance: str = ""
    preferred_initial_actions: tuple[str, ...] = ()

    def to_dict(self, *, include_guidance: bool = False, guidance_limit: int = 600) -> dict[str, str]:
        payload = {
            "name": self.name,
            "description": self.description,
            "path": self.path,
        }
        if self.preferred_initial_actions:
            payload["preferred_initial_actions"] = list(self.preferred_initial_actions)
        if include_guidance and self.guidance:
            payload["guidance_excerpt"] = self.guidance[:guidance_limit].strip()
        return payload


@dataclass(frozen=True, slots=True)
class CapabilityMatch:
    """One scored skill match for an objective."""

    manifest: SkillManifest
    score: int

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = self.manifest.to_dict()
        payload["score"] = self.score
        return payload


class SkillRegistry:
    """Discovers skills from configured roots."""

    def __init__(self, skill_roots: Iterable[Path | str]) -> None:
        self.skill_roots = [Path(root).expanduser().resolve() for root in skill_roots if str(root).strip()]
        self._cached_manifests: list[SkillManifest] | None = None

    def list_manifests(self) -> list[SkillManifest]:
        if self._cached_manifests is None:
            self._cached_manifests = self._discover_manifests()
        return list(self._cached_manifests)

    def refresh(self) -> list[SkillManifest]:
        self._cached_manifests = self._discover_manifests()
        return list(self._cached_manifests)

    def _discover_manifests(self) -> list[SkillManifest]:
        manifests: list[SkillManifest] = []
        seen_paths: set[Path] = set()
        for root in self.skill_roots:
            if not root.exists():
                continue
            for skill_file in sorted(root.rglob("SKILL.md")):
                resolved = skill_file.resolve()
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                manifest = _parse_skill_manifest(resolved)
                if manifest is not None:
                    manifests.append(manifest)
        manifests.sort(key=lambda item: item.name.lower())
        return manifests


class CapabilityResolver:
    """Ranks installed skills against an objective."""

    def __init__(self, registry: SkillRegistry, max_matches: int = 3) -> None:
        self.registry = registry
        self.max_matches = max(1, min(int(max_matches), 8))

    def resolve(self, objective: str) -> list[SkillManifest]:
        return [match.manifest for match in self.resolve_matches(objective)]

    def resolve_matches(self, objective: str) -> list[CapabilityMatch]:
        objective_tokens = _tokenize(objective)
        if not objective_tokens:
            return []

        scored: list[CapabilityMatch] = []
        for manifest in self.registry.list_manifests():
            score = _score_manifest(manifest, objective_tokens, objective)
            if score > 0:
                scored.append(CapabilityMatch(manifest=manifest, score=score))
        scored.sort(key=lambda item: (-item.score, item.manifest.name.lower()))
        return scored[: self.max_matches]


def serialize_skill_context(
    skills: Iterable[SkillManifest],
    *,
    include_guidance: bool = True,
    guidance_limit: int = 600,
) -> list[dict[str, str]]:
    """Serialize resolved skills into planner-safe prompt context."""

    return [
        skill.to_dict(include_guidance=include_guidance, guidance_limit=guidance_limit)
        for skill in skills
    ]


def default_skill_roots_from_env(raw: str) -> list[Path]:
    """Parse a configurable skill-root string into concrete paths."""

    if not raw.strip():
        return []
    normalized = raw.replace("\r\n", "\n").replace("\n", os.pathsep).replace(",", os.pathsep)
    parts = [part.strip() for part in normalized.split(os.pathsep)]
    unique: list[Path] = []
    seen: set[Path] = set()
    for part in parts:
        if not part:
            continue
        path = Path(part).expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _parse_skill_manifest(path: Path) -> SkillManifest | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None

    frontmatter, body = _split_frontmatter(raw)
    name = str(frontmatter.get("name", "")).strip() or _heading_name(body) or path.parent.name
    description = (
        str(frontmatter.get("description", "")).strip()
        or _first_meaningful_body_line(body)
        or f"Skill instructions from {path.parent.name}."
    )
    guidance = body.strip()
    if not name:
        return None
    return SkillManifest(
        name=name,
        description=description,
        path=str(path),
        guidance=guidance,
        preferred_initial_actions=_parse_action_list(
            str(frontmatter.get("preferred_initial_actions", ""))
        ),
    )


def _split_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    stripped = raw.lstrip()
    if not stripped.startswith(_FRONTMATTER_BOUNDARY):
        return {}, raw

    lines = stripped.splitlines()
    if not lines or lines[0].strip() != _FRONTMATTER_BOUNDARY:
        return {}, raw

    frontmatter: dict[str, str] = {}
    end_idx = -1
    for idx in range(1, len(lines)):
        line = lines[idx]
        if line.strip() == _FRONTMATTER_BOUNDARY:
            end_idx = idx
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        cleaned = value.strip().strip("'\"")
        frontmatter[key.strip()] = cleaned
    if end_idx < 0:
        return {}, raw
    body = "\n".join(lines[end_idx + 1 :]).strip()
    return frontmatter, body


def _heading_name(body: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def _first_meaningful_body_line(body: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return stripped
    return ""


def _score_manifest(manifest: SkillManifest, objective_tokens: set[str], objective: str) -> int:
    name_tokens = _tokenize(manifest.name)
    description_tokens = _tokenize(manifest.description)
    guidance_tokens = _tokenize(manifest.guidance[:1600])

    score = 0
    score += 5 * len(objective_tokens & name_tokens)
    score += 3 * len(objective_tokens & description_tokens)
    score += 1 * len(objective_tokens & guidance_tokens)

    lowered_objective = objective.lower()
    lowered_name = manifest.name.lower()
    if lowered_name and lowered_name in lowered_objective:
        score += 4
    if manifest.description and manifest.description.lower() in lowered_objective:
        score += 2
    return score


def _tokenize(value: str) -> set[str]:
    tokens: set[str] = set()
    for raw_token in _TOKEN_RE.findall(value.lower()):
        token = _normalize_token(raw_token)
        if len(token) < 3 or token in _STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _normalize_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 4 and not token.endswith("ses"):
        return token[:-2]
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        return token[:-1]
    return token


def _parse_action_list(raw: str) -> tuple[str, ...]:
    if not raw.strip():
        return ()
    actions: list[str] = []
    seen: set[str] = set()
    for chunk in raw.replace("|", ",").split(","):
        action = chunk.strip().lower().replace(" ", "_")
        if not action or action in seen:
            continue
        seen.add(action)
        actions.append(action)
    return tuple(actions)
