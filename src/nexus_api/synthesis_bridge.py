"""Bridge Nexus skill APIs to an external Synthesis checkout."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SynthesisBridge:
    """Thin wrapper around a live external ``SynthesisClient`` instance."""

    root: Path
    client: Any
    host_root: Path
    canonical_repo_path: Path | None = None

    @classmethod
    def from_settings(
        cls,
        *,
        synthesis_root: str,
        host_root: str,
        canonical_repo_path: str = "",
        provider_type: str = "mock",
        api_key: str = "",
        model: str = "",
        base_url: str = "",
    ) -> "SynthesisBridge | None":
        root = Path(synthesis_root).expanduser().resolve()
        if not synthesis_root.strip() or not root.exists():
            return None

        host = Path(host_root).expanduser().resolve()
        host.mkdir(parents=True, exist_ok=True)
        canonical = Path(canonical_repo_path).expanduser().resolve() if canonical_repo_path.strip() else None

        root_text = str(root)
        inserted = False
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
            inserted = True
        try:
            module = importlib.import_module("synthesis.client")
            client_cls = getattr(module, "SynthesisClient")
            provider_kwargs: dict[str, Any] = {}
            if api_key.strip():
                provider_kwargs["api_key"] = api_key.strip()
            if model.strip():
                provider_kwargs["model"] = model.strip()
            if base_url.strip():
                provider_kwargs["base_url"] = base_url.strip()
            client = client_cls(
                provider_type=provider_type.strip() or "mock",
                canonical_repo_path=str(canonical) if canonical else None,
                host_root=str(host),
                **provider_kwargs,
            )
            return cls(
                root=root,
                client=client,
                host_root=host,
                canonical_repo_path=canonical,
            )
        except Exception:
            return None
        finally:
            if inserted:
                try:
                    sys.path.remove(root_text)
                except ValueError:
                    pass

    async def acquire_skill(self, intent: str, requirements: str = "") -> dict[str, Any]:
        result = await self.client.acquire_skill(intent=intent, requirements=requirements)
        if hasattr(result, "to_dict"):
            return result.to_dict()
        if hasattr(result, "model_dump"):
            return result.model_dump()
        if isinstance(result, dict):
            return result
        return {"success": False, "error": "Unexpected Synthesis result payload"}


def synthesis_skill_paths(
    *,
    host_root: str,
    canonical_repo_path: str,
) -> list[Path]:
    """Return extra skill roots contributed by a Synthesis checkout."""

    paths: list[Path] = []
    if host_root.strip():
        paths.append(Path(host_root).expanduser().resolve())
    if canonical_repo_path.strip():
        paths.append((Path(canonical_repo_path).expanduser().resolve() / "skills").resolve())
    return paths
