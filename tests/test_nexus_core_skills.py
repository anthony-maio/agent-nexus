from __future__ import annotations

from pathlib import Path

from nexus_core.skills import CapabilityResolver, SkillRegistry


def _write_skill(root: Path, folder: str, *, name: str, description: str) -> Path:
    skill_dir = root / folder
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                f"name: {name}",
                f'description: "{description}"',
                "---",
                "",
                f"# {name}",
                "",
                description,
                "",
                "Use this when the task clearly matches the skill.",
            ]
        ),
        encoding="utf-8",
    )
    return skill_dir / "SKILL.md"


def test_skill_registry_discovers_manifests_and_resolves_relevant_skills(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "chart-maker",
        name="chart-maker",
        description="Generate charts and plots from CSV or tabular data.",
    )
    _write_skill(
        tmp_path,
        "browser-agent",
        name="browser-agent",
        description="Navigate websites, click controls, and fill forms in the browser.",
    )

    registry = SkillRegistry([tmp_path])
    manifests = registry.list_manifests()

    assert [manifest.name for manifest in manifests] == ["browser-agent", "chart-maker"]

    resolver = CapabilityResolver(registry, max_matches=2)
    resolved = resolver.resolve("Generate a chart from CSV sales data and summarize it")

    assert [skill.name for skill in resolved] == ["chart-maker"]
    assert resolved[0].description == "Generate charts and plots from CSV or tabular data."

