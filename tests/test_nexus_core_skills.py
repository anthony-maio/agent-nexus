from __future__ import annotations

from pathlib import Path

from nexus_core.skills import CapabilityResolver, SkillRegistry


def _write_skill(
    root: Path,
    folder: str,
    *,
    name: str,
    description: str,
    preferred_initial_actions: str = "",
    preferred_follow_up_actions: str = "",
    external_tools: str = "",
    verification_signals: str = "",
    required_artifact_kinds: str = "",
) -> Path:
    skill_dir = root / folder
    skill_dir.mkdir(parents=True, exist_ok=True)
    frontmatter = [
        "---",
        f"name: {name}",
        f'description: "{description}"',
    ]
    if preferred_initial_actions:
        frontmatter.append(f"preferred_initial_actions: {preferred_initial_actions}")
    if preferred_follow_up_actions:
        frontmatter.append(f"preferred_follow_up_actions: {preferred_follow_up_actions}")
    if external_tools:
        frontmatter.append(f"external_tools: {external_tools}")
    if verification_signals:
        frontmatter.append(f"verification_signals: {verification_signals}")
    if required_artifact_kinds:
        frontmatter.append(f"required_artifact_kinds: {required_artifact_kinds}")
    frontmatter.extend(["---", ""])
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            frontmatter
            + [
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
    matches = resolver.resolve_matches("Generate a chart from CSV sales data and summarize it")

    assert [skill.name for skill in resolved] == ["chart-maker"]
    assert resolved[0].description == "Generate charts and plots from CSV or tabular data."
    assert matches[0].manifest.name == "chart-maker"
    assert matches[0].score > 0


def test_skill_registry_parses_preferred_initial_actions(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "chart-maker",
        name="chart-maker",
        description="Generate charts and plots from CSV or tabular data.",
        preferred_initial_actions="list_files, read_file",
    )

    registry = SkillRegistry([tmp_path])
    manifests = registry.list_manifests()

    assert manifests[0].preferred_initial_actions == ("list_files", "read_file")
    assert manifests[0].to_dict()["preferred_initial_actions"] == ["list_files", "read_file"]


def test_skill_registry_parses_preferred_follow_up_actions(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "report-maker",
        name="report-maker",
        description="Produce polished reports from gathered findings.",
        preferred_follow_up_actions="generate_report, export",
    )

    registry = SkillRegistry([tmp_path])
    manifests = registry.list_manifests()

    assert manifests[0].preferred_follow_up_actions == ("generate_report", "export")
    assert manifests[0].to_dict()["preferred_follow_up_actions"] == [
        "generate_report",
        "export",
    ]


def test_skill_registry_parses_external_tool_dependencies(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "memory-helper",
        name="memory-helper",
        description="Retrieve scoped memory and repo maps from external tools.",
        external_tools="mnemos.retrieve, cartographer.map_repo",
    )

    registry = SkillRegistry([tmp_path])
    manifests = registry.list_manifests()

    assert manifests[0].external_tools == ("mnemos.retrieve", "cartographer.map_repo")
    assert manifests[0].to_dict()["external_tools"] == [
        "mnemos.retrieve",
        "cartographer.map_repo",
    ]


def test_skill_registry_parses_verification_requirements(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "chart-maker",
        name="chart-maker",
        description="Generate charts from tabular data.",
        verification_signals="citations, artifact",
        required_artifact_kinds="chart, report",
    )

    registry = SkillRegistry([tmp_path])
    manifests = registry.list_manifests()

    assert manifests[0].verification_signals == ("citations", "artifact")
    assert manifests[0].required_artifact_kinds == ("chart", "report")
    assert manifests[0].to_dict()["verification_signals"] == ["citations", "artifact"]
    assert manifests[0].to_dict()["required_artifact_kinds"] == ["chart", "report"]
