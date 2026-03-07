"""Tests for PiecesOS activity response parsing."""

import json

from nexus.integrations.pieces import ActivityDigest, parse_activity_response

# =====================================================================
# Structured JSON responses (summaries + events)
# =====================================================================

# Realistic PiecesOS response matching the format seen in production.
SAMPLE_STRUCTURED = json.dumps(
    {
        "summaries": [
            {
                "created": "2026-02-23 13:25:55.076568Z",
                "score": 0.284,
                "combined_string": (
                    "Automated Summary:\n"
                    "Created: 5 hrs ago\n"
                    "## TL;DR\n"
                    "Yesterday was a high-output session balancing a deep dive "
                    "into medical data review with significant technical milestones "
                    "for CoDA-GQA-L, including the merge of the dynamic bank "
                    "expansion branch.\n"
                    "### Core Tasks & Projects\n"
                    "- CoDA-GQA-L Optimization: Finalized Triton kernel updates\n"
                    "- Agent Nexus Infrastructure: Updated configuration\n"
                    "- Medical Data Review: Processed clinical dataset\n"
                ),
            },
            {
                "created": "2026-02-22 18:55:04.546658Z",
                "score": 0.214,
                "combined_string": (
                    "Automated Summary:\n"
                    "### Core Tasks & Projects\n"
                    "- CoDA-GQA-L Triton Kernel Optimization: Finalized updates\n"
                ),
            },
        ],
        "events": [
            {
                "created": "2026-02-23 05:56:19.748255Z",
                "app_title": "Discord.exe",
                "window_title": "#human | Secret Club - Discord",
                "score": 1.22,
            },
            {
                "created": "2026-02-23 05:50:00.000000Z",
                "app_title": "Code.exe",
                "window_title": "pieces.py - agent-nexus",
                "score": 0.9,
            },
        ],
    }
)


class TestStructuredParsing:
    def test_extracts_projects(self):
        digest = parse_activity_response(SAMPLE_STRUCTURED)
        assert "CoDA-GQA-L Optimization" in digest.projects
        assert "Agent Nexus Infrastructure" in digest.projects
        assert "Medical Data Review" in digest.projects

    def test_extracts_tldr_summary(self):
        digest = parse_activity_response(SAMPLE_STRUCTURED)
        assert "high-output session" in digest.summary
        assert "CoDA-GQA-L" in digest.summary

    def test_extracts_recent_focus(self):
        digest = parse_activity_response(SAMPLE_STRUCTURED)
        # Most recent summary (by created timestamp) provides recent_focus.
        assert digest.recent_focus
        assert len(digest.recent_focus) > 0

    def test_focus_uses_most_recent_not_highest_scored(self):
        """The most recently created summary should drive focus, not score."""
        raw = json.dumps(
            {
                "summaries": [
                    {
                        "created": "2026-02-20 10:00:00Z",
                        "score": 0.9,
                        "combined_string": ("## TL;DR\nOld high-scored summary.\n"),
                    },
                    {
                        "created": "2026-02-23 10:00:00Z",
                        "score": 0.1,
                        "combined_string": ("## TL;DR\nFresh low-scored summary.\n"),
                    },
                ],
                "events": [],
            }
        )
        digest = parse_activity_response(raw)
        assert "Fresh low-scored" in digest.recent_focus
        assert "Old high-scored" not in digest.recent_focus

    def test_extracts_active_apps(self):
        digest = parse_activity_response(SAMPLE_STRUCTURED)
        assert "Discord" in digest.active_apps
        assert "Code" in digest.active_apps

    def test_raw_summaries_populated(self):
        digest = parse_activity_response(SAMPLE_STRUCTURED)
        assert len(digest.raw_summaries) == 2

    def test_has_timestamp(self):
        digest = parse_activity_response(SAMPLE_STRUCTURED)
        assert digest.timestamp

    def test_not_empty(self):
        digest = parse_activity_response(SAMPLE_STRUCTURED)
        assert not digest.is_empty

    def test_projects_deduplication(self):
        """CoDA-GQA-L appears in both summaries — should not duplicate."""
        digest = parse_activity_response(SAMPLE_STRUCTURED)
        project_lower = [p.lower() for p in digest.projects]
        # Check no exact duplicates exist.
        assert len(project_lower) == len(set(project_lower))

    def test_metadata_stripped_from_focus(self):
        """'Automated Summary:' and 'Created:' should not appear in focus."""
        digest = parse_activity_response(SAMPLE_STRUCTURED)
        assert "Automated Summary" not in digest.recent_focus
        assert "Created:" not in digest.recent_focus

    def test_metadata_stripped_from_summary(self):
        digest = parse_activity_response(SAMPLE_STRUCTURED)
        assert "Automated Summary" not in digest.summary

    def test_metadata_stripped_from_raw_summaries(self):
        digest = parse_activity_response(SAMPLE_STRUCTURED)
        for raw in digest.raw_summaries:
            assert not raw.startswith("Automated Summary")

    def test_most_recent_at_populated(self):
        digest = parse_activity_response(SAMPLE_STRUCTURED)
        assert digest.most_recent_at == "2026-02-23 13:25:55.076568Z"


# =====================================================================
# Garbage app title filtering
# =====================================================================


class TestGarbageFiltering:
    def test_filters_could_not_retrieve(self):
        raw = json.dumps(
            {
                "summaries": [
                    {
                        "created": "2026-02-23",
                        "score": 0.5,
                        "combined_string": "## TL;DR\nWorking.\n",
                    }
                ],
                "events": [
                    {
                        "app_title": "[COULD NOT RETRIEVE APP TITLE]",
                        "window_title": "something",
                        "score": 1.0,
                    },
                    {
                        "app_title": "Code.exe",
                        "window_title": "file.py",
                        "score": 0.9,
                    },
                ],
            }
        )
        digest = parse_activity_response(raw)
        assert "[COULD NOT RETRIEVE APP TITLE]" not in digest.active_apps
        assert "Code" in digest.active_apps


# =====================================================================
# Plain text responses
# =====================================================================


class TestPlainTextParsing:
    def test_plain_text_summary(self):
        raw = "The user has been working on a Python project involving AI agents."
        digest = parse_activity_response(raw)
        assert "Python project" in digest.summary
        assert digest.projects == []  # No structured project extraction.

    def test_plain_text_not_empty(self):
        digest = parse_activity_response("Some activity text here.")
        assert not digest.is_empty

    def test_plain_text_with_project_sections(self):
        raw = (
            "## TL;DR\n"
            "Working on Agent-Nexus refactoring.\n"
            "### Core Tasks & Projects\n"
            "- MyProject Alpha: Major refactor in progress\n"
            "- Backend-Service: API cleanup\n"
        )
        digest = parse_activity_response(raw)
        assert "MyProject Alpha" in digest.projects
        assert "Backend-Service" in digest.projects
        assert "Agent-Nexus" in digest.summary or "refactoring" in digest.summary


# =====================================================================
# Edge cases
# =====================================================================


class TestEdgeCases:
    def test_empty_string(self):
        digest = parse_activity_response("")
        assert digest.is_empty

    def test_whitespace_only(self):
        digest = parse_activity_response("   \n  \n  ")
        assert digest.is_empty

    def test_malformed_json(self):
        digest = parse_activity_response("{invalid json here")
        # Falls back to plain text.
        assert digest.summary
        assert not digest.is_empty

    def test_json_without_summaries_key(self):
        raw = json.dumps({"other_key": "value"})
        digest = parse_activity_response(raw)
        # Falls back to plain text.
        assert digest.summary

    def test_empty_summaries_array(self):
        raw = json.dumps({"summaries": [], "events": []})
        digest = parse_activity_response(raw)
        assert digest.projects == []
        assert digest.active_apps == []

    def test_truncation_safety(self):
        raw = json.dumps(
            {
                "summaries": [
                    {
                        "created": "2026-01-01",
                        "score": 1.0,
                        "combined_string": "x" * 10000,
                    }
                ],
                "events": [],
            }
        )
        digest = parse_activity_response(raw)
        # raw_summaries should be capped.
        assert len(digest.raw_summaries[0]) <= 2000

    def test_events_without_app_title(self):
        raw = json.dumps(
            {
                "summaries": [
                    {
                        "created": "2026-01-01",
                        "score": 0.5,
                        "combined_string": "## TL;DR\nWorking on tests.\n",
                    }
                ],
                "events": [{"window_title": "terminal - bash", "score": 0.5}],
            }
        )
        digest = parse_activity_response(raw)
        assert "terminal - bash" in digest.active_apps


# =====================================================================
# ActivityDigest dataclass
# =====================================================================


class TestActivityDigest:
    def test_is_empty_default(self):
        digest = ActivityDigest()
        assert digest.is_empty

    def test_is_empty_with_summary(self):
        digest = ActivityDigest(summary="hello")
        assert not digest.is_empty

    def test_is_empty_with_projects(self):
        digest = ActivityDigest(projects=["MyProject"])
        assert not digest.is_empty

    def test_age_hours_none_when_no_timestamp(self):
        digest = ActivityDigest()
        assert digest.age_hours is None

    def test_age_description_format(self):
        digest = ActivityDigest(
            most_recent_at="2020-01-01T00:00:00+00:00",
        )
        # Very old — should show days.
        assert "d ago" in digest.age_description

    def test_is_stale_with_old_timestamp(self):
        digest = ActivityDigest(
            most_recent_at="2020-01-01T00:00:00+00:00",
        )
        assert digest.is_stale

    def test_not_stale_without_timestamp(self):
        digest = ActivityDigest()
        # No timestamp means unknown, not stale.
        assert not digest.is_stale
