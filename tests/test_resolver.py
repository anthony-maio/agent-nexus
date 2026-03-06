"""Tests for MRA ContradictionClassifier, ResolutionEngine, and Pipeline."""


from continuity_core.mra.pipeline import MRAResolutionPipeline
from continuity_core.mra.resolver import (
    ContradictionClassifier,
    ContradictionType,
    ResolutionEngine,
)
from continuity_core.mra.stress import StressResult

# ── Classifier ──────────────────────────────────────────────────────


class TestContradictionClassifier:

    def setup_method(self):
        self.clf = ContradictionClassifier()

    def test_truncation_substring(self):
        full = "The architecture uses a two-tier model system with swarm intelligence"
        truncated = "The architecture uses a two-tier model"
        result = self.clf.classify(truncated, full, score=0.6, similarity=0.8)
        assert result.type == ContradictionType.TRUNCATION

    def test_truncation_high_overlap_length_ratio(self):
        short = "Redis cache handles sessions"
        long = (
            "Redis cache handles sessions and also provides pub/sub "
            "messaging for inter-service communication with low latency"
        )
        result = self.clf.classify(short, long, score=0.5, similarity=0.7)
        assert result.type == ContradictionType.TRUNCATION

    def test_truncation_marker(self):
        s1 = "The system processes requests through a pipeline that..."
        s2 = (
            "The system processes requests through a pipeline that handles auth, "
            "validation, routing, and response formatting"
        )
        result = self.clf.classify(s1, s2, score=0.4, similarity=0.7)
        assert result.type == ContradictionType.TRUNCATION

    def test_duplicate_high_similarity_low_contradiction(self):
        s1 = "Python is a dynamically typed language"
        s2 = "Python is dynamically typed"
        result = self.clf.classify(s1, s2, score=0.2, similarity=0.92)
        assert result.type == ContradictionType.DUPLICATE

    def test_outdated_with_timestamps(self):
        s1 = "The API uses REST endpoints"
        s2 = "The API now uses GraphQL endpoints"
        meta = {"timestamp_s1": 1000.0, "timestamp_s2": 5000.0}
        result = self.clf.classify(s1, s2, score=0.6, similarity=0.5, metadata=meta)
        assert result.type == ContradictionType.OUTDATED

    def test_outdated_requires_time_diff(self):
        """Timestamps too close → not outdated."""
        s1 = "Use approach A"
        s2 = "Use approach B"
        meta = {"timestamp_s1": 1000.0, "timestamp_s2": 1500.0}
        result = self.clf.classify(s1, s2, score=0.7, similarity=0.3, metadata=meta)
        assert result.type != ContradictionType.OUTDATED

    def test_genuine_high_score(self):
        s1 = "The system should always validate input"
        s2 = "Input validation is not needed for internal APIs"
        result = self.clf.classify(s1, s2, score=0.75, similarity=0.4, metadata={})
        assert result.type == ContradictionType.GENUINE
        assert result.confidence == 0.75

    def test_low_score_treated_as_noise(self):
        """Below genuine threshold → classified as duplicate/noise."""
        s1 = "Foo"
        s2 = "Bar"
        result = self.clf.classify(s1, s2, score=0.3, similarity=0.2, metadata={})
        assert result.type == ContradictionType.DUPLICATE

    def test_no_metadata_still_works(self):
        s1 = "X is true"
        s2 = "X is not true"
        result = self.clf.classify(s1, s2, score=0.7, similarity=0.5)
        assert result.type == ContradictionType.GENUINE


# ── Resolution Engine ───────────────────────────────────────────────


class TestResolutionEngine:

    def setup_method(self):
        self.clf = ContradictionClassifier()
        self.engine = ResolutionEngine()

    def test_truncation_keeps_longer(self):
        short = "The architecture uses a two-tier model"
        full = "The architecture uses a two-tier model system with swarm intelligence"
        classified = self.clf.classify(short, full, score=0.5, similarity=0.8)
        result = self.engine.resolve(classified)
        assert result.resolved is True
        assert result.action_taken == "superseded_shorter"
        assert len(result.graph_ops) == 1
        assert result.graph_ops[0]["op"] == "supersede"

    def test_duplicate_merged(self):
        classified = self.clf.classify(
            "Redis handles caching for the application layer",
            "Redis provides caching at the application layer",
            score=0.2, similarity=0.92,
        )
        result = self.engine.resolve(classified)
        assert result.resolved is True
        assert result.action_taken == "merged"

    def test_outdated_supersedes_older(self):
        classified = self.clf.classify(
            "Use REST API for communication",
            "Use GraphQL API for communication",
            score=0.6, similarity=0.5,
            metadata={"timestamp_s1": 1000.0, "timestamp_s2": 5000.0},
        )
        result = self.engine.resolve(classified)
        assert result.resolved is True
        assert result.action_taken == "superseded_older"

    def test_genuine_escalated(self):
        classified = self.clf.classify(
            "Always use strict mode",
            "Strict mode is not needed in production",
            score=0.8, similarity=0.4,
        )
        result = self.engine.resolve(classified)
        assert result.resolved is False
        assert result.action_taken == "escalated"
        assert len(result.graph_ops) == 0


# ── Pipeline ────────────────────────────────────────────────────────


class TestMRAResolutionPipeline:

    def test_empty_stress(self):
        stress = StressResult(
            s_omega=0.0, d_log=0.0, d_sem=0.0, v_top=0.0,
        )
        pipeline = MRAResolutionPipeline()
        report = pipeline.run(stress)
        assert report.total_processed == 0
        assert report.auto_resolved == 0
        assert report.escalated == 0
        assert report.summary == ""

    def test_mixed_contradictions(self):
        stress = StressResult(
            s_omega=0.5, d_log=0.5, d_sem=0.0, v_top=0.0,
            contradictions=[
                # Truncation artifact
                ("short text", "short text with more details added on", 0.6),
                # Genuine
                ("Always validate", "Never validate internal calls", 0.8),
            ],
            deep_tensions=[],
        )
        pipeline = MRAResolutionPipeline()
        report = pipeline.run(stress)

        assert report.total_processed == 2
        assert report.auto_resolved == 1
        assert report.escalated == 1
        assert report.has_escalations
        assert "auto-resolved" in report.summary
        assert "escalated" in report.summary

    def test_deep_tensions_processed_first(self):
        stress = StressResult(
            s_omega=0.7, d_log=0.7, d_sem=0.0, v_top=0.0,
            contradictions=[
                ("A is true", "A is false", 0.9),
            ],
            deep_tensions=[
                ("A is true", "A is false", 0.9, 0.7),
            ],
        )
        pipeline = MRAResolutionPipeline()
        report = pipeline.run(stress)

        # Same pair in both lists — should only be processed once
        assert report.total_processed == 1

    def test_summary_readable(self):
        stress = StressResult(
            s_omega=0.4, d_log=0.4, d_sem=0.0, v_top=0.0,
            contradictions=[
                ("short", "short with more content appended to it here", 0.5),
                ("Python is good", "Python is great for scripting", 0.2),
            ],
            deep_tensions=[],
        )
        pipeline = MRAResolutionPipeline()
        report = pipeline.run(stress)

        assert "Knowledge maintenance" in report.summary
        assert report.auto_resolved == 2

    def test_graph_ops_collected(self):
        stress = StressResult(
            s_omega=0.5, d_log=0.5, d_sem=0.0, v_top=0.0,
            contradictions=[
                ("short", "short with a lot more content and details", 0.5),
            ],
            deep_tensions=[],
        )
        pipeline = MRAResolutionPipeline()
        report = pipeline.run(stress)

        assert len(report.graph_ops) >= 1
        assert report.graph_ops[0]["op"] == "supersede"

    def test_void_info_included(self):
        from continuity_core.mra.voids import VoidReport

        stress = StressResult(
            s_omega=0.1, d_log=0.1, d_sem=0.0, v_top=0.0,
        )
        voids = VoidReport(
            void_pairs=[("cluster_a", "cluster_b")],
            questions=["How do A and B relate?"],
        )
        pipeline = MRAResolutionPipeline()
        report = pipeline.run(stress, voids=voids)

        assert report.void_count == 1
        assert "knowledge gap" in report.summary.lower()
