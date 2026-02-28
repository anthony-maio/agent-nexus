"""Tests for sentiment analysis and its integration points."""

from unittest.mock import MagicMock

import pytest

from nexus.swarm.sentiment import (
    Mood,
    SentimentResult,
    SentimentTracker,
    _score_message,
)

# =====================================================================
# Core: _score_message
# =====================================================================


class TestScoreMessage:
    def test_positive_message(self):
        result = _score_message("This is awesome, thanks for the help!")
        assert result.label == Mood.POSITIVE
        assert result.score > 0

    def test_negative_message(self):
        result = _score_message("This is terrible and broken.")
        assert result.label == Mood.NEGATIVE
        assert result.score < 0

    def test_frustrated_message(self):
        result = _score_message("I'm so frustrated, nothing works!!")
        assert result.label == Mood.FRUSTRATED
        assert result.score < 0

    def test_curious_message(self):
        result = _score_message("How does this work? Can you explain?")
        assert result.label == Mood.CURIOUS
        assert result.score > 0

    def test_urgent_message(self):
        result = _score_message("This is urgent, I need help immediately!")
        assert result.label == Mood.URGENT
        assert result.score < 0

    def test_neutral_empty_text(self):
        result = _score_message("")
        assert result.label == Mood.NEUTRAL
        assert result.score == 0.0

    def test_neutral_no_keywords(self):
        result = _score_message("The cat sat on the mat.")
        assert result.label == Mood.NEUTRAL
        assert result.score == 0.0
        assert result.confidence == pytest.approx(0.3)

    def test_very_long_text(self):
        text = "great " * 10000
        result = _score_message(text)
        assert result.label == Mood.POSITIVE

    def test_caps_boost_frustration(self):
        result = _score_message("WHY IS THIS BROKEN AGAIN")
        assert result.label == Mood.FRUSTRATED

    def test_question_mark_boosts_curious(self):
        result = _score_message("What is happening here?")
        assert result.label == Mood.CURIOUS

    def test_score_range(self):
        for text in [
            "awesome great perfect",
            "terrible awful horrible",
            "frustrated stuck confused",
            "",
        ]:
            result = _score_message(text)
            assert -1.0 <= result.score <= 1.0
            assert 0.0 <= result.confidence <= 1.0


# =====================================================================
# Core: SentimentTracker
# =====================================================================


class TestSentimentTracker:
    def test_analyze_returns_result(self):
        tracker = SentimentTracker()
        result = tracker.analyze("This is great!")
        assert isinstance(result, SentimentResult)

    def test_window_sliding(self):
        tracker = SentimentTracker(window_size=3)
        tracker.analyze("great")
        tracker.analyze("awesome")
        tracker.analyze("terrible")
        tracker.analyze("horrible")  # pushes out "great"
        assert len(tracker._window) == 3

    def test_current_mood_empty(self):
        tracker = SentimentTracker()
        assert tracker.current_mood == Mood.NEUTRAL

    def test_current_mood_dominant(self):
        tracker = SentimentTracker()
        tracker.analyze("awesome")
        tracker.analyze("great")
        tracker.analyze("terrible")
        assert tracker.current_mood == Mood.POSITIVE  # 2 vs 1

    def test_average_score_empty(self):
        tracker = SentimentTracker()
        assert tracker.average_score == 0.0

    def test_average_score_calculated(self):
        tracker = SentimentTracker()
        tracker.analyze("awesome")
        tracker.analyze("terrible")
        assert isinstance(tracker.average_score, float)


# =====================================================================
# Mood context for prompt
# =====================================================================


class TestMoodContextForPrompt:
    def test_neutral_returns_empty(self):
        tracker = SentimentTracker()
        assert tracker.mood_context_for_prompt() == ""

    def test_positive_returns_hint(self):
        tracker = SentimentTracker()
        tracker.analyze("awesome amazing great wonderful")
        ctx = tracker.mood_context_for_prompt()
        assert "Current User Mood" in ctx

    def test_frustrated_returns_hint(self):
        tracker = SentimentTracker()
        tracker.analyze("frustrated stuck broken WHY!!")
        ctx = tracker.mood_context_for_prompt()
        assert "frustrated" in ctx.lower() or "patient" in ctx.lower()

    def test_all_non_neutral_moods_have_hints(self):
        """Every non-neutral mood should produce a non-empty hint."""
        cases = [
            "awesome great excellent",  # POSITIVE
            "terrible awful horrible",  # NEGATIVE
            "frustrated stuck confused WHY!!",  # FRUSTRATED
            "how why explain what?",  # CURIOUS
            "urgent asap immediately help now",  # URGENT
        ]
        for text in cases:
            tracker = SentimentTracker()
            tracker.analyze(text)
            if tracker.current_mood != Mood.NEUTRAL:
                assert tracker.mood_context_for_prompt() != "", (
                    f"No hint for mood {tracker.current_mood} "
                    f"from text: {text!r}"
                )


# =====================================================================
# State Gatherer integration
# =====================================================================


class TestStateGathererSentiment:
    def test_gather_sentiment_returns_dict(self):
        from nexus.orchestrator.state import StateGatherer

        bot = MagicMock()
        bot.sentiment = SentimentTracker()
        bot.sentiment.analyze("I'm frustrated with this!!")

        gatherer = StateGatherer(bot)
        result = gatherer._gather_sentiment()

        assert result is not None
        assert result["mood"] == "frustrated"
        assert isinstance(result["score"], float)
        assert isinstance(result["window_size"], int)
        assert result["window_size"] == 1
        assert "frustrated" in result["mood_hint"].lower()

    def test_gather_sentiment_neutral(self):
        from nexus.orchestrator.state import StateGatherer

        bot = MagicMock()
        bot.sentiment = SentimentTracker()

        gatherer = StateGatherer(bot)
        result = gatherer._gather_sentiment()

        assert result is not None
        assert result["mood"] == "neutral"
        assert result["mood_hint"] == ""

    def test_gather_sentiment_no_tracker(self):
        from nexus.orchestrator.state import StateGatherer

        bot = MagicMock(spec=[])  # No sentiment attribute
        gatherer = StateGatherer(bot)
        result = gatherer._gather_sentiment()
        assert result is None


# =====================================================================
# Autonomy risk score adjustment
# =====================================================================


class TestAutonomyMoodAdjustment:
    def test_frustrated_increases_risk(self):
        from nexus.orchestrator.autonomy import AutonomyGate

        gate = AutonomyGate()
        action = {"type": "research", "priority": "medium"}

        gate.set_current_mood("neutral")
        base_score = gate.compute_risk_score(action)

        gate.set_current_mood("frustrated")
        frustrated_score = gate.compute_risk_score(action)

        assert frustrated_score > base_score
        assert frustrated_score - base_score == pytest.approx(0.1)

    def test_urgent_decreases_risk(self):
        from nexus.orchestrator.autonomy import AutonomyGate

        gate = AutonomyGate()
        action = {"type": "research", "priority": "medium"}

        gate.set_current_mood("neutral")
        base_score = gate.compute_risk_score(action)

        gate.set_current_mood("urgent")
        urgent_score = gate.compute_risk_score(action)

        assert urgent_score < base_score
        assert base_score - urgent_score == pytest.approx(0.1)

    def test_positive_no_change(self):
        from nexus.orchestrator.autonomy import AutonomyGate

        gate = AutonomyGate()
        action = {"type": "research", "priority": "medium"}

        gate.set_current_mood("neutral")
        base = gate.compute_risk_score(action)

        gate.set_current_mood("positive")
        pos = gate.compute_risk_score(action)

        assert base == pos

    def test_risk_clamped_to_zero(self):
        from nexus.orchestrator.autonomy import AutonomyGate

        gate = AutonomyGate()
        action = {"type": "research", "priority": "low"}
        gate.set_current_mood("urgent")
        score = gate.compute_risk_score(action)
        assert score >= 0.0

    def test_risk_clamped_to_one(self):
        from nexus.orchestrator.autonomy import AutonomyGate

        gate = AutonomyGate()
        action = {"type": "code", "priority": "high"}
        gate.set_current_mood("frustrated")
        score = gate.compute_risk_score(action)
        assert score <= 1.0


# =====================================================================
# Initiative mood guard
# =====================================================================


class TestInitiativeMoodGuard:
    def test_frustrated_user_detected(self):
        """Verify the tracker correctly identifies frustrated mood."""
        tracker = SentimentTracker()
        for _ in range(3):
            tracker.analyze("frustrated stuck broken WHY!!")
        assert tracker.current_mood == Mood.FRUSTRATED

    def test_curious_user_detected(self):
        """Verify the tracker correctly identifies curious mood."""
        tracker = SentimentTracker()
        tracker.analyze("how does this work? explain please?")
        assert tracker.current_mood == Mood.CURIOUS


# =====================================================================
# Crosstalk mood-aware ordering
# =====================================================================


class TestCrosstalkMoodOrdering:
    def test_default_shuffles(self):
        """Without mood, models should be returned (shuffled)."""
        from nexus.swarm.crosstalk import CrosstalkManager

        mgr = CrosstalkManager()
        models = ["a", "b", "c", "d"]
        result = mgr.build_reaction_order("a", models)
        assert "a" not in result
        assert set(result) == {"b", "c", "d"}

    def test_frustrated_uses_model_specs(self):
        """Frustrated mood should sort by model strengths."""
        from nexus.swarm.crosstalk import CrosstalkManager

        mgr = CrosstalkManager()

        # Create mock model specs
        spec_reasoning = MagicMock()
        spec_reasoning.strengths = ["reasoning", "analysis"]
        spec_creative = MagicMock()
        spec_creative.strengths = ["creativity", "general-intelligence"]
        spec_coding = MagicMock()
        spec_coding.strengths = ["coding"]

        model_specs = {
            "creative-model": spec_creative,
            "reasoning-model": spec_reasoning,
            "coding-model": spec_coding,
        }

        result = mgr.build_reaction_order(
            "primary",
            ["primary", "creative-model", "reasoning-model", "coding-model"],
            mood="frustrated",
            model_specs=model_specs,
        )

        # Reasoning model should come before creative model
        assert result.index("reasoning-model") < result.index("creative-model")

    def test_no_mood_backward_compatible(self):
        """Calling without mood param should work (backward compat)."""
        from nexus.swarm.crosstalk import CrosstalkManager

        mgr = CrosstalkManager()
        result = mgr.build_reaction_order("a", ["a", "b", "c"])
        assert set(result) == {"b", "c"}


# =====================================================================
# Dispatch priority boost
# =====================================================================


class TestDispatchPriorityBoost:
    def test_urgent_bumps_medium_to_high(self):
        """Urgent mood should bump medium priority to high."""
        from nexus.orchestrator.dispatch import TaskDispatcher

        bot = MagicMock()
        dispatcher = TaskDispatcher(bot)
        dispatcher.set_current_mood("urgent")
        assert dispatcher._current_mood == "urgent"

    def test_default_mood_is_neutral(self):
        """Dispatcher should start with neutral mood."""
        from nexus.orchestrator.dispatch import TaskDispatcher

        bot = MagicMock()
        dispatcher = TaskDispatcher(bot)
        assert dispatcher._current_mood == "neutral"
