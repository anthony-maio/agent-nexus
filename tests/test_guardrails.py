"""Tests for anti-hallucination guardrails."""

from nexus.orchestrator.guardrails import (
    IdleLoopDetector,
    check_entity_grounding,
    validate_task_output,
)

# =====================================================================
# Entity Grounding
# =====================================================================


class TestEntityGrounding:
    def test_passes_grounded_action(self):
        actions = [{"type": "research", "description": "Analyze recent conversation patterns"}]
        result = check_entity_grounding(actions, "Recent conversation about patterns")
        assert len(result) == 1

    def test_drops_ungrounded_pr(self):
        actions = [{"type": "code", "description": "Review PR #487 for security issues"}]
        result = check_entity_grounding(actions, "No recent activity detected.")
        assert len(result) == 0

    def test_keeps_pr_when_in_state(self):
        actions = [{"type": "code", "description": "Review PR #487"}]
        result = check_entity_grounding(actions, "Discussion about PR #487 in the channel")
        assert len(result) == 1

    def test_drops_jira_ticket(self):
        actions = [{"type": "research", "description": "Investigate SEC-2847 security concern"}]
        result = check_entity_grounding(actions, "Epistemic stress level: 0.012")
        assert len(result) == 0

    def test_drops_fabricated_url(self):
        actions = [{"type": "research", "description": "Check https://github.com/example/repo"}]
        result = check_entity_grounding(actions, "No activity.")
        assert len(result) == 0

    def test_keeps_url_when_in_state(self):
        url = "https://github.com/example/repo"
        actions = [{"type": "research", "description": f"Check {url}"}]
        result = check_entity_grounding(actions, f"User shared {url}")
        assert len(result) == 1

    def test_mixed_actions_partial_filter(self):
        actions = [
            {"type": "research", "description": "Analyze conversation patterns"},
            {"type": "code", "description": "Fix PR #999 regression"},
        ]
        result = check_entity_grounding(actions, "Recent conversation about patterns.")
        assert len(result) == 1
        assert "patterns" in result[0]["description"]

    def test_empty_actions_unchanged(self):
        assert check_entity_grounding([], "some state") == []

    def test_drops_commit_sha(self):
        actions = [{"type": "code", "description": "Revert commit abc1234def"}]
        result = check_entity_grounding(actions, "No commits mentioned.")
        assert len(result) == 0


# =====================================================================
# Task Output Validation
# =====================================================================


class TestTaskOutputValidation:
    def test_valid_output_passes(self):
        is_valid, reason = validate_task_output(
            "The codebase uses an event-driven architecture with async handlers.",
            "Analyze the architecture",
        )
        assert is_valid
        assert reason == ""

    def test_medical_terminology_fails(self):
        is_valid, reason = validate_task_output(
            "The transcatheter aortic valve procedure requires cardiac "
            "catheter insertion through the arterial pathway.",
            "Summarize recent activity",
        )
        assert not is_valid
        assert "medical" in reason

    def test_two_medical_terms_passes(self):
        """Threshold is 3 — two medical words shouldn't trigger."""
        is_valid, _ = validate_task_output(
            "The patient data pipeline has a cardiac monitoring module.",
            "Analyze the pipeline",
        )
        assert is_valid

    def test_fabricated_pr_number_fails(self):
        is_valid, reason = validate_task_output(
            "PR #487 has been reviewed and approved by the security team.",
            "Analyze code changes",
        )
        assert not is_valid
        assert "fabricated" in reason

    def test_jira_ticket_fails(self):
        is_valid, reason = validate_task_output(
            "Ticket SEC-2847 has been escalated to the security lead.",
            "Check security status",
        )
        assert not is_valid
        assert "fabricated" in reason

    def test_word_count_filler_fails(self):
        is_valid, reason = validate_task_output(
            "The analysis shows strong patterns in the data. (498 words)",
            "Research the topic",
        )
        assert not is_valid
        assert "filler" in reason

    def test_confidence_without_evidence_fails(self):
        is_valid, reason = validate_task_output(
            "Confirmed: the deployment was successfully resolved. "
            "Action completed per protocol.",
            "Check system status",
        )
        assert not is_valid
        assert "confidence" in reason

    def test_single_confidence_word_passes(self):
        is_valid, _ = validate_task_output(
            "The pattern confirmed by the logs suggests a retry mechanism.",
            "Analyze logs",
        )
        assert is_valid

    def test_legal_terminology_fails(self):
        is_valid, reason = validate_task_output(
            "The plaintiff filed a subpoena for the defendant's deposition records.",
            "Summarize activity",
        )
        assert not is_valid
        assert "legal" in reason


# =====================================================================
# Idle-Loop Detection
# =====================================================================


def _make_state(has_human=False, has_activity=False):
    state = {"recent_messages": [], "activity": None}
    if has_human:
        state["recent_messages"] = [{"author": "human", "content": "hello"}]
    if has_activity:
        state["activity"] = "User edited file.py"
    return state


class TestIdleLoopDetector:
    def test_first_cycle_never_stale(self):
        detector = IdleLoopDetector()
        actions = [{"description": "Analyze conversation patterns in the codebase"}]
        assert not detector.check_cycle(actions, _make_state())

    def test_different_topics_not_stale(self):
        detector = IdleLoopDetector()
        a1 = [{"description": "Analyze conversation patterns in the codebase"}]
        a2 = [{"description": "Research quantum computing fundamentals deeply"}]
        detector.check_cycle(a1, _make_state())
        assert not detector.check_cycle(a2, _make_state())
        assert detector.staleness_counter == 0

    def test_same_topic_increments_counter(self):
        detector = IdleLoopDetector()
        actions = [{"description": "Analyze conversation patterns in the codebase"}]
        detector.check_cycle(actions, _make_state())
        detector.check_cycle(actions, _make_state())
        assert detector.staleness_counter == 1

    def test_three_stale_cycles_suppresses(self):
        detector = IdleLoopDetector(stale_cycle_limit=3)
        actions = [{"description": "Analyze conversation patterns in the codebase"}]
        state = _make_state()
        detector.check_cycle(actions, state)  # baseline
        detector.check_cycle(actions, state)  # stale 1
        detector.check_cycle(actions, state)  # stale 2
        assert detector.check_cycle(actions, state)  # stale 3 -> suppress
        assert detector.is_suppressed

    def test_human_message_resets(self):
        detector = IdleLoopDetector(stale_cycle_limit=3)
        actions = [{"description": "Analyze conversation patterns in the codebase"}]
        detector.check_cycle(actions, _make_state())
        detector.check_cycle(actions, _make_state())
        assert detector.staleness_counter == 1
        # Human arrives
        detector.check_cycle(actions, _make_state(has_human=True))
        assert detector.staleness_counter == 0
        assert not detector.is_suppressed

    def test_activity_resets(self):
        detector = IdleLoopDetector(stale_cycle_limit=3)
        actions = [{"description": "Analyze conversation patterns in the codebase"}]
        detector.check_cycle(actions, _make_state())
        detector.check_cycle(actions, _make_state())
        assert detector.staleness_counter == 1
        # PiecesOS activity detected
        detector.check_cycle(actions, _make_state(has_activity=True))
        assert detector.staleness_counter == 0

    def test_empty_actions_not_stale(self):
        detector = IdleLoopDetector()
        detector.check_cycle([], _make_state())
        detector.check_cycle([], _make_state())
        assert detector.staleness_counter == 0

    def test_reset_clears_state(self):
        detector = IdleLoopDetector(stale_cycle_limit=2)
        actions = [{"description": "Analyze conversation patterns in the codebase"}]
        detector.check_cycle(actions, _make_state())
        detector.check_cycle(actions, _make_state())
        assert detector.staleness_counter == 1
        detector.reset()
        assert detector.staleness_counter == 0
        assert not detector.is_suppressed
