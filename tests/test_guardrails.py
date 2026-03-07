"""Tests for anti-hallucination guardrails."""

from nexus.orchestrator.guardrails import (
    FailureCircuitBreaker,
    IdleLoopDetector,
    check_capability,
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
            "Confirmed: the deployment was successfully resolved. Action completed per protocol.",
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

    def test_fabricated_infrastructure_fails(self):
        is_valid, reason = validate_task_output(
            "I accessed /app/agents/task_agent/manifest.json and found "
            "the config at /config/agent_config.yaml. The PID: 12345 "
            "shows the agent is running.",
            "Extract config files",
        )
        assert not is_valid
        assert "infrastructure" in reason

    def test_fabricated_shell_commands_fails(self):
        is_valid, reason = validate_task_output(
            "Running cat /etc/passwd and grep -r 'secret' /app/config "
            "reveals the deployment structure.",
            "Analyze system",
        )
        assert not is_valid
        assert "infrastructure" in reason

    def test_single_path_mention_passes(self):
        """One path reference alone shouldn't trigger — need 2+ signals."""
        is_valid, _ = validate_task_output(
            "The config is typically stored at /etc/nginx/nginx.conf on Linux systems.",
            "Research nginx config",
        )
        assert is_valid

    def test_legal_terminology_fails(self):
        is_valid, reason = validate_task_output(
            "The plaintiff filed a subpoena for the defendant's deposition records.",
            "Summarize activity",
        )
        assert not is_valid
        assert "legal" in reason

    def test_assigned_to_task_agents_passes(self):
        """'assigned to task agents' is normal English, not a fabricated name."""
        is_valid, _ = validate_task_output(
            "The work was assigned to task agents for further processing.",
            "Summarize task dispatch strategy",
        )
        assert is_valid

    def test_create_evidence_passes(self):
        """'Create evidence' is normal English, not a SQL CREATE statement."""
        is_valid, _ = validate_task_output(
            "Create evidence-based documentation of the approach. "
            "Match source material with identified patterns.",
            "Document findings",
        )
        assert is_valid

    def test_actual_sql_still_fails(self):
        """Real SQL syntax should still be caught."""
        is_valid, reason = validate_task_output(
            "I ran SELECT * FROM users and also ran INSERT INTO logs the findings.",
            "Query data",
        )
        assert not is_valid
        assert "infrastructure" in reason

    def test_fabricated_personnel_with_full_name_fails(self):
        """'reviewed by John Smith' with a full name is fabricated."""
        is_valid, reason = validate_task_output(
            "The code was reviewed by John Smith and approved by Jane Doe.",
            "Check review status",
        )
        assert not is_valid
        assert "fabricated" in reason


# =====================================================================
# Idle-Loop Detection
# =====================================================================


def _make_state(has_human=False, has_activity=False, stale_activity=False):
    state = {"recent_messages": [], "activity": None}
    if has_human:
        state["recent_messages"] = [{"author": "human", "content": "hello"}]
    if has_activity:
        from nexus.integrations.pieces import ActivityDigest

        digest = ActivityDigest(summary="User edited file.py")
        if stale_activity:
            digest.most_recent_at = "2020-01-01T00:00:00+00:00"
        state["activity"] = digest
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

    def test_activity_does_not_reset(self):
        """PiecesOS activity should NOT reset idle detection.

        An active IDE session would permanently prevent loop detection
        otherwise.  Only human messages reset the staleness counter.
        """
        detector = IdleLoopDetector(stale_cycle_limit=3)
        actions = [{"description": "Analyze conversation patterns in the codebase"}]
        detector.check_cycle(actions, _make_state())
        detector.check_cycle(actions, _make_state())
        assert detector.staleness_counter == 1
        # PiecesOS activity should NOT reset
        detector.check_cycle(actions, _make_state(has_activity=True))
        assert detector.staleness_counter == 2

    def test_stale_activity_does_not_reset(self):
        detector = IdleLoopDetector(stale_cycle_limit=3)
        actions = [{"description": "Analyze conversation patterns in the codebase"}]
        detector.check_cycle(actions, _make_state())
        detector.check_cycle(actions, _make_state())
        assert detector.staleness_counter == 1
        # Stale PiecesOS activity should NOT reset
        detector.check_cycle(
            actions,
            _make_state(has_activity=True, stale_activity=True),
        )
        assert detector.staleness_counter == 2

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


# =====================================================================
# Capability Filtering
# =====================================================================


class TestCapabilityFiltering:
    def test_passes_normal_tasks(self):
        actions = [
            {"type": "research", "description": "Summarize recent security trends"},
            {"type": "analyze", "description": "Evaluate the conversation patterns"},
        ]
        result = check_capability(actions)
        assert len(result) == 2

    def test_drops_file_access_tasks(self):
        actions = [
            {"type": "analyze", "description": "Analyze code in tools/extraction.py"},
        ]
        result = check_capability(actions)
        assert len(result) == 0

    def test_code_type_passes_through(self):
        """Code-type actions bypass capability filter (routed to TDD engine)."""
        actions = [
            {"type": "code", "description": "Analyze code in tools/extraction.py"},
        ]
        result = check_capability(actions)
        assert len(result) == 1

    def test_drops_file_inspect_task(self):
        actions = [
            {"type": "extract", "description": "Read the configuration from config.yaml"},
        ]
        result = check_capability(actions)
        assert len(result) == 0

    def test_drops_database_inspection(self):
        actions = [
            {"type": "analyze", "description": "Query the Neo4j database for recent events"},
        ]
        result = check_capability(actions)
        assert len(result) == 0

    def test_drops_log_inspection(self):
        actions = [
            {"type": "extract", "description": "Check the logs for error patterns"},
        ]
        result = check_capability(actions)
        assert len(result) == 0

    def test_drops_command_execution(self):
        actions = [
            {"type": "analyze", "description": "Run the test suite and report failures"},
        ]
        result = check_capability(actions)
        assert len(result) == 0

    def test_drops_self_diagnosis(self):
        actions = [
            {"type": "analyze", "description": "Diagnose the bug in the extraction tool"},
        ]
        result = check_capability(actions)
        assert len(result) == 0

    def test_drops_root_cause_analysis(self):
        actions = [
            {"type": "research", "description": "Root cause analysis of empty results"},
        ]
        result = check_capability(actions)
        assert len(result) == 0

    def test_drops_trace_data_flow(self):
        actions = [
            {"type": "analyze", "description": "Trace the data flow from extraction to reasoning"},
        ]
        result = check_capability(actions)
        assert len(result) == 0

    def test_drops_cross_reference(self):
        actions = [
            {"type": "analyze", "description": "Cross-reference the two audit findings"},
        ]
        result = check_capability(actions)
        assert len(result) == 0

    def test_mixed_actions_partial_filter(self):
        actions = [
            {"type": "research", "description": "Summarize Python best practices for async"},
            {"type": "extract", "description": "Inspect the extraction code in parser.py"},
            {"type": "analyze", "description": "Evaluate user's architectural approach"},
        ]
        result = check_capability(actions)
        assert len(result) == 2
        descriptions = [a["description"] for a in result]
        assert any("best practices" in d for d in descriptions)
        assert any("architectural" in d for d in descriptions)

    def test_empty_actions_unchanged(self):
        assert check_capability([]) == []


# =====================================================================
# Failure Circuit Breaker
# =====================================================================


class TestFailureCircuitBreaker:
    def test_not_tripped_initially(self):
        breaker = FailureCircuitBreaker()
        assert not breaker.is_tripped
        assert not breaker.should_suppress()

    def test_trips_after_threshold(self):
        breaker = FailureCircuitBreaker(failure_threshold=3, cooldown_cycles=2)
        breaker.record_result(False)
        breaker.record_result(False)
        assert not breaker.is_tripped
        breaker.record_result(False)
        assert breaker.is_tripped

    def test_success_resets_counter(self):
        breaker = FailureCircuitBreaker(failure_threshold=3)
        breaker.record_result(False)
        breaker.record_result(False)
        breaker.record_result(True)  # Reset
        assert breaker.consecutive_failures == 0
        assert not breaker.is_tripped

    def test_cooldown_suppresses_then_resumes(self):
        breaker = FailureCircuitBreaker(failure_threshold=2, cooldown_cycles=2)
        breaker.record_result(False)
        breaker.record_result(False)
        assert breaker.is_tripped

        assert breaker.should_suppress()  # cooldown 1
        assert breaker.should_suppress()  # cooldown 0
        assert not breaker.should_suppress()  # expired -- resumed

    def test_reset_clears_all(self):
        breaker = FailureCircuitBreaker(failure_threshold=2)
        breaker.record_result(False)
        breaker.record_result(False)
        assert breaker.is_tripped
        breaker.reset()
        assert not breaker.is_tripped
        assert breaker.consecutive_failures == 0
