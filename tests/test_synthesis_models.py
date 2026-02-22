def test_synthesis_models_import():
    """Core synthesis models should be importable."""
    from nexus.synthesis.models import (
        CapabilityCategory,
        RiskLevel,
        SynthesisStatus,
        TestCase,
        TestResult,
        TestSuite,
        ValidationIssue,
        ValidationResult,
        SynthesisAttempt,
    )
    assert SynthesisStatus.COMPLETE.value == "complete"


def test_test_case_creation():
    from nexus.synthesis.models import TestCase
    tc = TestCase(
        name="test_add",
        inputs={"a": 1, "b": 2},
        expected_output=3,
    )
    assert tc.name == "test_add"
    assert tc.expected_output == 3


def test_synthesis_attempt_pass_rate():
    from nexus.synthesis.models import SynthesisAttempt, SynthesisStatus
    attempt = SynthesisAttempt(
        id="test-1",
        requirement="add two numbers",
        status=SynthesisStatus.COMPLETE,
        tests_generated=5,
        tests_passed=4,
        total_tests=5,
    )
    assert attempt.test_pass_rate == 0.8
    assert not attempt.is_success  # needs all tests passing
