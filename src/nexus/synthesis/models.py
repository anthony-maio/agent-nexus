"""
Core data models for the Synthesis TDD engine.

Ported from synthesis.core.models, trimmed to only the types needed
by Agent Nexus.
"""

from datetime import datetime
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class CapabilityCategory(str, Enum):
    """Categories of capabilities."""

    DATA_PROCESSING = "data_processing"
    COMPUTATION = "computation"
    INTEGRATION = "integration"
    ANALYSIS = "analysis"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"
    IO = "io"
    OTHER = "other"


class RiskLevel(str, Enum):
    """Risk assessment for capabilities."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SynthesisStatus(str, Enum):
    """Status of a synthesis attempt."""

    PENDING = "pending"
    GENERATING_TESTS = "generating_tests"
    GENERATING_CODE = "generating_code"
    RUNNING_TESTS = "running_tests"
    REFINING = "refining"
    COMPLETE = "complete"
    FAILED = "failed"


class ExecutionStatus(str, Enum):
    """Status of capability execution."""

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"


class TestCase(BaseModel):
    """A single test case."""

    name: str = Field(description="Test name")
    description: Optional[str] = Field(default=None)
    inputs: dict[str, Any] = Field(description="Input arguments")
    expected_output: Any = Field(description="Expected output")


class TestResult(BaseModel):
    """Result of running a test."""

    test_case: TestCase
    passed: bool
    actual_output: Optional[Any] = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0


class TestSuite(BaseModel):
    """Collection of test cases."""

    name: str
    tests: List[TestCase]


class ValidationIssue(BaseModel):
    """A validation issue found in code."""

    message: str = Field(description="Human-readable description")
    severity: str = Field(default="error", description="critical, high, medium, low, error")
    line: Optional[int] = Field(default=None, description="Line number if applicable")


class ValidationResult(BaseModel):
    """Result of code validation."""

    is_valid: bool
    issues: List[ValidationIssue] = Field(default_factory=list)


class SynthesisAttempt(BaseModel):
    """Record of a single synthesis attempt."""

    id: str = Field(default="", description="Unique attempt ID")
    requirement: str = Field(default="", description="What was requested")
    status: SynthesisStatus = Field(default=SynthesisStatus.PENDING)
    category: Optional[CapabilityCategory] = None

    # Generated artifacts
    generated_code: Optional[str] = None
    test_suite: Optional[TestSuite] = None

    # Metrics
    tests_generated: int = 0
    tests_passed: int = 0
    total_tests: int = 0
    iterations: int = 0

    # Results
    test_results: List[TestResult] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    # Timestamps
    completed_at: Optional[datetime] = None

    @property
    def test_pass_rate(self) -> float:
        """Percentage of tests passed."""
        if self.total_tests == 0:
            return 0.0
        return self.tests_passed / self.total_tests

    @property
    def is_success(self) -> bool:
        """Whether synthesis was successful (all tests must pass)."""
        return (
            self.status == SynthesisStatus.COMPLETE
            and self.total_tests > 0
            and self.tests_passed == self.total_tests
        )
