"""TDD Synthesis Engine for Agent Nexus.

Generates code through iterative test-driven development:
generate tests -> generate implementation -> run in sandbox -> refine.

Adapted from d:/Development/synthesis/synthesis/core/synthesis.py
"""

from __future__ import annotations

import ast
import json
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from nexus.synthesis.models import (
    CapabilityCategory,
    SynthesisAttempt,
    SynthesisStatus,
    TestCase,
    TestResult,
    TestSuite,
    ValidationResult,
)
from nexus.synthesis.sandbox import SandboxConfig, SandboxRuntime
from nexus.synthesis.validator import CodeValidator

log = logging.getLogger(__name__)


# -- LLM abstraction -------------------------------------------------------


class LLMResponse(BaseModel):
    content: str
    finish_reason: str
    tokens_used: int | None = None


class LLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse: ...


class NexusLLMAdapter(LLMProvider):
    """Wraps Nexus's OpenRouterClient for the TDD synthesizer."""

    def __init__(self, openrouter: Any, model: str | None = None) -> None:
        self._client = openrouter
        self._model = model

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        model = self._model or getattr(self._client, "default_model", None)
        if model is None:
            raise ValueError("No model specified for NexusLLMAdapter")

        resp = await self._client.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return LLMResponse(
            content=resp.content,
            finish_reason=resp.finish_reason or "stop",
            tokens_used=(resp.input_tokens or 0) + (resp.output_tokens or 0),
        )


# -- TDD Engine -------------------------------------------------------------


class TDDEngine:
    """Generates code through iterative test-driven development.

    Usage::

        engine = TDDEngine(llm=NexusLLMAdapter(openrouter, model="..."))
        result = await engine.synthesize("A function that adds two numbers")
    """

    def __init__(
        self,
        llm: LLMProvider,
        max_iterations: int = 5,
        sandbox_timeout: float = 30.0,
    ) -> None:
        self.llm = llm
        self.max_iterations = max_iterations
        self.validator = CodeValidator()
        self.sandbox = SandboxRuntime(SandboxConfig(timeout_seconds=sandbox_timeout))

    async def synthesize(
        self,
        requirement: str,
        category: CapabilityCategory | None = None,
    ) -> SynthesisAttempt:
        """Run full TDD synthesis loop for a requirement."""
        attempt = SynthesisAttempt(
            id=uuid.uuid4().hex[:12],
            requirement=requirement,
            category=category,
            status=SynthesisStatus.GENERATING_TESTS,
        )

        try:
            # Phase 1: Generate tests
            test_suite = await self._generate_tests(requirement)
            attempt.test_suite = test_suite
            attempt.tests_generated = len(test_suite.tests)
            attempt.total_tests = len(test_suite.tests)

            # Phase 2: Generate initial implementation
            attempt.status = SynthesisStatus.GENERATING_CODE
            code = await self._generate_implementation(requirement, test_suite)
            attempt.generated_code = code

            # Phase 3: Iterate until tests pass
            for i in range(self.max_iterations):
                attempt.iterations = i + 1
                attempt.status = SynthesisStatus.RUNNING_TESTS

                test_results, validation = await self._run_tests(code, test_suite)
                attempt.test_results = test_results
                attempt.tests_passed = sum(1 for t in test_results if t.passed)

                all_pass = all(t.passed for t in test_results) and validation.is_valid

                if all_pass:
                    attempt.status = SynthesisStatus.COMPLETE
                    attempt.generated_code = code
                    attempt.completed_at = datetime.now(timezone.utc)
                    break

                if i < self.max_iterations - 1:
                    attempt.status = SynthesisStatus.REFINING
                    code = await self._refine_implementation(
                        requirement, code, test_suite, test_results, validation, i,
                    )
                    attempt.generated_code = code
                else:
                    attempt.status = SynthesisStatus.FAILED
                    attempt.errors.append(
                        f"Failed after {self.max_iterations} iterations"
                    )

        except Exception as exc:
            attempt.status = SynthesisStatus.FAILED
            attempt.errors.append(str(exc))
            log.warning("TDD synthesis failed: %s", exc)

        return attempt

    async def _generate_tests(self, requirement: str) -> TestSuite:
        prompt = f"""Generate a JSON test suite for this requirement:

{requirement}

Return ONLY valid JSON in this format:
```json
{{
  "name": "test_suite_name",
  "tests": [
    {{
      "name": "test_case_name",
      "description": "what this tests",
      "inputs": {{"param1": "value1"}},
      "expected_output": "expected_value"
    }}
  ]
}}
```

Generate 3-5 test cases covering normal cases, edge cases, and error cases."""

        resp = await self.llm.complete(prompt, temperature=0.3)
        content = resp.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        try:
            data = json.loads(content.strip())
            tests = [TestCase(**t) for t in data.get("tests", [])]
        except (json.JSONDecodeError, Exception) as exc:
            raise ValueError(f"Failed to parse LLM test suite response: {exc}") from exc
        return TestSuite(
            name=data.get("name", "generated_tests"),
            tests=tests,
        )

    async def _generate_implementation(
        self, requirement: str, test_suite: TestSuite,
    ) -> str:
        tests_desc = "\n".join(
            f"- {t.name}: inputs={t.inputs}, expected={t.expected_output}"
            for t in test_suite.tests
        )
        prompt = f"""Write a Python function for this requirement:

{requirement}

It must pass these tests:
{tests_desc}

Return ONLY the function code in a ```python block. No imports of os, subprocess, sys, etc."""

        resp = await self.llm.complete(prompt, temperature=0.3)
        return self._extract_code(resp.content)

    async def _run_tests(
        self, code: str, test_suite: TestSuite,
    ) -> tuple[list[TestResult], ValidationResult]:
        validation = await self.validator.validate(code)
        results: list[TestResult] = []

        func_name = self._find_function_name(code)
        if func_name is None:
            for tc in test_suite.tests:
                results.append(TestResult(
                    test_case=tc,
                    passed=False,
                    error_message="No callable function found in generated code",
                    execution_time_ms=0.0,
                ))
            return results, validation

        for tc in test_suite.tests:
            try:
                exec_result = await self.sandbox.execute(
                    code=code,
                    function_name=func_name,
                    arguments=tc.inputs,
                )
                passed = (
                    exec_result.status.value == "success"
                    and exec_result.output == tc.expected_output
                )
                results.append(TestResult(
                    test_case=tc,
                    passed=passed,
                    actual_output=exec_result.output,
                    error_message=exec_result.error,
                    execution_time_ms=exec_result.execution_time_ms,
                ))
            except Exception as exc:
                results.append(TestResult(
                    test_case=tc,
                    passed=False,
                    error_message=str(exc),
                    execution_time_ms=0.0,
                ))

        return results, validation

    async def _refine_implementation(
        self,
        requirement: str,
        current_code: str,
        test_suite: TestSuite,
        test_results: list[TestResult],
        validation: ValidationResult,
        iteration: int,
    ) -> str:
        failures = [
            f"- {r.test_case.name}: expected={r.test_case.expected_output}, "
            f"got={r.actual_output}, error={r.error_message}"
            for r in test_results if not r.passed
        ]
        issues = [f"- {i.message}" for i in validation.issues] if not validation.is_valid else []

        failures_text = "\n".join(failures) if failures else "None"
        issues_text = "\n".join(issues) if issues else "None"

        prompt = f"""Fix this Python function (iteration {iteration + 1}):

Requirement: {requirement}

Current code:
```python
{current_code}
```

Test failures:
{failures_text}

Validation issues:
{issues_text}

Return ONLY the fixed function in a ```python block."""

        resp = await self.llm.complete(prompt, temperature=0.2)
        return self._extract_code(resp.content)

    @staticmethod
    def _extract_code(content: str) -> str:
        if "```python" in content:
            return content.split("```python")[1].split("```")[0].strip()
        if "```" in content:
            return content.split("```")[1].split("```")[0].strip()
        return content.strip()

    @staticmethod
    def _find_function_name(code: str) -> str | None:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return node.name
        except SyntaxError:
            pass
        return None
