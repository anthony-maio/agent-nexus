"""
Code validation and safety analysis.

Validates generated code before execution using AST analysis.
Ported from synthesis.core.validator.
"""

import ast
import re
from typing import List, Set

from nexus.synthesis.models import ValidationIssue, ValidationResult


class CodeValidator:
    """Validates Python code for safety and quality."""

    # Libraries that are forbidden for security reasons
    FORBIDDEN_LIBRARIES: Set[str] = {
        "os",
        "subprocess",
        "sys",
        "shutil",
        "pathlib",
        "importlib",
        "__import__",
        "socket",
        "socketserver",
        "threading",
        "multiprocessing",
        "pickle",
        "shelve",
        "marshal",
        "pty",
        "ctypes",
        "cffi",
    }

    # Dangerous built-in functions that must be blocked
    FORBIDDEN_FUNCTIONS: Set[str] = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "globals",
        "locals",
        "getattr",
        "setattr",
        "delattr",
        "breakpoint",
    }

    def __init__(self) -> None:
        """Initialize the validator."""
        self._issues: List[ValidationIssue] = []

    async def validate(self, code: str, strict_mode: bool = False) -> ValidationResult:
        """
        Validate Python code for safety.

        Args:
            code: Python code to validate.
            strict_mode: If True, be more restrictive about what is allowed.

        Returns:
            ValidationResult with findings.
        """
        self._issues = []

        # Check syntax first
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            self._issues.append(
                ValidationIssue(
                    message=f"Syntax error: {e.msg}",
                    severity="critical",
                    line=e.lineno,
                )
            )
            return self._build_result()

        # AST-based checks
        self._check_imports(tree)
        self._check_dangerous_calls(tree)
        self._check_patterns(code)

        return self._build_result()

    def _check_imports(self, tree: ast.AST) -> None:
        """Check for dangerous imports via AST walking."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module in self.FORBIDDEN_LIBRARIES:
                        self._issues.append(
                            ValidationIssue(
                                message=f"Forbidden library: {module}",
                                severity="critical",
                                line=getattr(node, "lineno", None),
                            )
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    if module in self.FORBIDDEN_LIBRARIES:
                        self._issues.append(
                            ValidationIssue(
                                message=f"Forbidden library: {module}",
                                severity="critical",
                                line=getattr(node, "lineno", None),
                            )
                        )

    def _check_dangerous_calls(self, tree: ast.AST) -> None:
        """Check for dangerous function calls via AST walking."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.FORBIDDEN_FUNCTIONS:
                        self._issues.append(
                            ValidationIssue(
                                message=f"Dangerous function call: {node.func.id}()",
                                severity="critical",
                                line=getattr(node, "lineno", None),
                            )
                        )

    def _check_patterns(self, code: str) -> None:
        """Check for dangerous patterns via regex."""
        if re.search(r"\b(eval|exec)\s*\(", code):
            self._issues.append(
                ValidationIssue(
                    message="Dynamic code execution detected (eval/exec)",
                    severity="critical",
                )
            )
        if re.search(r"os\.(system|popen|spawn)", code):
            self._issues.append(
                ValidationIssue(
                    message="Shell command execution detected",
                    severity="critical",
                )
            )

    def _build_result(self) -> ValidationResult:
        """Build validation result from collected issues."""
        critical_issues = [i for i in self._issues if i.severity == "critical"]
        return ValidationResult(
            is_valid=len(critical_issues) == 0,
            issues=self._issues,
        )
