#!/usr/bin/env bash
set -euo pipefail

base_ref="${1:-origin/main}"

if ! git rev-parse --verify "$base_ref" >/dev/null 2>&1; then
  echo "Base ref not found: $base_ref"
  exit 1
fi

mapfile -t python_files < <(
  git diff --name-only --diff-filter=ACMRTUXB "$base_ref"...HEAD -- "*.py"
)

if [[ ${#python_files[@]} -eq 0 ]]; then
  echo "No touched Python files between $base_ref and HEAD."
  exit 0
fi

echo "Running Ruff on ${#python_files[@]} touched Python file(s):"
printf " - %s\n" "${python_files[@]}"

uv run ruff check "${python_files[@]}"
