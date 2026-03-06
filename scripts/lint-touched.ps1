param(
    [string]$BaseRef = "origin/main"
)

$ErrorActionPreference = "Stop"

$null = & git rev-parse --verify $BaseRef 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Base ref not found: $BaseRef"
}

$pythonFiles = & git diff --name-only --diff-filter=ACMRTUXB "$BaseRef...HEAD" -- "*.py"
if (-not $pythonFiles -or $pythonFiles.Count -eq 0) {
    Write-Host "No touched Python files between $BaseRef and HEAD."
    exit 0
}

Write-Host "Running Ruff on $($pythonFiles.Count) touched Python file(s):"
foreach ($file in $pythonFiles) {
    Write-Host " - $file"
}

& uv run ruff check @pythonFiles
