param(
    [ValidateSet("local", "docker")]
    [string]$SandboxBackend = "local",
    [switch]$NoBuild
)

$ErrorActionPreference = "Stop"

function Test-Command {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Wait-HttpOk {
    param(
        [string]$Url,
        [int]$TimeoutSec = 120
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $resp = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
            if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 300) {
                return
            }
        } catch {
            Start-Sleep -Seconds 2
        }
    }
    throw "Timed out waiting for $Url"
}

if (-not (Test-Command "docker")) {
    throw "Docker CLI is required. Install Docker Desktop and retry."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$composeFile = Join-Path $repoRoot "docker/docker-compose.yml"

$profileArgs = @()
if ($SandboxBackend -eq "docker") {
    $env:SANDBOX_EXECUTION_BACKEND = "docker"
    if (-not $env:SANDBOX_DOCKER_HOST) {
        $env:SANDBOX_DOCKER_HOST = "tcp://nexus-sandbox-dind:2376"
    }
    if (-not $env:SANDBOX_DOCKER_TLS_VERIFY) {
        $env:SANDBOX_DOCKER_TLS_VERIFY = "1"
    }
    if (-not $env:SANDBOX_DOCKER_CERT_PATH) {
        $env:SANDBOX_DOCKER_CERT_PATH = "/certs/client"
    }
    $profileArgs = @("--profile", "sandbox-docker")
} else {
    $env:SANDBOX_EXECUTION_BACKEND = "local"
}

$upArgs = @("compose", "-f", $composeFile) + $profileArgs + @("up", "-d")
if (-not $NoBuild) {
    $upArgs += "--build"
}

Write-Host "Starting Agent Nexus stack (sandbox backend: $SandboxBackend)..."
& docker @upArgs

Wait-HttpOk -Url "http://localhost:8020/health" -TimeoutSec 180
Wait-HttpOk -Url "http://localhost:8000/health" -TimeoutSec 180

Write-Host ""
Write-Host "Agent Nexus is up."
Write-Host "API:      http://localhost:8000/health"
Write-Host "Sandbox:  http://localhost:8020/health"
Write-Host "Admin user defaults come from config/.env (APP_ADMIN_USERNAME / APP_ADMIN_PASSWORD)."
