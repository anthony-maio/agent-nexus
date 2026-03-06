param(
    [ValidateSet("local", "docker", "docker-host")]
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

function Ensure-SandboxStepImage {
    param([string]$RepoRoot)

    if (-not $env:SANDBOX_DOCKER_IMAGE) {
        Write-Host "Building local browser step image..."
        & docker build -f (Join-Path $RepoRoot "docker/Dockerfile.sandbox.step") -t agent-nexus-sandbox-step:local $RepoRoot
        $env:SANDBOX_DOCKER_IMAGE = "agent-nexus-sandbox-step:local"
    }
    if (-not $env:SANDBOX_DOCKER_ALLOWED_IMAGES) {
        $env:SANDBOX_DOCKER_ALLOWED_IMAGES = $env:SANDBOX_DOCKER_IMAGE
    }
    if ($env:SANDBOX_DOCKER_IMAGE -eq "agent-nexus-sandbox-step:local" -and -not $env:SANDBOX_DOCKER_ALLOW_UNPINNED_LOCAL) {
        $env:SANDBOX_DOCKER_ALLOW_UNPINNED_LOCAL = "1"
    }
    if (-not $env:SANDBOX_BROWSER_MODE) {
        $env:SANDBOX_BROWSER_MODE = "auto"
    }
}
    throw "Timed out waiting for $Url"
}

if (-not (Test-Command "docker")) {
    throw "Docker CLI is required. Install Docker Desktop and retry."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$composeFile = Join-Path $repoRoot "docker/docker-compose.yml"
$hostSocketComposeFile = Join-Path $repoRoot "docker/docker-compose.host-socket.yml"

$composeArgs = @("-f", $composeFile)

$profileArgs = @()
if ($SandboxBackend -eq "docker") {
    Ensure-SandboxStepImage -RepoRoot $repoRoot
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
} elseif ($SandboxBackend -eq "docker-host") {
    Ensure-SandboxStepImage -RepoRoot $repoRoot
    $env:SANDBOX_EXECUTION_BACKEND = "docker"
    if (-not $env:SANDBOX_DOCKER_HOST) {
        $env:SANDBOX_DOCKER_HOST = "unix:///var/run/docker.sock"
    }
    if (-not $env:SANDBOX_DOCKER_TLS_VERIFY) {
        $env:SANDBOX_DOCKER_TLS_VERIFY = "0"
    }
    if (-not $env:SANDBOX_DOCKER_CERT_PATH) {
        $env:SANDBOX_DOCKER_CERT_PATH = ""
    }
    $composeArgs += @("-f", $hostSocketComposeFile)
} else {
    $env:SANDBOX_EXECUTION_BACKEND = "local"
}

$upArgs = @("compose") + $composeArgs + $profileArgs + @("up", "-d")
if (-not $NoBuild) {
    $upArgs += "--build"
}

Write-Host "Starting Agent Nexus stack (sandbox backend: $SandboxBackend)..."
if ($SandboxBackend -eq "docker-host") {
    Write-Host "Warning: host-socket mode grants sandbox runner broad access to host Docker daemon."
}
& docker @upArgs

Wait-HttpOk -Url "http://localhost:8020/health" -TimeoutSec 180
Wait-HttpOk -Url "http://localhost:8000/health" -TimeoutSec 180

Write-Host ""
Write-Host "Agent Nexus is up."
Write-Host "API:      http://localhost:8000/health"
Write-Host "Sandbox:  http://localhost:8020/health"
Write-Host "Admin user defaults come from config/.env (APP_ADMIN_USERNAME / APP_ADMIN_PASSWORD)."
