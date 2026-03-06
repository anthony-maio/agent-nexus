#!/usr/bin/env bash
set -euo pipefail

backend="${1:-docker}"
if [[ "$backend" != "local" && "$backend" != "docker" && "$backend" != "docker-host" ]]; then
  echo "Usage: scripts/prod-up.sh [local|docker|docker-host]"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker CLI is required. Install Docker and retry."
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compose_file="$repo_root/docker/docker-compose.yml"
prod_compose_file="$repo_root/docker/docker-compose.prod.yml"
host_socket_compose_file="$repo_root/docker/docker-compose.host-socket.yml"
profile_args=()
compose_args=(-f "$compose_file" -f "$prod_compose_file")

ensure_sandbox_step_image() {
  if [[ -z "${SANDBOX_DOCKER_IMAGE:-}" ]]; then
    echo "Building local browser step image..."
    docker build -f "$repo_root/docker/Dockerfile.sandbox.step" -t agent-nexus-sandbox-step:local "$repo_root"
    export SANDBOX_DOCKER_IMAGE="agent-nexus-sandbox-step:local"
  fi
  export SANDBOX_DOCKER_ALLOWED_IMAGES="${SANDBOX_DOCKER_ALLOWED_IMAGES:-$SANDBOX_DOCKER_IMAGE}"
  if [[ "$SANDBOX_DOCKER_IMAGE" == "agent-nexus-sandbox-step:local" ]]; then
    export SANDBOX_DOCKER_ALLOW_UNPINNED_LOCAL="${SANDBOX_DOCKER_ALLOW_UNPINNED_LOCAL:-1}"
  fi
  export SANDBOX_BROWSER_MODE="${SANDBOX_BROWSER_MODE:-auto}"
}

if [[ "$backend" == "docker" ]]; then
  ensure_sandbox_step_image
  export SANDBOX_EXECUTION_BACKEND="docker"
  export SANDBOX_DOCKER_HOST="${SANDBOX_DOCKER_HOST:-tcp://nexus-sandbox-dind:2376}"
  export SANDBOX_DOCKER_TLS_VERIFY="${SANDBOX_DOCKER_TLS_VERIFY:-1}"
  export SANDBOX_DOCKER_CERT_PATH="${SANDBOX_DOCKER_CERT_PATH:-/certs/client}"
  profile_args=(--profile sandbox-docker)
elif [[ "$backend" == "docker-host" ]]; then
  ensure_sandbox_step_image
  export SANDBOX_EXECUTION_BACKEND="docker"
  export SANDBOX_DOCKER_HOST="${SANDBOX_DOCKER_HOST:-unix:///var/run/docker.sock}"
  export SANDBOX_DOCKER_TLS_VERIFY="${SANDBOX_DOCKER_TLS_VERIFY:-0}"
  export SANDBOX_DOCKER_CERT_PATH="${SANDBOX_DOCKER_CERT_PATH:-}"
  compose_args+=(-f "$host_socket_compose_file")
else
  export SANDBOX_EXECUTION_BACKEND="local"
fi

http_port="${NEXUS_HTTP_PORT:-80}"
app_url="http://localhost:${http_port}"
api_health_url="${app_url}/api/health"

echo "Starting Agent Nexus production stack (sandbox backend: $backend)..."
if [[ "$backend" == "docker-host" ]]; then
  echo "Warning: host-socket mode grants sandbox runner broad access to host Docker daemon."
fi
docker compose "${compose_args[@]}" "${profile_args[@]}" up -d --build

wait_http_ok() {
  local url="$1"
  local timeout="${2:-240}"
  local end=$((SECONDS + timeout))
  while (( SECONDS < end )); do
    if curl -kfsSL "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for $url"
  return 1
}

wait_http_ok "$api_health_url" 240

echo
echo "Agent Nexus production stack is up."
echo "App URL:  $app_url"
echo "API URL:  $app_url/api"
echo "Set NEXUS_PUBLIC_HOST and ACME_EMAIL in config/.env for automatic TLS certificates."
