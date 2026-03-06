#!/usr/bin/env bash
set -euo pipefail

backend="${1:-local}"
if [[ "$backend" != "local" && "$backend" != "docker" ]]; then
  echo "Usage: scripts/dev-up.sh [local|docker]"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker CLI is required. Install Docker and retry."
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compose_file="$repo_root/docker/docker-compose.yml"
profile_args=()

if [[ "$backend" == "docker" ]]; then
  export SANDBOX_EXECUTION_BACKEND="docker"
  export SANDBOX_DOCKER_HOST="${SANDBOX_DOCKER_HOST:-tcp://nexus-sandbox-dind:2376}"
  export SANDBOX_DOCKER_TLS_VERIFY="${SANDBOX_DOCKER_TLS_VERIFY:-1}"
  export SANDBOX_DOCKER_CERT_PATH="${SANDBOX_DOCKER_CERT_PATH:-/certs/client}"
  profile_args=(--profile sandbox-docker)
else
  export SANDBOX_EXECUTION_BACKEND="local"
fi

echo "Starting Agent Nexus stack (sandbox backend: $backend)..."
docker compose -f "$compose_file" "${profile_args[@]}" up -d --build

wait_http_ok() {
  local url="$1"
  local timeout="${2:-180}"
  local end=$((SECONDS + timeout))
  while (( SECONDS < end )); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for $url"
  return 1
}

wait_http_ok "http://localhost:8020/health" 180
wait_http_ok "http://localhost:8000/health" 180

echo
echo "Agent Nexus is up."
echo "API:      http://localhost:8000/health"
echo "Sandbox:  http://localhost:8020/health"
echo "Admin user defaults come from config/.env (APP_ADMIN_USERNAME / APP_ADMIN_PASSWORD)."
