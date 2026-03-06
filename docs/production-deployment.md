# Agent Nexus Production Deployment

This deployment path targets the app-first stack with the web UI fronted by Caddy.

On a fresh install, open the web app first. If `config/.env` is missing, the app presents a bootstrap screen that writes the initial config and restarts the API.

## Prerequisites

- Docker Engine + Docker Compose v2
- A configured `config/.env` (copy from `config/.env.example`)
- If using a public domain with automatic TLS:
  - DNS A/AAAA record pointing to host
  - `NEXUS_PUBLIC_HOST` set to that domain
  - `ACME_EMAIL` set

## One-Command Start

Linux/macOS:

```bash
./scripts/prod-up.sh docker
```

Windows PowerShell:

```powershell
scripts/prod-up.ps1 -SandboxBackend docker
```

Trusted local/dev hosts can use host-socket mode:

```bash
./scripts/prod-up.sh docker-host
```

```powershell
scripts/prod-up.ps1 -SandboxBackend docker-host
```

## Compose Stack

Production stack combines:

- `docker/docker-compose.yml`
- `docker/docker-compose.prod.yml`

Optional host-socket mode additionally applies:

- `docker/docker-compose.host-socket.yml`

## Health Checks

- App/API via proxy: `http://localhost:${NEXUS_HTTP_PORT:-80}/api/health`
- Direct API (local bind): `http://127.0.0.1:${APP_PORT:-8000}/health`
- Sandbox runner (local bind): `http://127.0.0.1:${SANDBOX_PORT:-8020}/health`

## Notes

- `nexus-proxy` routes `/api/*` to `nexus-api` and all other paths to `nexus-frontend`.
- `nexus-api` and `nexus-sandbox-runner` are bound to loopback in production override.
- Promotion safety checks require artifacts to stay under `APP_SANDBOX_ARTIFACT_ROOT`.
- `docker` and `docker-host` helper scripts auto-build `agent-nexus-sandbox-step:local`, a Playwright-capable step image used for real browser execution.
