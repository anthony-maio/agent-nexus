# Security Hardening Checklist

Use this checklist before exposing Agent Nexus outside a trusted local network.

## Identity and Access

- [ ] Set a strong `APP_ADMIN_PASSWORD` (do not use defaults).
- [ ] Reduce `APP_SESSION_TTL_HOURS` to operational minimum.
- [ ] Rotate credentials and restart services after secret changes.

## Transport and Network

- [ ] Set `NEXUS_PUBLIC_HOST` to the production domain.
- [ ] Set `ACME_EMAIL` for TLS certificate management.
- [ ] Restrict inbound firewall rules to `NEXUS_HTTP_PORT`/`NEXUS_HTTPS_PORT`.
- [ ] Keep `nexus-api` and `nexus-sandbox-runner` loopback-only binds in production override.

## Sandbox Execution

- [ ] Prefer `SANDBOX_EXECUTION_BACKEND=docker` with isolated dind profile.
- [ ] Keep digest-pinned `SANDBOX_DOCKER_IMAGE` and explicit allowlist.
- [ ] Keep `SANDBOX_DOCKER_NETWORK=none` unless a controlled exception is required.
- [ ] Use `SANDBOX_BROWSER_MODE=real` only with browser-capable pinned image.
- [ ] Avoid `docker-host` mode except trusted local/dev hosts.

## Data and Promotion

- [ ] Keep `APP_SANDBOX_ARTIFACT_ROOT` on isolated storage.
- [ ] Confirm canonical workspace (`APP_CANONICAL_WORKSPACE`) backup policy.
- [ ] Monitor promotion events for unexpected volume/patterns.

## Observability and Operations

- [ ] Centralize logs for `nexus-api`, `nexus-sandbox-runner`, `nexus-proxy`.
- [ ] Alert on repeated auth failures and promotion denials.
- [ ] Keep base images updated and rebuild regularly.

## Optional Discord Bridge

- [ ] Set dedicated bridge token and channel permissions.
- [ ] Restrict bridge bot role to approval/status channels only.
- [ ] Keep `APP_API_TOKEN` scoped to minimum required permissions.
