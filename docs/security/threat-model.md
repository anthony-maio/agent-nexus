# Agent Nexus Threat Model (v1)

## Scope

- App-first control plane (`nexus-api`)
- Sandbox execution service (`nexus-sandbox-runner`)
- Web UI + reverse proxy deployment path
- Optional Discord bridge

## Assets

- Admin credentials and session tokens
- Run plans, outputs, citations, and artifacts
- Canonical workspace files (post-promotion)
- Sandbox execution environment and host boundary

## Trust Boundaries

- Browser/client to API (bearer token boundary)
- API to sandbox runner (optional shared token)
- Sandbox artifacts to canonical workspace (promotion boundary)
- Optional Discord bridge to API
- Host Docker daemon boundary (especially host-socket mode)

## Main Threats

1. Credential compromise or offline hash cracking.
2. Session token theft from database.
3. Unauthorized promotion of tampered artifacts.
4. Path traversal from sandbox artifact names/paths into canonical workspace.
5. Sandbox escape or over-privileged runtime configuration.
6. Prompt injection causing unsafe high-risk automation.

## Current Mitigations

- Passwords stored with PBKDF2-SHA256 + per-password salt.
- Bearer tokens stored hashed at rest (SHA256), with legacy migration path.
- Session TTL + revoked flag handling on expiration.
- Promotion requires:
  - authenticated API caller,
  - artifact ownership by run,
  - completed source step,
  - approval for high-risk source step,
  - sandbox-root path enforcement,
  - SHA256 integrity match,
  - canonical target path safety checks,
  - idempotency guard against duplicate promotion.
- High-risk actions require supervised approval by default policy.
- Docker sandbox mode defaults to isolated flags (no-new-privileges, caps dropped, read-only rootfs, pids/memory/cpu limits, isolated network).

## Residual Risks

- Host-socket mode gives broad host control via Docker daemon and should only be used in trusted environments.
- Browser automation in `auto` mode can fallback to simulated behavior if browser dependencies are missing.
- Single-admin model has no MFA or multi-user RBAC in v1.
- No WAF/rate-limit layer is included by default.

## Next Security Milestones

1. Add configurable login rate limiting and lockout policy.
2. Add optional MFA for admin login.
3. Move session hashing to HMAC(secret) with key rotation support.
4. Add signed audit log chain for approvals/promotions.
5. Add sandbox egress controls and allowlisted outbound domains for browser runs.
