"""Lightweight app-first release smoke check.

Validates critical API paths without starting external services:
- health
- bootstrap status
- session auth
- run create/list
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from nexus_api.app import create_app
from nexus_api.config import ApiSettings
from nexus_api.service import build_context


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="agent-nexus-smoke-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        settings = ApiSettings(
            APP_DATABASE_URL=f"sqlite:///{tmp_path / 'smoke.db'}",
            APP_CANONICAL_WORKSPACE=str(tmp_path / "workspace"),
            APP_SANDBOX_ARTIFACT_ROOT=str(tmp_path / "sandbox"),
            APP_ADMIN_USERNAME="admin",
            APP_ADMIN_PASSWORD="secret",
            APP_SESSION_TTL_HOURS=24,
            APP_CONFIG_PATH=str(tmp_path / "config" / ".env"),
            APP_BOOTSTRAP_EXIT_AFTER_CONFIGURE=False,
            APP_ENABLE_MODEL_REPLANNER=False,
        )
        context = build_context(settings)
        with TestClient(create_app(context)) as client:
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json().get("status") == "ok"

            bootstrap = client.get("/bootstrap/status")
            assert bootstrap.status_code == 200
            assert "setup_required" in bootstrap.json()

            session = client.post(
                "/sessions",
                json={"username": "admin", "password": "secret"},
            )
            assert session.status_code == 200
            token = session.json()["token"]
            headers = {"Authorization": f"Bearer {token}"}

            create = client.post(
                "/runs",
                headers=headers,
                json={
                    "objective": "release smoke run",
                    "mode": "manual",
                    "steps": [{"action_type": "navigate", "instruction": "open status page"}],
                },
            )
            assert create.status_code == 200
            run_id = create.json()["id"]
            assert run_id

            listed = client.get("/runs?limit=5&offset=0", headers=headers)
            assert listed.status_code == 200
            payload = listed.json()
            assert payload["total"] >= 1
            assert any(item["id"] == run_id for item in payload["items"])

        context.db_engine.dispose()

    print("release smoke check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
