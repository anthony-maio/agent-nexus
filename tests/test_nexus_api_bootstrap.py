"""Integration tests for app-first bootstrap endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from nexus_api.app import create_app
from nexus_api.config import ApiSettings
from nexus_api.service import build_context


def _client(tmp_path: Path, admin_password: str = "change-me-now") -> TestClient:
    config_path = tmp_path / "config" / ".env"
    settings = ApiSettings(
        APP_DATABASE_URL=f"sqlite:///{tmp_path / 'bootstrap.db'}",
        APP_CANONICAL_WORKSPACE=str(tmp_path / "workspace"),
        APP_SANDBOX_ARTIFACT_ROOT=str(tmp_path / "sandbox"),
        APP_ADMIN_USERNAME="admin",
        APP_ADMIN_PASSWORD=admin_password,
        APP_SESSION_TTL_HOURS=24,
        APP_CONFIG_PATH=str(config_path),
        APP_BOOTSTRAP_EXIT_AFTER_CONFIGURE=False,
    )
    ctx = build_context(settings)
    return TestClient(create_app(ctx))


def test_bootstrap_status_requires_setup_when_config_missing(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.get("/bootstrap/status")

        assert response.status_code == 200
        data = response.json()
        assert data["setup_required"] is True
        assert data["configured"] is False
        assert Path(data["config_path"]).name == ".env"


def test_bootstrap_configure_writes_app_first_env_and_blocks_repeat(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        payload = {
            "admin_username": "owner",
            "admin_password": "replace-this-now",
            "sandbox_backend": "docker",
            "browser_mode": "real",
            "openrouter_api_key": "sk-or-v1-test",
            "discord_token": "discord-token",
            "discord_bridge_channel": "approvals",
        }

        response = client.post("/bootstrap/configure", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["configured"] is True
        assert data["restart_required"] is True

        config_path = Path(data["config_path"])
        content = config_path.read_text(encoding="utf-8")
        assert "APP_ADMIN_USERNAME=owner" in content
        assert "APP_ADMIN_PASSWORD=replace-this-now" in content
        assert "SANDBOX_EXECUTION_BACKEND=docker" in content
        assert "SANDBOX_BROWSER_MODE=real" in content
        assert "SANDBOX_DOCKER_IMAGE=agent-nexus-sandbox-step:local" in content
        assert "SANDBOX_DOCKER_ALLOWED_IMAGES=agent-nexus-sandbox-step:local" in content
        assert "SANDBOX_DOCKER_ALLOW_UNPINNED_LOCAL=1" in content
        assert "OPENROUTER_API_KEY=sk-or-v1-test" in content
        assert "DISCORD_TOKEN=discord-token" in content
        assert "DISCORD_BRIDGE_CHANNEL=approvals" in content

        status = client.get("/bootstrap/status")
        assert status.status_code == 200
        assert status.json()["setup_required"] is False
        assert status.json()["configured"] is True

        repeat = client.post("/bootstrap/configure", json=payload)
        assert repeat.status_code == 409
        assert "already configured" in repeat.json()["detail"]
