"""Configuration for Nexus API service."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    """Settings for app-first backend services."""

    model_config = SettingsConfigDict(
        env_file="config/.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_DATABASE_URL: str = Field(
        default="sqlite:///./data/app/nexus_app.db",
        description="Database URL for app control-plane state.",
    )
    APP_HOST: str = Field(default="0.0.0.0")
    APP_PORT: int = Field(default=8000)
    APP_ADMIN_USERNAME: str = Field(default="admin")
    APP_ADMIN_PASSWORD: str = Field(default="change-me-now")
    APP_SESSION_TTL_HOURS: int = Field(default=24)
    APP_CANONICAL_WORKSPACE: str = Field(default="workspace/app")
    APP_SANDBOX_ARTIFACT_ROOT: str = Field(default="data/sandbox")
    APP_CONFIG_PATH: str = Field(default="config/.env")
    APP_BOOTSTRAP_EXIT_AFTER_CONFIGURE: bool = Field(default=True)
    APP_ENABLE_MODEL_REPLANNER: bool = Field(default=True)
    APP_REPLANNER_MAX_STEPS: int = Field(default=4)
    APP_REPLANNER_TIMEOUT_SEC: float = Field(default=12.0)
    OPENROUTER_API_KEY: str = Field(default="")
    OPENROUTER_BASE_URL: str = Field(default="https://openrouter.ai/api/v1")
    OPENROUTER_MODEL: str = Field(default="openai/gpt-4.1-mini")
    SANDBOX_RUNNER_URL: str = Field(default="http://localhost:8020")
    SANDBOX_RUNNER_TOKEN: str = Field(
        default="",
        description="Optional shared token for API->sandbox runner authentication.",
    )

    @property
    def canonical_workspace_path(self) -> Path:
        return Path(self.APP_CANONICAL_WORKSPACE).resolve()

    @property
    def sandbox_artifact_root_path(self) -> Path:
        return Path(self.APP_SANDBOX_ARTIFACT_ROOT).resolve()

    @property
    def config_path(self) -> Path:
        return Path(self.APP_CONFIG_PATH).resolve()
