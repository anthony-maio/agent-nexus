"""Request/response schemas for Nexus API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from nexus_core.models import ApprovalDecision, RunMode, StepDefinition


class BootstrapStatusResponse(BaseModel):
    setup_required: bool
    configured: bool
    config_path: str
    uses_default_admin_password: bool


class BootstrapConfigureRequest(BaseModel):
    admin_username: str = Field(min_length=1, max_length=120)
    admin_password: str = Field(min_length=8, max_length=200)
    sandbox_backend: str = Field(default="docker", pattern="^(local|docker|docker-host)$")
    browser_mode: str = Field(default="auto", pattern="^(simulated|auto|real)$")
    openrouter_api_key: str = Field(default="", max_length=240)
    discord_token: str = Field(default="", max_length=240)
    discord_bridge_channel: str = Field(default="human", max_length=120)
    public_host: str = Field(default="", max_length=255)
    acme_email: str = Field(default="", max_length=255)


class BootstrapConfigureResponse(BaseModel):
    configured: bool
    restart_required: bool
    config_path: str


class SessionCreateRequest(BaseModel):
    username: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=1, max_length=200)


class SessionCreateResponse(BaseModel):
    session_id: str
    token: str
    username: str
    expires_at: str


class RunCreateRequest(BaseModel):
    objective: str = Field(min_length=1, max_length=2000)
    mode: RunMode = RunMode.SUPERVISED
    steps: list[StepDefinition] = Field(default_factory=list)


class ApprovalRequest(BaseModel):
    decision: ApprovalDecision
    reason: str = ""


class PromotionRequest(BaseModel):
    promoted_by: str = Field(default="", max_length=120)


class PendingApprovalItem(BaseModel):
    run_id: str
    step_id: str
    action_type: str
    instruction: str
    risk_tier: str
