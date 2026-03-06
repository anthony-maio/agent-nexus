"""Request/response schemas for Nexus API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from nexus_core.models import ApprovalDecision, RunMode, StepDefinition


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
    promoted_by: str = Field(min_length=1, max_length=120)


class PendingApprovalItem(BaseModel):
    run_id: str
    step_id: str
    action_type: str
    instruction: str
    risk_tier: str
