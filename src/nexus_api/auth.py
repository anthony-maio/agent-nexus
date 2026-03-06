"""Simple single-admin auth for API sessions."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from nexus_api.models import AdminUser, SessionToken


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def ensure_admin_user(session: Session, username: str, password: str) -> AdminUser:
    user = session.scalar(select(AdminUser).where(AdminUser.username == username))
    if user is not None:
        return user
    user = AdminUser(username=username, password_hash=hash_password(password))
    session.add(user)
    session.flush()
    return user


def create_session_token(session: Session, user: AdminUser, ttl_hours: int) -> SessionToken:
    token = SessionToken(
        user_id=user.id,
        token=secrets.token_urlsafe(40),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=ttl_hours),
    )
    session.add(token)
    session.flush()
    return token


def authenticate_user(session: Session, username: str, password: str) -> AdminUser | None:
    user = session.scalar(select(AdminUser).where(AdminUser.username == username))
    if user is None:
        return None
    if user.password_hash != hash_password(password):
        return None
    return user


def validate_bearer_token(session: Session, token: str) -> AdminUser | None:
    now = datetime.now(timezone.utc)
    s = session.scalar(select(SessionToken).where(SessionToken.token == token))
    if s is None or s.revoked:
        return None
    expires_at = s.expires_at
    if expires_at is not None and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at is not None and expires_at < now:
        return None
    return session.get(AdminUser, s.user_id)
