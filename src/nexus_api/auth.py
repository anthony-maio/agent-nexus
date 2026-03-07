"""Simple single-admin auth for API sessions."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from nexus_api.models import AdminUser, SessionToken

_PASSWORD_SCHEME = "pbkdf2_sha256"
_PASSWORD_ITERATIONS = 310_000


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        _PASSWORD_ITERATIONS,
    )
    return f"{_PASSWORD_SCHEME}${_PASSWORD_ITERATIONS}${salt.hex()}${digest.hex()}"


def _hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _verify_password(password: str, password_hash: str) -> tuple[bool, bool]:
    if password_hash.startswith(f"{_PASSWORD_SCHEME}$"):
        parts = password_hash.split("$", 3)
        if len(parts) != 4:
            return False, False
        _, iterations_raw, salt_hex, expected_hex = parts
        try:
            iterations = int(iterations_raw)
            salt = bytes.fromhex(salt_hex)
            expected = bytes.fromhex(expected_hex)
        except ValueError:
            return False, False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations,
        )
        return secrets.compare_digest(digest, expected), False

    # Legacy unsalted SHA256 support for in-place migration.
    legacy = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return secrets.compare_digest(password_hash, legacy), True


def ensure_admin_user(session: Session, username: str, password: str) -> AdminUser:
    user = session.scalar(select(AdminUser).where(AdminUser.username == username))
    if user is not None:
        return user
    user = AdminUser(username=username, password_hash=hash_password(password))
    session.add(user)
    session.flush()
    return user


def create_session_token(
    session: Session,
    user: AdminUser,
    ttl_hours: int,
) -> tuple[SessionToken, str]:
    raw_token = secrets.token_urlsafe(40)
    token_hash = _hash_session_token(raw_token)
    token = SessionToken(
        user_id=user.id,
        token=token_hash,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=ttl_hours),
    )
    session.add(token)
    session.flush()
    return token, raw_token


def authenticate_user(session: Session, username: str, password: str) -> AdminUser | None:
    user = session.scalar(select(AdminUser).where(AdminUser.username == username))
    if user is None:
        return None
    verified, should_upgrade = _verify_password(password, user.password_hash)
    if not verified:
        return None
    if should_upgrade:
        user.password_hash = hash_password(password)
        session.flush()
    return user


def validate_bearer_token(session: Session, token: str) -> AdminUser | None:
    now = datetime.now(timezone.utc)
    token_hash = _hash_session_token(token)
    s = session.scalar(select(SessionToken).where(SessionToken.token.in_([token_hash, token])))
    if s is None or s.revoked:
        return None
    # Migrate legacy plaintext token-at-rest values in-place.
    if s.token != token_hash:
        s.token = token_hash
        session.flush()
    expires_at = s.expires_at
    if expires_at is not None and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at is not None and expires_at < now:
        s.revoked = True
        session.flush()
        return None
    return session.get(AdminUser, s.user_id)
