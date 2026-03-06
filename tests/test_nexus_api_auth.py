"""Unit tests for API auth/session hardening behavior."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from nexus_api.auth import (
    authenticate_user,
    create_session_token,
    ensure_admin_user,
    hash_password,
    validate_bearer_token,
)
from nexus_api.db import Base, build_engine, build_session_factory, session_scope
from nexus_api.models import SessionToken


def _session_factory(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'auth.db'}"
    engine = build_engine(db_url)
    Base.metadata.create_all(engine)
    return build_session_factory(engine)


def test_hash_password_uses_pbkdf2_format() -> None:
    one = hash_password("secret")
    two = hash_password("secret")
    assert one.startswith("pbkdf2_sha256$")
    assert two.startswith("pbkdf2_sha256$")
    assert one != two


def test_authenticate_user_migrates_legacy_sha256_hash(tmp_path) -> None:
    session_factory = _session_factory(tmp_path)
    legacy_hash = hashlib.sha256("secret".encode("utf-8")).hexdigest()

    with session_scope(session_factory) as session:
        user = ensure_admin_user(session, username="admin", password="other-secret")
        user.password_hash = legacy_hash
        session.flush()

    with session_scope(session_factory) as session:
        authed = authenticate_user(session, username="admin", password="secret")
        assert authed is not None
        assert authed.password_hash.startswith("pbkdf2_sha256$")
        assert authed.password_hash != legacy_hash


def test_create_session_token_hashes_token_at_rest(tmp_path) -> None:
    session_factory = _session_factory(tmp_path)
    with session_scope(session_factory) as session:
        user = ensure_admin_user(session, username="admin", password="secret")
        tok, raw_token = create_session_token(session, user, ttl_hours=24)
        assert tok.token != raw_token
        assert len(tok.token) == 64
        assert validate_bearer_token(session, raw_token) is not None


def test_validate_bearer_token_migrates_legacy_plaintext_token(tmp_path) -> None:
    session_factory = _session_factory(tmp_path)
    raw_token = "legacy-plaintext-token"

    with session_scope(session_factory) as session:
        user = ensure_admin_user(session, username="admin", password="secret")
        session.add(
            SessionToken(
                user_id=user.id,
                token=raw_token,
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            )
        )
        session.flush()

    with session_scope(session_factory) as session:
        validated = validate_bearer_token(session, raw_token)
        assert validated is not None
        stored = session.query(SessionToken).first()
        assert stored is not None
        assert stored.token != raw_token
        assert len(stored.token) == 64


def test_validate_bearer_token_revokes_expired_tokens(tmp_path) -> None:
    session_factory = _session_factory(tmp_path)
    with session_scope(session_factory) as session:
        user = ensure_admin_user(session, username="admin", password="secret")
        _, raw_token = create_session_token(session, user, ttl_hours=-1)
        assert validate_bearer_token(session, raw_token) is None
        stored = session.query(SessionToken).first()
        assert stored is not None
        assert stored.revoked is True
