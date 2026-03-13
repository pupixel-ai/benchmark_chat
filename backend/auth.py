"""
Simple username/password auth and cookie-backed sessions for prototype use.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timedelta

from fastapi import Cookie, HTTPException, Response, status
from sqlalchemy import delete, select

from backend.db import SessionLocal
from backend.models import SessionRecord, UserRecord
from config import AUTH_SESSION_COOKIE_NAME, AUTH_SESSION_DAYS, COOKIE_SECURE


PBKDF2_ITERATIONS = 120_000


def _utcnow() -> datetime:
    return datetime.utcnow()


def _hash_password(password: str, salt: str | None = None) -> str:
    resolved_salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        resolved_salt.encode("utf-8"),
        PBKDF2_ITERATIONS,
    )
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${resolved_salt}${digest.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, iteration_text, salt, digest = password_hash.split("$", 3)
    except ValueError:
        return False

    if algorithm != "pbkdf2_sha256":
        return False

    iterations = int(iteration_text)
    candidate = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    ).hex()
    return hmac.compare_digest(candidate, digest)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _session_expiry() -> datetime:
    return _utcnow() + timedelta(days=AUTH_SESSION_DAYS)


def _serialize_user(user: UserRecord) -> dict:
    return {
        "user_id": user.user_id,
        "username": user.username,
        "created_at": user.created_at.isoformat(),
    }


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=AUTH_SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        max_age=AUTH_SESSION_DAYS * 24 * 60 * 60,
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(
        key=AUTH_SESSION_COOKIE_NAME,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        path="/",
    )


def _create_session(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    record = SessionRecord(
        session_id=uuid.uuid4().hex,
        user_id=user_id,
        token_hash=_hash_token(token),
        created_at=_utcnow(),
        expires_at=_session_expiry(),
    )
    with SessionLocal() as session:
        session.add(record)
        session.commit()
    return token


def register_user(username: str, password: str) -> dict:
    normalized_username = username.strip().lower()
    if len(normalized_username) < 3:
        raise HTTPException(status_code=400, detail="用户名至少 3 个字符")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="密码至少 8 个字符")

    with SessionLocal() as session:
        existing = session.execute(
            select(UserRecord).where(UserRecord.username == normalized_username)
        ).scalar_one_or_none()
        if existing is not None:
            raise HTTPException(status_code=409, detail="用户名已存在")

        now = _utcnow()
        user = UserRecord(
            user_id=uuid.uuid4().hex,
            username=normalized_username,
            password_hash=_hash_password(password),
            created_at=now,
            updated_at=now,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return _serialize_user(user)


def login_user(username: str, password: str) -> tuple[dict, str]:
    normalized_username = username.strip().lower()
    with SessionLocal() as session:
        user = session.execute(
            select(UserRecord).where(UserRecord.username == normalized_username)
        ).scalar_one_or_none()
        if user is None or not verify_password(password, user.password_hash):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")

        session.execute(
            delete(SessionRecord).where(
                SessionRecord.user_id == user.user_id,
                SessionRecord.expires_at <= _utcnow(),
            )
        )
        session.commit()
        return _serialize_user(user), _create_session(user.user_id)


def authenticate_response(response: Response, user: dict, session_token: str) -> dict:
    _set_session_cookie(response, session_token)
    return {"user": user}


def logout_current_session(session_token: str | None, response: Response) -> None:
    clear_session_cookie(response)
    if not session_token:
        return

    token_hash = _hash_token(session_token)
    with SessionLocal() as session:
        session.execute(delete(SessionRecord).where(SessionRecord.token_hash == token_hash))
        session.commit()


def get_current_user(session_token: str | None = Cookie(default=None, alias=AUTH_SESSION_COOKIE_NAME)) -> dict:
    if not session_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="请先登录")

    token_hash = _hash_token(session_token)
    with SessionLocal() as session:
        record = session.execute(
            select(SessionRecord, UserRecord)
            .join(UserRecord, UserRecord.user_id == SessionRecord.user_id)
            .where(SessionRecord.token_hash == token_hash)
        ).first()

        if record is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录已失效")

        session_record, user_record = record
        if session_record.expires_at <= _utcnow():
            session.execute(delete(SessionRecord).where(SessionRecord.session_id == session_record.session_id))
            session.commit()
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录已失效")

        return _serialize_user(user_record)
