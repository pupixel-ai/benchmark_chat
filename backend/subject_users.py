"""
Subject-user helpers.

`user_id` in downstream/task memory APIs means the photo owner's identity, not the
logged-in operator/admin account.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import select

from backend.db import SessionLocal
from backend.models import SubjectUserRecord, UserRecord


def _utcnow() -> datetime:
    return datetime.utcnow()


def normalize_subject_username(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text.lower()


def resolve_subject_identity(
    *,
    operator_user_id: str,
    operator_username: str | None,
    subject_username: str | None,
) -> dict:
    normalized_subject_username = normalize_subject_username(subject_username)
    normalized_operator_username = normalize_subject_username(operator_username)

    if not normalized_subject_username or normalized_subject_username == normalized_operator_username:
        return {
            "user_id": str(operator_user_id or "").strip(),
            "operator_user_id": None,
            "subject_username": normalized_operator_username,
            "subject_source": "self",
        }

    with SessionLocal() as session:
        auth_user = session.execute(
            select(UserRecord).where(UserRecord.username == normalized_subject_username)
        ).scalar_one_or_none()
        if auth_user is not None:
            _upsert_subject_user(
                session,
                user_id=auth_user.user_id,
                username=normalized_subject_username,
                display_name=normalized_subject_username,
                linked_auth_user_id=auth_user.user_id,
            )
            session.commit()
            return {
                "user_id": auth_user.user_id,
                "operator_user_id": None if auth_user.user_id == operator_user_id else operator_user_id,
                "subject_username": normalized_subject_username,
                "subject_source": "auth_user",
            }

        subject_user = session.execute(
            select(SubjectUserRecord).where(SubjectUserRecord.username == normalized_subject_username)
        ).scalar_one_or_none()
        if subject_user is None:
            subject_user = SubjectUserRecord(
                user_id=uuid.uuid4().hex,
                username=normalized_subject_username,
                display_name=normalized_subject_username,
                linked_auth_user_id=None,
                created_at=_utcnow(),
                updated_at=_utcnow(),
            )
            session.add(subject_user)
        else:
            subject_user.updated_at = _utcnow()
            if not subject_user.display_name:
                subject_user.display_name = normalized_subject_username

        session.commit()
        return {
            "user_id": subject_user.user_id,
            "operator_user_id": None if subject_user.user_id == operator_user_id else operator_user_id,
            "subject_username": subject_user.username,
            "subject_source": "subject_registry",
        }


def get_subject_user(user_id: str) -> dict | None:
    normalized_user_id = str(user_id or "").strip()
    if not normalized_user_id:
        return None

    with SessionLocal() as session:
        auth_user = session.get(UserRecord, normalized_user_id)
        if auth_user is not None:
            return {
                "user_id": auth_user.user_id,
                "username": auth_user.username,
                "display_name": auth_user.username,
                "linked_auth_user_id": auth_user.user_id,
                "source": "auth_user",
            }

        subject_user = session.get(SubjectUserRecord, normalized_user_id)
        if subject_user is None:
            return None
        return {
            "user_id": subject_user.user_id,
            "username": subject_user.username,
            "display_name": subject_user.display_name or subject_user.username,
            "linked_auth_user_id": subject_user.linked_auth_user_id,
            "source": "subject_registry",
        }


def ensure_subject_user(user_id: str, *, username: str | None = None, display_name: str | None = None) -> dict:
    normalized_user_id = str(user_id or "").strip()
    if not normalized_user_id:
        raise ValueError("subject user_id is required")

    existing = get_subject_user(normalized_user_id)
    if existing is not None:
        return existing

    normalized_username = normalize_subject_username(username) or normalized_user_id
    normalized_display_name = str(display_name or normalized_username).strip() or normalized_username

    with SessionLocal() as session:
        record = SubjectUserRecord(
            user_id=normalized_user_id,
            username=normalized_username,
            display_name=normalized_display_name,
            linked_auth_user_id=None,
            created_at=_utcnow(),
            updated_at=_utcnow(),
        )
        session.add(record)
        session.commit()
        return {
            "user_id": record.user_id,
            "username": record.username,
            "display_name": record.display_name or record.username,
            "linked_auth_user_id": record.linked_auth_user_id,
            "source": "subject_registry",
        }


def _upsert_subject_user(
    session,
    *,
    user_id: str,
    username: str,
    display_name: str | None,
    linked_auth_user_id: Optional[str],
) -> None:
    record = session.get(SubjectUserRecord, user_id)
    if record is None:
        record = SubjectUserRecord(
            user_id=user_id,
            username=username,
            display_name=display_name or username,
            linked_auth_user_id=linked_auth_user_id,
            created_at=_utcnow(),
            updated_at=_utcnow(),
        )
        session.add(record)
        return

    record.username = username
    record.display_name = display_name or record.display_name or username
    record.linked_auth_user_id = linked_auth_user_id
    record.updated_at = _utcnow()
    session.add(record)
