"""
Face feedback persistence.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, Iterable, Optional

from sqlalchemy import select

from backend.db import SessionLocal
from backend.models import FaceRecognitionImagePolicyRecord, FaceReviewRecord


class FaceReviewStore:
    def upsert_face_review(
        self,
        *,
        user_id: str,
        task_id: str,
        face_id: str,
        image_id: str,
        person_id: str | None,
        source_hash: str | None,
        is_inaccurate: bool | None,
        comment_text: str | None,
    ) -> Dict:
        now = datetime.now()
        with SessionLocal() as session:
            record = session.execute(
                select(FaceReviewRecord).where(
                    FaceReviewRecord.user_id == user_id,
                    FaceReviewRecord.task_id == task_id,
                    FaceReviewRecord.face_id == face_id,
                )
            ).scalar_one_or_none()
            if record is None:
                record = FaceReviewRecord(
                    review_id=uuid.uuid4().hex,
                    user_id=user_id,
                    task_id=task_id,
                    face_id=face_id,
                    image_id=image_id,
                    person_id=person_id,
                    source_hash=source_hash,
                    is_inaccurate=bool(is_inaccurate),
                    comment_text=comment_text,
                    created_at=now,
                    updated_at=now,
                )
            else:
                if is_inaccurate is not None:
                    record.is_inaccurate = bool(is_inaccurate)
                record.comment_text = comment_text
                record.image_id = image_id
                record.person_id = person_id
                record.source_hash = source_hash
                record.updated_at = now

            session.add(record)
            session.commit()
            session.refresh(record)
            return self._serialize_review(record)

    def upsert_image_policy(
        self,
        *,
        user_id: str,
        source_hash: str,
        is_abandoned: bool,
        last_task_id: str | None,
        last_image_id: str | None,
    ) -> Dict:
        now = datetime.now()
        with SessionLocal() as session:
            record = session.execute(
                select(FaceRecognitionImagePolicyRecord).where(
                    FaceRecognitionImagePolicyRecord.user_id == user_id,
                    FaceRecognitionImagePolicyRecord.source_hash == source_hash,
                )
            ).scalar_one_or_none()
            if record is None:
                record = FaceRecognitionImagePolicyRecord(
                    policy_id=uuid.uuid4().hex,
                    user_id=user_id,
                    source_hash=source_hash,
                    is_abandoned=bool(is_abandoned),
                    last_task_id=last_task_id,
                    last_image_id=last_image_id,
                    created_at=now,
                    updated_at=now,
                )
            else:
                record.is_abandoned = bool(is_abandoned)
                record.last_task_id = last_task_id
                record.last_image_id = last_image_id
                record.updated_at = now

            session.add(record)
            session.commit()
            session.refresh(record)
            return self._serialize_policy(record)

    def get_task_feedback(self, task_id: str, user_id: str, source_hashes: Iterable[str] = ()) -> Dict:
        with SessionLocal() as session:
            reviews = session.execute(
                select(FaceReviewRecord).where(
                    FaceReviewRecord.user_id == user_id,
                    FaceReviewRecord.task_id == task_id,
                )
            ).scalars().all()
            review_map = {record.face_id: self._serialize_review(record) for record in reviews}

            hashes = {value for value in source_hashes if value}
            policy_map: Dict[str, Dict] = {}
            if hashes:
                policies = session.execute(
                    select(FaceRecognitionImagePolicyRecord).where(
                        FaceRecognitionImagePolicyRecord.user_id == user_id,
                        FaceRecognitionImagePolicyRecord.source_hash.in_(hashes),
                    )
                ).scalars().all()
                policy_map = {record.source_hash: self._serialize_policy(record) for record in policies}

        return {"reviews": review_map, "policies": policy_map}

    def is_image_abandoned(self, user_id: str | None, source_hash: str | None) -> bool:
        if not user_id or not source_hash:
            return False
        with SessionLocal() as session:
            record = session.execute(
                select(FaceRecognitionImagePolicyRecord).where(
                    FaceRecognitionImagePolicyRecord.user_id == user_id,
                    FaceRecognitionImagePolicyRecord.source_hash == source_hash,
                    FaceRecognitionImagePolicyRecord.is_abandoned.is_(True),
                )
            ).scalar_one_or_none()
            return record is not None

    def _serialize_review(self, record: FaceReviewRecord) -> Dict:
        return {
            "review_id": record.review_id,
            "user_id": record.user_id,
            "task_id": record.task_id,
            "face_id": record.face_id,
            "image_id": record.image_id,
            "person_id": record.person_id,
            "source_hash": record.source_hash,
            "is_inaccurate": bool(record.is_inaccurate),
            "comment_text": record.comment_text or "",
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
        }

    def _serialize_policy(self, record: FaceRecognitionImagePolicyRecord) -> Dict:
        return {
            "policy_id": record.policy_id,
            "user_id": record.user_id,
            "source_hash": record.source_hash,
            "is_abandoned": bool(record.is_abandoned),
            "last_task_id": record.last_task_id,
            "last_image_id": record.last_image_id,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
        }
