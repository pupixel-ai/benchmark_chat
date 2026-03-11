"""
Database models.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.db import Base


class TaskRecord(Base):
    __tablename__ = "tasks"

    task_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    upload_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    task_dir: Mapped[str] = mapped_column(String(512), nullable=False)
    progress: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    uploads: Mapped[list | None] = mapped_column(JSON, nullable=True)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
