"""
SQLAlchemy database setup.
"""
from __future__ import annotations

import uuid

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import declarative_base, sessionmaker

from config import DATABASE_URL, SQL_ECHO, DEFAULT_TASK_VERSION


engine = create_engine(
    DATABASE_URL,
    echo=SQL_ECHO,
    future=True,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


def ensure_schema() -> None:
    """Apply lightweight schema changes needed for deploy-time compatibility."""
    Base.metadata.create_all(bind=engine)
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    if "tasks" not in table_names:
        return

    task_columns = {column["name"] for column in inspector.get_columns("tasks")}

    def add_task_column(name: str, ddl: str) -> None:
        if name in task_columns:
            return
        try:
            with engine.begin() as connection:
                connection.execute(text(f"ALTER TABLE tasks ADD COLUMN {name} {ddl}"))
        except OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise
        task_columns.add(name)

    nullable_suffix = "" if DATABASE_URL.startswith("sqlite") else " NULL"

    if not DATABASE_URL.startswith("sqlite"):
        with engine.begin() as connection:
            foreign_keys = connection.execute(
                text(
                    """
                    SELECT CONSTRAINT_NAME
                    FROM information_schema.KEY_COLUMN_USAGE
                    WHERE TABLE_SCHEMA = DATABASE()
                      AND TABLE_NAME = 'tasks'
                      AND COLUMN_NAME = 'user_id'
                      AND REFERENCED_TABLE_NAME = 'users'
                    """
                )
            ).scalars().all()
            for constraint_name in foreign_keys:
                connection.execute(text(f"ALTER TABLE tasks DROP FOREIGN KEY `{constraint_name}`"))

    add_task_column("user_id", f"VARCHAR(64){nullable_suffix}")
    add_task_column("operator_user_id", f"VARCHAR(64){nullable_suffix}")
    add_task_column("version", f"VARCHAR(16){nullable_suffix}")
    add_task_column("options", f"JSON{nullable_suffix}")
    add_task_column("result_summary", f"JSON{nullable_suffix}")
    add_task_column("asset_manifest", f"JSON{nullable_suffix}")
    add_task_column("worker_instance_id", f"VARCHAR(64){nullable_suffix}")
    add_task_column("worker_private_ip", f"VARCHAR(64){nullable_suffix}")
    add_task_column("worker_status", f"VARCHAR(32){nullable_suffix}")
    add_task_column("delete_state", f"VARCHAR(32){nullable_suffix}")
    add_task_column("expires_at", f"DATETIME{nullable_suffix}")
    add_task_column("deleted_at", f"DATETIME{nullable_suffix}")
    add_task_column("last_worker_sync_at", f"DATETIME{nullable_suffix}")
    with engine.begin() as connection:
        connection.execute(
            text("UPDATE tasks SET version = :default_version WHERE version IS NULL"),
            {"default_version": DEFAULT_TASK_VERSION},
        )

    if "artifacts" in table_names:
        artifact_columns = {column["name"] for column in inspector.get_columns("artifacts")}

        def add_artifact_column(name: str, ddl: str) -> None:
            if name in artifact_columns:
                return
            try:
                with engine.begin() as connection:
                    connection.execute(text(f"ALTER TABLE artifacts ADD COLUMN {name} {ddl}"))
            except OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise
            artifact_columns.add(name)

        add_artifact_column("stage", f"VARCHAR(64){nullable_suffix}")
        add_artifact_column("content_type", f"VARCHAR(255){nullable_suffix}")
        add_artifact_column("size_bytes", f"INTEGER{nullable_suffix}")
        add_artifact_column("sha256", f"VARCHAR(64){nullable_suffix}")
        add_artifact_column("storage_backend", f"VARCHAR(32){nullable_suffix}")
        add_artifact_column("object_key", f"VARCHAR(512){nullable_suffix}")
        add_artifact_column("asset_url", f"VARCHAR(512){nullable_suffix}")
        add_artifact_column("metadata", f"JSON{nullable_suffix}")

    from sqlalchemy import select

    from backend.models import SubjectUserRecord, TaskRecord, UserRecord

    with SessionLocal() as session:
        tasks = session.execute(select(TaskRecord)).scalars().all()
        if not tasks:
            return

        subjects_by_username = {
            record.username: record
            for record in session.execute(select(SubjectUserRecord)).scalars().all()
            if record.username
        }
        auth_users_by_username = {
            record.username: record
            for record in session.execute(select(UserRecord)).scalars().all()
            if record.username
        }

        for task in tasks:
            options = dict(task.options or {})
            original_user_id = str(task.user_id or "").strip() or None
            legacy_subject_user_id = str(options.pop("subject_user_id", "") or "").strip() or None
            options.pop("operator_user_id", None)
            survey_username = str(options.get("survey_username") or "").strip().lower() or None

            if legacy_subject_user_id:
                task.user_id = legacy_subject_user_id

            if not str(task.user_id or "").strip() and survey_username:
                auth_user = auth_users_by_username.get(survey_username)
                if auth_user is not None:
                    task.user_id = auth_user.user_id
                    subject = subjects_by_username.get(survey_username)
                    if subject is None:
                        subject = SubjectUserRecord(
                            user_id=auth_user.user_id,
                            username=survey_username,
                            display_name=survey_username,
                            linked_auth_user_id=auth_user.user_id,
                            created_at=auth_user.created_at,
                            updated_at=auth_user.updated_at,
                        )
                        session.add(subject)
                        subjects_by_username[survey_username] = subject
                else:
                    subject = subjects_by_username.get(survey_username)
                    if subject is None:
                        subject = SubjectUserRecord(
                            user_id=uuid.uuid4().hex,
                            username=survey_username,
                            display_name=survey_username,
                            linked_auth_user_id=None,
                            created_at=task.created_at,
                            updated_at=task.updated_at,
                        )
                        session.add(subject)
                        subjects_by_username[survey_username] = subject
                    task.user_id = subject.user_id

            normalized_owner_user_id = str(task.user_id or "").strip() or None
            if normalized_owner_user_id:
                task.user_id = normalized_owner_user_id
            if not str(task.operator_user_id or "").strip():
                task.operator_user_id = original_user_id or normalized_owner_user_id

            task.options = options
            session.add(task)

        session.commit()
