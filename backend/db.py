"""
SQLAlchemy database setup.
"""
from __future__ import annotations

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

from config import DATABASE_URL, SQL_ECHO


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
    inspector = inspect(engine)
    if "tasks" not in inspector.get_table_names():
        return

    task_columns = {column["name"] for column in inspector.get_columns("tasks")}

    def add_task_column(name: str, ddl: str) -> None:
        if name in task_columns:
            return
        with engine.begin() as connection:
            connection.execute(text(f"ALTER TABLE tasks ADD COLUMN {name} {ddl}"))
        task_columns.add(name)

    nullable_suffix = "" if DATABASE_URL.startswith("sqlite") else " NULL"

    add_task_column("user_id", f"VARCHAR(64){nullable_suffix}")
    add_task_column("result_summary", f"JSON{nullable_suffix}")
    add_task_column("asset_manifest", f"JSON{nullable_suffix}")
    add_task_column("worker_instance_id", f"VARCHAR(64){nullable_suffix}")
    add_task_column("worker_private_ip", f"VARCHAR(64){nullable_suffix}")
    add_task_column("worker_status", f"VARCHAR(32){nullable_suffix}")
    add_task_column("delete_state", f"VARCHAR(32){nullable_suffix}")
    add_task_column("expires_at", f"DATETIME{nullable_suffix}")
    add_task_column("deleted_at", f"DATETIME{nullable_suffix}")
    add_task_column("last_worker_sync_at", f"DATETIME{nullable_suffix}")
