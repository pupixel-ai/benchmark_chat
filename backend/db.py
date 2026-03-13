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
    if "user_id" in task_columns:
        return

    with engine.begin() as connection:
        if DATABASE_URL.startswith("sqlite"):
            connection.execute(text("ALTER TABLE tasks ADD COLUMN user_id VARCHAR(64)"))
        else:
            connection.execute(text("ALTER TABLE tasks ADD COLUMN user_id VARCHAR(64) NULL"))
