"""
SQLAlchemy database setup.
"""
from __future__ import annotations

from sqlalchemy import create_engine
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
