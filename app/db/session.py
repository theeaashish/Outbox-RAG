from __future__ import annotations

from collections.abc import Generator, Iterator
from contextlib import contextmanager

from sqlalchemy.orm import Session, sessionmaker

from app.db.database import engine

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


@contextmanager
def managed_session() -> Iterator[Session]:
    """
    Provide an application-owned database session with deterministic cleanup.

    The caller owns transaction boundaries and must explicitly commit successful
    work. Exceptions trigger a rollback before the session is closed.
    """

    db = SessionLocal()

    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db() -> Generator[Session]:
    """Provide a database session for each request"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
