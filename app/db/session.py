from collections.abc import Generator

from sqlalchemy.orm import Session, sessionmaker

from app.db.database import engine

SessionLocal = sessionmaker(
    bind=engine, autoflush=False, autocommit=False, expire_on_commit=False
)


def get_db() -> Generator[Session]:
    """Provide a database session for each request"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
