from __future__ import annotations

from typing import Final

import psycopg.errors
from sqlalchemy.exc import (
    DataError,
    DisconnectionError,
    IntegrityError,
    InvalidRequestError,
    ProgrammingError,
)
from sqlalchemy.exc import (
    TimeoutError as SATimeoutError,
)

from app.core.exceptions import (
    DatabaseException,
    TransientDatabaseException,
)

# PostgreSQL SQLSTATE prefixes representing transient connection failures
_TRANSIENT_SQLSTATE_PREFIXES: Final[tuple[str, ...]] = ("08",)

# Explicit PostgreSQL SQLSTATE codes that are retryable
_TRANSIENT_SQLSTATE_CODES: Final[frozenset[str]] = frozenset(
    {
        "40001",  # Serialization failure
        "40P01",  # Deadlock detected
        "53300",  # Too many connections
        "57014",  # Query canceled (e.g. statement_timeout, lock_timeout)
        "57P01",  # Admin shutdown
        "57P02",  # Crash shutdown
        "57P03",  # Cannot connect now
    }
)

# PostgreSQL SQLSTATE prefixes and codes that are permanent / non-retryable
_PERMANENT_SQLSTATE_PREFIXES: Final[tuple[str, ...]] = (
    "22",  # Data exception
    "23",  # Integrity constraint violation (23505 unique, 23503 foreign key, 23502 not null, 23514 check)
    "28",  # Invalid authorization (bad password, user not allowed)
    "42",  # Syntax error or access rule violation (42P01 undefined table, 42703 undefined column)
    "53",  # Insufficient resources other than 53300 (e.g. 53100 disk full, 53200 OOM)
)

_TRANSIENT_PSYCOPG_CLASSES: Final[tuple[type[BaseException], ...]] = (
    psycopg.errors.AdminShutdown,
    psycopg.errors.CrashShutdown,
    psycopg.errors.CannotConnectNow,
    psycopg.errors.QueryCanceled,
    psycopg.errors.DeadlockDetected,
    psycopg.errors.SerializationFailure,
    psycopg.errors.ConnectionException,
    psycopg.errors.TooManyConnections,
)

_PERMANENT_PSYCOPG_CLASSES: Final[tuple[type[BaseException], ...]] = (
    psycopg.errors.IntegrityError,
    psycopg.errors.DataError,
    psycopg.errors.ProgrammingError,
    psycopg.errors.SyntaxError,
    psycopg.errors.FeatureNotSupported,
    psycopg.errors.DiskFull,
    psycopg.errors.InsufficientResources,
    psycopg.errors.OutOfMemory,
)


def _get_sqlstate(exc: BaseException) -> str | None:
    """Extract PostgreSQL SQLSTATE code from exception or DBAPI wrapper."""
    sqlstate = getattr(exc, "sqlstate", None) or getattr(exc, "pgcode", None)
    if isinstance(sqlstate, str):
        return sqlstate

    orig = getattr(exc, "orig", None)
    if orig is not None:
        orig_sqlstate = getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
        if isinstance(orig_sqlstate, str):
            return orig_sqlstate

    return None


def is_transient_database_error(exc: BaseException | None) -> bool:
    """
    Determine whether a database exception represents a transient, retryable failure
    based on explicit PostgreSQL SQLSTATE codes, psycopg error types, and SQLAlchemy hierarchy.
    """
    if exc is None:
        return False

    if isinstance(exc, TransientDatabaseException):
        return True
    if isinstance(exc, DatabaseException):
        return False

    # Connection checkout or execution timeouts
    if isinstance(
        exc, (SATimeoutError, DisconnectionError, TimeoutError, ConnectionError)
    ):
        return True

    # Check PostgreSQL SQLSTATE if available
    sqlstate = _get_sqlstate(exc)
    if sqlstate:
        if sqlstate in _TRANSIENT_SQLSTATE_CODES:
            return True
        if any(sqlstate.startswith(prefix) for prefix in _TRANSIENT_SQLSTATE_PREFIXES):
            return True
        if any(sqlstate.startswith(prefix) for prefix in _PERMANENT_SQLSTATE_PREFIXES):
            return False

    # Check direct psycopg error types
    orig = getattr(exc, "orig", exc)
    if isinstance(orig, _TRANSIENT_PSYCOPG_CLASSES):
        return True
    if isinstance(orig, _PERMANENT_PSYCOPG_CLASSES):
        return False

    # Permanent SQLAlchemy error categories
    if isinstance(
        exc, (IntegrityError, DataError, ProgrammingError, InvalidRequestError)
    ):
        return False

    # Check underlying causes recursively
    cause = getattr(exc, "__cause__", None)
    if cause is not None and is_transient_database_error(cause):
        return True

    context = getattr(exc, "__context__", None)
    return bool(context is not None and is_transient_database_error(context))


def classify_database_exception(
    exc: Exception,
    *,
    transient_message: str = "Database temporarily unavailable",
    permanent_message: str = "Database operation failed",
) -> DatabaseException:
    """
    Wrap an underlying database exception into either TransientDatabaseException
    (if transient) or DatabaseException (if permanent).
    """
    if isinstance(exc, (TransientDatabaseException, DatabaseException)):
        return exc

    if is_transient_database_error(exc):
        return TransientDatabaseException(transient_message)

    return DatabaseException(permanent_message)
