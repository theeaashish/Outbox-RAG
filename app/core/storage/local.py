from __future__ import annotations

import errno
import logging
import os
import tempfile
from pathlib import Path
from typing import Final

from app.core.exceptions import (
    ResourceNotFoundException,
    StorageException,
    TransientStorageException,
)
from app.core.storage.base import StorageService, StoredObjectMetadata

logger = logging.getLogger(__name__)

_TRANSIENT_ERRNOS: Final[frozenset[int]] = frozenset(
    {
        errno.ETIMEDOUT,
        errno.EBUSY,
        errno.EAGAIN,
        errno.EWOULDBLOCK,
        errno.EIO,
        errno.EINTR,
        errno.ENOLCK,
        errno.ESTALE,
    }
)


def _is_transient_fs_error(exc: Exception) -> bool:
    """Determine if a filesystem exception is transient based on type and errno."""
    if isinstance(exc, (TimeoutError, InterruptedError, BlockingIOError)):
        return True
    return isinstance(exc, OSError) and exc.errno in _TRANSIENT_ERRNOS


class LocalFilesystemStorage(StorageService):
    """Filesystem-backed storage rooted at a configured upload directory."""

    def __init__(self, *, root_directory: str) -> None:
        self._root = Path(root_directory).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve_safe_path(self, path: str) -> Path:
        """Resolve a relative key and refuse path traversal outside the root."""

        if not path or path.startswith("/") or ".." in Path(path).parts:
            raise StorageException("Invalid storage path")

        resolved = (self._root / path).resolve()

        try:
            resolved.relative_to(self._root)
        except ValueError as exc:
            raise StorageException("Invalid storage path") from exc

        return resolved

    def save(self, key: str, content: bytes) -> None:
        """Write content via a temp file and atomic rename where practical."""

        target = self._resolve_safe_path(key)
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent,
                delete=False,
                prefix=f".{target.name}.",
                suffix=".tmp",
            ) as tmp:
                tmp.write(content)
                tmp.flush()
                os.fsync(tmp.fileno())
                temp_name = tmp.name

            os.replace(temp_name, target)
        except StorageException:
            raise
        except PermissionError as exc:
            logger.exception(
                "Storage save permission denied",
                extra={"path": key, "size": len(content)},
            )
            raise StorageException("Permission denied saving file to storage") from exc
        except Exception as exc:
            logger.exception(
                "Storage save failed",
                extra={"path": key, "size": len(content)},
            )
            if _is_transient_fs_error(exc):
                raise TransientStorageException(
                    "Failed to save file to storage"
                ) from exc
            raise StorageException("Failed to save file to storage") from exc

        logger.info("Stored file", extra={"path": key, "size": len(content)})

    def delete(self, key: str) -> None:
        """Delete a stored file if present."""

        target = self._resolve_safe_path(key)

        try:
            if target.is_file():
                target.unlink()
                logger.info("Deleted stored file", extra={"path": key})
        except StorageException:
            raise
        except PermissionError as exc:
            logger.exception("Storage delete permission denied", extra={"path": key})
            raise StorageException(
                "Permission denied deleting file from storage"
            ) from exc
        except Exception as exc:
            logger.exception("Storage delete failed", extra={"path": key})
            if _is_transient_fs_error(exc):
                raise TransientStorageException(
                    "Failed to delete file from storage"
                ) from exc
            raise StorageException("Failed to delete file from storage") from exc

    def head(self, key: str) -> StoredObjectMetadata:
        """Return metadata for a stored local file."""

        target = self._resolve_safe_path(key)

        try:
            if not target.is_file():
                raise ResourceNotFoundException("Stored object not found")

            stat_result = target.stat()

            return StoredObjectMetadata(
                size=stat_result.st_size,
                content_type=None,
                etag=None,
            )

        except (ResourceNotFoundException, StorageException):
            raise
        except FileNotFoundError as exc:
            logger.warning(
                "Stored object not found",
                extra={"key": key},
            )
            raise ResourceNotFoundException("Stored object not found") from exc
        except OSError as exc:
            logger.exception(
                "Storage head failed",
                extra={"key": key},
            )

            if _is_transient_fs_error(exc):
                raise TransientStorageException(
                    "Failed to retrieve stored object metadata"
                ) from exc

            raise StorageException("Failed to retrieve stored object metadata") from exc

    def read(self, key: str) -> bytes:
        """Read file content from local filesystem storage."""

        target = self._resolve_safe_path(key)

        try:
            if not target.is_file():
                raise ResourceNotFoundException("Stored object not found")
            return target.read_bytes()
        except (ResourceNotFoundException, StorageException):
            raise
        except FileNotFoundError as exc:
            logger.warning(
                "Stored file not found",
                extra={"path": key},
            )
            raise ResourceNotFoundException("Stored object not found") from exc
        except PermissionError as exc:
            logger.exception("Storage read permission denied", extra={"path": key})
            raise StorageException("Permission denied accessing storage file") from exc
        except Exception as exc:
            logger.exception(
                "Storage read failed",
                extra={"path": key},
            )
            if _is_transient_fs_error(exc):
                raise TransientStorageException(
                    "Failed to read file from storage"
                ) from exc
            raise StorageException("Failed to read file from storage") from exc
