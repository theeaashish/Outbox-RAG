from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from app.core.exceptions import StorageException
from app.core.storage.base import StorageService

logger = logging.getLogger(__name__)


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

    def save(self, path: str, content: bytes) -> None:
        """Write content via a temp file and atomic rename where practical."""

        target = self._resolve_safe_path(path)
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
        except Exception as exc:
            logger.exception(
                "Storage save failed",
                extra={"path": path, "size": len(content)},
            )
            raise StorageException("Failed to save file to storage") from exc

        logger.info("Stored file", extra={"path": path, "size": len(content)})

    def delete(self, path: str) -> None:
        """Delete a stored file if present."""

        target = self._resolve_safe_path(path)

        try:
            if target.is_file():
                target.unlink()
                logger.info("Deleted stored file", extra={"path": path})
        except StorageException:
            raise
        except Exception as exc:
            logger.exception("Storage delete failed", extra={"path": path})
            raise StorageException("Failed to delete file from storage") from exc

    def exists(self, path: str) -> bool:
        """Return whether a relative key exists on disk."""

        try:
            return self._resolve_safe_path(path).is_file()
        except StorageException:
            return False

    def read(self, path: str) -> bytes:
        """Read file content from local filesystem storage."""

        target = self._resolve_safe_path(path)

        try:
            return target.read_bytes()
        except FileNotFoundError as exc:
            logger.warning(
                "Stored file not found",
                extra={"path": path},
            )
            raise StorageException("Stored file not found") from exc
        except Exception as exc:
            logger.exception(
                "Storage read failed",
                extra={"path": path},
            )
            raise StorageException("Failed to read file from storage") from exc
