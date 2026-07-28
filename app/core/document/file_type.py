from __future__ import annotations

from pathlib import Path

from app.core.exceptions import ValidationException


class FileTypeResolver:
    """Utility for resolving normalized file extensions."""

    @staticmethod
    def get_extension(filename: str) -> str:
        """Return the normalized file extension (lowercase, no leading dot)."""

        filename = filename.strip()

        extension = Path(filename).suffix.lower()

        if not extension:
            raise ValidationException("File must have a valid extension.")

        if filename.startswith(".") and filename == extension:
            raise ValidationException("File must have a valid extension.")

        return extension
