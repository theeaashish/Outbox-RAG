from __future__ import annotations

from app.core.config import settings
from app.core.document.file_type import FileTypeResolver
from app.core.document.incoming_file import IncomingFile
from app.core.document.parsers.registry import DocumentParserRegistry
from app.core.exceptions import ValidationException


class UploadValidator:
    """Validates upload files against configured rules."""

    def __init__(
        self,
        *,
        parser_registry: DocumentParserRegistry,
    ) -> None:
        self._parser_registry = parser_registry

    def validate_metadata(
        self,
        *,
        filename: str,
        size: int,
    ) -> str:
        """Validate upload metadata before file bytes are available."""

        if not filename or not filename.strip():
            raise ValidationException("File name is required")

        if len(filename) > 255:
            raise ValidationException(
                "File name exceeds the maximum length of 255 characters"
            )

        if size < 1:
            raise ValidationException("Empty file is not allowed")

        max_bytes = settings.max_upload_size_mb * 1024 * 1024

        if size > max_bytes:
            raise ValidationException(
                f"File exceeds maximum size of {settings.max_upload_size_mb}MB"
            )

        extension = FileTypeResolver.get_extension(filename)

        if not self._parser_registry.supports(extension):
            raise ValidationException(f"Unsupported document type: {extension}")

        return extension

    def validate(self, file: IncomingFile) -> str:
        """Validate a complete file before processing or storage."""

        if not file.content:
            raise ValidationException("Empty file is not allowed")

        if len(file.content) != file.size:
            raise ValidationException("File size does not match uploaded content")

        return self.validate_metadata(
            filename=file.filename,
            size=file.size,
        )
