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

    def validate(self, file: IncomingFile) -> str:
        """
        Validate the upload file and return the normalized file extension.

        Args:
            file: The uploaded file to validate

        Returns:
            The normalized file extension (e.g., ".pdf", ".txt")

        Raises:
            ValidationException: If the file is invalid or unsupported
        """

        if not file.filename:
            raise ValidationException("File name is required")

        if file.size == 0 or not file.content:
            raise ValidationException("Empty file is not allowed")

        max_bytes = settings.max_upload_size_mb * 1024 * 1024
        if file.size > max_bytes:
            raise ValidationException(
                f"File exceeds maximum size of {settings.max_upload_size_mb}MB"
            )

        extension = FileTypeResolver.get_extension(file.filename)

        if not self._parser_registry.supports(extension):
            raise ValidationException(f"Unsupported document type: {extension}")

        return extension
