from __future__ import annotations

from fastapi import UploadFile

from app.core.document.file_type import FileTypeResolver
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

    def validate(self, file: UploadFile) -> str:
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

        extension = FileTypeResolver.get_extension(file.filename)

        if not self._parser_registry.supports(extension):
            raise ValidationException(f"Unsupported document type: {extension}")

        return extension
