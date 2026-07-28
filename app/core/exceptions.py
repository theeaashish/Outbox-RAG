from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class AppException(Exception):
    """Base exception for the application"""

    def __init__(self, message: str, status_code: int = HTTPStatus.BAD_REQUEST) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class ValidationException(AppException):
    """Raised when validation fails"""

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            status_code=HTTPStatus.BAD_REQUEST,
        )


class UnsupportedDocumentTypeError(ValidationException):
    """Raised when an unsupported document type is requested"""

    def __init__(self, extension: str) -> None:
        super().__init__(message=f"Unsupported document type: {extension}")



class NotFoundException(AppException):
    """Raised when a resource is not found"""

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_FOUND,
        )


class ConflictException(AppException):
    """Raised when a resource already exists"""

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            status_code=HTTPStatus.CONFLICT,
        )


class UnauthorizedException(AppException):
    """Raised when a user is not authorized to perform an action"""

    def __init__(self, message: str = "Unauthorized") -> None:
        super().__init__(
            message=message,
            status_code=HTTPStatus.UNAUTHORIZED,
        )


class ForbiddenException(AppException):
    """Raised when a user is not allowed to perform an action"""

    def __init__(self, message: str = "Forbidden") -> None:
        super().__init__(
            message=message,
            status_code=HTTPStatus.FORBIDDEN,
        )


class AIServiceException(AppException):
    """Raised when an AI service fails"""

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


class DatabaseException(AppException):
    """Raised when a database operation fails"""

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


async def app_exception_handler(_: Request, exc: AppException) -> JSONResponse:
    """Handle AppException and return JSON response"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "message": exc.message,
            },
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register exception handlers"""
    app.add_exception_handler(AppException, app_exception_handler)
