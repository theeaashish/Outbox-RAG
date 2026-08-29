from __future__ import annotations

from app.modules.auth import mapper
from app.modules.auth.schemas import RegisterRequest, RegisterResponse
from app.modules.auth.service import AuthService


class AuthController:
    """Thin controller responsible for authentication request orchestration."""

    def __init__(self, *, auth_service: AuthService) -> None:
        self._auth_service = auth_service

    def register(self, request: RegisterRequest) -> RegisterResponse:
        """Register a new local user."""

        user = self._auth_service.register(
            email=request.email, password=request.password, name=request.name
        )

        return mapper.to_register_response(user)
