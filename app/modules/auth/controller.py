from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from app.core.config import settings
from app.modules.auth import mapper
from app.modules.auth.schemas import (
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    RegisterResponse,
    UserResponse,
)
from app.modules.auth.service import AuthenticatedSession, AuthService

if TYPE_CHECKING:
    from app.db.models.user import User


class AuthController:
    """Thin controller responsible for authentication request orchestration."""

    def __init__(self, *, auth_service: AuthService) -> None:
        self._auth_service = auth_service

    def register(self, request: RegisterRequest) -> RegisterResponse:
        """Register a new local user."""

        user = self._auth_service.register(
            email=request.email,
            password=request.password,
            name=request.name,
        )

        return mapper.to_register_response(user)

    def login(
        self,
        request: LoginRequest,
        *,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> tuple[LoginResponse, AuthenticatedSession]:
        """Authenticate a user and return the mapped response alongside the internal session."""

        authenticated_session = self._auth_service.login(
            email=request.email,
            password=request.password,
            user_agent=user_agent,
            ip_address=ip_address,
            session_lifetime=timedelta(days=settings.session_lifetime_days),
        )

        response = mapper.to_login_response(authenticated_session.user)

        return response, authenticated_session

    def get_me(self, *, user: User) -> UserResponse:
        """Return the public profile for the authenticated user."""

        return mapper.to_user_response(user)
