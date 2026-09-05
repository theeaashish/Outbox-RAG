from __future__ import annotations

from app.db.models.user import User
from app.modules.auth.schemas import LoginResponse, RegisterResponse, UserResponse


def to_register_response(user: User) -> RegisterResponse:
    """Map a User model into the registration response."""

    return RegisterResponse(
        id=user.id,
        email=user.email,
        name=user.name,
    )


def to_login_response(user: User) -> LoginResponse:
    """Map an authenticated User model into the login response."""

    return LoginResponse(
        id=user.id,
        email=user.email,
        name=user.name,
    )


def to_user_response(user: User) -> UserResponse:
    """Map an authenticated User model into the public user profile response."""

    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
    )
