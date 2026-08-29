from __future__ import annotations

from app.db.models.user import User
from app.modules.auth.schemas import RegisterResponse


def to_register_response(user: User) -> RegisterResponse:
    """Map a User model into the registration response."""

    return RegisterResponse(
        id=user.id,
        email=user.email,
        name=user.name,
    )
