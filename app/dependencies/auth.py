from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import Depends, Request

from app.core.exceptions import UnauthorizedException


def get_authenticated_user_id(request: Request) -> UUID:
    """Read the user ID placed on request state by the auth milestone."""
    user_id = getattr(request.state, "user_id", None)
    if isinstance(user_id, UUID):
        return user_id
    if isinstance(user_id, str):
        try:
            return UUID(user_id)
        except ValueError:
            pass
    raise UnauthorizedException("Authentication required")


AuthenticatedUserIdDep = Annotated[
    UUID,
    Depends(get_authenticated_user_id),
]
