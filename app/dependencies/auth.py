from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.core.config import settings
from app.core.exceptions import UnauthorizedException
from app.db.models.user import User
from app.dependencies.services import AuthServiceDep


def get_current_user(
    request: Request,
    auth_service: AuthServiceDep,
) -> User:
    """Resolve the authenticated user from the session cookie."""

    token = request.cookies.get(settings.session_cookie_name)

    if not token:
        raise UnauthorizedException("Authentication required")

    return auth_service.authenticate_session(token=token)


CurrentUser = Annotated[User, Depends(get_current_user)]
