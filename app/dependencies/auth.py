from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.core.config import settings
from app.core.exceptions import UnauthorizedException
from app.db.models.session import Session as SessionModel
from app.db.models.user import User
from app.dependencies.services import (
    AuthServiceDep,
    SessionRepositoryDep,
    SessionTokenServiceDep,
)


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


def get_current_session(
    request: Request,
    session_token_service: SessionTokenServiceDep,
    session_repository: SessionRepositoryDep,
) -> SessionModel:
    """Resolve the active authenticated session from the session cookie."""

    token = request.cookies.get(settings.session_cookie_name)

    if not token:
        raise UnauthorizedException("Authentication required")

    token_hash = session_token_service.hash_token(token)
    session = session_repository.get_active_by_token_hash(token_hash=token_hash)

    if session is None:
        raise UnauthorizedException("Authentication required")

    return session


CurrentSession = Annotated[SessionModel, Depends(get_current_session)]
