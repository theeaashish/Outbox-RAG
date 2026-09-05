from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.core.config import settings
from app.core.exceptions import UnauthorizedException
from app.db.models.session import Session as SessionModel
from app.db.models.user import User
from app.dependencies.services import AuthServiceDep


def _resolve_session(
    request: Request,
    auth_service: AuthServiceDep,
) -> SessionModel:
    """Resolve the active session from the session cookie."""

    token = request.cookies.get(settings.session_cookie_name)

    if not token:
        raise UnauthorizedException("Authentication required")

    return auth_service.authenticate_session(token=token)


def get_current_session(
    session: Annotated[SessionModel, Depends(_resolve_session)],
    auth_service: AuthServiceDep,
) -> SessionModel:
    """Resolve the active session and record authenticated activity."""

    auth_service.record_session_activity(session=session)
    return session


CurrentSession = Annotated[SessionModel, Depends(get_current_session)]


def get_current_user(
    current_session: Annotated[SessionModel, Depends(get_current_session)],
) -> User:
    """Resolve the authenticated user from the active session."""

    return current_session.user


CurrentUser = Annotated[User, Depends(get_current_user)]


def get_current_session_no_touch(
    session: Annotated[SessionModel, Depends(_resolve_session)],
) -> SessionModel:
    """Resolve the active session without recording terminal-request activity."""

    return session


CurrentSessionNoTouch = Annotated[
    SessionModel,
    Depends(get_current_session_no_touch),
]


def get_current_user_no_touch(
    current_session: Annotated[
        SessionModel,
        Depends(get_current_session_no_touch),
    ],
) -> User:
    """Resolve the authenticated user without recording terminal-request activity."""

    return current_session.user


CurrentUserNoTouch = Annotated[User, Depends(get_current_user_no_touch)]
