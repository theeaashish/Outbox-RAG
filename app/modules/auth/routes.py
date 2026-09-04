from __future__ import annotations

from typing import Literal, cast

from fastapi import APIRouter, Request, Response, status

from app.core.config import settings
from app.dependencies.controllers import AuthControllerDep
from app.modules.auth.schemas import (
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    RegisterResponse,
)

router = APIRouter(
    prefix="/auth",
    tags=["auth"],
)


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    request: RegisterRequest,
    controller: AuthControllerDep,
) -> RegisterResponse:
    """Register a new local user."""

    return controller.register(request=request)


@router.post(
    "/login",
    response_model=LoginResponse,
    status_code=status.HTTP_200_OK,
)
def login(
    request: LoginRequest,
    controller: AuthControllerDep,
    http_request: Request,
    response: Response,
) -> LoginResponse:
    """Authenticate a user and establish a session."""

    login_response, authenticated_session = controller.login(
        request=request,
        user_agent=http_request.headers.get("user-agent"),
        ip_address=http_request.client.host if http_request.client else None,
    )

    response.set_cookie(
        key=settings.session_cookie_name,
        value=authenticated_session.token,
        max_age=settings.session_lifetime_days * 24 * 60 * 60,
        httponly=settings.session_cookie_httponly,
        secure=settings.session_cookie_secure,
        samesite=cast(
            Literal["lax", "strict", "none"],
            settings.session_cookie_samesite,
        ),
        path=settings.session_cookie_path,
    )

    return login_response
