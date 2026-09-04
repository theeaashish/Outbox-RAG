from __future__ import annotations

from fastapi import APIRouter, Request, status

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
) -> LoginResponse:
    """Authenticate a user and establish a session."""

    response, _ = controller.login(
        request=request,
        user_agent=http_request.headers.get("user-agent"),
        ip_address=http_request.client.host if http_request.client else None,
    )

    return response
