from __future__ import annotations

from fastapi import APIRouter, status

from app.dependencies.controllers import AuthControllerDep
from app.modules.auth.schemas import RegisterRequest, RegisterResponse

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
