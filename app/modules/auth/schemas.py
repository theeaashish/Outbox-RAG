from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class RegisterRequest(BaseModel):
    """Request payload for local user registration."""

    email: EmailStr
    password: str = Field(
        min_length=12,
        max_length=1024,
    )
    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
    )


class RegisterResponse(BaseModel):
    """Response returned after successful registration."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: EmailStr
    name: str | None
