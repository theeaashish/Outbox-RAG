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


class UserResponse(BaseModel):
    """Response returned for authenticated user information."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: EmailStr
    name: str | None


class RegisterResponse(UserResponse):
    """Response returned after successful registration."""


class LoginRequest(BaseModel):
    """Request payload for user login."""

    email: EmailStr
    password: str = Field(
        min_length=12,
        max_length=1024,
    )


class LoginResponse(UserResponse):
    """Response returned after successful authentication."""
