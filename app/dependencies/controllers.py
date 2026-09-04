from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from app.dependencies.services import (
    AuthServiceDep,
    DocumentServiceDep,
    ProjectServiceDep,
)
from app.modules.auth.controller import AuthController
from app.modules.document.controller import DocumentController
from app.modules.projects.controller import ProjectController


def get_document_controller(
    service: DocumentServiceDep,
) -> DocumentController:
    """Return a configured DocumentController."""

    return DocumentController(service)


DocumentControllerDep = Annotated[DocumentController, Depends(get_document_controller)]


def get_auth_controller(
    auth_service: AuthServiceDep,
) -> AuthController:
    """Return a configured AuthController."""

    return AuthController(
        auth_service=auth_service,
    )


AuthControllerDep = Annotated[
    AuthController,
    Depends(get_auth_controller),
]


def get_project_controller(
    project_service: ProjectServiceDep,
) -> ProjectController:
    """Return a configured ProjectController."""
    return ProjectController(project_service=project_service)


ProjectControllerDep = Annotated[
    ProjectController,
    Depends(get_project_controller),
]
