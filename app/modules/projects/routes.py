from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Query, Response, status

from app.dependencies.auth import CurrentUser
from app.dependencies.controllers import ProjectControllerDep
from app.modules.projects.schemas import (
    ProjectCreateRequest,
    ProjectListResponse,
    ProjectResponse,
    ProjectUpdateRequest,
)

router = APIRouter(prefix="/projects", tags=["Projects"])


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_project(
    request: ProjectCreateRequest,
    controller: ProjectControllerDep,
    current_user: CurrentUser,
) -> ProjectResponse:
    return controller.create_project(user_id=current_user.id, request=request)


@router.get("", response_model=ProjectListResponse)
def list_projects(
    controller: ProjectControllerDep,
    current_user: CurrentUser,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> ProjectListResponse:
    return controller.list_projects(user_id=current_user.id, limit=limit, offset=offset)


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(
    project_id: UUID,
    controller: ProjectControllerDep,
    current_user: CurrentUser,
) -> ProjectResponse:
    return controller.get_project(user_id=current_user.id, project_id=project_id)


@router.patch("/{project_id}", response_model=ProjectResponse)
def update_project(
    project_id: UUID,
    request: ProjectUpdateRequest,
    controller: ProjectControllerDep,
    current_user: CurrentUser,
) -> ProjectResponse:
    return controller.update_project(
        user_id=current_user.id,
        project_id=project_id,
        request=request,
    )


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(
    project_id: UUID,
    controller: ProjectControllerDep,
    current_user: CurrentUser,
) -> Response:
    controller.delete_project(user_id=current_user.id, project_id=project_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
