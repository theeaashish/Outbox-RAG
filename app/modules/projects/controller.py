from __future__ import annotations

from uuid import UUID

from app.modules.projects import mapper
from app.modules.projects.schemas import (
    ProjectCreateRequest,
    ProjectListResponse,
    ProjectResponse,
    ProjectUpdateRequest,
)
from app.modules.projects.service import ProjectService


class ProjectController:
    """Thin controller for project request orchestration."""

    def __init__(self, *, project_service: ProjectService) -> None:
        self._project_service = project_service

    def create_project(
        self,
        *,
        user_id: UUID,
        request: ProjectCreateRequest,
    ) -> ProjectResponse:
        project = self._project_service.create_project(
            user_id=user_id,
            name=request.name,
            description=request.description,
        )
        return mapper.to_project_response(project)

    def list_projects(
        self,
        *,
        user_id: UUID,
        limit: int,
        offset: int,
    ) -> ProjectListResponse:
        projects = self._project_service.list_projects(
            user_id=user_id,
            limit=limit,
            offset=offset,
        )
        return mapper.to_project_list_response(projects)

    def get_project(self, *, user_id: UUID, project_id: UUID) -> ProjectResponse:
        return mapper.to_project_response(
            self._project_service.get_project(user_id=user_id, project_id=project_id)
        )

    def update_project(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        request: ProjectUpdateRequest,
    ) -> ProjectResponse:
        project = self._project_service.update_project(
            user_id=user_id,
            project_id=project_id,
            updates=request.model_dump(exclude_unset=True),
        )
        return mapper.to_project_response(project)

    def delete_project(self, *, user_id: UUID, project_id: UUID) -> None:
        self._project_service.delete_project(user_id=user_id, project_id=project_id)
