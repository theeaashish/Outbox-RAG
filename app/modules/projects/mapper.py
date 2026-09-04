from __future__ import annotations

from collections.abc import Sequence

from app.db.models import Project
from app.modules.projects.schemas import ProjectListResponse, ProjectResponse


def to_project_response(project: Project) -> ProjectResponse:
    return ProjectResponse.model_validate(project)


def to_project_list_response(projects: Sequence[Project]) -> ProjectListResponse:
    return ProjectListResponse(
        results=[to_project_response(project) for project in projects]
    )
