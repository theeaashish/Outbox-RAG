from __future__ import annotations

from collections.abc import Sequence

from app.db.models import Project
from app.modules.projects.schemas import ProjectListResponse, ProjectResponse


def to_project_response(project: Project) -> ProjectResponse:
    if project.knowledge_base is None:
        raise RuntimeError("Project is missing its knowledge base")

    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        knowledge_base_id=project.knowledge_base.id,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


def to_project_list_response(projects: Sequence[Project]) -> ProjectListResponse:
    return ProjectListResponse(
        results=[to_project_response(project) for project in projects]
    )
