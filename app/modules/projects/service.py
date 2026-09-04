from __future__ import annotations

import logging
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.exceptions import (
    ConflictException,
    ResourceNotFoundException,
    ValidationException,
)
from app.db.models import KnowledgeBase, Project
from app.repositories.knowledge_base import KnowledgeBaseRepository
from app.repositories.project import ProjectRepository

logger = logging.getLogger(__name__)


class ProjectService:
    """Application service for project lifecycle operations."""

    def __init__(
        self,
        *,
        db: Session,
        project_repository: ProjectRepository,
        knowledge_base_repository: KnowledgeBaseRepository,
    ) -> None:
        self._db = db
        self._project_repository = project_repository
        self._knowledge_base_repository = knowledge_base_repository

    @staticmethod
    def _normalize_name(name: str) -> str:
        normalized = name.strip()
        if not normalized:
            raise ValidationException("Project name cannot be blank")
        return normalized

    def _get_project(self, *, user_id: UUID, project_id: UUID) -> Project:
        project = self._project_repository.get_by_user_and_id(
            user_id=user_id,
            project_id=project_id,
        )
        if project is None:
            raise ResourceNotFoundException("Project not found")
        return project

    def create_project(
        self,
        *,
        user_id: UUID,
        name: str,
        description: str | None,
    ) -> Project:
        normalized_name = self._normalize_name(name)
        try:
            project = self._project_repository.create(
                Project(
                    user_id=user_id,
                    name=normalized_name,
                    description=description,
                )
            )
            self._project_repository.flush()
            self._knowledge_base_repository.create(
                KnowledgeBase(
                    user_id=user_id,
                    project_id=project.id,
                    name=normalized_name,
                )
            )
            self._db.commit()
        except IntegrityError as exc:
            self._db.rollback()
            raise ConflictException("A project with this name already exists") from exc
        except Exception:
            self._db.rollback()
            raise
        self._project_repository.refresh(project)
        return project

    def list_projects(
        self,
        *,
        user_id: UUID,
        limit: int,
        offset: int,
    ) -> Sequence[Project]:
        return self._project_repository.list_by_user(
            user_id=user_id,
            limit=limit,
            offset=offset,
        )

    def get_project(self, *, user_id: UUID, project_id: UUID) -> Project:
        return self._get_project(user_id=user_id, project_id=project_id)

    def update_project(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        updates: dict[str, object],
    ) -> Project:
        project = self._get_project(user_id=user_id, project_id=project_id)
        if "name" in updates:
            raw_name = updates["name"]
            if not isinstance(raw_name, str):
                raise ValidationException("Project name cannot be blank")
            project.name = self._normalize_name(raw_name)
        if "description" in updates:
            raw_description = updates["description"]
            if raw_description is not None and not isinstance(raw_description, str):
                raise ValidationException(
                    "Project description must be a string or null"
                )
            project.description = raw_description
        try:
            self._db.commit()
        except IntegrityError as exc:
            self._db.rollback()
            raise ConflictException("A project with this name already exists") from exc
        except Exception:
            self._db.rollback()
            raise
        self._project_repository.refresh(project)
        return project

    def delete_project(self, *, user_id: UUID, project_id: UUID) -> None:
        project = self._get_project(user_id=user_id, project_id=project_id)
        try:
            self._project_repository.delete(project)
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise
        logger.info("Project deleted", extra={"project_id": str(project_id)})
