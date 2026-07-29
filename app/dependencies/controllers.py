from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from app.dependencies.services import DocumentServiceDep
from app.modules.document.controller import DocumentController


def get_document_controller(
    service: DocumentServiceDep,
) -> DocumentController:
    """Return a configured DocumentController."""

    return DocumentController(service)


DocumentControllerDep = Annotated[DocumentController, Depends(get_document_controller)]
