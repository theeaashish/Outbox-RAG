from fastapi import APIRouter

from app.modules.document.routes import router as document_router

api_router = APIRouter()

api_router.include_router(document_router)
