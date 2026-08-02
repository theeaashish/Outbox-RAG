from fastapi import APIRouter

from app.modules.conversations import router as conversation_router
from app.modules.document.routes import router as document_router
from app.modules.retrieval.routes import router as retrieval_router

api_router = APIRouter()

api_router.include_router(document_router)
api_router.include_router(retrieval_router)
api_router.include_router(conversation_router)
