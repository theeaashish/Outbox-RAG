from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Application lifecycle"""

    # Startup logic - configure logging
    configure_logging()
    yield
    # shutdown logic


app = FastAPI(title=settings.app_name, debug=settings.app_debug, lifespan=lifespan)

register_exception_handlers(app)

app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    """root endpoint"""
    return {
        "message": f"Welcome to {settings.app_name}",
    }


@app.get("/health", tags=["Health"])
async def health() -> dict[str, str]:
    """health check endpoint"""
    return {
        "status": "healthy",
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
