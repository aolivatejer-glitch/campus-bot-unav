from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from rag_chatbot import __version__
from rag_chatbot.api.routes import router
from rag_chatbot.config import get_settings
from rag_chatbot.logging_config import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings)
    app = FastAPI(
        title="rag-chatbot API",
        version=__version__,
        description="Local-first API for the document RAG chatbot.",
    )
    app.include_router(router)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):  # noqa: ANN001
        detail = exc.errors() if settings.api_include_debug_errors else "Invalid request"
        return JSONResponse(status_code=400, content={"detail": detail})

    return app


app = create_app()
