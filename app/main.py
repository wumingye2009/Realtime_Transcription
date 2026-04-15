from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes_devices import router as devices_router
from app.api.routes_sessions import router as sessions_router
from app.api.routes_ui import router as ui_router
from app.api.websocket import router as websocket_router
from app.core.config import get_settings
from app.core.logging import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        summary="Local Windows-first realtime transcription scaffold",
    )

    app.mount("/static", StaticFiles(directory="app/static"), name="static")

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(ui_router)
    app.include_router(devices_router)
    app.include_router(sessions_router)
    app.include_router(websocket_router)
    return app


app = create_app()
