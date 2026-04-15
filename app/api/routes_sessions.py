from fastapi import APIRouter, Depends, HTTPException, status

from app.models.session import (
    SessionControlResponse,
    SessionCreateRequest,
    SessionStatusResponse,
)
from app.services.sessions.session_manager import SessionManager, get_session_manager


router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("/current", response_model=SessionStatusResponse)
async def get_current_session(
    manager: SessionManager = Depends(get_session_manager),
) -> SessionStatusResponse:
    return manager.get_status()


@router.post("/start", response_model=SessionControlResponse)
async def start_session(
    payload: SessionCreateRequest,
    manager: SessionManager = Depends(get_session_manager),
) -> SessionControlResponse:
    try:
        return manager.start(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/pause", response_model=SessionControlResponse)
async def pause_session(
    manager: SessionManager = Depends(get_session_manager),
) -> SessionControlResponse:
    try:
        return manager.pause()
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/resume", response_model=SessionControlResponse)
async def resume_session(
    manager: SessionManager = Depends(get_session_manager),
) -> SessionControlResponse:
    try:
        return manager.resume()
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/stop", response_model=SessionControlResponse)
async def stop_session(
    manager: SessionManager = Depends(get_session_manager),
) -> SessionControlResponse:
    try:
        return manager.stop()
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
