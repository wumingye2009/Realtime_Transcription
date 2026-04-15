import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.sessions.session_manager import get_session_manager


router = APIRouter(tags=["ws"])


@router.websocket("/ws/transcript")
async def transcript_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    manager = get_session_manager()
    last_snapshot = None

    try:
        while True:
            snapshot = manager.get_stream_snapshot()
            if snapshot != last_snapshot:
                await websocket.send_text(json.dumps(snapshot))
                last_snapshot = snapshot
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return
