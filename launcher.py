from __future__ import annotations

import logging
import socket
from threading import Thread
from time import sleep
import webbrowser

import uvicorn

from app.core.config import get_settings


logger = logging.getLogger("launcher")


def _wait_for_server(host: str, port: int, timeout_seconds: float = 10.0) -> bool:
    deadline = timeout_seconds / 0.1
    for _ in range(int(deadline)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            if sock.connect_ex((host, port)) == 0:
                return True
        sleep(0.1)
    return False


def _open_browser_when_ready(url: str, host: str, port: int) -> None:
    if _wait_for_server(host, port):
        webbrowser.open(url)
    else:
        logger.warning("Launcher could not confirm the local server was ready before opening the browser.")


def main() -> None:
    settings = get_settings()
    host = settings.host
    port = settings.port
    url = f"http://{host}:{port}"

    # Ensure the main runtime/output folders exist before the server starts.
    settings.default_output_dir.mkdir(parents=True, exist_ok=True)
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.temp_dir.mkdir(parents=True, exist_ok=True)

    browser_thread = Thread(
        target=_open_browser_when_ready,
        args=(url, host, port),
        name="launcher-browser-open",
        daemon=True,
    )
    browser_thread.start()

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
