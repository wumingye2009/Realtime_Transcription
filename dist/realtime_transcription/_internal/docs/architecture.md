# Architecture

This project is split into deliberately small layers so Windows-specific audio work does not leak into the rest of the app.

- `app/main.py` starts FastAPI and wires routes.
- `app/api/` contains browser-facing HTTP and websocket endpoints.
- `app/models/` holds request, response, and transcript data models.
- `app/services/audio/` owns device discovery and future capture implementations.
- `app/services/transcription/` owns the engine abstraction and faster-whisper adapter.
- `app/services/output/` writes Markdown and TXT files.
- `app/services/sessions/` coordinates session state across capture, transcription, and output.

Risk is isolated around `windows_loopback_capture.py`, which is the main Windows-specific integration point for system audio capture.
