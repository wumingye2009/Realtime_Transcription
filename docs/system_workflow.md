# System Workflow

This document explains the current MVP runtime flow at a high level.

## 1. Configuration Load

- App settings are loaded through [config.py](d:/Code/Git_Project/Realtime_Transcription/app/core/config.py)
- Runtime mode, transcription profiles, and debug strategy are resolved there
- The important split is:
  - realtime profile for live preview
  - final profile for the official final transcript

## 2. Session Creation

- The frontend sends a session start request to the sessions API
- [session_manager.py](d:/Code/Git_Project/Realtime_Transcription/app/services/sessions/session_manager.py) builds:
  - session id
  - capture plan
  - capture provider
  - audio source
  - transcription engine
- Current MVP routing:
  - microphone enabled -> microphone capture path
  - otherwise -> Windows system loopback path

## 3. Capture Start

- The chosen capture provider starts
- Audio chunks are buffered and fed into the session manager queue
- Runtime diagnostics begin tracking capture status and queue depth

## 4. Realtime Transcription Loop

- The active transcription engine consumes queued audio chunks
- Realtime rolling transcription produces preview segments
- The UI polls current session state and renders `transcript_segments`

This preview is intentionally lightweight and may be less accurate than the final transcript.

## 5. Stop Request

- The user clicks `Stop`
- Session state changes to `stopping`
- Capture providers are stopped
- Remaining queued audio is allowed to drain

## 6. Drain and Finalize

- The realtime engine is stopped cleanly
- Final realtime preview state is captured
- Finalized audio artifacts are written as needed
- A single finalized audio file path is chosen for the official final transcript path

That finalized audio source may come from:
- processed loopback audio for system-loopback sessions
- processed microphone audio for microphone sessions

## 7. Offline Final Transcription

- The final transcription pass runs on the finalized audio artifact
- It uses the final transcription profile, not the realtime profile
- It can also use a different final-only model size

This is the step that produces the official saved transcript.

## 8. Output Generation

- The final transcript becomes the primary saved transcript
- Markdown and optional TXT outputs are written
- A final transcript sidecar text file is also written

Saved Markdown is transcript-first, then metadata.

## 9. Session Completion

- Finalization stage becomes `complete`
- Session state becomes `stopped`
- The UI can distinguish:
  - running preview
  - stopping/finalizing
  - final transcript ready

## Practical Summary

The current MVP is built around this product rule:

- realtime transcript = live preview
- final transcript = official saved result
