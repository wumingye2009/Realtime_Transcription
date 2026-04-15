# File / Function Map

This is a practical guide to where the main MVP behavior lives.

## Configuration and Runtime Strategy

- [app/core/config.py](d:/Code/Git_Project/Realtime_Transcription/app/core/config.py)
  - `RuntimeStrategySettings`
  - `TranscriptionProfileSettings`
  - `DebugStrategySettings`
  - `TranscriptionSettings.get_realtime_profile()`
  - `TranscriptionSettings.get_final_profile()`
  - `TranscriptionSettings.get_final_model_size()`

## App Entrypoint and API

- [app/main.py](d:/Code/Git_Project/Realtime_Transcription/app/main.py)
  - FastAPI app creation
- [app/api/routes_sessions.py](d:/Code/Git_Project/Realtime_Transcription/app/api/routes_sessions.py)
  - session control endpoints
- [app/api/routes_devices.py](d:/Code/Git_Project/Realtime_Transcription/app/api/routes_devices.py)
  - device listing endpoint

## Session Lifecycle

- [app/services/sessions/session_manager.py](d:/Code/Git_Project/Realtime_Transcription/app/services/sessions/session_manager.py)
  - `start()`
  - `pause()`
  - `resume()`
  - `stop()`
  - `_create_engine()`
  - `_create_capture_providers()`
  - `_finalize_stop()`
  - `_run_final_transcription()`
  - `_rewrite_session_outputs()`
  - `_build_saved_output_metadata()`

This file is the main orchestration layer for the product.

## Audio Capture Abstractions

- [app/services/audio/capture_base.py](d:/Code/Git_Project/Realtime_Transcription/app/services/audio/capture_base.py)
  - `AudioCaptureBase`
  - `BufferedAudioSource`
  - `CompositeAudioSource`

## Windows System Loopback

- [app/services/audio/windows_loopback_capture.py](d:/Code/Git_Project/Realtime_Transcription/app/services/audio/windows_loopback_capture.py)
  - Windows loopback capture provider
  - PyAudioWPatch WASAPI path
  - soundcard fallback path
  - debug/finalization audio artifacts

## Microphone Capture

- [app/services/audio/microphone_capture.py](d:/Code/Git_Project/Realtime_Transcription/app/services/audio/microphone_capture.py)
  - microphone capture provider
  - microphone audio artifact generation for final transcription

## Device Discovery

- [app/services/audio/device_discovery.py](d:/Code/Git_Project/Realtime_Transcription/app/services/audio/device_discovery.py)
  - device enumeration
  - host API selection
  - frontend device descriptions / hints

## Transcription Engine Abstraction

- [app/services/transcription/engine_base.py](d:/Code/Git_Project/Realtime_Transcription/app/services/transcription/engine_base.py)
  - shared transcription engine interface

## Faster-Whisper Implementation

- [app/services/transcription/faster_whisper_engine.py](d:/Code/Git_Project/Realtime_Transcription/app/services/transcription/faster_whisper_engine.py)
  - rolling realtime transcription
  - offline full-pass transcription
  - final-only option overrides

## Output Writing

- [app/services/output/markdown_writer.py](d:/Code/Git_Project/Realtime_Transcription/app/services/output/markdown_writer.py)
  - primary saved Markdown output
- [app/services/output/txt_writer.py](d:/Code/Git_Project/Realtime_Transcription/app/services/output/txt_writer.py)
  - optional TXT export

## Frontend

- [app/static/js/app.js](d:/Code/Git_Project/Realtime_Transcription/app/static/js/app.js)
  - device list rendering
  - session control requests
  - status polling
  - realtime transcript rendering

## Models

- [app/models/session.py](d:/Code/Git_Project/Realtime_Transcription/app/models/session.py)
  - session request/response models
  - diagnostics model
- [app/models/audio.py](d:/Code/Git_Project/Realtime_Transcription/app/models/audio.py)
  - audio device and chunk models
- [app/models/transcript.py](d:/Code/Git_Project/Realtime_Transcription/app/models/transcript.py)
  - transcript segment model

## Useful Tests

- [tests/test_session_api.py](d:/Code/Git_Project/Realtime_Transcription/tests/test_session_api.py)
  - session lifecycle
  - saved output expectations
- [tests/test_audio_capture.py](d:/Code/Git_Project/Realtime_Transcription/tests/test_audio_capture.py)
  - loopback capture
  - microphone capture
  - capture-related session behavior
- [tests/test_transcription_engines.py](d:/Code/Git_Project/Realtime_Transcription/tests/test_transcription_engines.py)
  - faster-whisper behavior
  - final-model override behavior
- [tests/test_transcript_writer.py](d:/Code/Git_Project/Realtime_Transcription/tests/test_transcript_writer.py)
  - saved markdown writer shape
