from __future__ import annotations

from threading import Lock
from time import monotonic

import sounddevice as sd

from app.core.config import get_settings
from app.models.audio import AudioChunk, CaptureConfig
from app.services.audio.capture_base import AudioCaptureBase


class MicrophoneCapture(AudioCaptureBase):
    """Minimal real microphone capture provider for Phase 1 validation."""

    def __init__(self, config: CaptureConfig) -> None:
        super().__init__(config)
        self._buffered_chunks: list[AudioChunk] = []
        self._buffer_lock = Lock()
        self._stream: sd.RawInputStream | None = None
        self._started_at_monotonic: float | None = None
        debug_strategy = get_settings().transcription.get_debug_strategy()
        self._dump_raw_wav_enabled = bool(debug_strategy.dump_raw_wav)
        self._dump_processed_wav_enabled = bool(debug_strategy.dump_processed_wav)
        self._debug_raw_pcm16 = bytearray()
        self._debug_processed_pcm16 = bytearray()
        self._runtime_diagnostics: dict[str, object] = {
            "sample_rate": int(self.config.sample_rate),
            "channels": int(self.config.channels),
            "dtype": "int16",
            "chunk_frames": int(self.config.chunk_size),
            "backend_path": "sounddevice microphone input",
        }

    def start(self) -> None:
        if self.state == "running":
            return

        try:
            if self._stream is None:
                self._stream = self._create_stream()

            self._started_at_monotonic = monotonic()
            self._debug_raw_pcm16 = bytearray()
            self._debug_processed_pcm16 = bytearray()
            self._stream.start()
            self.state = "running"
        except Exception as exc:
            self._safe_close_stream()
            self.state = "idle"
            raise ValueError(
                f"Unable to open microphone device {self.config.device_id}. "
                "Please choose another input device."
            ) from exc

    def pause(self) -> None:
        if self.state != "running":
            return

        if self._stream is not None:
            self._stream.stop()
        self.state = "paused"

    def resume(self) -> None:
        if self.state != "paused":
            return

        if self._stream is None:
            self._stream = self._create_stream()
        self._stream.start()
        self.state = "running"

    def stop(self) -> None:
        self._safe_close_stream()
        self.state = "stopped"

    def get_buffered_chunks(self) -> list[AudioChunk]:
        with self._buffer_lock:
            chunks = list(self._buffered_chunks)
            self._buffered_chunks.clear()
        return chunks

    def get_runtime_diagnostics(self) -> dict[str, object]:
        return dict(self._runtime_diagnostics)

    def get_debug_audio_artifacts(self) -> dict[str, object]:
        return {
            "raw_microphone": {
                "sample_rate": int(self.config.sample_rate),
                "channels": int(self.config.channels),
                "sample_width_bytes": 2,
                "data": bytes(self._debug_raw_pcm16),
            },
            "processed_microphone": {
                "sample_rate": int(self.config.sample_rate),
                "channels": int(self.config.channels),
                "sample_width_bytes": 2,
                "data": bytes(self._debug_processed_pcm16),
            },
        }

    def _create_stream(self) -> sd.RawInputStream:
        return sd.RawInputStream(
            samplerate=int(self.config.sample_rate),
            blocksize=self.config.chunk_size,
            device=self._resolve_device(),
            channels=self.config.channels,
            dtype="int16",
            callback=self._on_audio_chunk,
        )

    def _resolve_device(self) -> int | None:
        try:
            return int(self.config.device_id)
        except (TypeError, ValueError):
            return None

    def _on_audio_chunk(self, indata, frames, time_info, status) -> None:
        if self.state != "running":
            return

        timestamp_ms = 0
        if self._started_at_monotonic is not None:
            timestamp_ms = int((monotonic() - self._started_at_monotonic) * 1000)

        chunk = AudioChunk(
            source=self.config.source,
            sample_rate=int(self.config.sample_rate),
            channels=self.config.channels,
            frames=frames,
            timestamp_ms=timestamp_ms,
            data=bytes(indata),
        )
        self._runtime_diagnostics["chunk_frames"] = int(frames)
        if self._dump_raw_wav_enabled:
            self._debug_raw_pcm16.extend(bytes(indata))
        if self._dump_processed_wav_enabled:
            self._debug_processed_pcm16.extend(bytes(indata))
        with self._buffer_lock:
            self._buffered_chunks.append(chunk)

    def _safe_close_stream(self) -> None:
        if self._stream is None:
            return

        try:
            self._stream.stop()
        except Exception:
            pass

        try:
            self._stream.close()
        except Exception:
            pass

        self._stream = None
