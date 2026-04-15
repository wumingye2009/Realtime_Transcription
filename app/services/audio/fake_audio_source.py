from __future__ import annotations

from time import monotonic

from app.models.audio import AudioChunk, CaptureConfig, CaptureState
from app.services.audio.capture_base import BufferedAudioSource


class FakeAudioSource(BufferedAudioSource):
    """Generates deterministic dummy audio chunks for the mock runtime flow."""

    def __init__(
        self,
        source: str = "fake",
        chunk_interval_seconds: float = 0.25,
        max_chunks: int = 16,
    ) -> None:
        self.config = CaptureConfig(
            source=source,
            device_id="fake-audio-source",
            sample_rate=16000,
            channels=1 if source == "microphone" else 2,
            chunk_size=1024,
            enabled=True,
        )
        self.state: CaptureState = "idle"
        self.chunk_interval_seconds = chunk_interval_seconds
        self.max_chunks = max_chunks
        self.started_at_monotonic: float | None = None
        self.paused_at_monotonic: float | None = None
        self.accumulated_paused_seconds = 0.0
        self.emitted_chunks = 0

    def start(self) -> None:
        self.state = "running"
        self.started_at_monotonic = monotonic()
        self.paused_at_monotonic = None
        self.accumulated_paused_seconds = 0.0
        self.emitted_chunks = 0

    def pause(self) -> None:
        if self.state != "running":
            return
        self.state = "paused"
        self.paused_at_monotonic = monotonic()

    def resume(self) -> None:
        if self.state != "paused" or self.paused_at_monotonic is None:
            return
        self.accumulated_paused_seconds += monotonic() - self.paused_at_monotonic
        self.paused_at_monotonic = None
        self.state = "running"

    def stop(self) -> None:
        self.state = "stopped"

    def get_buffered_chunks(self) -> list[AudioChunk]:
        if self.started_at_monotonic is None:
            return []

        if self.state == "stopped":
            target_count = self.max_chunks
        elif self.state == "paused" and self.paused_at_monotonic is not None:
            elapsed = self.paused_at_monotonic - self.started_at_monotonic - self.accumulated_paused_seconds
            target_count = int(elapsed // self.chunk_interval_seconds)
        else:
            elapsed = monotonic() - self.started_at_monotonic - self.accumulated_paused_seconds
            target_count = int(elapsed // self.chunk_interval_seconds)

        target_count = min(self.max_chunks, max(0, target_count))
        if target_count <= self.emitted_chunks:
            return []

        chunks: list[AudioChunk] = []
        for index in range(self.emitted_chunks, target_count):
            chunks.append(
                AudioChunk(
                    source=self.config.source,
                    sample_rate=self.config.sample_rate,
                    channels=self.config.channels,
                    frames=self.config.chunk_size,
                    timestamp_ms=int(index * self.chunk_interval_seconds * 1000),
                    data=b"fake-audio",
                )
            )

        self.emitted_chunks = target_count
        return chunks
