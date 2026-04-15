from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from app.models.audio import AudioChunk, CaptureConfig, CaptureState


@runtime_checkable
class BufferedAudioSource(Protocol):
    """Future handoff contract from capture layer to transcription layer.

    Engines should depend on this buffered-audio view instead of knowing
    individual capture provider details. For Phase 1, providers expose
    `get_buffered_chunks()` and engines may ignore the returned audio while
    mock behavior remains active.
    """

    def get_buffered_chunks(self) -> list[AudioChunk]:
        """Return buffered audio chunks accumulated so far."""


class AudioCaptureBase(ABC):
    def __init__(self, config: CaptureConfig) -> None:
        self.config = config
        self.state: CaptureState = "idle"

    @abstractmethod
    def start(self) -> None:
        """Start audio capture."""

    @abstractmethod
    def pause(self) -> None:
        """Pause audio capture."""

    @abstractmethod
    def resume(self) -> None:
        """Resume audio capture."""

    @abstractmethod
    def stop(self) -> None:
        """Stop audio capture."""

    @abstractmethod
    def get_buffered_chunks(self) -> list[AudioChunk]:
        """Return any captured or placeholder audio chunks accumulated so far."""

    def get_runtime_diagnostics(self) -> dict[str, object]:
        """Return lightweight capture diagnostics for logs/status/metadata."""
        return {}

    def get_debug_audio_artifacts(self) -> dict[str, object]:
        """Return optional audio artifacts captured for diagnostics."""
        return {}


class CompositeAudioSource(BufferedAudioSource):
    """Aggregates multiple capture providers into one engine-facing source."""

    def __init__(self, providers: list[BufferedAudioSource]) -> None:
        self.providers = providers

    def get_buffered_chunks(self) -> list[AudioChunk]:
        chunks: list[AudioChunk] = []
        for provider in self.providers:
            chunks.extend(provider.get_buffered_chunks())
        return chunks
