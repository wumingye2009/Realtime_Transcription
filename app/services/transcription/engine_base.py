from abc import ABC, abstractmethod
from pathlib import Path

from app.models.audio import AudioChunk
from app.models.transcript import TranscriptSegment
from app.services.audio.capture_base import BufferedAudioSource


class TranscriptionEngineBase(ABC):
    @abstractmethod
    def start(self, audio_source: BufferedAudioSource | None = None) -> None:
        """Begin transcript generation for the active session.

        `audio_source` is the future handoff point between capture providers and
        transcription engines. Engines may inspect buffered `AudioChunk` values
        through `get_buffered_chunks()` without knowing provider-specific details.
        """

    @abstractmethod
    def process_chunk(self, chunk: AudioChunk) -> None:
        """Consume one audio chunk from the capture layer."""

    @abstractmethod
    def pause(self) -> None:
        """Pause transcript generation without discarding progress."""

    @abstractmethod
    def resume(self) -> None:
        """Resume transcript generation after a pause."""

    @abstractmethod
    def stop(self) -> None:
        """Stop transcript generation and finalize any remaining state."""

    @abstractmethod
    def get_segments(self) -> list[TranscriptSegment]:
        """Return currently available transcript segments."""

    def get_runtime_diagnostics(self) -> dict[str, object]:
        """Return lightweight engine diagnostics for session status/metadata."""
        return {}

    def run_offline_transcription(
        self,
        audio_path: Path,
        model_size_override: str | None = None,
        options: dict[str, object] | None = None,
    ) -> list[TranscriptSegment]:
        """Optionally run one full-pass offline transcription for diagnostics."""
        return []
