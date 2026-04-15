from __future__ import annotations

from dataclasses import dataclass

from app.models.audio import AudioChunk
from app.models.transcript import TranscriptSegment
from app.services.audio.capture_base import BufferedAudioSource
from app.services.transcription.engine_base import TranscriptionEngineBase


@dataclass(frozen=True)
class MockTranscriptTemplate:
    duration: float
    text: str


class MockTranscriptionEngine(TranscriptionEngineBase):
    """Deterministic in-memory engine used for local Phase 1 development."""

    def __init__(self, chunks_per_segment: int = 4) -> None:
        self.chunks_per_segment = chunks_per_segment
        self.templates = [
            MockTranscriptTemplate(
                duration=2.4,
                text="Mock system audio detected. This is the first transcript segment.",
            ),
            MockTranscriptTemplate(
                duration=2.8,
                text="The frontend is polling the backend and receiving transcript updates live.",
            ),
            MockTranscriptTemplate(
                duration=2.2,
                text="Microphone and Windows loopback remain deferred in this vertical slice.",
            ),
            MockTranscriptTemplate(
                duration=3.1,
                text="Transcript output will be written to disk when the session is stopped.",
            ),
        ]
        self.active = False
        self.paused = False
        self.stopped = False
        self.audio_source: BufferedAudioSource | None = None
        self.processed_chunks = 0
        self.emitted_segment_count = 0
        self.segments: list[TranscriptSegment] = []

    def start(self, audio_source: BufferedAudioSource | None = None) -> None:
        self.audio_source = audio_source
        self.active = True
        self.paused = False
        self.stopped = False
        self.processed_chunks = 0
        self.emitted_segment_count = 0
        self.segments = []

    def process_chunk(self, chunk: AudioChunk) -> None:
        if not self.active or self.paused or self.stopped:
            return

        self.processed_chunks += 1
        while (
            self.emitted_segment_count < len(self.templates)
            and self.processed_chunks >= (self.emitted_segment_count + 1) * self.chunks_per_segment
        ):
            self._emit_next_segment()

    def pause(self) -> None:
        if not self.active or self.paused or self.stopped:
            return
        self.paused = True

    def resume(self) -> None:
        if not self.active or not self.paused or self.stopped:
            return
        self.paused = False

    def stop(self) -> None:
        if not self.active:
            return
        self.active = False
        self.stopped = True
        while self.emitted_segment_count < len(self.templates):
            self._emit_next_segment()

    def get_segments(self) -> list[TranscriptSegment]:
        return list(self.segments)

    def _emit_next_segment(self) -> None:
        template = self.templates[self.emitted_segment_count]
        current_start = 0.0
        if self.segments:
            current_start = self.segments[-1].end

        self.segments.append(
            TranscriptSegment(
                start=round(current_start, 2),
                end=round(current_start + template.duration, 2),
                text=template.text,
                speaker=None,
            )
        )
        self.emitted_segment_count += 1
