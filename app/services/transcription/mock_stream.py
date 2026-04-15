from dataclasses import dataclass

from app.models.transcript import TranscriptSegment


@dataclass(frozen=True)
class MockTranscriptTemplate:
    duration: float
    text: str


class MockStreamingTranscriptionService:
    """Deterministic mock transcript generator for the first vertical slice."""

    def __init__(self, segment_interval_seconds: float = 1.0) -> None:
        self.segment_interval_seconds = segment_interval_seconds
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

    def build_segments(self, elapsed_seconds: float) -> list[TranscriptSegment]:
        visible_count = min(
            len(self.templates),
            max(0, int(elapsed_seconds // self.segment_interval_seconds)),
        )
        segments: list[TranscriptSegment] = []
        current_start = 0.0

        for template in self.templates[:visible_count]:
            segment = TranscriptSegment(
                start=round(current_start, 2),
                end=round(current_start + template.duration, 2),
                text=template.text,
                speaker=None,
            )
            segments.append(segment)
            current_start = segment.end

        return segments
