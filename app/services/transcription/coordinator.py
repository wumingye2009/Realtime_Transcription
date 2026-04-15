from app.models.transcript import TranscriptSegment
from app.services.transcription.engine_base import TranscriptionEngineBase


class TranscriptionCoordinator:
    def __init__(self, engine: TranscriptionEngineBase) -> None:
        self.engine = engine

    def transcribe_placeholder(self) -> list[TranscriptSegment]:
        return [
            TranscriptSegment(
                start=0.0,
                end=2.5,
                text="Live transcript streaming is scaffolded. Real audio-to-text wiring comes next.",
                speaker=None,
            )
        ]
