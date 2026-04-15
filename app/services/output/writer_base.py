from abc import ABC, abstractmethod
from pathlib import Path

from app.models.transcript import TranscriptSegment


class OutputWriterBase(ABC):
    @abstractmethod
    def write(
        self,
        destination: Path,
        metadata: dict[str, str],
        segments: list[TranscriptSegment],
    ) -> Path:
        """Write the transcript to disk and return the created file path."""
