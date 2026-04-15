from datetime import datetime
from pathlib import Path

from app.models.transcript import TranscriptSegment
from app.services.output.writer_base import OutputWriterBase


class TxtWriter(OutputWriterBase):
    def write(
        self,
        destination: Path,
        metadata: dict[str, str],
        segments: list[TranscriptSegment],
    ) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        lines = []

        for key, value in metadata.items():
            lines.append(f"{key}: {value}")

        lines.append("")

        for segment in segments:
            lines.append(
                f"[{self._format_timestamp(segment.start)} - {self._format_timestamp(segment.end)}] "
                f"{segment.text}"
            )

        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return destination

    @staticmethod
    def _format_timestamp(value: float) -> str:
        return datetime.utcfromtimestamp(value).strftime("%H:%M:%S")
