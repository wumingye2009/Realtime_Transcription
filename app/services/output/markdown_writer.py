from datetime import datetime
from pathlib import Path

from app.models.transcript import TranscriptSegment
from app.services.output.writer_base import OutputWriterBase


class MarkdownWriter(OutputWriterBase):
    def write(
        self,
        destination: Path,
        metadata: dict[str, str],
        segments: list[TranscriptSegment],
    ) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# Session Transcript",
            "",
        ]

        saved_source = metadata.get("saved_transcript_source", "realtime_preview")
        if saved_source == "final_transcript":
            lines.append("Final transcript generated during session finalization.")
        else:
            lines.append("Realtime preview transcript. Final transcript was not available.")

        lines.extend(["", "## Transcript", ""])

        for segment in segments:
            lines.append(
                f"- [{self._format_timestamp(segment.start)} - {self._format_timestamp(segment.end)}] "
                f"{segment.text}"
            )

        lines.extend(["", "## Metadata", ""])

        for key, value in metadata.items():
            lines.append(f"- **{key}**: {value}")

        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return destination

    @staticmethod
    def _format_timestamp(value: float) -> str:
        return datetime.utcfromtimestamp(value).strftime("%H:%M:%S")
