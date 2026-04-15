from pathlib import Path

from app.models.transcript import TranscriptSegment
from app.services.output.markdown_writer import MarkdownWriter


def test_markdown_writer_creates_segmented_output() -> None:
    destination = Path("runtime/test_artifacts/sample.md")
    writer = MarkdownWriter()

    writer.write(
        destination=destination,
        metadata={"language_mode": "auto"},
        segments=[
            TranscriptSegment(start=0.0, end=5.0, text="Hello world", speaker=None),
        ],
    )

    content = destination.read_text(encoding="utf-8")
    assert "## Metadata" in content
    assert "## Transcript" in content
    assert "- **language_mode**: auto" in content
    assert "[00:00:00 - 00:00:05] Hello world" in content
