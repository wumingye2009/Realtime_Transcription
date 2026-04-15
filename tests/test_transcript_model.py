from app.models.transcript import TranscriptSegment


def test_transcript_segment_reserves_speaker_field() -> None:
    segment = TranscriptSegment(
        start=1.25,
        end=3.5,
        text="Mock transcript segment",
        speaker=None,
    )

    payload = segment.model_dump()
    assert payload == {
        "start": 1.25,
        "end": 3.5,
        "text": "Mock transcript segment",
        "speaker": None,
    }
