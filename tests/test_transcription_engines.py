import time
from pathlib import Path

from app.core.config import get_settings
from app.models.audio import AudioChunk
from app.models.session import SessionCreateRequest
from app.services.audio.capture_base import BufferedAudioSource
from app.services.sessions.session_manager import SessionManager
from app.services.transcription.engine_base import TranscriptionEngineBase
from app.services.transcription.faster_whisper_engine import FasterWhisperEngine
from app.services.transcription.mock_engine import MockTranscriptionEngine


class StubAudioSource(BufferedAudioSource):
    def __init__(self) -> None:
        self.calls = 0

    def get_buffered_chunks(self) -> list:
        self.calls += 1
        return []


class FakeWhisperSegment:
    def __init__(self, start: float, end: float, text: str):
        self.start = start
        self.end = end
        self.text = text


class FakeWhisperModel:
    def __init__(self, *args, **kwargs):
        return

    def transcribe(self, audio, **kwargs):
        return ([FakeWhisperSegment(0.0, 1.6, "hello from mic with enough context")], {"language": kwargs.get("language")})


def wait_for_segments(engine: FasterWhisperEngine, expected_count: int, timeout_seconds: float = 1.0):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        segments = engine.get_segments()
        if len(segments) >= expected_count:
            return segments
        time.sleep(0.02)
    return engine.get_segments()


def wait_for_manager_state(manager: SessionManager, expected_state: str, timeout_seconds: float = 1.0):
    deadline = time.time() + timeout_seconds
    status = manager.get_status()
    while time.time() < deadline:
        status = manager.get_status()
        if status.state == expected_state:
            return status
        time.sleep(0.02)
    return status


def wait_for_manager_segments(manager: SessionManager, expected_count: int, timeout_seconds: float = 1.0):
    deadline = time.time() + timeout_seconds
    status = manager.get_status()
    while time.time() < deadline:
        status = manager.get_status()
        if len(status.transcript_segments) >= expected_count:
            return status
        time.sleep(0.02)
    return status


def test_mock_engine_generates_expected_segment_shape_and_pause_resume_behavior() -> None:
    engine = MockTranscriptionEngine(chunks_per_segment=2)
    audio_source = StubAudioSource()
    engine.start(audio_source)
    engine.process_chunk(AudioChunk(source="system_loopback", sample_rate=16000, channels=2, frames=1024))
    first_segments = engine.get_segments()
    assert first_segments == []

    engine.process_chunk(AudioChunk(source="system_loopback", sample_rate=16000, channels=2, frames=1024))
    first_segments = engine.get_segments()
    assert len(first_segments) >= 1
    assert first_segments[0].start == 0.0
    assert first_segments[0].end > first_segments[0].start
    assert first_segments[0].text
    assert first_segments[0].speaker is None

    engine.pause()
    paused_segments = engine.get_segments()
    engine.process_chunk(AudioChunk(source="system_loopback", sample_rate=16000, channels=2, frames=1024))
    assert engine.get_segments() == paused_segments

    engine.resume()
    engine.process_chunk(AudioChunk(source="system_loopback", sample_rate=16000, channels=2, frames=1024))
    engine.process_chunk(AudioChunk(source="system_loopback", sample_rate=16000, channels=2, frames=1024))
    resumed_segments = engine.get_segments()
    assert len(resumed_segments) >= len(paused_segments)

    engine.stop()
    final_segments = engine.get_segments()
    assert len(final_segments) == len(engine.templates)


def _install_fake_loopback(monkeypatch) -> None:
    def fake_start(self):
        self.state = "running"
        self._emitted = False

    def fake_pause(self):
        if self.state == "running":
            self.state = "paused"

    def fake_resume(self):
        if self.state == "paused":
            self.state = "running"

    def fake_stop(self):
        self.state = "stopped"

    def fake_get_buffered_chunks(self):
        if self.state != "running" or getattr(self, "_emitted", False):
            return []
        self._emitted = True
        frames = int(self.config.sample_rate * 8.2)
        return [
            AudioChunk(
                source="system_loopback",
                sample_rate=self.config.sample_rate,
                channels=1,
                frames=frames,
                timestamp_ms=8200,
                data=(b"\x00\x01" * frames),
            )
        ]

    def fake_get_runtime_diagnostics(self):
        return {
            "raw_sample_rate": 48000,
            "raw_channels": 2,
            "raw_dtype": "float32",
            "chunk_frames": 39360,
            "host_api": "Windows WASAPI",
            "backend_path": "soundcard.mediafoundation WASAPI loopback",
            "discontinuity_warnings": 0,
            "processed_sample_rate": 16000,
            "processed_channels": 1,
            "conversion_path": "test loopback conversion",
        }

    def fake_get_debug_audio_artifacts(self):
        return {
            "raw_loopback": {
                "sample_rate": 48000,
                "channels": 2,
                "sample_width_bytes": 2,
                "data": b"\x00\x01" * 100,
            },
            "processed_loopback": {
                "sample_rate": 16000,
                "channels": 1,
                "sample_width_bytes": 2,
                "data": b"\x00\x01" * 100,
            },
        }

    monkeypatch.setattr("app.services.audio.windows_loopback_capture.WindowsLoopbackCapture.start", fake_start)
    monkeypatch.setattr("app.services.audio.windows_loopback_capture.WindowsLoopbackCapture.pause", fake_pause)
    monkeypatch.setattr("app.services.audio.windows_loopback_capture.WindowsLoopbackCapture.resume", fake_resume)
    monkeypatch.setattr("app.services.audio.windows_loopback_capture.WindowsLoopbackCapture.stop", fake_stop)
    monkeypatch.setattr(
        "app.services.audio.windows_loopback_capture.WindowsLoopbackCapture.get_buffered_chunks",
        fake_get_buffered_chunks,
    )
    monkeypatch.setattr(
        "app.services.audio.windows_loopback_capture.WindowsLoopbackCapture.get_runtime_diagnostics",
        fake_get_runtime_diagnostics,
    )
    monkeypatch.setattr(
        "app.services.audio.windows_loopback_capture.WindowsLoopbackCapture.get_debug_audio_artifacts",
        fake_get_debug_audio_artifacts,
    )


def test_session_manager_uses_engine_abstraction_with_loopback_capture(monkeypatch) -> None:
    _install_fake_loopback(monkeypatch)
    monkeypatch.setattr("app.services.transcription.faster_whisper_engine.WhisperModel", FakeWhisperModel)

    manager = SessionManager()
    payload = SessionCreateRequest(
        system_output_device_id="0",
        microphone_enabled=False,
        language_mode="auto",
        output_dir="runtime/test_outputs/engine_abstraction",
        export_formats=["md"],
    )

    start_response = manager.start(payload)
    assert start_response.state == "running"
    assert isinstance(manager.engine, TranscriptionEngineBase)
    assert isinstance(manager.engine, FasterWhisperEngine)
    assert manager.audio_source is not None

    status = wait_for_manager_segments(manager, 1)
    assert status.state == "running"
    assert len(status.transcript_segments) >= 1
    assert status.transcript_segments[0].text == "hello from mic with enough context"
    assert status.diagnostics.total_system_loopback_chunks_received >= 1

    stop_response = manager.stop()
    assert stop_response.state == "stopping"
    assert wait_for_manager_state(manager, "stopped").state == "stopped"


def test_faster_whisper_engine_emits_real_segments_from_microphone_chunks(monkeypatch) -> None:
    monkeypatch.setattr("app.services.transcription.faster_whisper_engine.WhisperModel", FakeWhisperModel)
    engine = FasterWhisperEngine(
        language_mode="english",
        buffer_window_seconds=0.1,
        transcription_cadence_seconds=0.1,
        flush_minimum_seconds=0.1,
        beam_size=2,
        emit_min_chars=1,
        emit_min_duration_seconds=0.0,
        merge_gap_seconds=0.0,
    )
    engine.start()

    chunk = AudioChunk(
        source="microphone",
        sample_rate=16000,
        channels=1,
        frames=1600,
        timestamp_ms=100,
        data=(b"\x00\x01" * 1600),
    )
    engine.process_chunk(chunk)
    segments = wait_for_segments(engine, 1)

    assert len(segments) == 1
    assert segments[0].text == "hello from mic with enough context"
    assert segments[0].speaker is None


def test_faster_whisper_engine_merges_short_adjacent_segments(monkeypatch) -> None:
    class MergeWhisperModel:
        def __init__(self, *args, **kwargs):
            return

        def transcribe(self, audio, **kwargs):
            return (
                [
                    FakeWhisperSegment(0.0, 0.2, "hello"),
                    FakeWhisperSegment(0.25, 0.7, "world"),
                ],
                {"language": kwargs.get("language")},
            )

    monkeypatch.setattr("app.services.transcription.faster_whisper_engine.WhisperModel", MergeWhisperModel)
    engine = FasterWhisperEngine(
        language_mode="english",
        buffer_window_seconds=0.1,
        transcription_cadence_seconds=0.1,
        flush_minimum_seconds=0.1,
        beam_size=2,
        emit_min_chars=12,
        emit_min_duration_seconds=0.8,
        merge_gap_seconds=0.5,
    )
    engine.start()
    engine.process_chunk(
        AudioChunk(
            source="microphone",
            sample_rate=16000,
            channels=1,
            frames=1600,
            timestamp_ms=100,
            data=(b"\x00\x01" * 1600),
        )
    )
    engine.stop()

    segments = engine.get_segments()
    assert len(segments) == 1
    assert segments[0].text == "hello world"


def test_faster_whisper_engine_suppresses_overlap_between_neighboring_windows(monkeypatch) -> None:
    class OverlapWhisperModel:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def transcribe(self, audio, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return ([FakeWhisperSegment(0.0, 1.0, "hello world")], {"language": kwargs.get("language")})
            return ([FakeWhisperSegment(0.0, 1.0, "world again")], {"language": kwargs.get("language")})

    monkeypatch.setattr("app.services.transcription.faster_whisper_engine.WhisperModel", OverlapWhisperModel)
    engine = FasterWhisperEngine(
        language_mode="english",
        buffer_window_seconds=0.1,
        transcription_cadence_seconds=0.1,
        flush_minimum_seconds=0.1,
        beam_size=2,
        emit_min_chars=1,
        emit_min_duration_seconds=0.0,
        merge_gap_seconds=0.0,
    )
    engine.start()

    chunk = AudioChunk(
        source="microphone",
        sample_rate=16000,
        channels=1,
        frames=1600,
        timestamp_ms=100,
        data=(b"\x00\x01" * 1600),
    )
    engine.process_chunk(chunk)
    wait_for_segments(engine, 1)
    engine.process_chunk(chunk)

    segments = wait_for_segments(engine, 2)
    assert len(segments) == 2
    assert segments[0].text == "hello world"
    assert segments[1].text == "again"


def test_faster_whisper_engine_uses_rolling_context_and_emits_only_new_tail(monkeypatch) -> None:
    class RollingWhisperModel:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def transcribe(self, audio, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return (
                    [FakeWhisperSegment(0.0, 3.2, "this is the start of a spoken sentence with context")],
                    {"language": kwargs.get("language")},
                )
            return (
                [
                    FakeWhisperSegment(0.0, 3.2, "this is the start of a spoken sentence with context"),
                    FakeWhisperSegment(2.8, 5.2, "with more detail at the tail end"),
                ],
                {"language": kwargs.get("language")},
            )

    monkeypatch.setattr("app.services.transcription.faster_whisper_engine.WhisperModel", RollingWhisperModel)
    engine = FasterWhisperEngine(
        language_mode="english",
        buffer_window_seconds=0.2,
        rolling_window_seconds=0.5,
        transcription_cadence_seconds=0.1,
        flush_minimum_seconds=0.1,
        beam_size=1,
        emit_min_chars=1,
        emit_min_duration_seconds=0.0,
        merge_gap_seconds=0.0,
    )
    engine.start()

    chunk = AudioChunk(
        source="system_loopback",
        sample_rate=16000,
        channels=1,
        frames=3200,
        timestamp_ms=200,
        data=(b"\x00\x01" * 3200),
    )
    engine.process_chunk(chunk)
    wait_for_segments(engine, 1)
    engine.process_chunk(chunk)

    segments = wait_for_segments(engine, 2)
    assert len(segments) == 2
    assert segments[0].text == "this is the start of a spoken sentence with context"
    assert segments[1].text.endswith("tail end")


def test_faster_whisper_engine_passes_quality_threshold_settings(monkeypatch) -> None:
    captured_calls = []

    class ThresholdWhisperModel:
        def __init__(self, *args, **kwargs):
            return

        def transcribe(self, audio, **kwargs):
            captured_calls.append(dict(kwargs))
            return ([FakeWhisperSegment(0.0, 1.6, "hello from mic with enough context")], {"language": kwargs.get("language")})

    monkeypatch.setattr("app.services.transcription.faster_whisper_engine.WhisperModel", ThresholdWhisperModel)
    engine = FasterWhisperEngine(
        language_mode="english",
        buffer_window_seconds=0.1,
        transcription_cadence_seconds=0.1,
        flush_minimum_seconds=0.1,
        beam_size=1,
        vad_filter=True,
        compression_ratio_threshold=2.0,
        log_prob_threshold=-0.4,
        no_speech_threshold=0.45,
        condition_on_previous_text=False,
        emit_min_chars=1,
        emit_min_duration_seconds=0.0,
        merge_gap_seconds=0.0,
    )
    engine.start()
    engine.process_chunk(
        AudioChunk(
            source="system_loopback",
            sample_rate=16000,
            channels=1,
            frames=1600,
            timestamp_ms=100,
            data=(b"\x00\x01" * 1600),
        )
    )

    deadline = time.time() + 1.0
    while time.time() < deadline and not captured_calls:
        time.sleep(0.02)

    assert any(call["compression_ratio_threshold"] == 2.0 for call in captured_calls)
    assert any(call["log_prob_threshold"] == -0.4 for call in captured_calls)
    assert any(call["no_speech_threshold"] == 0.45 for call in captured_calls)
    assert any(call["condition_on_previous_text"] is False for call in captured_calls)


def test_faster_whisper_engine_can_run_offline_full_pass_from_wav(monkeypatch) -> None:
    monkeypatch.setattr("app.services.transcription.faster_whisper_engine.WhisperModel", FakeWhisperModel)
    wav_path = Path("runtime/test_outputs/offline_comparison/processed_loopback_debug.wav")
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    import wave
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x01" * 3200)

    engine = FasterWhisperEngine(language_mode="english", beam_size=1)
    segments = engine.run_offline_transcription(wav_path)

    assert len(segments) == 1
    assert segments[0].text == "hello from mic with enough context"


def test_faster_whisper_engine_can_run_offline_full_pass_with_model_override(monkeypatch) -> None:
    class ModelAwareFakeWhisperModel:
        def __init__(self, model_size, *args, **kwargs):
            self.model_size = model_size

        def transcribe(self, audio, **kwargs):
            return (
                [FakeWhisperSegment(0.0, 1.0, f"offline transcript from {self.model_size}")],
                {"language": kwargs.get("language")},
            )

    monkeypatch.setattr(
        "app.services.transcription.faster_whisper_engine.WhisperModel",
        ModelAwareFakeWhisperModel,
    )
    wav_path = Path("runtime/test_outputs/offline_comparison/processed_loopback_debug.wav")
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    import wave
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x01" * 3200)

    engine = FasterWhisperEngine(language_mode="english", model_size="base", beam_size=1)
    segments = engine.run_offline_transcription(wav_path, model_size_override="small")

    assert len(segments) == 1
    assert segments[0].text == "offline transcript from small"


def test_faster_whisper_engine_enforces_single_flight_transcription(monkeypatch) -> None:
    class SlowWhisperModel:
        active_calls = 0
        max_active_calls = 0
        total_calls = 0

        def __init__(self, *args, **kwargs):
            return

        def transcribe(self, audio, **kwargs):
            type(self).active_calls += 1
            type(self).max_active_calls = max(type(self).max_active_calls, type(self).active_calls)
            type(self).total_calls += 1
            time.sleep(0.05)
            type(self).active_calls -= 1
            return ([FakeWhisperSegment(0.0, 0.8, "steady transcript output")], {"language": kwargs.get("language")})

    monkeypatch.setattr("app.services.transcription.faster_whisper_engine.WhisperModel", SlowWhisperModel)
    engine = FasterWhisperEngine(
        language_mode="english",
        buffer_window_seconds=0.1,
        transcription_cadence_seconds=0.1,
        flush_minimum_seconds=0.1,
        beam_size=1,
        emit_min_chars=1,
        emit_min_duration_seconds=0.0,
        merge_gap_seconds=0.0,
    )
    engine.start()

    for _ in range(6):
        engine.process_chunk(
            AudioChunk(
                source="system_loopback",
                sample_rate=16000,
                channels=1,
                frames=1600,
                timestamp_ms=100,
                data=(b"\x00\x01" * 1600),
            )
        )
        time.sleep(0.01)

    time.sleep(0.25)
    engine.stop()

    diagnostics = engine.get_runtime_diagnostics()
    assert SlowWhisperModel.max_active_calls == 1
    assert diagnostics["transcription_jobs_started"] == SlowWhisperModel.total_calls
    assert diagnostics["transcription_jobs_skipped_in_flight"] >= 1
    assert diagnostics["final_drain_completed"] is True


def test_session_manager_uses_configured_transcription_settings(monkeypatch) -> None:
    class FakeRawInputStream:
        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]

        def start(self):
            return

        def stop(self):
            return

        def close(self):
            return

    monkeypatch.setattr("app.services.audio.microphone_capture.sd.RawInputStream", FakeRawInputStream)
    monkeypatch.setattr("app.services.transcription.faster_whisper_engine.WhisperModel", FakeWhisperModel)

    settings = get_settings()
    transcription = settings.transcription
    original = transcription.model_copy()
    transcription.model_size = "base"
    transcription.buffer_seconds = 7.0
    transcription.beam_size = 3
    transcription.vad_filter = False

    try:
        manager = SessionManager()
        payload = SessionCreateRequest(
            system_output_device_id="0",
            microphone_enabled=True,
            microphone_input_device_id="1",
            language_mode="english",
            output_dir="runtime/test_outputs/configured_transcription",
            export_formats=["md"],
        )
        start_response = manager.start(payload)
        assert start_response.state == "running"
        assert isinstance(manager.engine, FasterWhisperEngine)
        assert manager.session_metadata["transcription_model"] == "base"
        assert manager.session_metadata["transcription_buffer_seconds"] == "7.0"
        assert manager.session_metadata["transcription_rolling_window_seconds"] == str(transcription.rolling_window_seconds)
        assert manager.session_metadata["transcription_cadence_seconds"] == str(transcription.transcription_cadence_seconds)
        assert manager.session_metadata["transcription_beam_size"] == "3"
        assert manager.session_metadata["transcription_vad_filter"] == "No"
        assert manager.session_metadata["transcription_condition_on_previous_text"] == "Yes"
        assert manager.session_metadata["transcription_log_prob_threshold"] == str(transcription.log_prob_threshold)
        assert manager.session_metadata["transcription_no_speech_threshold"] == str(transcription.no_speech_threshold)
        assert manager.stop().state == "stopping"
        assert wait_for_manager_state(manager, "stopped").state == "stopped"
    finally:
        settings.transcription = original


def test_final_transcription_can_use_final_only_model_override(monkeypatch) -> None:
    class ModelAwareFakeWhisperModel:
        def __init__(self, model_size, *args, **kwargs):
            self.model_size = model_size

        def transcribe(self, audio, **kwargs):
            return (
                [FakeWhisperSegment(0.0, 1.0, f"final transcript from {self.model_size}")],
                {"language": kwargs.get("language")},
            )

    monkeypatch.setattr("app.services.transcription.faster_whisper_engine.WhisperModel", ModelAwareFakeWhisperModel)
    _install_fake_loopback(monkeypatch)

    settings = get_settings()
    transcription = settings.transcription
    original = transcription.model_copy(deep=True)
    transcription.model_size = "base"
    transcription.final_model_size = "small"
    transcription.runtime.final.beam_size = 4
    transcription.runtime.final.vad_filter = True

    try:
        manager = SessionManager()
        payload = SessionCreateRequest(
            system_output_device_id="0",
            microphone_enabled=False,
            language_mode="english",
            output_dir="runtime/test_outputs/final_model_override",
            export_formats=["md"],
        )
        start_response = manager.start(payload)
        assert start_response.state == "running"
        assert manager.stop().state == "stopping"
        status = wait_for_manager_state(manager, "stopped")

        assert status.diagnostics.final_transcript_ready is True
        assert status.diagnostics.final_transcript_model_size == "small"
        assert status.final_transcript_segments[0].text == "final transcript from small"
        final_result = Path(status.diagnostics.final_transcript_result_path).read_text(encoding="utf-8")
        assert "model_size**: small (beam=4, vad=True, language=english)" in final_result
    finally:
        settings.transcription = original


def test_default_transcription_model_is_base() -> None:
    assert get_settings().transcription.model_size == "base"
    assert get_settings().transcription.get_final_model_size() == "small"
