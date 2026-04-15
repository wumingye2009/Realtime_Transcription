import time
from pathlib import Path

import numpy as np

from app.core.config import get_settings
from app.models.audio import AudioChunk, AudioDevice, CaptureConfig, CapturePlan
from app.models.session import SessionCreateRequest
from app.services.audio.capture_base import AudioCaptureBase, BufferedAudioSource, CompositeAudioSource
from app.services.audio.device_discovery import DeviceDiscoveryService
from app.services.audio.fake_audio_source import FakeAudioSource
from app.services.audio.microphone_capture import MicrophoneCapture
from app.services.transcription.faster_whisper_engine import FasterWhisperEngine
from app.services.audio.windows_loopback_capture import WindowsLoopbackCapture
from app.services.sessions.session_manager import SessionManager


def wait_for_manager_state(manager: SessionManager, expected_state: str, timeout_seconds: float = 1.0):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if manager.get_status().state == expected_state:
            return manager.get_status()
        time.sleep(0.02)
    return manager.get_status()


def test_audio_models_cover_chunk_device_and_capture_plan_shapes() -> None:
    chunk = AudioChunk(
        source="microphone",
        sample_rate=16000,
        channels=1,
        frames=1024,
        timestamp_ms=250,
        data=b"\x00\x01",
    )
    device = AudioDevice(
        id="2",
        name="USB Mic",
        description="Microphone input device.",
        kind="input",
        channels=1,
        default_samplerate=16000,
        hostapi="WASAPI",
        is_default=False,
        capture_source="microphone",
        capture_provider="microphone",
        supports_loopback=False,
    )
    plan = CapturePlan(
        system_loopback=CaptureConfig(source="system_loopback", device_id="10", channels=2),
        microphone=CaptureConfig(source="microphone", device_id="2", channels=1),
    )

    assert chunk.source == "microphone"
    assert chunk.frames == 1024
    assert device.capture_provider == "microphone"
    assert plan.system_loopback.source == "system_loopback"
    assert plan.microphone is not None


def test_windows_loopback_capture_reads_real_time_chunks_via_soundcard_loopback(monkeypatch) -> None:
    settings = get_settings()
    original_runtime = settings.transcription.runtime.model_copy(deep=True)
    settings.transcription.runtime.mode = "development"
    settings.transcription.runtime.debug.dump_raw_wav = True
    settings.transcription.runtime.debug.dump_processed_wav = True

    class FakeRecorder:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def record(self, numframes):
            return np.ones((numframes, 2), dtype=np.float32) * 0.25

    class FakeLoopbackMicrophone:
        channels = 2

        def recorder(self, samplerate, channels, blocksize):
            assert samplerate == 48000
            assert channels == 2
            assert blocksize >= 9600
            return FakeRecorder()

    class FakeSpeaker:
        def __init__(self, name: str):
            self.name = name

    class FakeSoundcard:
        def default_speaker(self):
            return FakeSpeaker("Speakers")

        def get_speaker(self, name: str):
            return FakeSpeaker(name)

        def all_speakers(self):
            return [FakeSpeaker("Speakers")]

        def get_microphone(self, name: str, include_loopback: bool):
            assert include_loopback is True
            return FakeLoopbackMicrophone()

    monkeypatch.setattr(
        WindowsLoopbackCapture,
        "_load_soundcard_module",
        staticmethod(lambda: FakeSoundcard()),
    )
    monkeypatch.setattr(
        WindowsLoopbackCapture,
        "_load_pyaudiowpatch_module",
        staticmethod(lambda: (_ for _ in ()).throw(ValueError("PyAudioWPatch unavailable for this test"))),
    )
    monkeypatch.setattr(
        "app.services.audio.windows_loopback_capture.sd.query_devices",
        lambda device_id: {"name": "Speakers", "default_samplerate": 48000, "hostapi": 0},
    )
    monkeypatch.setattr(
        "app.services.audio.windows_loopback_capture.sd.query_hostapis",
        lambda: [{"name": "Windows WASAPI"}],
    )

    try:
        loopback = WindowsLoopbackCapture(CaptureConfig(source="system_loopback", device_id="5"))
        assert isinstance(loopback, AudioCaptureBase)
        loopback.start()
        time.sleep(0.03)
        chunks = loopback.get_buffered_chunks()

        assert loopback.state == "running"
        assert chunks
        assert chunks[0].source == "system_loopback"
        assert chunks[0].sample_rate == 16000
        assert chunks[0].channels == 1
        diagnostics = loopback.get_runtime_diagnostics()
        assert diagnostics["raw_sample_rate"] == 48000
        assert diagnostics["raw_channels"] == 2
        assert diagnostics["host_api"] == "Windows WASAPI"
        assert diagnostics["backend_path"] == "soundcard.mediafoundation WASAPI loopback"
        assert diagnostics["processed_sample_rate"] == 16000
        artifacts = loopback.get_debug_audio_artifacts()
        assert artifacts["raw_loopback"]["data"]
        assert artifacts["processed_loopback"]["data"]

        loopback.pause()
        assert loopback.state == "paused"
        loopback.resume()
        assert loopback.state == "running"
        loopback.stop()
        assert loopback.state == "stopped"
    finally:
        settings.transcription.runtime = original_runtime


def test_windows_loopback_capture_prefers_pyaudiowpatch_when_available(monkeypatch) -> None:
    class FakePyAudioStream:
        def __init__(self):
            self.started = False
            self.closed = False

        def start_stream(self):
            self.started = True

        def is_active(self):
            return self.started and not self.closed

        def stop_stream(self):
            self.started = False

        def close(self):
            self.closed = True

        def read(self, numframes, exception_on_overflow=False):
            frames = np.ones((numframes, 2), dtype=np.int16) * 1024
            return frames.tobytes()

    class FakePyAudioManager:
        def __init__(self):
            self.stream = FakePyAudioStream()

        def get_wasapi_loopback_analogue_by_index(self, index):
            return {
                "index": 12,
                "name": "Realtek HD Audio 2nd output (Realtek(R) Audio) (loopback)",
                "defaultSampleRate": 48000,
                "maxInputChannels": 2,
            }

        def get_loopback_device_info_generator(self):
            yield from []

        def open(self, **kwargs):
            assert kwargs["input"] is True
            assert kwargs["input_device_index"] == 12
            assert kwargs["channels"] == 2
            assert kwargs["rate"] == 48000
            return self.stream

        def terminate(self):
            return

    class FakePyAudioModule:
        paInt16 = 8

        @staticmethod
        def PyAudio():
            return FakePyAudioManager()

    monkeypatch.setattr(
        WindowsLoopbackCapture,
        "_load_pyaudiowpatch_module",
        staticmethod(lambda: FakePyAudioModule),
    )
    monkeypatch.setattr(
        "app.services.audio.windows_loopback_capture.sd.query_devices",
        lambda device_id: {
            "name": "Realtek HD Audio 2nd output (Realtek(R) Audio)",
            "default_samplerate": 48000,
            "hostapi": 0,
        },
    )
    monkeypatch.setattr(
        "app.services.audio.windows_loopback_capture.sd.query_hostapis",
        lambda: [{"name": "Windows WASAPI"}],
    )

    loopback = WindowsLoopbackCapture(CaptureConfig(source="system_loopback", device_id="7"))
    loopback.start()
    time.sleep(0.03)
    chunks = loopback.get_buffered_chunks()

    assert chunks
    assert chunks[0].source == "system_loopback"
    diagnostics = loopback.get_runtime_diagnostics()
    assert diagnostics["backend_path"] == "PyAudioWPatch WASAPI loopback"
    assert diagnostics["raw_dtype"] == "int16"
    assert diagnostics["host_api"] == "Windows WASAPI"
    loopback.stop()


def test_windows_loopback_capture_sanitizes_invalid_values_before_pcm_conversion() -> None:
    settings = get_settings()
    original_runtime = settings.transcription.runtime.model_copy(deep=True)
    settings.transcription.runtime.mode = "development"
    settings.transcription.runtime.debug.dump_raw_wav = True
    settings.transcription.runtime.debug.dump_processed_wav = True
    try:
        loopback = WindowsLoopbackCapture(CaptureConfig(source="system_loopback", device_id="5"))
        loopback._source_sample_rate = 16000
        chunks = loopback._to_audio_chunks(np.array([[np.nan, np.inf], [-np.inf, 0.5]] * 1024, dtype=np.float32))

        assert chunks
        assert chunks[0].source == "system_loopback"
        assert chunks[0].frames > 0
        artifacts = loopback.get_debug_audio_artifacts()
        assert artifacts["raw_loopback"]["data"]
        assert artifacts["processed_loopback"]["data"]
    finally:
        settings.transcription.runtime = original_runtime


def test_windows_loopback_capture_patches_soundcard_numpy_fromstring_for_numpy2() -> None:
    class FakeNumpyModule:
        def __init__(self):
            self.fromstring = lambda *args, **kwargs: None

    class FakeMediaFoundation:
        numpy = FakeNumpyModule()

    class FakeSoundcard:
        mediafoundation = FakeMediaFoundation()

    WindowsLoopbackCapture._patch_soundcard_numpy_compatibility(FakeSoundcard)
    patched = FakeSoundcard.mediafoundation.numpy.fromstring
    data = (np.array([1, 2, 3], dtype=np.int16)).tobytes()
    result = patched(data, dtype=np.int16)

    assert getattr(patched, "_rt_numpy_compat", False) is True
    assert result.tolist() == [1, 2, 3]


def test_microphone_capture_reads_real_time_chunks_via_sounddevice_stream(monkeypatch) -> None:
    callbacks = {}

    class FakeRawInputStream:
        def __init__(self, **kwargs):
            callbacks["callback"] = kwargs["callback"]
            self.started = False
            self.stopped = False
            self.closed = False

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def close(self):
            self.closed = True

    monkeypatch.setattr("app.services.audio.microphone_capture.sd.RawInputStream", FakeRawInputStream)

    mic = MicrophoneCapture(CaptureConfig(source="microphone", device_id="1", channels=1, chunk_size=4))
    mic.start()
    callbacks["callback"](b"\x01\x02\x03\x04", 4, None, None)
    chunks = mic.get_buffered_chunks()

    assert mic.state == "running"
    assert len(chunks) == 1
    assert chunks[0].source == "microphone"
    assert chunks[0].frames == 4
    assert chunks[0].data == b"\x01\x02\x03\x04"

    mic.pause()
    assert mic.state == "paused"
    mic.resume()
    assert mic.state == "running"
    mic.stop()
    assert mic.state == "stopped"


def test_composite_audio_source_exposes_engine_facing_buffered_audio_contract() -> None:
    mic = MicrophoneCapture(CaptureConfig(source="microphone", device_id="1"))
    loopback = WindowsLoopbackCapture(CaptureConfig(source="system_loopback", device_id="5"))
    source = CompositeAudioSource([mic, loopback])

    assert isinstance(source, BufferedAudioSource)
    assert source.get_buffered_chunks() == []


def test_fake_audio_source_generates_dummy_chunks_over_time() -> None:
    source = FakeAudioSource(chunk_interval_seconds=0.1, max_chunks=4)
    source.start()
    time.sleep(0.22)
    first_batch = source.get_buffered_chunks()
    assert len(first_batch) >= 2
    assert first_batch[0].frames == 1024
    assert first_batch[0].data == b"fake-audio"

    source.pause()
    paused_batch = source.get_buffered_chunks()
    time.sleep(0.15)
    assert source.get_buffered_chunks() == paused_batch == []

    source.resume()
    time.sleep(0.12)
    assert len(source.get_buffered_chunks()) >= 1

    source.stop()
    final_batch = source.get_buffered_chunks()
    assert len(final_batch) >= 0


def test_device_discovery_shape_supports_future_capture_provider_selection(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.audio.device_discovery.sd.query_devices",
        lambda: [
            {
                "name": "Microsoft Sound Mapper - Output",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000,
                "hostapi": 2,
            },
            {
                "name": "Realtek HD Audio 2nd output (Realtek(R) Audio)",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000,
                "hostapi": 2,
            },
            {
                "name": "Realtek HD Audio 2nd output (Realtek(R) Audio)",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000,
                "hostapi": 0,
            },
            {
                "name": "Speakers",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000,
                "hostapi": 1,
            },
            {
                "name": "Microphone",
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 16000,
                "hostapi": 0,
            },
        ],
    )
    monkeypatch.setattr(
        "app.services.audio.device_discovery.sd.query_hostapis",
        lambda: [
            {"name": "Windows WASAPI"},
            {"name": "Windows DirectSound"},
            {"name": "MME"},
        ],
    )
    monkeypatch.setattr("app.services.audio.device_discovery.sd.default.device", (4, 2))

    response = DeviceDiscoveryService().list_devices()
    assert len(response.system_output_devices) == 2
    assert len(response.microphone_input_devices) == 1
    assert response.system_output_devices[0].capture_provider == "windows_loopback"
    assert response.system_output_devices[0].supports_loopback is True
    assert response.system_output_devices[0].name == "Realtek HD Audio 2nd output (Realtek(R) Audio)"
    assert response.system_output_devices[0].hostapi == "Windows WASAPI"
    assert response.system_output_devices[0].description == "Wired headphones / headset output."
    assert response.microphone_input_devices[0].capture_provider == "microphone"
    assert response.microphone_input_devices[0].capture_source == "microphone"
    assert response.microphone_input_devices[0].description == "Microphone input device."


def test_session_manager_prepares_capture_configuration_for_microphone_only_session(monkeypatch) -> None:
    class FakeRawInputStream:
        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]

        def start(self):
            return

        def stop(self):
            return

        def close(self):
            return

    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            return

        def transcribe(self, audio, **kwargs):
            return ([], {"language": kwargs.get("language")})

    monkeypatch.setattr("app.services.audio.microphone_capture.sd.RawInputStream", FakeRawInputStream)
    monkeypatch.setattr("app.services.transcription.faster_whisper_engine.WhisperModel", FakeWhisperModel)
    manager = SessionManager()
    payload = SessionCreateRequest(
        system_output_device_id="10",
        microphone_enabled=True,
        microphone_input_device_id="2",
        language_mode="auto",
        output_dir="runtime/test_outputs/audio_capture",
        export_formats=["md"],
    )

    start_response = manager.start(payload)
    assert start_response.state == "running"
    assert manager.capture_plan is not None
    assert manager.capture_plan.system_loopback.device_id == "10"
    assert manager.capture_plan.microphone is not None
    assert manager.capture_plan.microphone.device_id == "2"
    assert len(manager.capture_providers) == 1
    assert manager.audio_source is not None
    assert manager.diagnostics.system_loopback_start_attempted is False
    assert manager.diagnostics.system_loopback_start_succeeded is False
    assert manager.diagnostics.microphone_capture_start_attempted is True
    assert manager.diagnostics.microphone_capture_start_succeeded is True
    assert all(provider.state == "running" for provider in manager.capture_providers)

    stop_response = manager.stop()
    assert stop_response.state == "stopping"
    assert wait_for_manager_state(manager, "stopped").state == "stopped"
    assert all(provider.state == "stopped" for provider in manager.capture_providers)


def test_session_manager_uses_microphone_capture_as_audio_source_when_enabled(monkeypatch) -> None:
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
    monkeypatch.setattr(
        "app.services.transcription.faster_whisper_engine.WhisperModel",
        lambda *args, **kwargs: type("Model", (), {"transcribe": lambda self, audio, **kw: ([], {})})(),
    )

    manager = SessionManager()
    payload = SessionCreateRequest(
        system_output_device_id="10",
        microphone_enabled=True,
        microphone_input_device_id="2",
        language_mode="auto",
        output_dir="runtime/test_outputs/audio_capture_mic",
        export_formats=["md"],
    )

    start_response = manager.start(payload)
    assert start_response.state == "running"
    assert isinstance(manager.audio_source, MicrophoneCapture)
    assert isinstance(manager.engine, FasterWhisperEngine)
    assert manager.diagnostics.active_audio_source == "MicrophoneCapture"
    assert len(manager.capture_providers) == 1

    stop_response = manager.stop()
    assert stop_response.state == "stopping"
    assert wait_for_manager_state(manager, "stopped").state == "stopped"


def test_microphone_only_session_produces_final_transcript_artifact(monkeypatch) -> None:
    callbacks = {}

    class FakeRawInputStream:
        def __init__(self, **kwargs):
            callbacks["callback"] = kwargs["callback"]

        def start(self):
            return

        def stop(self):
            return

        def close(self):
            return

    class FakeWhisperModel:
        def __init__(self, model_size="base", *args, **kwargs):
            self.model_size = model_size

        def transcribe(self, audio, **kwargs):
            return ([type("Seg", (), {"start": 0.0, "end": 1.0, "text": f"final transcript from {self.model_size}"})()], {})

    monkeypatch.setattr("app.services.audio.microphone_capture.sd.RawInputStream", FakeRawInputStream)
    monkeypatch.setattr("app.services.transcription.faster_whisper_engine.WhisperModel", FakeWhisperModel)

    manager = SessionManager()
    payload = SessionCreateRequest(
        system_output_device_id="10",
        microphone_enabled=True,
        microphone_input_device_id="2",
        language_mode="english",
        output_dir="runtime/test_outputs/microphone_final_path",
        export_formats=["md"],
    )

    start_response = manager.start(payload)
    assert start_response.state == "running"
    callbacks["callback"](b"\x01\x02" * 1600, 1600, None, None)
    time.sleep(0.1)

    stop_response = manager.stop()
    assert stop_response.state == "stopping"
    status = wait_for_manager_state(manager, "stopped")

    assert status.diagnostics.final_transcript_ready is True
    assert status.diagnostics.final_transcript_source == "offline_full_pass"
    assert status.diagnostics.final_transcription_audio_path
    assert Path(status.diagnostics.final_transcription_audio_path).exists()
    assert status.diagnostics.final_transcript_result_path
    assert Path(status.diagnostics.final_transcript_result_path).exists()
    assert status.final_transcript_segments
    assert status.final_transcript_segments[0].text == "final transcript from small"


def test_session_manager_resets_cleanly_when_microphone_start_fails(monkeypatch) -> None:
    class FailingRawInputStream:
        def __init__(self, **kwargs):
            raise RuntimeError("device open failed")

    monkeypatch.setattr("app.services.audio.microphone_capture.sd.RawInputStream", FailingRawInputStream)

    manager = SessionManager()
    payload = SessionCreateRequest(
        system_output_device_id="10",
        microphone_enabled=True,
        microphone_input_device_id="2",
        language_mode="auto",
        output_dir="runtime/test_outputs/audio_capture_fail",
        export_formats=["md"],
    )

    try:
        manager.start(payload)
        assert False, "Expected microphone startup to fail"
    except ValueError as exc:
        assert "Unable to open microphone device" in str(exc)

    assert manager.state == "idle"
    assert manager.active_session_id is None
    assert manager.audio_source is None
    assert manager.diagnostics.total_audio_chunks_received == 0
