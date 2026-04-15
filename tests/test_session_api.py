import time
from pathlib import Path
import re
import wave

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from app.models.audio import AudioChunk
from app.services.audio.device_discovery import DeviceDiscoveryService
from app.services.sessions.session_manager import SessionManager, get_session_manager


client = TestClient(app)


class FakeRawInputStream:
    def __init__(self, **kwargs):
        self.callback = kwargs["callback"]

    def start(self):
        return

    def stop(self):
        return

    def close(self):
        return


class FakeWhisperSegment:
    def __init__(self, start: float, end: float, text: str):
        self.start = start
        self.end = end
        self.text = text


class FakeWhisperModel:
    def __init__(self, *args, **kwargs):
        return

    def transcribe(self, audio, **kwargs):
        return ([FakeWhisperSegment(0.0, 1.6, "test transcript with enough context")], {"language": kwargs.get("language")})


def install_fake_loopback(monkeypatch) -> None:
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


def wait_for_state(expected_state: str, timeout_seconds: float = 1.0):
    deadline = time.time() + timeout_seconds
    payload = None
    while time.time() < deadline:
        response = client.get("/api/sessions/current")
        payload = response.json()
        if payload["state"] == expected_state:
            return payload
        time.sleep(0.02)
    return payload


def wait_for_offline_comparison(timeout_seconds: float = 3.0):
    deadline = time.time() + timeout_seconds
    payload = None
    while time.time() < deadline:
        response = client.get("/api/sessions/current")
        payload = response.json()
        diagnostics = payload["diagnostics"]
        if (
            diagnostics["offline_comparison_performed"]
            and diagnostics["offline_alternative_comparison_performed"]
            and diagnostics["offline_control_primary_result_path"]
            and diagnostics["offline_control_alternative_result_path"]
            and len(diagnostics["offline_additional_result_paths"]) >= 2
            and len(diagnostics["offline_control_additional_result_paths"]) >= 2
            and not diagnostics["offline_diagnostics_in_progress"]
        ):
            return payload
        time.sleep(0.05)
    return payload


def wait_for_no_offline_diagnostics(timeout_seconds: float = 3.0):
    deadline = time.time() + timeout_seconds
    payload = None
    manager = get_session_manager()
    while time.time() < deadline:
        response = client.get("/api/sessions/current")
        payload = response.json()
        if payload["state"] in {"running", "paused", "stopping"}:
            client.post("/api/sessions/stop")
        thread = getattr(manager, "_offline_diagnostics_thread", None)
        if (
            payload["state"] in {"idle", "stopped"}
            and not payload["diagnostics"].get("offline_diagnostics_in_progress", False)
            and (thread is None or not thread.is_alive())
        ):
            return payload
        time.sleep(0.05)
    if manager.state in {"running", "paused", "stopping"}:
        try:
            manager.stop()
        except ValueError:
            pass
    manager.diagnostics.offline_diagnostics_in_progress = False
    manager._offline_diagnostics_thread = None
    return payload


def normalize_manager_for_fresh_start() -> None:
    manager = get_session_manager()
    manager.state = "stopped"
    manager.diagnostics.offline_diagnostics_in_progress = False
    manager._offline_diagnostics_thread = None


def test_session_lifecycle_scaffold(monkeypatch) -> None:
    install_fake_loopback(monkeypatch)
    monkeypatch.setattr("app.services.transcription.faster_whisper_engine.WhisperModel", FakeWhisperModel)
    payload = {
        "system_output_device_id": "0",
        "microphone_enabled": False,
        "language_mode": "auto",
        "output_dir": "outputs",
        "export_formats": ["md"],
    }
    start_response = client.post("/api/sessions/start", json=payload)
    assert start_response.status_code == 200
    assert start_response.json()["state"] == "running"
    assert "transcript" in start_response.json()["message"].lower()

    time.sleep(1.1)
    current_response = client.get("/api/sessions/current")
    assert current_response.status_code == 200
    current_payload = current_response.json()
    assert re.fullmatch(r"\d{8}_\d{8}", current_payload["active_session_id"])
    assert current_payload["state"] == "running"
    assert "diagnostics" in current_payload
    assert len(current_payload["transcript_segments"]) >= 1
    first_segment = current_payload["transcript_segments"][0]
    assert first_segment["start"] == 0.0
    assert first_segment["speaker"] is None
    assert first_segment["text"]
    assert current_payload["diagnostics"]["capture_flow_active"] is True
    assert current_payload["diagnostics"]["total_audio_chunks_received"] >= 1
    assert current_payload["diagnostics"]["total_chunks_passed_to_engine"] >= 1

    stop_response = client.post("/api/sessions/stop")
    assert stop_response.status_code == 200
    assert stop_response.json()["state"] == "stopping"

    second_stop_response = client.post("/api/sessions/stop")
    assert second_stop_response.status_code == 200
    assert second_stop_response.json()["state"] in {"stopping", "stopped"}
    stopping_payload = client.get("/api/sessions/current").json()
    assert "queued_audio_chunks" in stopping_payload["diagnostics"]
    assert "producer_finished" in stopping_payload["diagnostics"]
    assert "finalize_complete" in stopping_payload["diagnostics"]
    assert "max_queue_depth" in stopping_payload["diagnostics"]
    assert "transcription_jobs_started" in stopping_payload["diagnostics"]
    assert "transcription_jobs_skipped_in_flight" in stopping_payload["diagnostics"]
    stopped_payload = wait_for_state("stopped")
    assert stopped_payload["state"] == "stopped"
    assert stopped_payload["diagnostics"]["final_drain_completed"] is True
    assert stopped_payload["diagnostics"]["debug_audio_dump_path"]
    assert Path(stopped_payload["diagnostics"]["debug_audio_dump_path"]).exists()
    assert stopped_payload["diagnostics"]["raw_loopback_debug_wav_path"]
    assert stopped_payload["diagnostics"]["processed_loopback_debug_wav_path"]
    assert Path(stopped_payload["diagnostics"]["raw_loopback_debug_wav_path"]).exists()
    assert Path(stopped_payload["diagnostics"]["processed_loopback_debug_wav_path"]).exists()
    assert stopped_payload["diagnostics"]["processed_loopback_debug_duration_seconds"] is not None
    assert stopped_payload["diagnostics"]["expected_audio_coverage_seconds"] is not None
    assert stopped_payload["diagnostics"]["raw_loopback_coverage_ratio"] is not None
    assert stopped_payload["diagnostics"]["processed_loopback_coverage_ratio"] is not None
    assert "raw_debug_window_limited" in stopped_payload["diagnostics"]
    assert "offline_diagnostics_in_progress" in stopped_payload["diagnostics"]
    assert stopped_payload["diagnostics"]["finalization_stage"] == "complete"
    assert stopped_payload["diagnostics"]["capture_stop_completed"] is True
    assert stopped_payload["diagnostics"]["final_transcript_ready"] is True
    assert stopped_payload["diagnostics"]["final_transcript_in_progress"] is False
    assert stopped_payload["diagnostics"]["final_transcript_source"] == "offline_full_pass"
    assert stopped_payload["diagnostics"]["final_transcription_audio_path"]
    assert stopped_payload["diagnostics"]["final_transcript_result_path"]
    assert stopped_payload["diagnostics"]["final_transcript_model_size"] == get_settings().transcription.get_final_model_size()
    assert Path(stopped_payload["diagnostics"]["final_transcript_result_path"]).exists()
    assert stopped_payload["diagnostics"]["final_transcript_segment_count"] >= 1
    assert len(stopped_payload["final_transcript_segments"]) >= 1
    final_result_content = Path(stopped_payload["diagnostics"]["final_transcript_result_path"]).read_text(
        encoding="utf-8"
    )
    assert "# Final Transcript" in final_result_content
    assert f"- **model_size**: {get_settings().transcription.get_final_model_size()}" in final_result_content
    assert "test transcript with enough context" in final_result_content


def test_session_controls_are_forgiving_when_repeated_or_out_of_order(monkeypatch) -> None:
    install_fake_loopback(monkeypatch)
    monkeypatch.setattr("app.services.transcription.faster_whisper_engine.WhisperModel", FakeWhisperModel)
    resume_without_session = client.post("/api/sessions/resume")
    assert resume_without_session.status_code == 200
    assert resume_without_session.json()["state"] in {"idle", "stopped"}

    payload = {
        "system_output_device_id": "0",
        "microphone_enabled": False,
        "language_mode": "auto",
        "output_dir": "outputs",
        "export_formats": ["md"],
    }
    client.post("/api/sessions/start", json=payload)

    first_pause = client.post("/api/sessions/pause")
    assert first_pause.status_code == 200
    assert first_pause.json()["state"] == "paused"

    second_pause = client.post("/api/sessions/pause")
    assert second_pause.status_code == 200
    assert second_pause.json()["state"] == "paused"

    first_resume = client.post("/api/sessions/resume")
    assert first_resume.status_code == 200
    assert first_resume.json()["state"] == "running"

    second_resume = client.post("/api/sessions/resume")
    assert second_resume.status_code == 200
    assert second_resume.json()["state"] == "running"

    stop_response = client.post("/api/sessions/stop")
    assert stop_response.status_code == 200
    assert stop_response.json()["state"] == "stopping"
    assert wait_for_state("stopped")["state"] == "stopped"


def test_stop_produces_final_transcript_before_session_is_fully_stopped(monkeypatch) -> None:
    install_fake_loopback(monkeypatch)
    monkeypatch.setattr("app.services.transcription.faster_whisper_engine.WhisperModel", FakeWhisperModel)
    wait_for_no_offline_diagnostics()
    normalize_manager_for_fresh_start()

    payload = {
        "system_output_device_id": "0",
        "microphone_enabled": False,
        "language_mode": "auto",
        "output_dir": "outputs",
        "export_formats": ["md"],
    }
    start_response = client.post("/api/sessions/start", json=payload)
    assert start_response.status_code == 200

    stop_response = client.post("/api/sessions/stop")
    assert stop_response.status_code == 200
    stopped_payload = wait_for_state("stopped")
    assert stopped_payload["state"] == "stopped"
    assert stopped_payload["diagnostics"]["final_transcript_ready"] is True
    assert stopped_payload["diagnostics"]["final_transcript_source"] == "offline_full_pass"


def test_session_settings_are_persisted_and_written_to_markdown_metadata(monkeypatch) -> None:
    wait_for_no_offline_diagnostics()
    normalize_manager_for_fresh_start()
    monkeypatch.setattr(
        DeviceDiscoveryService,
        "get_device_metadata",
        lambda self, system_output_device_id, microphone_input_device_id: {
            "system_output_device": f"Speakers (ID {system_output_device_id}, host API Test API)",
            "microphone_input_device": f"USB Mic (ID {microphone_input_device_id}, host API Test API)",
        },
    )
    monkeypatch.setattr("app.services.audio.microphone_capture.sd.RawInputStream", FakeRawInputStream)
    monkeypatch.setattr("app.services.transcription.faster_whisper_engine.WhisperModel", FakeWhisperModel)
    output_dir = Path("runtime/test_outputs/session_settings")
    payload = {
        "system_output_device_id": "7",
        "microphone_enabled": True,
        "microphone_input_device_id": "3",
        "language_mode": "chinese",
        "output_dir": str(output_dir),
        "export_formats": ["md", "txt"],
    }

    start_response = client.post("/api/sessions/start", json=payload)
    assert start_response.status_code == 200

    current_response = client.get("/api/sessions/current")
    assert current_response.status_code == 200
    current_payload = current_response.json()
    assert current_payload["session_options"]["system_output_device_id"] == "7"
    assert current_payload["session_options"]["microphone_enabled"] is True
    assert current_payload["session_options"]["microphone_input_device_id"] == "3"
    assert current_payload["session_options"]["language_mode"] == "chinese"
    assert current_payload["session_options"]["output_dir"] == str(output_dir)
    assert current_payload["session_options"]["export_formats"] == ["md", "txt"]

    stop_response = client.post("/api/sessions/stop")
    assert stop_response.status_code == 200
    assert stop_response.json()["state"] == "stopping"
    assert wait_for_state("stopped")["state"] == "stopped"

    markdown_paths = sorted(output_dir.glob("session_*.md"), key=lambda path: path.stat().st_mtime, reverse=True)
    assert markdown_paths
    markdown_path = markdown_paths[0]
    assert re.fullmatch(r"session_\d{8}_\d{8}\.md", markdown_path.name)
    markdown_content = markdown_path.read_text(encoding="utf-8")
    assert "## Transcript" in markdown_content
    assert f"- **session_id**: {current_payload['active_session_id']}" in markdown_content
    assert markdown_path.stem == f"session_{current_payload['active_session_id']}"
    assert "- **language_mode**: Chinese" in markdown_content
    assert "- **microphone_enabled**: Yes" in markdown_content
    assert "- **microphone_input_device**: USB Mic (ID 3, host API Test API)" in markdown_content
    assert "- **system_output_device**: Speakers (ID 7, host API Test API)" in markdown_content
    assert "- **capture_routing_mode**: microphone_only" in markdown_content
    assert "- **runtime_mode**:" in markdown_content
    assert "- **capture_verification**:" in markdown_content
    assert "- **realtime_transcription_mode**: rolling" in markdown_content
    assert "- **finalization_stage**: complete" in markdown_content
    assert "- **capture_stop_completed**: Yes" in markdown_content
    assert "- **final_transcript_ready**: Yes" in markdown_content
    assert "- **final_transcript_source**:" in markdown_content
    assert "- **final_transcript_segment_count**:" in markdown_content
    assert "- **saved_transcript_source**: final_transcript" in markdown_content
    assert "- **transcription_model**: base" in markdown_content
    assert "- **final_beam_size**: 3" in markdown_content
    assert "- **final_vad_filter**: Yes" in markdown_content
    assert "- **final_language_strategy**: en" in markdown_content
    assert "- **final_drain_completed**:" in markdown_content
    assert f"- **output_dir**: {output_dir.resolve()}" in markdown_content
    assert "- **export_formats**: md, txt" in markdown_content


def test_empty_output_dir_falls_back_to_default_and_mic_device_is_optional_when_disabled(monkeypatch) -> None:
    wait_for_no_offline_diagnostics()
    normalize_manager_for_fresh_start()
    install_fake_loopback(monkeypatch)
    monkeypatch.setattr("app.services.transcription.faster_whisper_engine.WhisperModel", FakeWhisperModel)
    payload = {
        "system_output_device_id": "4",
        "microphone_enabled": False,
        "microphone_input_device_id": "",
        "language_mode": "auto",
        "output_dir": "   ",
        "export_formats": [],
    }

    start_response = client.post("/api/sessions/start", json=payload)
    assert start_response.status_code == 200

    current_response = client.get("/api/sessions/current")
    current_payload = current_response.json()
    assert current_payload["session_options"]["microphone_input_device_id"] is None
    assert current_payload["session_options"]["output_dir"] == "outputs"
    assert current_payload["session_options"]["export_formats"] == ["md"]
    assert current_payload["diagnostics"]["microphone_capture_start_attempted"] is False

    stop_response = client.post("/api/sessions/stop")
    assert stop_response.status_code == 200
    assert stop_response.json()["state"] == "stopping"
    assert wait_for_state("stopped")["state"] == "stopped"


def test_invalid_language_mode_is_rejected() -> None:
    payload = {
        "system_output_device_id": "1",
        "microphone_enabled": False,
        "microphone_input_device_id": None,
        "language_mode": "spanish",
        "output_dir": "outputs",
        "export_formats": ["md"],
    }

    response = client.post("/api/sessions/start", json=payload)
    assert response.status_code == 422
