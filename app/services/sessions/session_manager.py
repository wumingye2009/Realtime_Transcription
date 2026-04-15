from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from time import sleep
import wave

from app.core.config import get_settings
from app.models.session import (
    SessionControlResponse,
    SessionCreateRequest,
    SessionDiagnostics,
    SessionState,
    SessionStatusResponse,
)
from app.models.audio import AudioChunk, CaptureConfig, CapturePlan
from app.models.transcript import TranscriptSegment
from app.services.audio.capture_base import AudioCaptureBase, BufferedAudioSource
from app.services.audio.device_discovery import DeviceDiscoveryService
from app.services.audio.fake_audio_source import FakeAudioSource
from app.services.audio.microphone_capture import MicrophoneCapture
from app.services.output.markdown_writer import MarkdownWriter
from app.services.output.txt_writer import TxtWriter
from app.services.storage.output_path_service import OutputPathService
from app.services.transcription.engine_base import TranscriptionEngineBase
from app.services.transcription.faster_whisper_engine import FasterWhisperEngine
from app.services.transcription.mock_engine import MockTranscriptionEngine
from app.services.audio.windows_loopback_capture import WindowsLoopbackCapture


logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(self) -> None:
        settings = get_settings()
        self.state: SessionState = "idle"
        self.active_session_id: str | None = None
        self.output_file_stem: str | None = None
        self.session_options: SessionCreateRequest | None = None
        self.session_metadata: dict[str, str] = {}
        self.transcript_segments: list[TranscriptSegment] = []
        self.final_transcript_segments: list[TranscriptSegment] = []
        self.diagnostics = SessionDiagnostics()
        self.engine: TranscriptionEngineBase | None = None
        self.capture_plan: CapturePlan | None = None
        self.capture_providers: list[AudioCaptureBase] = []
        self.audio_source: BufferedAudioSource | None = None
        self.default_output_dir = str(settings.default_output_dir)
        self.device_service = DeviceDiscoveryService()
        self.path_service = OutputPathService()
        self.markdown_writer = MarkdownWriter()
        self.txt_writer = TxtWriter()
        self._chunk_queue: Queue[AudioChunk] = Queue()
        self._session_lock = Lock()
        self._producer_stop_event = Event()
        self._producer_finished_event = Event()
        self._finalize_requested_event = Event()
        self._finalize_complete_event = Event()
        self._producer_thread: Thread | None = None
        self._consumer_thread: Thread | None = None
        self._offline_diagnostics_thread: Thread | None = None
        self._finalized = False
        self._debug_audio_dump_path: Path | None = None
        self._raw_loopback_debug_wav_path: Path | None = None
        self._processed_loopback_debug_wav_path: Path | None = None
        self._final_transcription_audio_wav_path: Path | None = None

    def get_status(self) -> SessionStatusResponse:
        if self.engine is not None:
            self.transcript_segments = self.engine.get_segments()
        self._merge_engine_diagnostics()
        self._merge_capture_diagnostics()
        self.diagnostics.queued_audio_chunks = self._chunk_queue.qsize()
        self.diagnostics.producer_finished = self._producer_finished_event.is_set()
        self.diagnostics.finalize_complete = self._finalize_complete_event.is_set()
        return SessionStatusResponse(
            state=self.state,
            active_session_id=self.active_session_id,
            session_options=self.session_options,
            transcript_segments=self.transcript_segments,
            final_transcript_segments=self.final_transcript_segments,
            diagnostics=self.diagnostics,
        )

    def start(self, payload: SessionCreateRequest) -> SessionControlResponse:
        if self.state in {"running", "paused", "stopping"}:
            raise ValueError("A session is already active.")
        if self.diagnostics.offline_diagnostics_in_progress:
            raise ValueError("The previous session is still finalizing offline diagnostics. Please wait a moment.")
        payload = self._normalized_session_options(payload)

        try:
            self.diagnostics = SessionDiagnostics()
            self._reset_runtime_flow()
            self.active_session_id = self._build_session_id()
            self.output_file_stem = f"session_{self.active_session_id}"
            self.session_options = payload
            self.capture_plan = self._build_capture_plan(payload)
            self.capture_providers = self._create_capture_providers(self.capture_plan)
            self.audio_source = self._create_audio_source()
            self._initialize_diagnostics()
            logger.info(
                "Session %s: start requested providers=%s active_source=%s",
                self.active_session_id,
                ", ".join(self.diagnostics.selected_capture_providers),
                self.diagnostics.active_audio_source,
            )
            self.session_metadata = self._build_session_metadata(payload)
            self.engine = self._create_engine()
            self.final_transcript_segments = []
            self.state = "running"
            self._start_capture_providers()
            self._start_audio_source()
            self.engine.start(self.audio_source)
            self._start_processing_threads()
        except Exception as exc:
            self._reset_failed_start()
            if isinstance(exc, ValueError):
                raise
            raise ValueError("Unable to start the session with the selected devices.") from exc

        return SessionControlResponse(
            state=self.state,
            message="Session started with live transcript streaming.",
        )

    def pause(self) -> SessionControlResponse:
        if self.state == "paused":
            return SessionControlResponse(
                state=self.state,
                message="Session is already paused.",
            )

        if self.state == "stopped":
            return SessionControlResponse(
                state=self.state,
                message="Session is stopped. Start a new session before pausing.",
            )

        if self.state == "stopping":
            return SessionControlResponse(
                state=self.state,
                message="Session is stopping and cannot be paused.",
            )

        if self.state == "idle":
            return SessionControlResponse(
                state=self.state,
                message="No session is running yet.",
            )

        assert self.engine is not None
        self._pause_capture_providers()
        self._pause_audio_source()
        self.engine.pause()
        self.transcript_segments = self.engine.get_segments()
        self.state = "paused"
        self.diagnostics.pause_count += 1
        logger.info("Session %s: paused", self.active_session_id)
        return SessionControlResponse(
            state=self.state,
            message="Session paused. Underlying audio backends will be implemented in a later step.",
        )

    def resume(self) -> SessionControlResponse:
        if self.state == "running":
            return SessionControlResponse(
                state=self.state,
                message="Session is already running.",
            )

        if self.state == "stopped":
            return SessionControlResponse(
                state=self.state,
                message="Session is stopped. Start a new session before resuming.",
            )

        if self.state == "stopping":
            return SessionControlResponse(
                state=self.state,
                message="Session is stopping and cannot be resumed.",
            )

        if self.state == "idle":
            return SessionControlResponse(
                state=self.state,
                message="No paused session is available to resume.",
            )

        assert self.engine is not None
        self._resume_capture_providers()
        self._resume_audio_source()
        self.engine.resume()
        self.transcript_segments = self.engine.get_segments()
        self.state = "running"
        self.diagnostics.resume_count += 1
        logger.info("Session %s: resumed", self.active_session_id)
        return SessionControlResponse(state=self.state, message="Session resumed.")

    def stop(self) -> SessionControlResponse:
        if self.state == "stopped":
            return SessionControlResponse(
                state=self.state,
                message="Session is already stopped.",
            )

        if self.state == "stopping":
            return SessionControlResponse(
                state=self.state,
                message="Stopping... processing remaining audio.",
            )

        if self.state not in {"running", "paused"}:
            raise ValueError("There is no active session to stop.")

        self.diagnostics.stop_requested = True
        self.state = "stopping"
        logger.info("Session %s: stop requested, draining remaining audio", self.active_session_id)
        self._stop_capture_providers()
        self.diagnostics.capture_stop_completed = True
        self._stop_audio_source()
        self._producer_stop_event.set()
        self._finalize_requested_event.set()
        return SessionControlResponse(
            state=self.state,
            message="Stopping... processing remaining audio.",
        )

    def get_stream_snapshot(self) -> dict[str, object]:
        self._refresh_transcript_segments()
        return {
            "state": self.state,
            "active_session_id": self.active_session_id,
            "segments": [segment.model_dump() for segment in self.transcript_segments],
        }

    def _write_outputs(
        self,
        output_dir: Path,
        file_stem: str,
        metadata: dict[str, str],
        segments: list[TranscriptSegment],
        export_formats: list[str],
    ) -> None:
        if "md" in export_formats:
            self.markdown_writer.write(output_dir / f"{file_stem}.md", metadata, segments)
        if "txt" in export_formats:
            self.txt_writer.write(output_dir / f"{file_stem}.txt", metadata, segments)

    def _rewrite_session_outputs(self, output_dir: Path) -> None:
        if self.session_options is None or self.output_file_stem is None:
            return
        metadata = self._build_saved_output_metadata(output_dir)
        output_segments = self.final_transcript_segments or self.transcript_segments
        self._write_outputs(
            output_dir,
            self.output_file_stem,
            metadata,
            output_segments,
            self.session_options.export_formats,
        )

    def _build_saved_output_metadata(self, output_dir: Path) -> dict[str, str]:
        full_metadata = self.session_metadata | self._build_diagnostics_metadata() | {"output_dir": str(output_dir)}
        preferred_keys = [
            "session_id",
            "language_mode",
            "microphone_enabled",
            "microphone_input_device",
            "system_output_device",
            "capture_routing_mode",
            "capture_sources",
            "runtime_mode",
            "transcription_model",
            "final_transcript_model_size",
            "transcription_device",
            "transcription_compute_type",
            "transcription_buffer_seconds",
            "transcription_rolling_window_seconds",
            "transcription_cadence_seconds",
            "realtime_beam_size",
            "realtime_vad_filter",
            "final_beam_size",
            "final_vad_filter",
            "final_language_strategy",
            "capture_verification",
            "realtime_transcription_mode",
            "capture_stop_completed",
            "finalization_stage",
            "final_transcript_ready",
            "final_transcript_source",
            "final_transcript_model_size",
            "final_transcript_segment_count",
            "final_transcription_audio_path",
            "final_transcription_audio_duration_seconds",
            "final_transcript_result_path",
            "saved_transcript_source",
            "loopback_backend_path",
            "loopback_host_api",
            "final_drain_completed",
            "stop_returned_partial_transcript_due_to_timeout",
            "output_dir",
            "export_formats",
        ]
        metadata: dict[str, str] = {}
        for key in preferred_keys:
            value = full_metadata.get(key)
            if value is None:
                continue
            if value in {"None", ""}:
                continue
            metadata[key] = value
        return metadata

    def _normalized_session_options(self, payload: SessionCreateRequest) -> SessionCreateRequest:
        normalized_output_dir = payload.output_dir.strip() or self.default_output_dir
        return payload.model_copy(update={"output_dir": normalized_output_dir})

    def _build_capture_plan(self, payload: SessionCreateRequest) -> CapturePlan:
        system_loopback = CaptureConfig(
            source="system_loopback",
            device_id=payload.system_output_device_id,
            channels=2,
            sample_rate=16000,
            chunk_size=1024,
            enabled=True,
        )
        microphone = None
        if payload.microphone_enabled and payload.microphone_input_device_id:
            microphone = CaptureConfig(
                source="microphone",
                device_id=payload.microphone_input_device_id,
                channels=1,
                sample_rate=16000,
                chunk_size=1024,
                enabled=True,
            )
        return CapturePlan(system_loopback=system_loopback, microphone=microphone)

    def _build_session_metadata(self, payload: SessionCreateRequest) -> dict[str, str]:
        device_metadata = self.device_service.get_device_metadata(
            system_output_device_id=payload.system_output_device_id,
            microphone_input_device_id=payload.microphone_input_device_id,
        )
        assert self.active_session_id is not None
        settings = get_settings().transcription
        realtime_profile = settings.get_realtime_profile()
        final_profile = settings.get_final_profile()
        final_model_size = settings.get_final_model_size()
        debug_strategy = settings.get_debug_strategy()
        return {
            "session_id": self.active_session_id,
            "language_mode": self._display_language(payload.language_mode),
            "microphone_enabled": "Yes" if payload.microphone_enabled else "No",
            "microphone_input_device": device_metadata["microphone_input_device"],
            "system_output_device": device_metadata["system_output_device"],
            "capture_routing_mode": "microphone_only" if payload.microphone_enabled else "system_loopback_only",
            "capture_sources": ", ".join(self._describe_capture_sources()),
            "runtime_mode": settings.runtime.mode,
            "transcription_model": settings.model_size,
            "final_transcript_model_size": final_model_size,
            "transcription_device": settings.device,
            "transcription_compute_type": settings.compute_type,
            "transcription_buffer_seconds": str(settings.buffer_seconds),
            "transcription_rolling_window_seconds": str(settings.rolling_window_seconds),
            "transcription_cadence_seconds": str(settings.transcription_cadence_seconds),
            "transcription_beam_size": str(realtime_profile.beam_size),
            "transcription_vad_filter": "Yes" if realtime_profile.vad_filter else "No",
            "transcription_condition_on_previous_text": "Yes"
            if realtime_profile.condition_on_previous_text
            else "No",
            "transcription_log_prob_threshold": str(realtime_profile.log_prob_threshold),
            "transcription_no_speech_threshold": str(realtime_profile.no_speech_threshold),
            "realtime_beam_size": str(realtime_profile.beam_size),
            "realtime_vad_filter": "Yes" if realtime_profile.vad_filter else "No",
            "final_beam_size": str(final_profile.beam_size),
            "final_vad_filter": "Yes" if final_profile.vad_filter else "No",
            "final_language_strategy": final_profile.language,
            "debug_dump_raw_wav": "Yes" if debug_strategy.dump_raw_wav else "No",
            "debug_dump_processed_wav": "Yes" if debug_strategy.dump_processed_wav else "No",
            "debug_log_queue_depth": "Yes" if debug_strategy.log_queue_depth else "No",
            "debug_log_chunks": "Yes" if debug_strategy.log_chunks else "No",
            "debug_log_timing": "Yes" if debug_strategy.log_timing else "No",
            "debug_verbose_worker": "Yes" if debug_strategy.verbose_worker else "No",
            "export_formats": ", ".join(payload.export_formats),
        }

    def _build_diagnostics_metadata(self) -> dict[str, str]:
        return {
            "capture_verification": "yes" if self.diagnostics.capture_flow_active else "no",
            "realtime_transcription_mode": self.diagnostics.realtime_transcription_mode or "None",
            "finalization_stage": self.diagnostics.finalization_stage or "None",
            "capture_stop_completed": self._yes_no(self.diagnostics.capture_stop_completed),
            "final_transcript_ready": self._yes_no(self.diagnostics.final_transcript_ready),
            "final_transcript_in_progress": self._yes_no(self.diagnostics.final_transcript_in_progress),
            "final_transcript_source": self.diagnostics.final_transcript_source or "None",
            "final_transcript_result_path": self.diagnostics.final_transcript_result_path or "None",
            "final_transcript_segment_count": str(self.diagnostics.final_transcript_segment_count),
            "final_transcription_audio_path": self.diagnostics.final_transcription_audio_path or "None",
            "final_transcription_audio_duration_seconds": str(
                self.diagnostics.final_transcription_audio_duration_seconds or "None"
            ),
            "final_transcript_model_size": self.diagnostics.final_transcript_model_size or "None",
            "saved_transcript_source": "final_transcript" if self.diagnostics.final_transcript_ready else "realtime_preview",
            "offline_comparison_performed": self._yes_no(self.diagnostics.offline_comparison_performed),
            "offline_diagnostics_in_progress": self._yes_no(self.diagnostics.offline_diagnostics_in_progress),
            "offline_comparison_source_path": self.diagnostics.offline_comparison_source_path or "None",
            "offline_comparison_source_duration_seconds": str(
                self.diagnostics.offline_comparison_source_duration_seconds or "None"
            ),
            "offline_comparison_result_path": self.diagnostics.offline_comparison_result_path or "None",
            "offline_comparison_segment_count": str(self.diagnostics.offline_comparison_segment_count),
            "offline_comparison_model_size": self.diagnostics.offline_comparison_model_size or "None",
            "offline_alternative_comparison_performed": self._yes_no(
                self.diagnostics.offline_alternative_comparison_performed
            ),
            "offline_alternative_model_size": self.diagnostics.offline_alternative_model_size or "None",
            "offline_alternative_result_path": self.diagnostics.offline_alternative_result_path or "None",
            "offline_alternative_segment_count": str(self.diagnostics.offline_alternative_segment_count),
            "offline_additional_model_sizes": ", ".join(self.diagnostics.offline_additional_model_sizes) or "None",
            "offline_additional_result_paths": ", ".join(self.diagnostics.offline_additional_result_paths) or "None",
            "offline_control_source_path": self.diagnostics.offline_control_source_path or "None",
            "offline_control_source_duration_seconds": str(
                self.diagnostics.offline_control_source_duration_seconds or "None"
            ),
            "offline_control_primary_result_path": self.diagnostics.offline_control_primary_result_path or "None",
            "offline_control_primary_segment_count": str(self.diagnostics.offline_control_primary_segment_count),
            "offline_control_alternative_result_path": self.diagnostics.offline_control_alternative_result_path
            or "None",
            "offline_control_alternative_segment_count": str(
                self.diagnostics.offline_control_alternative_segment_count
            ),
            "offline_control_additional_result_paths": ", ".join(
                self.diagnostics.offline_control_additional_result_paths
            )
            or "None",
            "capture_attempted_sources": ", ".join(self.diagnostics.capture_attempted_sources) or "None",
            "capture_started_sources": ", ".join(self.diagnostics.capture_started_sources) or "None",
            "selected_capture_providers": ", ".join(self.diagnostics.selected_capture_providers) or "None",
            "active_audio_source": self.diagnostics.active_audio_source or "None",
            "microphone_capture_start_attempted": self._yes_no(self.diagnostics.microphone_capture_start_attempted),
            "microphone_capture_start_succeeded": self._yes_no(self.diagnostics.microphone_capture_start_succeeded),
            "system_loopback_start_attempted": self._yes_no(self.diagnostics.system_loopback_start_attempted),
            "system_loopback_start_succeeded": self._yes_no(self.diagnostics.system_loopback_start_succeeded),
            "total_audio_chunks_received": str(self.diagnostics.total_audio_chunks_received),
            "microphone_chunks_received": str(self.diagnostics.total_microphone_chunks_received),
            "system_loopback_chunks_received": str(self.diagnostics.total_system_loopback_chunks_received),
            "fake_chunks_received": str(self.diagnostics.total_fake_chunks_received),
            "unknown_chunks_received": str(self.diagnostics.total_unknown_chunks_received),
            "engine_chunks_received": str(self.diagnostics.total_chunks_passed_to_engine),
            "max_queue_depth": str(self.diagnostics.max_queue_depth),
            "transcription_jobs_started": str(self.diagnostics.transcription_jobs_started),
            "transcription_jobs_skipped_in_flight": str(self.diagnostics.transcription_jobs_skipped_in_flight),
            "final_drain_completed": self._yes_no(self.diagnostics.final_drain_completed),
            "stop_returned_partial_transcript_due_to_timeout": self._yes_no(
                self.diagnostics.stop_returned_partial_transcript_due_to_timeout
            ),
            "debug_audio_dump_path": self.diagnostics.debug_audio_dump_path or "None",
            "raw_loopback_debug_wav_path": self.diagnostics.raw_loopback_debug_wav_path or "None",
            "raw_loopback_debug_duration_seconds": str(self.diagnostics.raw_loopback_debug_duration_seconds or "None"),
            "processed_loopback_debug_wav_path": self.diagnostics.processed_loopback_debug_wav_path or "None",
            "processed_loopback_debug_duration_seconds": str(
                self.diagnostics.processed_loopback_debug_duration_seconds or "None"
            ),
            "expected_audio_coverage_seconds": str(self.diagnostics.expected_audio_coverage_seconds or "None"),
            "raw_loopback_coverage_ratio": str(self.diagnostics.raw_loopback_coverage_ratio or "None"),
            "processed_loopback_coverage_ratio": str(self.diagnostics.processed_loopback_coverage_ratio or "None"),
            "raw_debug_window_limited": self._yes_no(self.diagnostics.raw_debug_window_limited),
            "debug_audio_tail_truncated": self._yes_no(self.diagnostics.debug_audio_tail_truncated),
            "loopback_raw_sample_rate": str(self.diagnostics.loopback_raw_sample_rate or "None"),
            "loopback_raw_channels": str(self.diagnostics.loopback_raw_channels or "None"),
            "loopback_raw_dtype": self.diagnostics.loopback_raw_dtype or "None",
            "loopback_chunk_frames": str(self.diagnostics.loopback_chunk_frames or "None"),
            "loopback_host_api": self.diagnostics.loopback_host_api or "None",
            "loopback_backend_path": self.diagnostics.loopback_backend_path or "None",
            "loopback_discontinuity_warnings": str(self.diagnostics.loopback_discontinuity_warnings),
            "loopback_processed_sample_rate": str(self.diagnostics.loopback_processed_sample_rate or "None"),
            "loopback_processed_channels": str(self.diagnostics.loopback_processed_channels or "None"),
            "loopback_conversion_path": self.diagnostics.loopback_conversion_path or "None",
            "first_chunk_time_offset": self._format_chunk_offset(self.diagnostics.first_chunk_timestamp_ms),
            "last_chunk_time_offset": self._format_chunk_offset(self.diagnostics.last_chunk_timestamp_ms),
            "last_chunk_source": self.diagnostics.last_chunk_source or "None",
        }

    @staticmethod
    def _display_language(language_mode: str) -> str:
        mapping = {
            "auto": "Auto",
            "english": "English",
            "chinese": "Chinese",
        }
        return mapping.get(language_mode, language_mode)

    @staticmethod
    def _build_output_file_stem() -> str:
        now = datetime.now()
        milliseconds_of_day = (
            ((now.hour * 60 + now.minute) * 60 + now.second) * 1000 + now.microsecond // 1000
        )
        return f"{now.strftime('%Y%m%d')}_{milliseconds_of_day:08d}"

    def _build_session_id(self) -> str:
        return self._build_output_file_stem()

    def _create_engine(self, engine_name: str = "mock") -> TranscriptionEngineBase:
        transcription_settings = get_settings().transcription
        realtime_profile = transcription_settings.get_realtime_profile()
        debug_strategy = transcription_settings.get_debug_strategy()
        engine_language_mode = self._resolve_profile_language_mode(realtime_profile.language)
        if engine_name == "faster_whisper":
            return FasterWhisperEngine(
                language_mode=engine_language_mode,
                model_size=transcription_settings.model_size,
                device=transcription_settings.device,
                compute_type=transcription_settings.compute_type,
                buffer_window_seconds=transcription_settings.buffer_seconds,
                rolling_window_seconds=transcription_settings.rolling_window_seconds,
                transcription_cadence_seconds=transcription_settings.transcription_cadence_seconds,
                flush_minimum_seconds=transcription_settings.flush_minimum_seconds,
                beam_size=realtime_profile.beam_size,
                vad_filter=realtime_profile.vad_filter,
                compression_ratio_threshold=realtime_profile.compression_ratio_threshold,
                log_prob_threshold=realtime_profile.log_prob_threshold,
                no_speech_threshold=realtime_profile.no_speech_threshold,
                condition_on_previous_text=realtime_profile.condition_on_previous_text,
                emit_min_chars=transcription_settings.emit_min_chars,
                emit_min_duration_seconds=transcription_settings.emit_min_duration_seconds,
                merge_gap_seconds=transcription_settings.merge_gap_seconds,
                log_timing=debug_strategy.log_timing,
                verbose_worker_logs=debug_strategy.verbose_worker,
            )
        if self.audio_source is not None and not isinstance(self.audio_source, FakeAudioSource):
            logger.info(
                "Session %s: using faster-whisper model=%s device=%s compute_type=%s buffer_seconds=%s rolling_window_seconds=%s cadence_seconds=%s beam_size=%s vad_filter=%s condition_on_previous_text=%s runtime_mode=%s",
                self.active_session_id,
                transcription_settings.model_size,
                transcription_settings.device,
                transcription_settings.compute_type,
                transcription_settings.buffer_seconds,
                transcription_settings.rolling_window_seconds,
                transcription_settings.transcription_cadence_seconds,
                realtime_profile.beam_size,
                realtime_profile.vad_filter,
                realtime_profile.condition_on_previous_text,
                transcription_settings.runtime.mode,
            )
            return FasterWhisperEngine(
                language_mode=engine_language_mode,
                model_size=transcription_settings.model_size,
                device=transcription_settings.device,
                compute_type=transcription_settings.compute_type,
                buffer_window_seconds=transcription_settings.buffer_seconds,
                rolling_window_seconds=transcription_settings.rolling_window_seconds,
                transcription_cadence_seconds=transcription_settings.transcription_cadence_seconds,
                flush_minimum_seconds=transcription_settings.flush_minimum_seconds,
                beam_size=realtime_profile.beam_size,
                vad_filter=realtime_profile.vad_filter,
                compression_ratio_threshold=realtime_profile.compression_ratio_threshold,
                log_prob_threshold=realtime_profile.log_prob_threshold,
                no_speech_threshold=realtime_profile.no_speech_threshold,
                condition_on_previous_text=realtime_profile.condition_on_previous_text,
                emit_min_chars=transcription_settings.emit_min_chars,
                emit_min_duration_seconds=transcription_settings.emit_min_duration_seconds,
                merge_gap_seconds=transcription_settings.merge_gap_seconds,
                log_timing=debug_strategy.log_timing,
                verbose_worker_logs=debug_strategy.verbose_worker,
            )
        return MockTranscriptionEngine()

    def _create_capture_providers(self, plan: CapturePlan) -> list[AudioCaptureBase]:
        # Current product behavior uses one active capture path per session.
        # If microphone is enabled we prioritize the microphone provider; otherwise
        # we use system loopback. This keeps finalization and saved output behavior
        # consistent while avoiding a larger mixed-source refactor in MVP scope.
        if plan.microphone is not None:
            return [MicrophoneCapture(plan.microphone)]
        return [WindowsLoopbackCapture(plan.system_loopback)]

    def _create_audio_source(self) -> BufferedAudioSource:
        if self.capture_providers:
            return self.capture_providers[0]
        return FakeAudioSource()

    def _start_capture_providers(self) -> None:
        for provider in self.capture_providers:
            source_name = provider.config.source
            self._record_capture_attempt(source_name)
            logger.info("Session %s: %s capture start attempted", self.active_session_id, source_name)
            try:
                provider.start()
            except Exception:
                logger.exception("Session %s: %s capture start failed", self.active_session_id, source_name)
                raise
            self._record_capture_started(source_name)
            self._merge_capture_diagnostics()
            logger.info("Session %s: %s capture started successfully", self.active_session_id, source_name)
            provider_diagnostics = provider.get_runtime_diagnostics()
            if provider_diagnostics:
                logger.info(
                    "Session %s: %s capture format raw_rate=%s raw_channels=%s raw_dtype=%s chunk_frames=%s host_api=%s backend=%s discontinuities=%s processed_rate=%s processed_channels=%s conversion=%s",
                    self.active_session_id,
                    source_name,
                    provider_diagnostics.get("raw_sample_rate"),
                    provider_diagnostics.get("raw_channels"),
                    provider_diagnostics.get("raw_dtype"),
                    provider_diagnostics.get("chunk_frames"),
                    provider_diagnostics.get("host_api"),
                    provider_diagnostics.get("backend_path"),
                    provider_diagnostics.get("discontinuity_warnings"),
                    provider_diagnostics.get("processed_sample_rate"),
                    provider_diagnostics.get("processed_channels"),
                    provider_diagnostics.get("conversion_path"),
                )

    def _pause_capture_providers(self) -> None:
        for provider in self.capture_providers:
            provider.pause()

    def _resume_capture_providers(self) -> None:
        for provider in self.capture_providers:
            provider.resume()

    def _stop_capture_providers(self) -> None:
        for provider in self.capture_providers:
            provider.stop()

    def _start_audio_source(self) -> None:
        if self.audio_source in self.capture_providers:
            return
        start = getattr(self.audio_source, "start", None)
        if callable(start):
            start()

    def _pause_audio_source(self) -> None:
        if self.audio_source in self.capture_providers:
            return
        pause = getattr(self.audio_source, "pause", None)
        if callable(pause):
            pause()

    def _resume_audio_source(self) -> None:
        if self.audio_source in self.capture_providers:
            return
        resume = getattr(self.audio_source, "resume", None)
        if callable(resume):
            resume()

    def _stop_audio_source(self) -> None:
        if self.audio_source in self.capture_providers:
            return
        stop = getattr(self.audio_source, "stop", None)
        if callable(stop):
            stop()

    def _start_processing_threads(self) -> None:
        self._producer_thread = Thread(target=self._producer_loop, name="session-audio-producer", daemon=True)
        self._consumer_thread = Thread(target=self._consumer_loop, name="session-audio-consumer", daemon=True)
        self._producer_thread.start()
        self._consumer_thread.start()

    def _producer_loop(self) -> None:
        idle_polls_after_stop = 0
        while True:
            if self.audio_source is None:
                break

            chunks = self.audio_source.get_buffered_chunks()
            if chunks:
                idle_polls_after_stop = 0
                for chunk in chunks:
                    self._chunk_queue.put(chunk)
                    self._update_queue_depth()
            elif self._producer_stop_event.is_set():
                idle_polls_after_stop += 1
                if idle_polls_after_stop >= 5:
                    break

            if self._producer_stop_event.is_set() and not chunks:
                sleep(0.05)
            else:
                sleep(0.02)

        self._producer_finished_event.set()

    def _consumer_loop(self) -> None:
        while True:
            if self.state == "paused":
                sleep(0.05)
                continue

            try:
                chunk = self._chunk_queue.get(timeout=0.05)
            except Empty:
                if self._finalize_requested_event.is_set() and self._producer_finished_event.is_set():
                    break
                continue

            if self.engine is not None:
                self._record_chunk(chunk)
                self.engine.process_chunk(chunk)
                self.diagnostics.total_chunks_passed_to_engine += 1
                self._merge_engine_diagnostics()
                if self._should_log_chunk_summary() and self.diagnostics.total_chunks_passed_to_engine % 10 == 0:
                    logger.info(
                        "Session %s: chunk summary total=%s mic=%s loopback=%s fake=%s engine=%s queue=%s max_queue=%s jobs=%s skipped_in_flight=%s",
                        self.active_session_id,
                        self.diagnostics.total_audio_chunks_received,
                        self.diagnostics.total_microphone_chunks_received,
                        self.diagnostics.total_system_loopback_chunks_received,
                        self.diagnostics.total_fake_chunks_received,
                        self.diagnostics.total_chunks_passed_to_engine,
                        self._chunk_queue.qsize(),
                        self.diagnostics.max_queue_depth,
                        self.diagnostics.transcription_jobs_started,
                        self.diagnostics.transcription_jobs_skipped_in_flight,
                    )

        self._finalize_stop()

    def _reset_failed_start(self) -> None:
        self._producer_stop_event.set()
        self._finalize_requested_event.set()
        for provider in self.capture_providers:
            try:
                provider.stop()
            except Exception:
                pass

        if self.audio_source is not None and self.audio_source not in self.capture_providers:
            stop = getattr(self.audio_source, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    pass

        self.state = "idle"
        self.active_session_id = None
        self.output_file_stem = None
        self.session_options = None
        self.session_metadata = {}
        self.transcript_segments = []
        self.diagnostics = SessionDiagnostics()
        self.engine = None
        self.capture_plan = None
        self.capture_providers = []
        self.audio_source = None
        self._reset_runtime_flow()

    def _describe_capture_sources(self) -> list[str]:
        if self.capture_plan is None:
            return []
        if self.capture_plan.microphone is not None:
            return [f"microphone:{self.capture_plan.microphone.device_id}"]
        return [f"system_loopback:{self.capture_plan.system_loopback.device_id}"]

    def _initialize_diagnostics(self) -> None:
        self.diagnostics.selected_capture_providers = [
            f"{provider.config.source}:{provider.__class__.__name__}" for provider in self.capture_providers
        ]
        self.diagnostics.active_audio_source = self.audio_source.__class__.__name__ if self.audio_source else None
        self.diagnostics.finalization_stage = "live_preview"
        if self.audio_source is not None and not isinstance(self.audio_source, FakeAudioSource):
            self.diagnostics.realtime_transcription_mode = "rolling"
        else:
            self.diagnostics.realtime_transcription_mode = "mock"

    def _refresh_transcript_segments(self) -> None:
        if self.engine is not None:
            self.transcript_segments = self.engine.get_segments()

    def _record_capture_attempt(self, source_name: str) -> None:
        if source_name not in self.diagnostics.capture_attempted_sources:
            self.diagnostics.capture_attempted_sources.append(source_name)
        if source_name == "microphone":
            self.diagnostics.microphone_capture_start_attempted = True
        elif source_name == "system_loopback":
            self.diagnostics.system_loopback_start_attempted = True

    def _record_capture_started(self, source_name: str) -> None:
        if source_name not in self.diagnostics.capture_started_sources:
            self.diagnostics.capture_started_sources.append(source_name)
        if source_name == "microphone":
            self.diagnostics.microphone_capture_start_succeeded = True
        elif source_name == "system_loopback":
            self.diagnostics.system_loopback_start_succeeded = True

    def _record_chunk(self, chunk: AudioChunk) -> None:
        self.diagnostics.total_audio_chunks_received += 1
        self.diagnostics.capture_flow_active = True

        if self.diagnostics.first_chunk_timestamp_ms is None:
            self.diagnostics.first_chunk_timestamp_ms = chunk.timestamp_ms
            logger.info("Session %s: first chunk received from %s", self.active_session_id, chunk.source)

        self.diagnostics.last_chunk_timestamp_ms = chunk.timestamp_ms
        self.diagnostics.last_chunk_source = chunk.source

        if chunk.source == "microphone":
            self.diagnostics.total_microphone_chunks_received += 1
        elif chunk.source == "system_loopback":
            self.diagnostics.total_system_loopback_chunks_received += 1
        elif chunk.source == "fake":
            self.diagnostics.total_fake_chunks_received += 1
        else:
            self.diagnostics.total_unknown_chunks_received += 1

    @staticmethod
    def _format_chunk_offset(value: int | None) -> str:
        if value is None:
            return "None"
        return f"{value} ms"

    @staticmethod
    def _yes_no(value: bool) -> str:
        return "Yes" if value else "No"

    def _should_log_chunk_summary(self) -> bool:
        debug_strategy = get_settings().transcription.get_debug_strategy()
        return bool(debug_strategy.log_queue_depth or debug_strategy.log_chunks)

    def _resolve_profile_language_mode(self, configured_language: str) -> str:
        session_language_mode = self.session_options.language_mode if self.session_options else "auto"
        if session_language_mode == "chinese":
            return "chinese"

        normalized = configured_language.strip().lower()
        if normalized in {"", "session"}:
            return session_language_mode
        if normalized in {"en", "english"}:
            return "english"
        if normalized in {"zh", "chinese"}:
            return "chinese"
        if normalized == "auto":
            return "auto"
        return session_language_mode

    def _reset_runtime_flow(self) -> None:
        self._chunk_queue = Queue()
        self._producer_stop_event = Event()
        self._producer_finished_event = Event()
        self._finalize_requested_event = Event()
        self._finalize_complete_event = Event()
        self._producer_thread = None
        self._consumer_thread = None
        self._finalized = False
        self._debug_audio_dump_path = None
        self._raw_loopback_debug_wav_path = None
        self._processed_loopback_debug_wav_path = None
        self._final_transcription_audio_wav_path = None
        self.final_transcript_segments = []
        self.diagnostics.finalization_stage = None
        self.diagnostics.capture_stop_completed = False
        self.diagnostics.final_transcript_ready = False
        self.diagnostics.final_transcript_in_progress = False
        self.diagnostics.final_transcript_source = None
        self.diagnostics.final_transcript_result_path = None
        self.diagnostics.final_transcript_segment_count = 0
        self.diagnostics.final_transcription_audio_path = None
        self.diagnostics.final_transcription_audio_duration_seconds = None
        self.diagnostics.final_transcript_model_size = None

    def _finalize_stop(self) -> None:
        with self._session_lock:
            if self._finalized:
                return
            self._finalized = True

        output_dir: Path | None = None
        self.diagnostics.finalization_stage = "draining_realtime"
        if self.engine is not None:
            self.engine.stop()
            self.transcript_segments = self.engine.get_segments()
            self._merge_engine_diagnostics()

        self._merge_capture_diagnostics()
        self.diagnostics.finalization_stage = "finalizing_audio"
        self._write_debug_audio_dump()

        if (
            self.session_options is not None
            and self.output_file_stem is not None
            and self.active_session_id is not None
        ):
            output_dir = self.path_service.resolve_output_dir(self.session_options.output_dir)
            self.diagnostics.finalization_stage = "final_transcription"
            self._run_final_transcription(output_dir)
            self.diagnostics.finalization_stage = "saving_output"
            self.diagnostics.finalization_stage = "complete"
            self._rewrite_session_outputs(output_dir)
            logger.info(
                "Session %s: final chunk summary total=%s mic=%s loopback=%s fake=%s engine=%s max_queue=%s jobs=%s skipped_in_flight=%s final_drain_completed=%s partial_stop=%s",
                self.active_session_id,
                self.diagnostics.total_audio_chunks_received,
                self.diagnostics.total_microphone_chunks_received,
                self.diagnostics.total_system_loopback_chunks_received,
                self.diagnostics.total_fake_chunks_received,
                self.diagnostics.total_chunks_passed_to_engine,
                self.diagnostics.max_queue_depth,
                self.diagnostics.transcription_jobs_started,
                self.diagnostics.transcription_jobs_skipped_in_flight,
                self.diagnostics.final_drain_completed,
                self.diagnostics.stop_returned_partial_transcript_due_to_timeout,
            )

        else:
            self.diagnostics.finalization_stage = "complete"
        self.state = "stopped"
        self._finalize_complete_event.set()

    def _start_offline_diagnostics(self, output_dir: Path) -> None:
        if self._offline_diagnostics_thread is not None and self._offline_diagnostics_thread.is_alive():
            return
        session_id = self.active_session_id
        self.diagnostics.offline_diagnostics_in_progress = True
        self._offline_diagnostics_thread = Thread(
            target=self._run_offline_diagnostics_task,
            args=(output_dir, session_id),
            name="offline-diagnostics",
            daemon=True,
        )
        self._offline_diagnostics_thread.start()

    def _run_offline_diagnostics_task(self, output_dir: Path, session_id: str | None) -> None:
        try:
            self._run_offline_comparison(output_dir)
            self._rewrite_session_outputs(output_dir)
            logger.info("Session %s: session outputs refreshed with offline diagnostics", session_id)
        except Exception:
            logger.exception("Session %s: offline diagnostics task failed", session_id)
        finally:
            self.diagnostics.offline_diagnostics_in_progress = False
            self._offline_diagnostics_thread = None

    def _run_offline_comparison(self, output_dir: Path) -> None:
        self.diagnostics.offline_comparison_performed = False
        self.diagnostics.offline_comparison_source_path = None
        self.diagnostics.offline_comparison_source_duration_seconds = None
        self.diagnostics.offline_comparison_result_path = None
        self.diagnostics.offline_comparison_segment_count = 0
        self.diagnostics.offline_comparison_model_size = None
        self.diagnostics.offline_alternative_comparison_performed = False
        self.diagnostics.offline_alternative_model_size = None
        self.diagnostics.offline_alternative_result_path = None
        self.diagnostics.offline_alternative_segment_count = 0
        self.diagnostics.offline_additional_model_sizes = []
        self.diagnostics.offline_additional_result_paths = []
        self.diagnostics.offline_control_source_path = None
        self.diagnostics.offline_control_source_duration_seconds = None
        self.diagnostics.offline_control_primary_result_path = None
        self.diagnostics.offline_control_primary_segment_count = 0
        self.diagnostics.offline_control_alternative_result_path = None
        self.diagnostics.offline_control_alternative_segment_count = 0
        self.diagnostics.offline_control_additional_result_paths = []

        if not isinstance(self.engine, FasterWhisperEngine):
            return
        if self._processed_loopback_debug_wav_path is None or not self._processed_loopback_debug_wav_path.exists():
            return
        if self.output_file_stem is None:
            return

        primary_model_size = get_settings().transcription.model_size
        configured_models = get_settings().transcription.offline_additional_comparison_model_sizes
        additional_model_sizes = [
            model_size
            for model_size in configured_models
            if model_size and model_size != primary_model_size
        ]
        self.diagnostics.offline_comparison_source_path = str(self._processed_loopback_debug_wav_path)
        self.diagnostics.offline_comparison_source_duration_seconds = self._get_wav_duration_seconds(
            self._processed_loopback_debug_wav_path
        )
        self.diagnostics.offline_comparison_model_size = primary_model_size
        self.diagnostics.offline_additional_model_sizes = additional_model_sizes.copy()
        if additional_model_sizes:
            self.diagnostics.offline_alternative_model_size = additional_model_sizes[0]
        logger.info(
            "Session %s: offline comparison transcription starting from %s duration=%.2fs primary_model=%s additional_models=%s",
            self.active_session_id,
            self._processed_loopback_debug_wav_path,
            self.diagnostics.offline_comparison_source_duration_seconds or 0.0,
            primary_model_size,
            ", ".join(additional_model_sizes) or "None",
        )
        try:
            primary_segments = self.engine.run_offline_transcription(
                self._processed_loopback_debug_wav_path,
                model_size_override=primary_model_size,
            )
        except Exception:
            logger.exception("Session %s: offline comparison transcription failed for primary model", self.active_session_id)
            return

        result_path = output_dir / f"{self.output_file_stem}_offline_comparison.txt"
        self._write_offline_comparison_output(
            result_path,
            primary_segments,
            title="Offline Comparison Transcript",
            source_path=self._processed_loopback_debug_wav_path,
            model_size=primary_model_size,
        )
        self.diagnostics.offline_comparison_performed = True
        self.diagnostics.offline_comparison_result_path = str(result_path)
        self.diagnostics.offline_comparison_segment_count = len(primary_segments)

        for index, comparison_model_size in enumerate(additional_model_sizes):
            try:
                comparison_segments = self.engine.run_offline_transcription(
                    self._processed_loopback_debug_wav_path,
                    model_size_override=comparison_model_size,
                )
            except Exception:
                logger.exception(
                    "Session %s: offline comparison transcription failed for comparison model %s",
                    self.active_session_id,
                    comparison_model_size,
                )
            else:
                comparison_result_path = output_dir / f"{self.output_file_stem}_offline_comparison_{comparison_model_size}.txt"
                self._write_offline_comparison_output(
                    comparison_result_path,
                    comparison_segments,
                    title="Offline Comparison Transcript",
                    source_path=self._processed_loopback_debug_wav_path,
                    model_size=comparison_model_size,
                )
                self.diagnostics.offline_additional_result_paths.append(str(comparison_result_path))
                if index == 0:
                    self.diagnostics.offline_alternative_comparison_performed = True
                    self.diagnostics.offline_alternative_result_path = str(comparison_result_path)
                    self.diagnostics.offline_alternative_segment_count = len(comparison_segments)

        self._run_offline_control_comparison(
            output_dir=output_dir,
            primary_model_size=primary_model_size,
            comparison_model_sizes=additional_model_sizes,
        )
        logger.info(
            "Session %s: offline comparison transcription finished primary_segments=%s first_additional_segments=%s result=%s",
            self.active_session_id,
            len(primary_segments),
            self.diagnostics.offline_alternative_segment_count,
            result_path,
        )

    def _run_final_transcription(self, output_dir: Path) -> None:
        self.final_transcript_segments = []
        self.diagnostics.final_transcript_ready = False
        self.diagnostics.final_transcript_in_progress = False
        self.diagnostics.final_transcript_source = None
        self.diagnostics.final_transcript_result_path = None
        self.diagnostics.final_transcript_segment_count = 0
        self.diagnostics.final_transcription_audio_path = None
        self.diagnostics.final_transcription_audio_duration_seconds = None
        self.diagnostics.final_transcript_model_size = None

        if not isinstance(self.engine, FasterWhisperEngine):
            self.final_transcript_segments = list(self.transcript_segments)
            self.diagnostics.final_transcript_ready = True
            self.diagnostics.final_transcript_source = "realtime_preview"
            self.diagnostics.final_transcript_segment_count = len(self.final_transcript_segments)
            self._write_final_transcript_artifact(output_dir, model_label="realtime_preview")
            return
        if self._final_transcription_audio_wav_path is None or not self._final_transcription_audio_wav_path.exists():
            self.final_transcript_segments = list(self.transcript_segments)
            self.diagnostics.final_transcript_ready = True
            self.diagnostics.final_transcript_source = "realtime_preview"
            self.diagnostics.final_transcript_segment_count = len(self.final_transcript_segments)
            self._write_final_transcript_artifact(output_dir, model_label="realtime_preview")
            return
        if self.output_file_stem is None:
            return

        transcription_settings = get_settings().transcription
        final_profile = transcription_settings.get_final_profile()
        final_language_mode = self._resolve_profile_language_mode(final_profile.language)
        final_model_size = transcription_settings.get_final_model_size()
        final_options = {
            "beam_size": final_profile.beam_size,
            "vad_filter": final_profile.vad_filter,
            "language": final_language_mode,
            "compression_ratio_threshold": final_profile.compression_ratio_threshold,
            "log_prob_threshold": final_profile.log_prob_threshold,
            "no_speech_threshold": final_profile.no_speech_threshold,
            "condition_on_previous_text": final_profile.condition_on_previous_text,
        }
        self.diagnostics.final_transcript_in_progress = True
        self.diagnostics.final_transcription_audio_path = str(self._final_transcription_audio_wav_path)
        self.diagnostics.final_transcription_audio_duration_seconds = self._get_wav_duration_seconds(
            self._final_transcription_audio_wav_path
        )
        self.diagnostics.final_transcript_model_size = final_model_size
        logger.info(
            "Session %s: final offline transcription starting from %s model=%s beam_size=%s vad_filter=%s language=%s",
            self.active_session_id,
            self._final_transcription_audio_wav_path,
            final_model_size,
            final_profile.beam_size,
            final_profile.vad_filter,
            final_language_mode,
        )
        try:
            final_segments = self.engine.run_offline_transcription(
                self._final_transcription_audio_wav_path,
                model_size_override=final_model_size,
                options=final_options,
            )
        except Exception:
            logger.exception("Session %s: final offline transcription failed; falling back to realtime preview", self.active_session_id)
            self.final_transcript_segments = list(self.transcript_segments)
            self.diagnostics.final_transcript_ready = True
            self.diagnostics.final_transcript_source = "realtime_preview_fallback"
            self.diagnostics.final_transcript_segment_count = len(self.final_transcript_segments)
            self.diagnostics.final_transcript_in_progress = False
            self._write_final_transcript_artifact(
                output_dir,
                model_label=f"{final_model_size} fallback_to_realtime_preview",
            )
            return

        self.final_transcript_segments = final_segments
        self.diagnostics.final_transcript_ready = True
        self.diagnostics.final_transcript_in_progress = False
        self.diagnostics.final_transcript_source = "offline_full_pass"
        self.diagnostics.final_transcript_segment_count = len(final_segments)
        result_path = self._write_final_transcript_artifact(
            output_dir,
            model_label=f"{final_model_size} (beam={final_profile.beam_size}, vad={final_profile.vad_filter}, language={final_language_mode})",
        )
        logger.info(
            "Session %s: final offline transcription finished segments=%s result=%s",
            self.active_session_id,
            len(final_segments),
            result_path,
        )

    def _write_final_transcript_artifact(self, output_dir: Path, model_label: str) -> Path | None:
        if self.output_file_stem is None:
            return None
        result_path = output_dir / f"{self.output_file_stem}_final_transcript.txt"
        source_path = self._final_transcription_audio_wav_path or Path("realtime_preview")
        self._write_offline_comparison_output(
            result_path,
            self.final_transcript_segments,
            title="Final Transcript",
            source_path=source_path,
            model_size=model_label,
        )
        self.diagnostics.final_transcript_result_path = str(result_path)
        return result_path

    def _run_offline_control_comparison(
        self,
        output_dir: Path,
        primary_model_size: str,
        comparison_model_sizes: list[str],
    ) -> None:
        if not isinstance(self.engine, FasterWhisperEngine):
            return

        temp_dir = get_settings().temp_dir
        raw_reference_path = temp_dir / "harvard_raw_reference.wav"
        processed_reference_path = temp_dir / "harvard_processed_reference.wav"
        control_source_path = raw_reference_path if raw_reference_path.exists() else processed_reference_path
        if not control_source_path.exists():
            return

        self.diagnostics.offline_control_source_path = str(control_source_path)
        self.diagnostics.offline_control_source_duration_seconds = self._get_wav_duration_seconds(control_source_path)
        try:
            primary_segments = self.engine.run_offline_transcription(
                control_source_path,
                model_size_override=primary_model_size,
            )
        except Exception:
            logger.exception("Session %s: offline control transcription failed for primary model", self.active_session_id)
            return

        primary_result_path = output_dir / f"{self.output_file_stem}_offline_control_{primary_model_size}.txt"
        self._write_offline_comparison_output(
            primary_result_path,
            primary_segments,
            title="Offline Control Transcript",
            source_path=control_source_path,
            model_size=primary_model_size,
        )
        self.diagnostics.offline_control_primary_result_path = str(primary_result_path)
        self.diagnostics.offline_control_primary_segment_count = len(primary_segments)

        for index, comparison_model_size in enumerate(comparison_model_sizes):
            try:
                comparison_segments = self.engine.run_offline_transcription(
                    control_source_path,
                    model_size_override=comparison_model_size,
                )
            except Exception:
                logger.exception(
                    "Session %s: offline control transcription failed for comparison model %s",
                    self.active_session_id,
                    comparison_model_size,
                )
            else:
                comparison_result_path = output_dir / f"{self.output_file_stem}_offline_control_{comparison_model_size}.txt"
                self._write_offline_comparison_output(
                    comparison_result_path,
                    comparison_segments,
                    title="Offline Control Transcript",
                    source_path=control_source_path,
                    model_size=comparison_model_size,
                )
                self.diagnostics.offline_control_additional_result_paths.append(str(comparison_result_path))
                if index == 0:
                    self.diagnostics.offline_control_alternative_result_path = str(comparison_result_path)
                    self.diagnostics.offline_control_alternative_segment_count = len(comparison_segments)

    @staticmethod
    def _write_offline_comparison_output(
        destination: Path,
        segments: list[TranscriptSegment],
        title: str,
        source_path: Path,
        model_size: str,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# {title}",
            "",
            "## Metadata",
            "",
            f"- **source_path**: {source_path}",
            f"- **model_size**: {model_size}",
            "",
            "## Segments",
            "",
        ]
        if not segments:
            lines.append("No offline comparison segments were produced.")
        else:
            for segment in segments:
                lines.append(
                    f"- [{SessionManager._format_seconds(segment.start)} - {SessionManager._format_seconds(segment.end)}] {segment.text}"
                )
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _format_seconds(value: float) -> str:
        hours = int(value // 3600)
        minutes = int((value % 3600) // 60)
        seconds = int(value % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _merge_engine_diagnostics(self) -> None:
        if self.engine is None:
            return
        runtime = self.engine.get_runtime_diagnostics()
        if not runtime:
            return
        self.diagnostics.transcription_jobs_started = int(runtime.get("transcription_jobs_started", 0))
        self.diagnostics.transcription_jobs_skipped_in_flight = int(
            runtime.get("transcription_jobs_skipped_in_flight", 0)
        )
        self.diagnostics.final_drain_completed = bool(runtime.get("final_drain_completed", False))
        self.diagnostics.stop_returned_partial_transcript_due_to_timeout = bool(
            runtime.get("stop_returned_partial_transcript_due_to_timeout", False)
        )

    def _merge_capture_diagnostics(self) -> None:
        provider = self.capture_providers[0] if self.capture_providers else None
        if provider is None:
            return
        runtime = provider.get_runtime_diagnostics()
        if not runtime:
            return
        raw_sample_rate = runtime.get("raw_sample_rate")
        raw_channels = runtime.get("raw_channels")
        chunk_frames = runtime.get("chunk_frames")
        processed_sample_rate = runtime.get("processed_sample_rate")
        processed_channels = runtime.get("processed_channels")
        self.diagnostics.loopback_raw_sample_rate = int(raw_sample_rate) if raw_sample_rate else None
        self.diagnostics.loopback_raw_channels = int(raw_channels) if raw_channels else None
        self.diagnostics.loopback_raw_dtype = (
            str(runtime.get("raw_dtype")) if runtime.get("raw_dtype") is not None else None
        )
        self.diagnostics.loopback_chunk_frames = int(chunk_frames) if chunk_frames else None
        self.diagnostics.loopback_host_api = str(runtime.get("host_api")) if runtime.get("host_api") is not None else None
        self.diagnostics.loopback_backend_path = (
            str(runtime.get("backend_path")) if runtime.get("backend_path") is not None else None
        )
        self.diagnostics.loopback_discontinuity_warnings = int(runtime.get("discontinuity_warnings", 0))
        self.diagnostics.loopback_processed_sample_rate = (
            int(processed_sample_rate) if processed_sample_rate else None
        )
        self.diagnostics.loopback_processed_channels = int(processed_channels) if processed_channels else None
        self.diagnostics.loopback_conversion_path = (
            str(runtime.get("conversion_path")) if runtime.get("conversion_path") is not None else None
        )

    def _update_queue_depth(self) -> None:
        queue_depth = self._chunk_queue.qsize()
        if queue_depth > self.diagnostics.max_queue_depth:
            self.diagnostics.max_queue_depth = queue_depth

    def _write_debug_audio_dump(self) -> None:
        if self.active_session_id is None:
            return
        provider = self.capture_providers[0] if self.capture_providers else None
        if provider is None:
            return
        artifacts = provider.get_debug_audio_artifacts()
        if not artifacts:
            return
        temp_dir = get_settings().temp_dir
        temp_dir.mkdir(parents=True, exist_ok=True)

        raw_artifact = artifacts.get("raw_loopback")
        if isinstance(raw_artifact, dict) and raw_artifact.get("data"):
            raw_path = temp_dir / f"session_{self.active_session_id}_raw_loopback_debug.wav"
            self._write_wav_artifact(raw_path, raw_artifact)
            self._raw_loopback_debug_wav_path = raw_path
            self.diagnostics.raw_loopback_debug_wav_path = str(raw_path)
            self.diagnostics.raw_loopback_debug_duration_seconds = self._get_wav_duration_seconds(raw_path)

        processed_artifact = artifacts.get("processed_loopback")
        if isinstance(processed_artifact, dict) and processed_artifact.get("data"):
            processed_path = temp_dir / f"session_{self.active_session_id}_processed_loopback_debug.wav"
            self._write_wav_artifact(processed_path, processed_artifact)
            self._processed_loopback_debug_wav_path = processed_path
            self._debug_audio_dump_path = processed_path
            self._final_transcription_audio_wav_path = processed_path
            self.diagnostics.processed_loopback_debug_wav_path = str(processed_path)
            self.diagnostics.debug_audio_dump_path = str(processed_path)
            self.diagnostics.processed_loopback_debug_duration_seconds = self._get_wav_duration_seconds(
                processed_path
            )

        raw_microphone_artifact = artifacts.get("raw_microphone")
        if isinstance(raw_microphone_artifact, dict) and raw_microphone_artifact.get("data"):
            raw_microphone_path = temp_dir / f"session_{self.active_session_id}_raw_microphone_debug.wav"
            self._write_wav_artifact(raw_microphone_path, raw_microphone_artifact)
            self.diagnostics.debug_audio_dump_path = str(raw_microphone_path)

        processed_microphone_artifact = artifacts.get("processed_microphone")
        if isinstance(processed_microphone_artifact, dict) and processed_microphone_artifact.get("data"):
            processed_microphone_path = temp_dir / f"session_{self.active_session_id}_processed_microphone_audio.wav"
            self._write_wav_artifact(processed_microphone_path, processed_microphone_artifact)
            self._debug_audio_dump_path = processed_microphone_path
            self._final_transcription_audio_wav_path = processed_microphone_path
            self.diagnostics.debug_audio_dump_path = str(processed_microphone_path)
        self._update_debug_audio_coverage_diagnostics()

    @staticmethod
    def _write_wav_artifact(path: Path, artifact: dict[str, object]) -> None:
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(int(artifact.get("channels", 1)))
            wav_file.setsampwidth(int(artifact.get("sample_width_bytes", 2)))
            wav_file.setframerate(int(artifact.get("sample_rate", 16000)))
            wav_file.writeframes(bytes(artifact.get("data", b"")))

    @staticmethod
    def _get_wav_duration_seconds(path: Path) -> float | None:
        if not path.exists():
            return None
        with wave.open(str(path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            if frame_rate <= 0:
                return None
            return round(wav_file.getnframes() / float(frame_rate), 2)

    def _update_debug_audio_coverage_diagnostics(self) -> None:
        expected_seconds = None
        if self.diagnostics.last_chunk_timestamp_ms is not None:
            expected_seconds = round(self.diagnostics.last_chunk_timestamp_ms / 1000.0, 2)
        self.diagnostics.expected_audio_coverage_seconds = expected_seconds

        raw_duration = self.diagnostics.raw_loopback_debug_duration_seconds
        processed_duration = self.diagnostics.processed_loopback_debug_duration_seconds

        self.diagnostics.raw_loopback_coverage_ratio = self._coverage_ratio(raw_duration, expected_seconds)
        self.diagnostics.processed_loopback_coverage_ratio = self._coverage_ratio(
            processed_duration,
            expected_seconds,
        )
        raw_limit_seconds = max(0.0, get_settings().transcription.debug_loopback_audio_dump_seconds)
        self.diagnostics.raw_debug_window_limited = bool(
            expected_seconds
            and raw_duration is not None
            and raw_limit_seconds > 0
            and raw_duration + 0.25 < expected_seconds
            and abs(raw_duration - raw_limit_seconds) <= 0.25
        )
        self.diagnostics.debug_audio_tail_truncated = bool(
            expected_seconds
            and processed_duration is not None
            and processed_duration + 0.25 < expected_seconds
        )

    @staticmethod
    def _coverage_ratio(duration_seconds: float | None, expected_seconds: float | None) -> float | None:
        if duration_seconds is None or expected_seconds is None or expected_seconds <= 0:
            return None
        return round(duration_seconds / expected_seconds, 3)


_session_manager = SessionManager()


def get_session_manager() -> SessionManager:
    return _session_manager
