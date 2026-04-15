from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.transcript import TranscriptSegment


LanguageMode = Literal["auto", "english", "chinese"]
ExportFormat = Literal["md", "txt"]
SessionState = Literal["idle", "running", "paused", "stopping", "stopped"]


class SessionCreateRequest(BaseModel):
    system_output_device_id: str = Field(min_length=1)
    microphone_enabled: bool = False
    microphone_input_device_id: str | None = None
    language_mode: LanguageMode = "auto"
    output_dir: str = "outputs"
    export_formats: list[ExportFormat] = Field(default_factory=lambda: ["md"])

    @field_validator("output_dir", mode="before")
    @classmethod
    def normalize_output_dir(cls, value: str | None) -> str:
        if value is None:
            return "outputs"

        normalized = str(value).strip()
        return normalized or "outputs"

    @field_validator("microphone_input_device_id", mode="before")
    @classmethod
    def normalize_microphone_device(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalized = str(value).strip()
        return normalized or None

    @field_validator("export_formats")
    @classmethod
    def ensure_export_formats(cls, value: list[ExportFormat]) -> list[ExportFormat]:
        if not value:
            return ["md"]

        ordered_unique: list[ExportFormat] = []
        for item in value:
            if item not in ordered_unique:
                ordered_unique.append(item)

        if "md" not in ordered_unique:
            ordered_unique.insert(0, "md")

        return ordered_unique

    @model_validator(mode="after")
    def validate_microphone_selection(self) -> "SessionCreateRequest":
        if not self.microphone_enabled:
            self.microphone_input_device_id = None
            return self

        if not self.microphone_input_device_id:
            raise ValueError("Microphone device selection is required when microphone is enabled.")

        return self


class SessionControlResponse(BaseModel):
    ok: bool = True
    state: SessionState
    message: str


class SessionDiagnostics(BaseModel):
    selected_capture_providers: list[str] = Field(default_factory=list)
    active_audio_source: str | None = None
    microphone_capture_start_attempted: bool = False
    microphone_capture_start_succeeded: bool = False
    system_loopback_start_attempted: bool = False
    system_loopback_start_succeeded: bool = False
    capture_attempted_sources: list[str] = Field(default_factory=list)
    capture_started_sources: list[str] = Field(default_factory=list)
    total_audio_chunks_received: int = 0
    total_microphone_chunks_received: int = 0
    total_system_loopback_chunks_received: int = 0
    total_fake_chunks_received: int = 0
    total_unknown_chunks_received: int = 0
    total_chunks_passed_to_engine: int = 0
    max_queue_depth: int = 0
    transcription_jobs_started: int = 0
    transcription_jobs_skipped_in_flight: int = 0
    first_chunk_timestamp_ms: int | None = None
    last_chunk_timestamp_ms: int | None = None
    last_chunk_source: str | None = None
    capture_flow_active: bool = False
    pause_count: int = 0
    resume_count: int = 0
    stop_requested: bool = False
    capture_stop_completed: bool = False
    queued_audio_chunks: int = 0
    producer_finished: bool = False
    finalize_complete: bool = False
    final_drain_completed: bool = False
    stop_returned_partial_transcript_due_to_timeout: bool = False
    debug_audio_dump_path: str | None = None
    raw_loopback_debug_wav_path: str | None = None
    raw_loopback_debug_duration_seconds: float | None = None
    processed_loopback_debug_wav_path: str | None = None
    processed_loopback_debug_duration_seconds: float | None = None
    expected_audio_coverage_seconds: float | None = None
    raw_loopback_coverage_ratio: float | None = None
    processed_loopback_coverage_ratio: float | None = None
    raw_debug_window_limited: bool = False
    debug_audio_tail_truncated: bool = False
    realtime_transcription_mode: str | None = None
    finalization_stage: str | None = None
    final_transcript_ready: bool = False
    final_transcript_in_progress: bool = False
    final_transcript_source: str | None = None
    final_transcript_result_path: str | None = None
    final_transcript_segment_count: int = 0
    final_transcription_audio_path: str | None = None
    final_transcription_audio_duration_seconds: float | None = None
    final_transcript_model_size: str | None = None
    offline_comparison_performed: bool = False
    offline_diagnostics_in_progress: bool = False
    offline_comparison_source_path: str | None = None
    offline_comparison_source_duration_seconds: float | None = None
    offline_comparison_result_path: str | None = None
    offline_comparison_segment_count: int = 0
    offline_comparison_model_size: str | None = None
    offline_alternative_comparison_performed: bool = False
    offline_alternative_model_size: str | None = None
    offline_alternative_result_path: str | None = None
    offline_alternative_segment_count: int = 0
    offline_additional_model_sizes: list[str] = Field(default_factory=list)
    offline_additional_result_paths: list[str] = Field(default_factory=list)
    offline_control_source_path: str | None = None
    offline_control_source_duration_seconds: float | None = None
    offline_control_primary_result_path: str | None = None
    offline_control_primary_segment_count: int = 0
    offline_control_alternative_result_path: str | None = None
    offline_control_alternative_segment_count: int = 0
    offline_control_additional_result_paths: list[str] = Field(default_factory=list)
    loopback_raw_sample_rate: int | None = None
    loopback_raw_channels: int | None = None
    loopback_raw_dtype: str | None = None
    loopback_chunk_frames: int | None = None
    loopback_host_api: str | None = None
    loopback_backend_path: str | None = None
    loopback_discontinuity_warnings: int = 0
    loopback_processed_sample_rate: int | None = None
    loopback_processed_channels: int | None = None
    loopback_conversion_path: str | None = None


class SessionStatusResponse(BaseModel):
    state: SessionState = "idle"
    active_session_id: str | None = None
    session_options: SessionCreateRequest | None = None
    transcript_segments: list[TranscriptSegment] = Field(default_factory=list)
    final_transcript_segments: list[TranscriptSegment] = Field(default_factory=list)
    diagnostics: SessionDiagnostics = Field(default_factory=SessionDiagnostics)
