from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator


load_dotenv()


RuntimeMode = Literal["production", "troubleshooting", "development"]


class TranscriptionProfileSettings(BaseModel):
    beam_size: int = 1
    vad_filter: bool = False
    language: str = "session"
    compression_ratio_threshold: float = 2.0
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.7
    condition_on_previous_text: bool = True


class DebugStrategySettings(BaseModel):
    dump_raw_wav: bool | None = None
    dump_processed_wav: bool | None = None
    log_queue_depth: bool | None = None
    log_chunks: bool | None = None
    log_timing: bool | None = None
    verbose_worker: bool | None = None


class RuntimeStrategySettings(BaseModel):
    mode: RuntimeMode = "production"
    realtime: TranscriptionProfileSettings = Field(default_factory=TranscriptionProfileSettings)
    final: TranscriptionProfileSettings = Field(
        default_factory=lambda: TranscriptionProfileSettings(
            beam_size=3,
            vad_filter=True,
            language="en",
            compression_ratio_threshold=2.0,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.5,
            condition_on_previous_text=False,
        )
    )
    debug: DebugStrategySettings = Field(default_factory=DebugStrategySettings)

    @model_validator(mode="after")
    def apply_mode_defaults(self) -> "RuntimeStrategySettings":
        mode_defaults = {
            "production": {
                "dump_raw_wav": False,
                "dump_processed_wav": True,
                "log_queue_depth": False,
                "log_chunks": False,
                "log_timing": False,
                "verbose_worker": False,
            },
            "troubleshooting": {
                "dump_raw_wav": False,
                "dump_processed_wav": True,
                "log_queue_depth": True,
                "log_chunks": False,
                "log_timing": True,
                "verbose_worker": False,
            },
            "development": {
                "dump_raw_wav": True,
                "dump_processed_wav": True,
                "log_queue_depth": True,
                "log_chunks": True,
                "log_timing": True,
                "verbose_worker": True,
            },
        }
        defaults = mode_defaults[self.mode]
        for field_name, default_value in defaults.items():
            if getattr(self.debug, field_name) is None:
                setattr(self.debug, field_name, default_value)
        return self


class TranscriptionSettings(BaseModel):
    model_size: str = "base"
    final_model_size: str | None = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    buffer_seconds: float = 8.0
    rolling_window_seconds: float = 10.0
    transcription_cadence_seconds: float = 2.0
    flush_minimum_seconds: float = 1.5
    beam_size: int = 1
    vad_filter: bool = False
    compression_ratio_threshold: float = 2.0
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.7
    condition_on_previous_text: bool = True
    emit_min_chars: int = 8
    emit_min_duration_seconds: float = 0.5
    merge_gap_seconds: float = 1.0
    debug_loopback_audio_dump_enabled: bool = True
    debug_loopback_audio_dump_seconds: float = 8.0
    offline_additional_comparison_model_sizes: list[str] = Field(default_factory=list)
    runtime: RuntimeStrategySettings = Field(default_factory=RuntimeStrategySettings)

    def get_realtime_profile(self) -> TranscriptionProfileSettings:
        return self.runtime.realtime.model_copy(
            update={
                "beam_size": self.beam_size,
                "vad_filter": self.vad_filter,
                "compression_ratio_threshold": self.compression_ratio_threshold,
                "log_prob_threshold": self.log_prob_threshold,
                "no_speech_threshold": self.no_speech_threshold,
                "condition_on_previous_text": self.condition_on_previous_text,
            }
        )

    def get_final_profile(self) -> TranscriptionProfileSettings:
        return self.runtime.final.model_copy()

    def get_debug_strategy(self) -> DebugStrategySettings:
        return self.runtime.debug.model_copy()

    def get_final_model_size(self) -> str:
        return self.final_model_size or self.model_size


class AppSettings(BaseModel):
    app_name: str = "Realtime Transcription"
    app_env: str = "development"
    host: str = "127.0.0.1"
    port: int = 8000
    default_output_dir: Path = Field(default=Path("outputs"))
    runtime_dir: Path = Field(default=Path("runtime"))
    logs_dir: Path = Field(default=Path("runtime/logs"))
    temp_dir: Path = Field(default=Path("runtime/temp"))
    default_export_format: str = "md"
    microphone_enabled_by_default: bool = False
    default_language_mode: str = "auto"
    transcription: TranscriptionSettings = Field(default_factory=TranscriptionSettings)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    settings = AppSettings()
    settings.default_output_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    return settings
