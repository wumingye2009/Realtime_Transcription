from typing import Literal

from pydantic import BaseModel, Field


DeviceKind = Literal["input", "output"]
CaptureSource = Literal["microphone", "system_loopback", "fake", "unknown"]
CaptureState = Literal["idle", "running", "paused", "stopped"]


class AudioDevice(BaseModel):
    id: str
    name: str
    description: str | None = None
    kind: DeviceKind
    channels: int = Field(default=0)
    default_samplerate: float | None = None
    hostapi: str | None = None
    is_default: bool = False
    capture_source: CaptureSource | None = None
    capture_provider: str | None = None
    supports_loopback: bool = False


class AudioChunk(BaseModel):
    source: CaptureSource
    sample_rate: int
    channels: int
    frames: int
    timestamp_ms: int = 0
    data: bytes = b""


class CaptureConfig(BaseModel):
    source: CaptureSource
    device_id: str
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    enabled: bool = True


class CapturePlan(BaseModel):
    system_loopback: CaptureConfig
    microphone: CaptureConfig | None = None


class AvailableDevicesResponse(BaseModel):
    system_output_devices: list[AudioDevice]
    microphone_input_devices: list[AudioDevice]
    warnings: list[str] = Field(default_factory=list)
