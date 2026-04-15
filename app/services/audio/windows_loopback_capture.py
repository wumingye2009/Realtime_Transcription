from __future__ import annotations

import importlib
import logging
import warnings
from threading import Event, Lock, Thread
from time import sleep

import numpy as np
import sounddevice as sd

from app.core.config import get_settings
from app.models.audio import AudioChunk, CaptureConfig
from app.services.audio.capture_base import AudioCaptureBase


logger = logging.getLogger(__name__)


class WindowsLoopbackCapture(AudioCaptureBase):
    """Minimal Windows loopback capture provider using SoundCard/WASAPI."""

    def __init__(self, config: CaptureConfig) -> None:
        super().__init__(config)
        self._buffered_chunks: list[AudioChunk] = []
        self._buffer_lock = Lock()
        self._stop_event = Event()
        self._reader_thread: Thread | None = None
        self._recorder = None
        self._recorder_cm = None
        self._pyaudio_manager = None
        self._pyaudio_stream = None
        self._speaker = None
        self._loopback_microphone = None
        self._source_sample_rate = 48000
        self._source_numframes = 3072
        self._source_channels = 2
        self._backend_name = "unknown"
        self._capture_read_seconds = 0.2
        self._processed_chunk_frames = int(self.config.chunk_size)
        self._processed_remainder = np.array([], dtype=np.float32)
        self._emitted_processed_frames = 0
        debug_strategy = get_settings().transcription.get_debug_strategy()
        debug_seconds = max(0.0, get_settings().transcription.debug_loopback_audio_dump_seconds)
        self._debug_raw_max_frames = int(self._source_sample_rate * debug_seconds)
        self._debug_processed_max_frames: int | None = None
        self._dump_raw_wav_enabled = bool(debug_strategy.dump_raw_wav)
        self._dump_processed_wav_enabled = bool(debug_strategy.dump_processed_wav)
        self._debug_raw_frames = 0
        self._debug_processed_frames = 0
        self._debug_raw_pcm16 = bytearray()
        self._debug_processed_pcm16 = bytearray()
        self._runtime_diagnostics: dict[str, object] = {
            "raw_sample_rate": None,
            "raw_channels": None,
            "raw_dtype": None,
            "chunk_frames": None,
            "processed_sample_rate": int(self.config.sample_rate),
            "processed_channels": 1,
            "host_api": None,
            "backend_path": None,
            "discontinuity_warnings": 0,
            "capture_read_seconds": self._capture_read_seconds,
            "conversion_path": "loopback_float -> nan_to_num(0) -> stereo_to_mono(mean) -> concat_remainder -> linear_interp_resample -> clip[-1,1] -> pcm16 -> fixed_chunk_split",
        }

    def start(self) -> None:
        if self.state == "running":
            return

        try:
            self._prepare_loopback_recorder()
        except Exception as exc:
            self._cleanup_recorder()
            self.state = "idle"
            raise ValueError(
                "Unable to open Windows loopback capture for the selected output device. "
                f"Underlying error: {exc}"
            ) from exc

        self._stop_event.clear()
        self._processed_remainder = np.array([], dtype=np.float32)
        self._emitted_processed_frames = 0
        self._debug_raw_frames = 0
        self._debug_processed_frames = 0
        self._debug_raw_pcm16 = bytearray()
        self._debug_processed_pcm16 = bytearray()
        self.state = "running"
        self._reader_thread = Thread(target=self._capture_loop, name="windows-loopback-capture", daemon=True)
        self._reader_thread.start()

    def pause(self) -> None:
        if self.state != "running":
            return
        self.state = "paused"

    def resume(self) -> None:
        if self.state != "paused":
            return
        self.state = "running"

    def stop(self) -> None:
        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None
        self._flush_processed_remainder_chunk()
        self._cleanup_recorder()
        self.state = "stopped"

    def get_buffered_chunks(self) -> list[AudioChunk]:
        with self._buffer_lock:
            chunks = list(self._buffered_chunks)
            self._buffered_chunks.clear()
        return chunks

    def _prepare_loopback_recorder(self) -> None:
        device_info = sd.query_devices(int(self.config.device_id))
        source_sample_rate = device_info.get("default_samplerate") or 48000
        self._source_sample_rate = int(source_sample_rate)
        self._runtime_diagnostics["host_api"] = self._lookup_hostapi_name(device_info.get("hostapi"))
        derived_target_frames = int(self._source_sample_rate * (self.config.chunk_size / float(self.config.sample_rate)))
        preferred_read_frames = int(self._source_sample_rate * self._capture_read_seconds)
        self._source_numframes = max(derived_target_frames, preferred_read_frames, 1)
        self._debug_raw_max_frames = int(
            self._source_sample_rate * max(0.0, get_settings().transcription.debug_loopback_audio_dump_seconds)
        )
        self._debug_processed_max_frames = None

        try:
            self._prepare_pyaudiowpatch_loopback(device_info)
            return
        except Exception as exc:
            logger.info("PyAudioWPatch loopback setup failed, falling back to soundcard: %s", exc)
            self._cleanup_pyaudio()

        self._prepare_soundcard_loopback(device_info)

    def _prepare_pyaudiowpatch_loopback(self, device_info: dict) -> None:
        pyaudio = self._load_pyaudiowpatch_module()
        manager = pyaudio.PyAudio()
        try:
            loopback_info = self._resolve_pyaudio_loopback_device(manager, device_info)
            self._source_sample_rate = int(loopback_info.get("defaultSampleRate") or self._source_sample_rate)
            self._source_channels = max(1, min(int(loopback_info.get("maxInputChannels") or 2), 2))
            derived_target_frames = int(
                self._source_sample_rate * (self.config.chunk_size / float(self.config.sample_rate))
            )
            preferred_read_frames = int(self._source_sample_rate * self._capture_read_seconds)
            self._source_numframes = max(derived_target_frames, preferred_read_frames, 1)
            self._runtime_diagnostics.update(
                {
                    "raw_sample_rate": self._source_sample_rate,
                    "raw_channels": self._source_channels,
                    "raw_dtype": "int16",
                    "chunk_frames": self._source_numframes,
                    "processed_sample_rate": int(self.config.sample_rate),
                    "processed_channels": 1,
                    "backend_path": "PyAudioWPatch WASAPI loopback",
                    "host_api": "Windows WASAPI",
                }
            )
            self._pyaudio_stream = manager.open(
                format=pyaudio.paInt16,
                channels=self._source_channels,
                rate=self._source_sample_rate,
                input=True,
                input_device_index=int(loopback_info["index"]),
                frames_per_buffer=self._source_numframes,
                start=False,
            )
            self._pyaudio_stream.start_stream()
            self._pyaudio_manager = manager
            self._backend_name = "pyaudiowpatch"
        except Exception:
            manager.terminate()
            raise

    def _prepare_soundcard_loopback(self, device_info: dict) -> None:
        soundcard = self._load_soundcard_module()
        self._speaker = self._resolve_speaker(soundcard)
        self._loopback_microphone = soundcard.get_microphone(self._speaker.name, include_loopback=True)

        loopback_channels = getattr(self._loopback_microphone, "channels", None)
        if isinstance(loopback_channels, int) and loopback_channels > 0:
            self._source_channels = max(1, min(loopback_channels, 2))
        else:
            self._source_channels = 2
        self._runtime_diagnostics.update(
            {
                "raw_sample_rate": self._source_sample_rate,
                "raw_channels": self._source_channels,
                "raw_dtype": "float32",
                "chunk_frames": self._source_numframes,
                "processed_sample_rate": int(self.config.sample_rate),
                "processed_channels": 1,
                "backend_path": "soundcard.mediafoundation WASAPI loopback",
            }
        )
        self._recorder_cm = self._loopback_microphone.recorder(
            samplerate=self._source_sample_rate,
            channels=self._source_channels,
            blocksize=self._source_numframes,
        )
        self._recorder = self._recorder_cm.__enter__()
        self._backend_name = "soundcard"

    def _resolve_speaker(self, soundcard_module):
        device_info = sd.query_devices(int(self.config.device_id))
        selected_name = str(device_info["name"])

        if "Microsoft Sound Mapper" in selected_name or "Primary Sound Driver" in selected_name:
            return soundcard_module.default_speaker()

        try:
            return soundcard_module.get_speaker(selected_name)
        except Exception:
            normalized_selected = self._normalize_name(selected_name)
            for speaker in soundcard_module.all_speakers():
                if normalized_selected in self._normalize_name(speaker.name) or self._normalize_name(speaker.name) in normalized_selected:
                    return speaker
            raise

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            if self.state == "paused":
                sleep(0.05)
                continue
            if self.state != "running":
                sleep(0.01)
                continue

            try:
                data = self._read_backend_frames()
                chunks = self._to_audio_chunks(data)
                if not chunks:
                    continue
                with self._buffer_lock:
                    self._buffered_chunks.extend(chunks)
            except Exception:
                logger.exception("Windows loopback capture thread failed")
                self.state = "stopped"
                self._stop_event.set()
                break

    def _read_backend_frames(self):
        if self._backend_name == "pyaudiowpatch":
            if self._pyaudio_stream is None:
                raise RuntimeError("PyAudioWPatch stream is not available.")
            raw_bytes = self._pyaudio_stream.read(self._source_numframes, exception_on_overflow=False)
            return np.frombuffer(raw_bytes, dtype=np.int16).reshape(-1, self._source_channels)

        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            data = self._recorder.record(numframes=self._source_numframes)
        discontinuities = [
            warning
            for warning in captured_warnings
            if "data discontinuity in recording" in str(warning.message).lower()
        ]
        if discontinuities:
            self._runtime_diagnostics["discontinuity_warnings"] = int(
                self._runtime_diagnostics.get("discontinuity_warnings", 0)
            ) + len(discontinuities)
        return data

    def _to_audio_chunks(self, data) -> list[AudioChunk]:
        raw_waveform = np.asarray(data)
        if raw_waveform.size == 0:
            return []

        raw_channels = int(raw_waveform.shape[1]) if raw_waveform.ndim > 1 else 1
        raw_dtype = str(raw_waveform.dtype)
        self._runtime_diagnostics.update(
            {
                "raw_channels": raw_channels,
                "raw_dtype": raw_dtype,
                "chunk_frames": int(raw_waveform.shape[0]) if raw_waveform.ndim >= 1 else 0,
            }
        )
        self._append_raw_debug_audio(raw_waveform, raw_channels)

        if np.issubdtype(raw_waveform.dtype, np.integer):
            waveform = raw_waveform.astype(np.float32) / 32768.0
        else:
            waveform = np.asarray(raw_waveform, dtype=np.float32)
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1, dtype=np.float32)

        waveform = self._resample_to_target_rate(waveform, self._source_sample_rate, int(self.config.sample_rate))
        if self._processed_remainder.size:
            waveform = np.concatenate([self._processed_remainder, waveform]).astype(np.float32, copy=False)
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        waveform = np.clip(waveform, -1.0, 1.0)
        if waveform.size == 0:
            return []

        chunks: list[AudioChunk] = []
        chunk_frames = max(1, self._processed_chunk_frames)
        complete_frames = (waveform.shape[0] // chunk_frames) * chunk_frames
        if complete_frames <= 0:
            self._processed_remainder = waveform.astype(np.float32, copy=True)
            return []

        complete_waveform = waveform[:complete_frames]
        self._processed_remainder = waveform[complete_frames:].astype(np.float32, copy=True)
        pcm16 = np.rint(complete_waveform * 32767.0).astype(np.int16)
        self._append_processed_debug_audio(pcm16)

        for offset in range(0, pcm16.shape[0], chunk_frames):
            frames = min(chunk_frames, pcm16.shape[0] - offset)
            chunk_pcm16 = pcm16[offset : offset + frames]
            timestamp_ms = int((self._emitted_processed_frames / float(self.config.sample_rate)) * 1000)
            chunks.append(
                AudioChunk(
                    source="system_loopback",
                    sample_rate=int(self.config.sample_rate),
                    channels=1,
                    frames=int(chunk_pcm16.shape[0]),
                    timestamp_ms=timestamp_ms,
                    data=chunk_pcm16.tobytes(),
                )
            )
            self._emitted_processed_frames += int(chunk_pcm16.shape[0])
        return chunks

    def get_runtime_diagnostics(self) -> dict[str, object]:
        return dict(self._runtime_diagnostics)

    def get_debug_audio_artifacts(self) -> dict[str, object]:
        return {
            "raw_loopback": {
                "sample_rate": self._source_sample_rate,
                "channels": int(self._runtime_diagnostics.get("raw_channels") or self._source_channels),
                "sample_width_bytes": 2,
                "data": bytes(self._debug_raw_pcm16),
            },
            "processed_loopback": {
                "sample_rate": int(self.config.sample_rate),
                "channels": 1,
                "sample_width_bytes": 2,
                "data": bytes(self._debug_processed_pcm16),
            },
        }

    def _append_raw_debug_audio(self, waveform: np.ndarray, channels: int) -> None:
        if not self._dump_raw_wav_enabled:
            return
        if self._debug_raw_frames >= self._debug_raw_max_frames:
            return

        raw_array = np.asarray(waveform)
        if np.issubdtype(raw_array.dtype, np.integer):
            pcm16 = raw_array.astype(np.int16, copy=False)
        else:
            raw = np.asarray(raw_array, dtype=np.float32)
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            raw = np.clip(raw, -1.0, 1.0)
            pcm16 = np.rint(raw * 32767.0).astype(np.int16)
        if pcm16.ndim == 1:
            pcm16 = pcm16.reshape(-1, 1)
        frames_available = pcm16.shape[0]
        frames_to_take = min(self._debug_raw_max_frames - self._debug_raw_frames, frames_available)
        if frames_to_take <= 0:
            return
        self._debug_raw_pcm16.extend(pcm16[:frames_to_take].tobytes())
        self._debug_raw_frames += frames_to_take

    def _append_processed_debug_audio(self, pcm16: np.ndarray) -> None:
        if not self._dump_processed_wav_enabled:
            return
        if self._debug_processed_max_frames is not None and self._debug_processed_frames >= self._debug_processed_max_frames:
            return
        frames_available = int(pcm16.shape[0])
        if self._debug_processed_max_frames is None:
            frames_to_take = frames_available
        else:
            frames_to_take = min(self._debug_processed_max_frames - self._debug_processed_frames, frames_available)
        if frames_to_take <= 0:
            return
        self._debug_processed_pcm16.extend(pcm16[:frames_to_take].tobytes())
        self._debug_processed_frames += frames_to_take

    def _flush_processed_remainder_chunk(self) -> None:
        if self._processed_remainder.size == 0:
            return

        waveform = np.nan_to_num(self._processed_remainder, nan=0.0, posinf=0.0, neginf=0.0)
        waveform = np.clip(waveform, -1.0, 1.0)
        pcm16 = np.rint(waveform * 32767.0).astype(np.int16)
        self._processed_remainder = np.array([], dtype=np.float32)
        if pcm16.size == 0:
            return

        self._append_processed_debug_audio(pcm16)
        timestamp_ms = int((self._emitted_processed_frames / float(self.config.sample_rate)) * 1000)
        chunk = AudioChunk(
            source="system_loopback",
            sample_rate=int(self.config.sample_rate),
            channels=1,
            frames=int(pcm16.shape[0]),
            timestamp_ms=timestamp_ms,
            data=pcm16.tobytes(),
        )
        with self._buffer_lock:
            self._buffered_chunks.append(chunk)
        self._emitted_processed_frames += int(pcm16.shape[0])

    @staticmethod
    def _resample_to_target_rate(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate == target_rate or waveform.size == 0:
            return waveform.astype(np.float32, copy=False)

        duration = waveform.shape[0] / float(source_rate)
        target_length = max(1, int(duration * target_rate))
        source_positions = np.linspace(0.0, 1.0, num=waveform.shape[0], endpoint=False)
        target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
        return np.interp(target_positions, source_positions, waveform).astype(np.float32)

    @staticmethod
    def _normalize_name(name: str) -> str:
        return "".join(ch.lower() for ch in name if ch.isalnum() or ch.isspace()).strip()

    @staticmethod
    def _lookup_hostapi_name(hostapi_index) -> str | None:
        if not isinstance(hostapi_index, int):
            return None
        try:
            hostapis = sd.query_hostapis()
        except Exception:
            return str(hostapi_index)
        if 0 <= hostapi_index < len(hostapis):
            return hostapis[hostapi_index].get("name")
        return str(hostapi_index)

    @staticmethod
    def _load_soundcard_module():
        try:
            soundcard = importlib.import_module("soundcard")
        except ImportError as exc:
            raise ValueError(
                "Windows loopback capture requires the 'soundcard' package. "
                "Install it in the transcription environment first."
            ) from exc
        WindowsLoopbackCapture._patch_soundcard_numpy_compatibility(soundcard)
        return soundcard

    @staticmethod
    def _load_pyaudiowpatch_module():
        try:
            return importlib.import_module("pyaudiowpatch")
        except ImportError as exc:
            raise ValueError(
                "PyAudioWPatch is not installed. Install it to enable the alternative WASAPI loopback backend."
            ) from exc

    def _resolve_pyaudio_loopback_device(self, manager, device_info: dict) -> dict:
        try:
            loopback = manager.get_wasapi_loopback_analogue_by_index(int(self.config.device_id))
            if loopback:
                return loopback
        except Exception:
            pass

        selected_name = str(device_info["name"])
        normalized_selected = self._normalize_name(selected_name)

        candidates = []
        for candidate in manager.get_loopback_device_info_generator():
            candidate_name = str(candidate.get("name", ""))
            normalized_candidate = self._normalize_name(candidate_name)
            if normalized_selected in normalized_candidate or normalized_candidate in normalized_selected:
                candidates.append(candidate)

        if candidates:
            return candidates[0]

        raise ValueError(
            f"Unable to resolve a PyAudioWPatch WASAPI loopback device for output endpoint '{selected_name}'."
        )

    @staticmethod
    def _patch_soundcard_numpy_compatibility(soundcard_module) -> None:
        mediafoundation = getattr(soundcard_module, "mediafoundation", None)
        if mediafoundation is None:
            return

        soundcard_numpy = getattr(mediafoundation, "numpy", None)
        if soundcard_numpy is None:
            return

        current_fromstring = getattr(soundcard_numpy, "fromstring", None)
        if current_fromstring is None or getattr(current_fromstring, "_rt_numpy_compat", False):
            return

        def _compat_fromstring(buffer, dtype=float, count=-1, sep="", offset=0):
            if sep == "":
                return np.frombuffer(buffer, dtype=dtype, count=count, offset=offset)
            return np.fromstring(buffer, dtype=dtype, count=count, sep=sep)

        _compat_fromstring._rt_numpy_compat = True  # type: ignore[attr-defined]
        soundcard_numpy.fromstring = _compat_fromstring

    def _cleanup_recorder(self) -> None:
        self._cleanup_pyaudio()
        if self._recorder_cm is not None:
            try:
                self._recorder_cm.__exit__(None, None, None)
            except Exception:
                pass
        self._recorder_cm = None
        self._recorder = None
        self._speaker = None
        self._loopback_microphone = None

    def _cleanup_pyaudio(self) -> None:
        if self._pyaudio_stream is not None:
            try:
                if self._pyaudio_stream.is_active():
                    self._pyaudio_stream.stop_stream()
            except Exception:
                pass
            try:
                self._pyaudio_stream.close()
            except Exception:
                pass
        if self._pyaudio_manager is not None:
            try:
                self._pyaudio_manager.terminate()
            except Exception:
                pass
        self._pyaudio_stream = None
        self._pyaudio_manager = None
