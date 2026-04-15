from __future__ import annotations

import logging
from pathlib import Path
from threading import Event, Lock, Thread
import wave

import numpy as np
from faster_whisper import WhisperModel

from app.models.audio import AudioChunk
from app.models.transcript import TranscriptSegment
from app.services.audio.capture_base import BufferedAudioSource
from app.services.transcription.engine_base import TranscriptionEngineBase


logger = logging.getLogger(__name__)


class FasterWhisperEngine(TranscriptionEngineBase):
    """Minimal buffered transcription engine for microphone or loopback audio."""

    def __init__(
        self,
        language_mode: str = "auto",
        model_size: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        buffer_window_seconds: float = 8.0,
        rolling_window_seconds: float | None = None,
        transcription_cadence_seconds: float = 2.0,
        flush_minimum_seconds: float = 1.5,
        beam_size: int = 5,
        vad_filter: bool = True,
        compression_ratio_threshold: float = 2.0,
        log_prob_threshold: float = -0.4,
        no_speech_threshold: float = 0.45,
        condition_on_previous_text: bool = False,
        emit_min_chars: int = 12,
        emit_min_duration_seconds: float = 0.8,
        merge_gap_seconds: float = 0.4,
        log_timing: bool = False,
        verbose_worker_logs: bool = False,
    ) -> None:
        self._segments: list[TranscriptSegment] = []
        self._audio_source: BufferedAudioSource | None = None
        self._language_mode = language_mode
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._buffer_window_seconds = buffer_window_seconds
        self._rolling_window_seconds = rolling_window_seconds or buffer_window_seconds
        self._transcription_cadence_seconds = transcription_cadence_seconds
        self._flush_minimum_seconds = flush_minimum_seconds
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._compression_ratio_threshold = compression_ratio_threshold
        self._log_prob_threshold = log_prob_threshold
        self._no_speech_threshold = no_speech_threshold
        self._condition_on_previous_text = condition_on_previous_text
        self._emit_min_chars = emit_min_chars
        self._emit_min_duration_seconds = emit_min_duration_seconds
        self._merge_gap_seconds = merge_gap_seconds
        self._log_timing = log_timing
        self._verbose_worker_logs = verbose_worker_logs
        self._model: WhisperModel | None = None
        self._rolling_pcm: list[np.ndarray] = []
        self._rolling_buffer_duration_seconds = 0.0
        self._captured_audio_seconds = 0.0
        self._audio_since_last_transcription_seconds = 0.0
        self._emitted_until_seconds = 0.0
        self._paused = False
        self._stopped = False
        self._pending_segment: TranscriptSegment | None = None
        self._state_lock = Lock()
        self._transcribe_event = Event()
        self._stop_event = Event()
        self._worker_thread: Thread | None = None
        self._transcription_in_flight = False
        self._pending_transcription_pass = False
        self._transcription_jobs_started = 0
        self._transcription_jobs_skipped_in_flight = 0
        self._final_drain_completed = False
        self._stop_returned_partial_transcript_due_to_timeout = False

    def start(self, audio_source: BufferedAudioSource | None = None) -> None:
        self._audio_source = audio_source
        with self._state_lock:
            self._segments = []
            self._rolling_pcm = []
            self._rolling_buffer_duration_seconds = 0.0
            self._captured_audio_seconds = 0.0
            self._audio_since_last_transcription_seconds = 0.0
            self._emitted_until_seconds = 0.0
            self._paused = False
            self._stopped = False
            self._pending_segment = None
            self._transcription_in_flight = False
            self._pending_transcription_pass = False
            self._transcription_jobs_started = 0
            self._transcription_jobs_skipped_in_flight = 0
            self._final_drain_completed = False
            self._stop_returned_partial_transcript_due_to_timeout = False
        self._stop_event.clear()
        self._transcribe_event.clear()
        self._ensure_model_loaded()
        self._worker_thread = Thread(target=self._worker_loop, name="faster-whisper-worker", daemon=True)
        self._worker_thread.start()

    def process_chunk(self, chunk: AudioChunk) -> None:
        if chunk.source not in {"microphone", "system_loopback"}:
            return

        pcm = self._chunk_to_float32_mono(chunk)
        if pcm.size == 0:
            return

        with self._state_lock:
            if self._paused or self._stopped:
                return
            chunk_duration = pcm.shape[0] / float(chunk.sample_rate)
            self._rolling_pcm.append(pcm)
            self._rolling_buffer_duration_seconds += chunk_duration
            self._captured_audio_seconds += chunk_duration
            self._audio_since_last_transcription_seconds += chunk_duration
            self._trim_rolling_buffer_locked()

            should_transcribe = (
                self._rolling_buffer_duration_seconds >= self._buffer_window_seconds
                and self._audio_since_last_transcription_seconds >= self._transcription_cadence_seconds
            )
            if should_transcribe:
                if self._transcription_in_flight:
                    self._transcription_jobs_skipped_in_flight += 1
                    if self._log_timing and not self._pending_transcription_pass:
                        logger.info(
                            "faster-whisper cadence tick arrived while a job was already running; queuing one pending pass"
                        )
                    self._pending_transcription_pass = True
                else:
                    self._pending_transcription_pass = False
                    self._transcribe_event.set()

    def pause(self) -> None:
        with self._state_lock:
            self._paused = True

    def resume(self) -> None:
        with self._state_lock:
            self._paused = False

    def stop(self) -> None:
        with self._state_lock:
            self._stopped = True
        self._stop_event.set()
        self._transcribe_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=1.0)
            if self._worker_thread.is_alive():
                logger.warning("faster-whisper worker did not stop within timeout; returning partial transcript")
                with self._state_lock:
                    self._stop_returned_partial_transcript_due_to_timeout = True
            self._worker_thread = None
        with self._state_lock:
            if self._pending_segment is not None:
                self._segments.append(self._pending_segment)
                self._emitted_until_seconds = max(self._emitted_until_seconds, self._pending_segment.end)
                self._pending_segment = None

    def get_segments(self) -> list[TranscriptSegment]:
        with self._state_lock:
            return list(self._segments)

    def get_runtime_diagnostics(self) -> dict[str, object]:
        with self._state_lock:
            return {
                "transcription_jobs_started": self._transcription_jobs_started,
                "transcription_jobs_skipped_in_flight": self._transcription_jobs_skipped_in_flight,
                "transcription_in_flight": self._transcription_in_flight,
                "pending_transcription_pass": self._pending_transcription_pass,
                "final_drain_completed": self._final_drain_completed,
                "stop_returned_partial_transcript_due_to_timeout": self._stop_returned_partial_transcript_due_to_timeout,
            }

    def run_offline_transcription(
        self,
        audio_path: Path,
        model_size_override: str | None = None,
        options: dict[str, object] | None = None,
    ) -> list[TranscriptSegment]:
        waveform = self._load_audio_file(audio_path)
        if waveform.size == 0:
            return []

        model_size = model_size_override or self._model_size
        model = self._resolve_offline_model(model_size)
        if model is None:
            return []

        options = options or {}
        beam_size = int(options.get("beam_size", self._beam_size))
        vad_filter = bool(options.get("vad_filter", self._vad_filter))
        compression_ratio_threshold = float(
            options.get("compression_ratio_threshold", self._compression_ratio_threshold)
        )
        log_prob_threshold = float(options.get("log_prob_threshold", self._log_prob_threshold))
        no_speech_threshold = float(options.get("no_speech_threshold", self._no_speech_threshold))
        condition_on_previous_text = bool(
            options.get("condition_on_previous_text", self._condition_on_previous_text)
        )
        language = self._resolve_language_override(options.get("language"))
        logger.info(
            "faster-whisper offline comparison started source=%s model_size=%s language=%s beam_size=%s vad_filter=%s",
            audio_path,
            model_size,
            language or "auto",
            beam_size,
            vad_filter,
        )
        segments, _info = model.transcribe(
            waveform,
            language=language,
            task="transcribe",
            beam_size=beam_size,
            best_of=beam_size,
            vad_filter=vad_filter,
            compression_ratio_threshold=compression_ratio_threshold,
            log_prob_threshold=log_prob_threshold,
            no_speech_threshold=no_speech_threshold,
            condition_on_previous_text=condition_on_previous_text,
        )

        recognized_segments: list[TranscriptSegment] = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            recognized_segments.append(
                TranscriptSegment(
                    start=round(float(segment.start), 2),
                    end=round(float(segment.end), 2),
                    text=text,
                    speaker=None,
                )
            )

        logger.info(
            "faster-whisper offline comparison finished source=%s model_size=%s emitted_segments=%s",
            audio_path,
            model_size,
            len(recognized_segments),
        )
        return recognized_segments

    def _resolve_offline_model(self, model_size: str) -> WhisperModel | None:
        if model_size == self._model_size:
            self._ensure_model_loaded()
            return self._model

        logger.info(
            "Initializing faster-whisper offline comparison model size=%s device=%s compute_type=%s",
            model_size,
            self._device,
            self._compute_type,
        )
        return WhisperModel(model_size, device=self._device, compute_type=self._compute_type)

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        logger.info(
            "Initializing faster-whisper model size=%s device=%s compute_type=%s buffer_seconds=%.2f beam_size=%s vad_filter=%s condition_on_previous_text=%s",
            self._model_size,
            self._device,
            self._compute_type,
            self._buffer_window_seconds,
            self._beam_size,
            self._vad_filter,
            self._condition_on_previous_text,
        )
        try:
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
        except Exception:
            logger.exception("faster-whisper model loading failed")
            raise ValueError(
                "Unable to initialize faster-whisper. "
                "Please confirm the model can be loaded in your local environment."
            )
        logger.info("faster-whisper model loaded successfully")

    def _transcribe_buffer(self, final: bool = False) -> None:
        if self._model is None:
            return
        with self._state_lock:
            if not self._rolling_pcm:
                return
            self._transcription_in_flight = True
            waveform = np.concatenate(self._rolling_pcm).astype(np.float32, copy=False)
            buffer_duration = self._rolling_buffer_duration_seconds
            buffer_start_seconds = max(0.0, self._captured_audio_seconds - buffer_duration)
            self._transcription_jobs_started += 1
            current_job_number = self._transcription_jobs_started
        language = self._map_language()
        if self._log_timing:
            logger.info(
                "faster-whisper transcription started job=%s duration=%.2fs language=%s rolling_start=%.2fs rolling_window=%.2fs cadence=%.2fs final=%s",
                current_job_number,
                buffer_duration,
                language or "auto",
                buffer_start_seconds,
                self._rolling_window_seconds,
                self._transcription_cadence_seconds,
                final,
            )

        try:
            segments, _info = self._model.transcribe(
                waveform,
                language=language,
                task="transcribe",
                beam_size=self._beam_size,
                best_of=self._beam_size,
                vad_filter=self._vad_filter,
                compression_ratio_threshold=self._compression_ratio_threshold,
                log_prob_threshold=self._log_prob_threshold,
                no_speech_threshold=self._no_speech_threshold,
                condition_on_previous_text=self._condition_on_previous_text,
            )
            recognized_segments: list[TranscriptSegment] = []
            for segment in segments:
                text = segment.text.strip()
                if not text:
                    continue
                recognized_segments.append(
                    TranscriptSegment(
                        start=round(buffer_start_seconds + float(segment.start), 2),
                        end=round(buffer_start_seconds + float(segment.end), 2),
                        text=text,
                        speaker=None,
                    )
                )
            recognized = self._append_recognized_segments(recognized_segments, final=final)
        except Exception:
            logger.exception("faster-whisper transcription failed")
            with self._state_lock:
                self._transcription_in_flight = False
            raise ValueError("Microphone transcription failed while processing buffered audio.")

        if self._log_timing:
            logger.info(
                "faster-whisper transcription finished job=%s duration=%.2fs emitted_segments=%s",
                current_job_number,
                buffer_duration,
                recognized,
            )
        with self._state_lock:
            self._audio_since_last_transcription_seconds = 0.0
            self._transcription_in_flight = False

    def _append_recognized_segments(
        self,
        segments: list[TranscriptSegment],
        final: bool,
    ) -> int:
        with self._state_lock:
            recognized = 0
            working: list[TranscriptSegment] = []

            if self._pending_segment is not None:
                working.append(self._pending_segment)
                self._pending_segment = None

            last_emitted = working[-1] if working else (self._segments[-1] if self._segments else None)
            for segment in segments:
                if segment.end <= self._emitted_until_seconds + 0.05:
                    continue
                if segment.start < self._emitted_until_seconds:
                    segment = TranscriptSegment(
                        start=round(self._emitted_until_seconds, 2),
                        end=segment.end,
                        text=segment.text,
                        speaker=segment.speaker,
                    )
                prepared = self._suppress_overlap(last_emitted, segment)
                if prepared is None:
                    continue
                working.append(prepared)
                last_emitted = prepared
            if not working:
                return recognized

            merged: list[TranscriptSegment] = []
            for segment in working:
                if not merged:
                    merged.append(segment)
                    continue

                previous = merged[-1]
                if self._should_merge(previous, segment):
                    merged[-1] = TranscriptSegment(
                        start=previous.start,
                        end=segment.end,
                        text=self._join_text(previous.text, segment.text),
                        speaker=None,
                    )
                else:
                    merged.append(segment)

            if not final and merged and self._should_hold_back(merged[-1]):
                self._pending_segment = merged.pop()

            self._segments.extend(merged)
            recognized += len(merged)
            if merged:
                self._emitted_until_seconds = max(self._emitted_until_seconds, merged[-1].end)

            if final and self._pending_segment is not None:
                self._segments.append(self._pending_segment)
                self._emitted_until_seconds = max(self._emitted_until_seconds, self._pending_segment.end)
                self._pending_segment = None
                recognized += 1

            return recognized

    def _trim_rolling_buffer_locked(self) -> None:
        while self._rolling_pcm and self._rolling_buffer_duration_seconds > self._rolling_window_seconds:
            oldest = self._rolling_pcm.pop(0)
            self._rolling_buffer_duration_seconds -= oldest.shape[0] / 16000.0
        self._rolling_buffer_duration_seconds = max(0.0, self._rolling_buffer_duration_seconds)

    def _worker_loop(self) -> None:
        while True:
            self._transcribe_event.wait(0.1)
            self._transcribe_event.clear()

            while True:
                run_final = False
                run_normal = False
                with self._state_lock:
                    if self._stop_event.is_set():
                        should_flush = self._rolling_buffer_duration_seconds >= self._flush_minimum_seconds
                        if should_flush and not self._transcription_in_flight:
                            run_final = True
                        else:
                            self._final_drain_completed = True
                            return
                    else:
                        should_transcribe = (
                            not self._paused
                            and not self._stopped
                            and not self._transcription_in_flight
                            and self._rolling_buffer_duration_seconds >= self._buffer_window_seconds
                            and self._audio_since_last_transcription_seconds >= self._transcription_cadence_seconds
                        )
                        if should_transcribe:
                            self._pending_transcription_pass = False
                            run_normal = True
                        elif self._pending_transcription_pass and not self._transcription_in_flight:
                            self._pending_transcription_pass = False
                            pending_ready = (
                                self._rolling_buffer_duration_seconds >= self._buffer_window_seconds
                                and self._audio_since_last_transcription_seconds >= self._transcription_cadence_seconds
                            )
                            if pending_ready:
                                if self._log_timing:
                                    logger.info("faster-whisper running deferred pending transcription pass")
                                run_normal = True
                            else:
                                continue
                        else:
                            break

                if run_final:
                    try:
                        self._transcribe_buffer(final=True)
                    except Exception:
                        logger.exception("faster-whisper final flush failed during stop")
                    with self._state_lock:
                        self._final_drain_completed = True
                    return

                if run_normal:
                    try:
                        self._transcribe_buffer()
                    except Exception:
                        logger.exception("faster-whisper transcription pass failed")
                    continue

                break

    def _suppress_overlap(
        self,
        previous: TranscriptSegment | None,
        current: TranscriptSegment,
    ) -> TranscriptSegment | None:
        if previous is None:
            return current

        previous_text = previous.text.strip().lower()
        current_text = current.text.strip().lower()
        if (
            current_text
            and previous_text
            and current_text in previous_text
            and current.start <= previous.end + self._merge_gap_seconds
        ):
            return None

        previous_words = self._tokenize(previous.text)
        current_words = self._tokenize(current.text)
        if not current_words:
            return None

        max_overlap = min(len(previous_words), len(current_words), 8)
        overlap_size = 0
        for size in range(max_overlap, 0, -1):
            if previous_words[-size:] == current_words[:size]:
                overlap_size = size
                break

        if overlap_size == len(current_words):
            return None

        if overlap_size == 0:
            return current

        trimmed_words = current_words[overlap_size:]
        if not trimmed_words:
            return None

        trimmed_text = " ".join(trimmed_words).strip()
        if not trimmed_text:
            return None

        if self._verbose_worker_logs:
            logger.info(
                "faster-whisper overlap suppression trimmed %s words from segment",
                overlap_size,
            )
        return TranscriptSegment(
            start=current.start,
            end=current.end,
            text=trimmed_text,
            speaker=current.speaker,
        )

    def _should_hold_back(self, segment: TranscriptSegment) -> bool:
        duration = max(0.0, segment.end - segment.start)
        return duration < self._emit_min_duration_seconds or len(segment.text.strip()) < self._emit_min_chars

    def _should_merge(self, left: TranscriptSegment, right: TranscriptSegment) -> bool:
        gap = max(0.0, right.start - left.end)
        return (
            gap <= self._merge_gap_seconds
            or self._should_hold_back(left)
            or self._should_hold_back(right)
        )

    @staticmethod
    def _join_text(left: str, right: str) -> str:
        left = left.strip()
        right = right.strip()
        if not left:
            return right
        if not right:
            return left
        if left == right or left.endswith(right):
            return left
        if right.startswith(left):
            return right
        return f"{left} {right}"

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [part for part in text.strip().split() if part]

    @staticmethod
    def _chunk_to_float32_mono(chunk: AudioChunk) -> np.ndarray:
        pcm = np.frombuffer(chunk.data, dtype=np.int16)
        if pcm.size == 0:
            return np.array([], dtype=np.float32)

        if chunk.channels > 1:
            pcm = pcm.reshape(-1, chunk.channels).mean(axis=1)

        waveform = (pcm.astype(np.float32) / 32768.0).flatten()
        return np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _load_audio_file(audio_path: Path) -> np.ndarray:
        with wave.open(str(audio_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            raw_bytes = wav_file.readframes(wav_file.getnframes())

        pcm = np.frombuffer(raw_bytes, dtype=np.int16)
        if pcm.size == 0:
            return np.array([], dtype=np.float32)

        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1)

        waveform = pcm.astype(np.float32) / 32768.0
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        if sample_rate and sample_rate != 16000:
            source_positions = np.arange(waveform.shape[0], dtype=np.float32)
            target_length = max(int(round(waveform.shape[0] * (16000.0 / float(sample_rate)))), 1)
            target_positions = np.linspace(0, waveform.shape[0] - 1, num=target_length, dtype=np.float32)
            waveform = np.interp(target_positions, source_positions, waveform).astype(np.float32)
        return waveform

    def _map_language(self) -> str | None:
        mapping = {
            "auto": None,
            "english": "en",
            "chinese": "zh",
        }
        return mapping.get(self._language_mode, None)

    def _resolve_language_override(self, override: object) -> str | None:
        if override is None:
            return self._map_language()

        normalized = str(override).strip().lower()
        if normalized in {"", "session"}:
            return self._map_language()
        if normalized == "auto":
            return None
        if normalized in {"en", "english"}:
            return "en"
        if normalized in {"zh", "chinese"}:
            return "zh"
        return normalized
