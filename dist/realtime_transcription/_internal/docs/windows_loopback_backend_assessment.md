# Windows Loopback Backend Assessment

## Summary

The current Windows loopback backend is the strongest remaining suspect for poor transcription quality.

Why:

- The source speech WAV is clean.
- The file-based processed reference WAV is also clean.
- Multiple `raw_loopback_debug.wav` captures from live playback are still audibly degraded.
- Queue backlog is no longer the primary issue.
- Repeated `SoundcardRuntimeWarning: data discontinuity in recording` warnings are still observed during live capture.

This means the degradation is already present before Whisper decoding and before the app's mono/resample preparation path.

## Repeatable Comparison Test

Use the same clean speech source for repeated manual comparison:

1. Use `harvard.wav` as the playback source.
2. Play it through the chosen system output device.
3. Start a loopback-only transcription session.
4. Stop the session after the debug WAVs are written.
5. Compare:
   - `runtime/temp/harvard_raw_reference.wav`
   - `runtime/temp/session_<id>_raw_loopback_debug.wav`
   - `runtime/temp/harvard_processed_reference.wav`
   - `runtime/temp/session_<id>_processed_loopback_debug.wav`

Interpretation:

- If the raw reference is clean but the raw loopback capture is degraded, the problem is in live loopback capture.
- If raw loopback is acceptable but processed loopback becomes degraded, the problem is in the conversion pipeline.
- If both raw and processed loopback are acceptable but transcript quality remains poor, the model becomes the next suspect.

## Repeated Raw Capture Result

Based on repeated runs with the same clean Harvard speech WAV played through the PC:

- `session_20260412_53672593_raw_loopback_debug.wav` is degraded
- `session_20260412_53708327_raw_loopback_debug.wav` is degraded

Both runs preserve the expected nominal format:

- sample rate: `44100`
- channels: `2`
- sample width: `16-bit PCM`
- duration: `8.0s`

But both still sound bad, which strongly suggests the current live backend path is the problem rather than file content or later processing.

Repeatability conclusion:

- the degradation reproduces across multiple captures of the same clean source
- this makes the current `soundcard` backend path the most likely root cause
- Whisper tuning should stay paused until the loopback backend itself is improved or replaced

## Current Backend Assessment

Current backend:

- Python package: `soundcard`
- path: Media Foundation / WASAPI loopback through `soundcard`

Current conclusion:

- The existing backend is likely the root cause of degraded raw capture on this machine.
- The issue appears to be related to live loopback acquisition itself rather than Whisper settings.
- The repeated discontinuity warnings further support this conclusion.

## Proposed Alternative Backend Path

Preferred replacement candidate:

- `PyAudioWPatch`

Why this is the best next candidate:

- It is explicitly built around PortAudio with WASAPI loopback support.
- It exposes loopback devices as input-style devices, which is a more direct fit for stable streaming capture.
- It is a better candidate for replacing the current `soundcard`-based endpoint recorder while keeping the rest of the app architecture intact.

Proposed architecture:

- Keep `WindowsLoopbackCapture` as the app-facing provider abstraction.
- Add a new backend implementation inside that provider or behind a small selector.
- Use `PyAudioWPatch` to enumerate the WASAPI loopback analogue of the selected output endpoint.
- Continue emitting the same `AudioChunk` model and the same raw/processed debug WAV diagnostics.

## Next Implementation Step

If backend replacement is approved, the next step should be:

1. Add `PyAudioWPatch` as an optional Windows dependency.
2. Implement a `PyAudioWPatch`-based loopback capture path.
3. Keep the current `soundcard` path behind a fallback or diagnostic switch.
4. Run the same Harvard WAV comparison again.
5. Compare the new raw loopback debug WAV against the current degraded raw loopback output.

Success criteria for that replacement:

- raw loopback debug WAV sounds noticeably cleaner
- fewer or no discontinuity warnings
- same downstream processing path can then be reused unchanged
