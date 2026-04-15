# Realtime Transcription

Local Windows-first realtime transcription MVP for:
- system audio capture through Windows loopback
- microphone capture
- live transcript preview during a session
- higher-quality final transcript generation after stop

The product direction is intentionally simple:
- realtime transcript is a live preview
- final transcript is the official saved transcript

## MVP Capabilities

- FastAPI backend with a lightweight browser UI
- Session lifecycle: `start`, `pause`, `resume`, `stop`, `stopping`, `stopped`
- Windows system loopback capture using PyAudioWPatch WASAPI loopback
- Microphone capture using `sounddevice`
- Realtime rolling transcription with `faster-whisper`
- Final offline full-pass transcription after stop
- Saved Markdown output based primarily on the final transcript
- Optional TXT export
- Runtime strategy profiles for production, troubleshooting, and development

## Platform Assumptions

- Windows local use
- CPU-first deployment
- Python environment with local audio device access
- No cloud dependencies
- No GPU required

The project is tuned for practical local use first, not for the largest possible model or maximum research flexibility.

## Setup

### 1. Create or activate your Python environment

Use your existing local environment, for example a Conda environment named `transcription`.

### 2. Install dependencies

From the project root:

```powershell
pip install -r requirements.txt
```

Important Windows dependency:
- `PyAudioWPatch` is used for the preferred Windows loopback backend

### 3. Start the app

```powershell
python -m uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Normal Usage

### System audio session

1. Leave `Microphone Enabled` off.
2. Choose the playback device your audio is actually coming from.
3. Click `Start`.
4. Play audio.
5. Click `Stop`.
6. Wait for finalization to complete.

### Microphone session

1. Turn `Microphone Enabled` on.
2. Choose the correct microphone device.
3. Click `Start`.
4. Speak into the chosen mic.
5. Click `Stop`.
6. Wait for finalization to complete.

## Choosing Audio Devices

### System output device

Pick the real playback endpoint:
- wired headphones/headset: choose the headphone / 2nd output device
- laptop speakers: choose the speakers device
- avoid generic aliases like `Microsoft Sound Mapper`

### Microphone device

- `Microphone Array` usually means the built-in laptop mic
- `Headset Microphone` usually means a wired headset mic
- `USB ... Microphone` usually means an external USB mic/headset
- avoid `Line In` and `Stereo Mix` for normal speech input

The frontend now appends short hints to device labels to make these choices clearer.

## Realtime vs Final Transcript

### Realtime transcript

- generated while the session is running
- optimized for responsiveness
- used as the live preview in the UI
- lower accuracy is acceptable

### Final transcript

- generated after stop
- based on finalized captured audio
- uses the final transcription profile
- is the primary saved transcript

This is why the saved Markdown output can be noticeably better than the live preview.

## Runtime Modes

Runtime mode is configured in [app/core/config.py](d:/Code/Git_Project/Realtime_Transcription/app/core/config.py).

### Production

Default mode for normal use.

- minimal logging
- raw WAV dump off
- processed finalized audio retained for final transcription
- intended for normal CPU-first local usage

### Troubleshooting

Use when diagnosing behavior without turning on everything.

- queue/timing logging on
- processed audio retained
- raw WAV dump still off by default

### Development

Use when actively debugging code changes.

- verbose logs
- chunk logs
- raw WAV dump on
- highest overhead of the three modes

## Finalized Default Configuration

Current practical defaults for normal Windows CPU-first use:

- runtime mode: `production`
- realtime model: `base`
- final model: `small`
- realtime beam size: `1`
- final beam size: `3`
- realtime VAD: `False`
- final VAD: `True`
- final language strategy: `en`

This keeps the live path responsive while giving the saved final transcript a better chance of being useful out of the box.

## Outputs

Outputs are saved under [outputs](d:/Code/Git_Project/Realtime_Transcription/outputs) by default.

Typical saved files:
- `session_<id>.md`
- `session_<id>.txt` if TXT export is enabled
- `session_<id>_final_transcript.txt`

Runtime audio artifacts are saved under [runtime/temp](d:/Code/Git_Project/Realtime_Transcription/runtime/temp) when needed by the current runtime mode and finalization path.

## Saved Markdown Behavior

The saved Markdown file is product-oriented:
- transcript body first
- concise metadata after the transcript
- final transcript is preferred whenever available

The main transcript body is no longer treated as a raw debug artifact.

## Workflow Overview

See:
- [docs/system_workflow.md](d:/Code/Git_Project/Realtime_Transcription/docs/system_workflow.md)
- [docs/file_function_map.md](d:/Code/Git_Project/Realtime_Transcription/docs/file_function_map.md)

## Known Limitations

- The app currently uses one active capture path per session:
  - microphone-only if microphone is enabled
  - system-loopback-only otherwise
- Realtime preview quality is still weaker than final offline transcription
- Final transcript latency increases with longer audio and larger final models
- The MVP is Windows-first; cross-platform behavior is not the focus

## Suggested Normal Usage

For normal daily use:
- keep runtime mode at `production`
- use system loopback for podcasts/videos/system audio
- use microphone mode only when you actually want spoken mic capture
- rely on the final transcript as the saved result
- treat realtime preview as a convenience, not as the authoritative transcript

## Development Notes

Main references for future maintenance:
- [docs/system_workflow.md](d:/Code/Git_Project/Realtime_Transcription/docs/system_workflow.md)
- [docs/file_function_map.md](d:/Code/Git_Project/Realtime_Transcription/docs/file_function_map.md)
