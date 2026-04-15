# Windows Loopback Notes

System audio capture on Windows is the riskiest Phase 1 item.

- The UI and backend already model system output device selection.
- The current scaffold lists output devices but does not claim end-to-end loopback capture works yet.
- The implementation should stay behind a dedicated `WindowsLoopbackCapture` abstraction.
- Different host APIs and drivers may expose output devices differently.
- Bluetooth-specific behavior is explicitly deferred to a later phase.
