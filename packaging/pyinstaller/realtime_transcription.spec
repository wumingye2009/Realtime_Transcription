# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


PROJECT_ROOT = Path.cwd()
APP_DIR = PROJECT_ROOT / 'app'
DOCS_DIR = PROJECT_ROOT / 'docs'
LAUNCHER_PATH = PROJECT_ROOT / 'launcher.py'

datas = [
    (str(APP_DIR / 'static'), 'app/static'),
    (str(APP_DIR / 'templates'), 'app/templates'),
    (str(PROJECT_ROOT / 'VERSION'), '.'),
    (str(PROJECT_ROOT / 'README.md'), '.'),
]

if DOCS_DIR.exists():
    datas.append((str(DOCS_DIR), 'docs'))

hiddenimports = [
    'app.main',
    'jinja2',
    'uvicorn.logging',
    'uvicorn.loops.auto',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets.auto',
    'sounddevice',
    'soundcard',
    'pyaudiowpatch',
    'faster_whisper',
]

a = Analysis(
    [str(LAUNCHER_PATH)],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='realtime_transcription',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='realtime_transcription',
)
