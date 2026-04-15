@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%\..\..") do set "PROJECT_ROOT=%%~fI"
set "SPEC_FILE=%SCRIPT_DIR%realtime_transcription.spec"

if not exist "%SPEC_FILE%" (
    echo Spec file not found:
    echo   %SPEC_FILE%
    exit /b 1
)

pushd "%PROJECT_ROOT%"

if defined CONDA_DEFAULT_ENV (
    echo Using active conda environment: %CONDA_DEFAULT_ENV%
) else if defined CONDA_EXE (
    echo Activating conda environment: transcription
    call conda activate transcription
    if errorlevel 1 (
        echo Failed to activate conda environment "transcription".
        popd
        exit /b 1
    )
) else (
    echo Conda not detected in this shell. Using the current Python environment.
)

echo Building portable package from:
echo   %SPEC_FILE%

pyinstaller "%SPEC_FILE%" --noconfirm
if errorlevel 1 (
    echo PyInstaller build failed.
    popd
    exit /b 1
)

echo.
echo Portable build completed.
echo Output should be under:
echo   %PROJECT_ROOT%\dist\realtime_transcription

popd
endlocal
