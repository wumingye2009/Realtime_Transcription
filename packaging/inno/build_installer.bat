@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%\..\..") do set "PROJECT_ROOT=%%~fI"
set "ISS_FILE=%SCRIPT_DIR%realtime_transcription.iss"
set "DIST_DIR=%PROJECT_ROOT%\dist\realtime_transcription"

if not exist "%ISS_FILE%" (
    echo Inno Setup script not found:
    echo   %ISS_FILE%
    exit /b 1
)

if not exist "%DIST_DIR%" (
    echo Portable PyInstaller output not found:
    echo   %DIST_DIR%
    echo Build the portable package first.
    exit /b 1
)

set "ISCC_PATH="
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC_PATH=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not defined ISCC_PATH if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC_PATH=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not defined ISCC_PATH if defined ISCC_EXE set "ISCC_PATH=%ISCC_EXE%"

if not defined ISCC_PATH (
    echo ISCC.exe was not found.
    echo Install Inno Setup 6 or set ISCC_EXE to the full path of ISCC.exe.
    exit /b 1
)

pushd "%PROJECT_ROOT%"
echo Compiling installer from:
echo   %ISS_FILE%
echo Using ISCC:
echo   %ISCC_PATH%

"%ISCC_PATH%" "%ISS_FILE%"
if errorlevel 1 (
    echo Inno Setup build failed.
    popd
    exit /b 1
)

echo.
echo Installer build completed.
echo Output should be under:
echo   %PROJECT_ROOT%\packaging\release

popd
endlocal
