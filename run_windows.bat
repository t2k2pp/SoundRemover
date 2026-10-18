@echo off
setlocal
title SoundRemover Studio

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    echo [SoundRemover] Starting application from .venv ...
    ".venv\Scripts\python.exe" -m sound_remover.main %*
    exit /b %errorlevel%
)

where uv >nul 2>nul
if %errorlevel% equ 0 (
    echo [SoundRemover] Starting application via uv ...
    uv run sound-remover %*
    exit /b %errorlevel%
)

if exist "%USERPROFILE%\.local\bin\uv.exe" (
    echo [SoundRemover] Starting application via user uv ...
    "%USERPROFILE%\.local\bin\uv.exe" run sound-remover %*
    exit /b %errorlevel%
)

echo [Error] Python virtual environment (.venv) or uv was not found.
pause
exit /b 1
