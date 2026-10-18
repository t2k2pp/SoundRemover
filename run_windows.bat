@echo off
chcp 65001 > nul
title SoundRemover - 吹奏楽・発表会動画 音声抽出＆ノイズ除去スタジオ

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    echo [SoundRemover] 仮想環境からアプリを起動しています...
    ".venv\Scripts\python.exe" -m sound_remover.main %*
    goto :eof
)

where uv >nul 2>nul
if %errorlevel% equ 0 (
    echo [SoundRemover] uv で環境を準備して起動しています...
    uv run sound-remover %*
    goto :eof
)

if exist "%USERPROFILE%\.local\bin\uv.exe" (
    echo [SoundRemover] uv で環境を準備して起動しています...
    "%USERPROFILE%\.local\bin\uv.exe" run sound-remover %*
    goto :eof
)

echo [エラー] Python仮想環境 (.venv) または uv が見つかりませんでした。
pause
