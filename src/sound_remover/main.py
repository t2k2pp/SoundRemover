"""
SoundRemover アプリケーションエントリーポイント (`main.py`)
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from sound_remover.core.ffmpeg_manager import FFmpegManager
from sound_remover.ui.main_window import MainWindow


def main() -> int:
    """デスクトップGUIを起動するメイン関数"""
    # FFmpegバイナリのパスを事前解決
    try:
        FFmpegManager.ensure_ffmpeg()
    except Exception as exc:
        print(f"[Warning] FFmpeg initialization warning: {exc}", file=sys.stderr)

    check_only = "--check" in sys.argv
    args = [a for a in sys.argv if a != "--check"]

    app = QApplication(args)
    app.setApplicationName("SoundRemover")
    app.setOrganizationName("WindBandAudioTools")

    window = MainWindow()
    if check_only:
        devs = [d.device_id for d in window.device_manager.available_devices]
        print(f"[SoundRemover] Self-check OK. Available devices: {devs}")
        window.close()
        return 0

    window.show()
    window.raise_()
    window.activateWindow()

    # コマンドライン引数で動画・音声ファイルが渡された場合は自動で開く
    if len(args) > 1:
        arg_path = Path(args[1])
        if arg_path.exists() and arg_path.is_file():
            window.load_media_file(arg_path)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
