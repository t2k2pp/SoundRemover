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

    app = QApplication(sys.argv)
    app.setApplicationName("SoundRemover")
    app.setOrganizationName("WindBandAudioTools")

    window = MainWindow()
    window.show()

    # コマンドライン引数で動画・音声ファイルが渡された場合は自動で開く
    if len(sys.argv) > 1:
        arg_path = Path(sys.argv[1])
        if arg_path.exists() and arg_path.is_file():
            window.load_media_file(arg_path)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
