"""
FFmpeg / FFprobe バイナリ自動検出・解決モジュール

他の保護者へ配布した際など、OSにFFmpegがインストールされていない環境でも、
Pythonパッケージ (`static-ffmpeg` または `imageio-ffmpeg`) 内蔵のバイナリを
自動的に見つけ出し、PATH環境変数への登録およびコマンドパス解決を行う。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional, Tuple


class FFmpegManager:
    """FFmpegおよびFFprobeの実行ファイルパスを管理・解決するクラス"""

    _initialized: bool = False
    _ffmpeg_path: Optional[str] = None
    _ffprobe_path: Optional[str] = None

    @classmethod
    def ensure_ffmpeg(cls) -> Tuple[str, Optional[str]]:
        """
        FFmpeg と FFprobe のパスを解決し、必要に応じて環境変数 PATH に追加する。

        Returns:
            (ffmpeg_path, ffprobe_path) のタプル
        """
        if cls._initialized and cls._ffmpeg_path and Path(cls._ffmpeg_path).exists():
            return cls._ffmpeg_path, cls._ffprobe_path

        # 1. まずOS標準のPATHから探す
        sys_ffmpeg = shutil.which("ffmpeg")
        sys_ffprobe = shutil.which("ffprobe")
        if sys_ffmpeg and sys_ffprobe:
            cls._ffmpeg_path = sys_ffmpeg
            cls._ffprobe_path = sys_ffprobe
            cls._initialized = True
            return cls._ffmpeg_path, cls._ffprobe_path

        # 2. static_ffmpeg パッケージ（ffmpeg と ffprobe の両方を提供）を試す
        try:
            import static_ffmpeg  # type: ignore

            static_ffmpeg.add_paths()
            sf_ffmpeg = shutil.which("ffmpeg")
            sf_ffprobe = shutil.which("ffprobe")
            if sf_ffmpeg:
                cls._ffmpeg_path = sf_ffmpeg
                cls._ffprobe_path = sf_ffprobe
                cls._initialized = True
                return cls._ffmpeg_path, cls._ffprobe_path
        except Exception:
            pass

        # 3. imageio_ffmpeg パッケージ（単体ffmpegバイナリを同梱）を試す
        try:
            import imageio_ffmpeg  # type: ignore

            iio_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            if iio_ffmpeg and Path(iio_ffmpeg).exists():
                ffmpeg_dir = str(Path(iio_ffmpeg).parent)
                if ffmpeg_dir not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
                cls._ffmpeg_path = iio_ffmpeg
                cls._ffprobe_path = shutil.which("ffprobe")
                cls._initialized = True
                return cls._ffmpeg_path, cls._ffprobe_path
        except Exception:
            pass

        if sys_ffmpeg:
            cls._ffmpeg_path = sys_ffmpeg
            cls._ffprobe_path = sys_ffprobe
            cls._initialized = True
            return cls._ffmpeg_path, cls._ffprobe_path

        raise RuntimeError(
            "FFmpegが見つかりませんでした。`static-ffmpeg` または `imageio-ffmpeg` をインストールするか、"
            "システムのPATHにFFmpegを追加してください。"
        )

    @classmethod
    def get_ffmpeg_cmd(cls) -> str:
        """FFmpeg実行ファイルのパスを返す"""
        ffmpeg_path, _ = cls.ensure_ffmpeg()
        return ffmpeg_path

    @classmethod
    def get_ffprobe_cmd(cls) -> Optional[str]:
        """FFprobe実行ファイルのパスを返す（存在しない場合はNone）"""
        _, ffprobe_path = cls.ensure_ffmpeg()
        return ffprobe_path
