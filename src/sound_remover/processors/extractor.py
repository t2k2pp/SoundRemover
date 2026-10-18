"""
動画・音声からのロスレス音声抽出・高品位PCM読み込み・書き出しエンジン (`extractor.py`)

主な機能:
1. 完全ストリームコピー抽出 (`ffmpeg -vn -acodec copy`): 再エンコード一切なしの100%無劣化抽出
2. ロスレスPCM (24-bit / 16-bit WAV) および 可逆圧縮 FLAC への直接抽出
3. 動画/音声ファイルから内部処理用の float32 (shape: [channels, samples]) 配列への高品位デコード
4. 加工後の float32 配列から WAV (24bit/16bit) / FLAC / AAC(M4A) / MP3 への書き出し
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

from sound_remover.core.ffmpeg_manager import FFmpegManager

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".mts", ".m2ts", ".webm", ".wmv", ".flv", ".mpg", ".mpeg"}
AUDIO_EXTENSIONS = {".wav", ".flac", ".m4a", ".aac", ".mp3", ".ogg", ".opus", ".ac3", ".wma", ".aiff"}

CODEC_TO_EXTENSION = {
    "aac": ".m4a",
    "alac": ".m4a",
    "mp3": ".mp3",
    "flac": ".flac",
    "vorbis": ".ogg",
    "opus": ".opus",
    "ac3": ".ac3",
    "eac3": ".eac3",
    "pcm_s16le": ".wav",
    "pcm_s24le": ".wav",
    "pcm_s32le": ".wav",
    "pcm_f32le": ".wav",
    "pcm_s16be": ".wav",
    "pcm_s24be": ".wav",
}


@dataclass
class AudioStreamInfo:
    """動画または音声ファイル内のオーディオストリーム情報"""
    codec_name: str
    sample_rate: int
    channels: int
    duration_sec: float
    recommended_ext: str
    is_video_container: bool


class LosslessExtractor:
    """動画からのロスレス音声抽出および高品位オーディオI/Oを担当するクラス"""

    @staticmethod
    def _subprocess_flags() -> int:
        """Windowsでのコンソールウィンドウ非表示フラグ"""
        import sys

        if sys.platform == "win32":
            return subprocess.CREATE_NO_WINDOW
        return 0

    @classmethod
    def probe_audio_stream(cls, file_path: Path | str) -> AudioStreamInfo:
        """
        入力ファイルの音声ストリーム情報（コーデック名、サンプルレート、チャンネル数等）を取得する。
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {path}")

        is_video = path.suffix.lower() in VIDEO_EXTENSIONS
        ffprobe_cmd = FFmpegManager.get_ffprobe_cmd()

        if ffprobe_cmd:
            cmd = [
                ffprobe_cmd,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name,sample_rate,channels,duration:format=duration",
                "-of",
                "json",
                str(path),
            ]
            try:
                res = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=cls._subprocess_flags(),
                    check=True,
                )
                data = json.loads(res.stdout)
                streams = data.get("streams", [])
                if streams:
                    st = streams[0]
                    codec_name = str(st.get("codec_name", "unknown")).lower()
                    sr = int(st.get("sample_rate", 48000))
                    ch = int(st.get("channels", 2))
                    dur_str = st.get("duration") or data.get("format", {}).get("duration") or "0"
                    try:
                        dur = float(dur_str)
                    except ValueError:
                        dur = 0.0
                    rec_ext = CODEC_TO_EXTENSION.get(codec_name, ".mka")
                    return AudioStreamInfo(
                        codec_name=codec_name,
                        sample_rate=sr,
                        channels=ch,
                        duration_sec=dur,
                        recommended_ext=rec_ext,
                        is_video_container=is_video,
                    )
            except Exception:
                pass

        # ffprobeが無い、または失敗した場合は ffmpeg -i の標準エラー出力から解析する
        ffmpeg_cmd = FFmpegManager.get_ffmpeg_cmd()
        res = subprocess.run(
            [ffmpeg_cmd, "-i", str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=cls._subprocess_flags(),
        )
        stderr_text = res.stderr or ""
        codec_name = "aac" if is_video else path.suffix.lstrip(".").lower()
        sr = 48000
        ch = 2
        import re

        m_audio = re.search(r"Audio:\s*([a-zA-Z0-9_]+).*?(\d+)\s*Hz,\s*(mono|stereo|\d+\s*channels)", stderr_text)
        if m_audio:
            codec_name = m_audio.group(1).lower()
            sr = int(m_audio.group(2))
            ch_str = m_audio.group(3)
            if ch_str == "mono":
                ch = 1
            elif ch_str == "stereo":
                ch = 2

        rec_ext = CODEC_TO_EXTENSION.get(codec_name, ".m4a" if is_video else ".wav")
        return AudioStreamInfo(
            codec_name=codec_name,
            sample_rate=sr,
            channels=ch,
            duration_sec=0.0,
            recommended_ext=rec_ext,
            is_video_container=is_video,
        )

    @classmethod
    def extract_stream_copy(
        cls,
        input_path: Path | str,
        output_path: Optional[Path | str] = None,
    ) -> Tuple[Path, AudioStreamInfo]:
        """
        動画（または音声）ファイルから、再エンコードを一切行わずに元の音声ストリームをそのまま抽出する
        (`ffmpeg -y -i <input> -vn -acodec copy <output>`)。

        Args:
            input_path: 入力動画/音声ファイルのパス
            output_path: 出力先パス（省略時は入力ファイル名 + `_lossless_raw` + 推奨拡張子）

        Returns:
            (出力されたファイルパス, AudioStreamInfo)
        """
        in_path = Path(input_path)
        info = cls.probe_audio_stream(in_path)
        ffmpeg_cmd = FFmpegManager.get_ffmpeg_cmd()

        if output_path is None:
            out_path = in_path.with_name(f"{in_path.stem}_lossless{info.recommended_ext}")
        else:
            out_path = Path(output_path)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            ffmpeg_cmd,
            "-y",
            "-i",
            str(in_path),
            "-vn",
            "-acodec",
            "copy",
            str(out_path),
        ]
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=cls._subprocess_flags(),
        )
        if res.returncode != 0:
            # 万が一指定コンテナにストリームコピーできない場合は、24bit WAVロスレスで出力する安全策
            fallback_path = out_path.with_suffix(".wav")
            cmd_wav = [
                ffmpeg_cmd,
                "-y",
                "-i",
                str(in_path),
                "-vn",
                "-acodec",
                "pcm_s24le",
                str(fallback_path),
            ]
            res2 = subprocess.run(
                cmd_wav,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=cls._subprocess_flags(),
            )
            if res2.returncode != 0:
                raise RuntimeError(f"音声抽出に失敗しました:\n{res.stderr}\n{res2.stderr}")
            return fallback_path, info

        return out_path, info

    @classmethod
    def extract_lossless_pcm(
        cls,
        input_path: Path | str,
        output_path: Path | str,
        codec: str = "pcm_s24le",
    ) -> Path:
        """
        動画から非圧縮PCM (24-bit WAV `pcm_s24le` / 16-bit WAV `pcm_s16le`) または可逆圧縮 `flac` として抽出する。
        元のサンプリングレートとチャンネル数を完全維持する。
        """
        in_path = Path(input_path)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_cmd = FFmpegManager.get_ffmpeg_cmd()

        cmd = [
            ffmpeg_cmd,
            "-y",
            "-i",
            str(in_path),
            "-vn",
            "-acodec",
            codec,
            str(out_path),
        ]
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=cls._subprocess_flags(),
        )
        if res.returncode != 0:
            raise RuntimeError(f"ロスレスPCM/FLAC抽出に失敗しました:\n{res.stderr}")
        return out_path

    @classmethod
    def load_audio(cls, file_path: Path | str) -> Tuple[np.ndarray, int, AudioStreamInfo]:
        """
        任意の動画・音声ファイルを読み込み、高品位 float32 配列 (shape: [channels, samples]) と
        サンプリングレート、ストリーム情報を返す。
        """
        path = Path(file_path)
        info = cls.probe_audio_stream(path)

        # WAV や FLAC の場合は soundfile で直接高速・無劣化読み込みを試みる
        if path.suffix.lower() in {".wav", ".flac", ".ogg"}:
            try:
                data, sr = sf.read(str(path), dtype="float32", always_2d=True)
                # soundfile は (samples, channels) を返すため (channels, samples) に転置
                return np.ascontiguousarray(data.T, dtype=np.float32), int(sr), info
            except Exception:
                pass

        # 動画ファイル (.mp4, .mov, .mts 等) や AAC/M4A 等は FFmpeg で 32-bit float WAV に一時デコードして読み込む
        ffmpeg_cmd = FFmpegManager.get_ffmpeg_cmd()
        with tempfile.TemporaryDirectory(prefix="sound_remover_") as tmp_dir:
            tmp_wav = Path(tmp_dir) / "decoded_f32.wav"
            cmd = [
                ffmpeg_cmd,
                "-y",
                "-i",
                str(path),
                "-vn",
                "-acodec",
                "pcm_f32le",
                str(tmp_wav),
            ]
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=cls._subprocess_flags(),
            )
            if res.returncode != 0 or not tmp_wav.exists():
                raise RuntimeError(f"音声データのデコードに失敗しました:\n{res.stderr}")

            data, sr = sf.read(str(tmp_wav), dtype="float32", always_2d=True)
            return np.ascontiguousarray(data.T, dtype=np.float32), int(sr), info

    @classmethod
    def save_audio(
        cls,
        audio: np.ndarray,
        sample_rate: int,
        output_path: Path | str,
        format_type: str = "wav24",
    ) -> Path:
        """
        内部音声配列 (shape: [channels, samples], float32) を指定フォーマットで保存する。

        Args:
            audio: (channels, samples) の float32 配列
            sample_rate: サンプリングレート (Hz)
            output_path: 保存先ファイルパス
            format_type:
                - 'wav24': 非圧縮 24-bit PCM WAV (吹奏楽・高音質マスター推奨)
                - 'wav16': 非圧縮 16-bit PCM WAV (CD標準)
                - 'flac': ロスレス可逆圧縮 FLAC (24-bit精度維持・ファイルサイズ約半分)
                - 'm4a': 高音質 AAC 320kbps (.m4a, スマホ配布・LINE共有等に便利)
                - 'mp3': 高音質 MP3 320kbps (.mp3)
        """
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if audio.ndim == 1:
            audio_2d = audio[np.newaxis, :]
        else:
            audio_2d = audio

        # クリッピング防止の最終安全ガード (-1.0 〜 +1.0)
        safe_audio = np.clip(audio_2d, -1.0, 1.0).astype(np.float32)
        # soundfile 用に (samples, channels) へ転置
        sf_data = np.ascontiguousarray(safe_audio.T)

        ext = out_path.suffix.lower()
        fmt = format_type.lower()

        if fmt == "wav24" or (ext == ".wav" and fmt not in {"wav16"}):
            sf.write(str(out_path), sf_data, sample_rate, subtype="PCM_24", format="WAV")
            return out_path
        if fmt == "wav16":
            sf.write(str(out_path), sf_data, sample_rate, subtype="PCM_16", format="WAV")
            return out_path
        if fmt == "flac" or ext == ".flac":
            sf.write(str(out_path), sf_data, sample_rate, subtype="PCM_24", format="FLAC")
            return out_path

        # AAC (.m4a) または MP3 (.mp3) の場合は一時24-bit WAV経由でFFmpegエンコード
        ffmpeg_cmd = FFmpegManager.get_ffmpeg_cmd()
        with tempfile.TemporaryDirectory(prefix="sound_remover_export_") as tmp_dir:
            tmp_wav = Path(tmp_dir) / "export_src.wav"
            sf.write(str(tmp_wav), sf_data, sample_rate, subtype="PCM_24", format="WAV")

            if fmt == "m4a" or ext in {".m4a", ".aac"}:
                codec_args = ["-c:a", "aac", "-b:a", "320k"]
            elif fmt == "mp3" or ext == ".mp3":
                codec_args = ["-c:a", "libmp3lame", "-b:a", "320k"]
            else:
                codec_args = ["-c:a", "pcm_s24le"]

            cmd = [ffmpeg_cmd, "-y", "-i", str(tmp_wav), *codec_args, str(out_path)]
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=cls._subprocess_flags(),
            )
            if res.returncode != 0:
                raise RuntimeError(f"音声ファイルの書き出しに失敗しました:\n{res.stderr}")

        return out_path
