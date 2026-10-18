"""
吹奏楽・屋外ステージ模擬サンプル動画生成スクリプト (`scripts/generate_sample_video.py`)

実物の動画ファイルが手元にない状態でもアプリのすべての機能（ロスレス抽出、ホワイトノイズ見本学習、
ピンポイント除去、屋外風切り音カット、ラウドネスノーマライズ、曲分割）をすぐにテストできるよう、
以下の構成を持つ10秒間のサンプル動画 (`sample_windband_stage.mp4`) を生成します:
- 0.0秒 〜 1.5秒: 演奏開始前の屋外環境ノイズ（サーッというホワイトノイズ ＋ 25Hzの風切り音）
- 1.5秒 〜 5.0秒: 1曲目（Bbメジャーの管楽器アンサンブル和音）※3.0〜4.2秒に擬似的な雑音バースト混入
- 5.0秒 〜 6.0秒: 曲間の静寂（環境ノイズのみ）
- 6.0秒 〜 10.0秒: 2曲目（Fメジャー〜Bbメジャーのファンファーレ和音）
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from sound_remover.core.ffmpeg_manager import FFmpegManager
from sound_remover.processors.extractor import LosslessExtractor


def generate_sample_video(output_mp4: Path | str = "sample_windband_stage.mp4") -> Path:
    sr = 48000
    duration_sec = 10.0
    n_samples = int(sr * duration_sec)
    t = np.linspace(0.0, duration_sec, n_samples, endpoint=False, dtype=np.float32)

    rng = np.random.default_rng(1234)
    # 1. 定常ホワイトノイズ + 屋外の風切り音 (25Hz)
    white_noise = rng.normal(0.0, 0.012, size=n_samples).astype(np.float32)
    wind_rumble = 0.08 * np.sin(2.0 * np.pi * 25.0 * t).astype(np.float32)

    # 2. 管楽器アンサンブル模擬和音（基音＋倍音）
    def brass_tone(freq: float, amp: float) -> np.ndarray:
        return (
            amp * np.sin(2.0 * np.pi * freq * t)
            + (amp * 0.5) * np.sin(2.0 * np.pi * (freq * 2.0) * t)
            + (amp * 0.25) * np.sin(2.0 * np.pi * (freq * 3.0) * t)
            + (amp * 0.12) * np.sin(2.0 * np.pi * (freq * 4.0) * t)
        ).astype(np.float32)

    song1 = (
        brass_tone(116.54, 0.14)  # Bb2 (Tuba / Euphonium)
        + brass_tone(233.08, 0.12)  # Bb3 (Trombone / Horn)
        + brass_tone(293.66, 0.11)  # D4 (Clarinet)
        + brass_tone(349.23, 0.11)  # F4 (Trumpet)
        + brass_tone(466.16, 0.10)  # Bb4 (Flute / Trumpet)
    )
    song2 = (
        brass_tone(87.31, 0.15)   # F2
        + brass_tone(174.61, 0.13)  # F3
        + brass_tone(220.00, 0.11)  # A3
        + brass_tone(261.63, 0.11)  # C4
        + brass_tone(349.23, 0.12)  # F4
    )

    env1 = np.zeros_like(t)
    env1[int(1.5 * sr) : int(5.0 * sr)] = 1.0
    env2 = np.zeros_like(t)
    env2[int(6.0 * sr) : int(9.8 * sr)] = 1.0

    music_l = song1 * env1 + song2 * env2
    music_r = np.roll(music_l, 18)

    left = music_l + white_noise + wind_rumble
    right = music_r + white_noise + wind_rumble
    stereo = np.vstack([left, right]).astype(np.float32)

    out_path = Path(output_mp4).resolve()
    tmp_wav = out_path.with_suffix(".tmp.wav")
    LosslessExtractor.save_audio(stereo, sr, tmp_wav, format_type="wav24")

    ffmpeg_cmd = FFmpegManager.get_ffmpeg_cmd()
    subprocess.run(
        [
            ffmpeg_cmd,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=#1e293b:s=640x360:d=10.0",
            "-i",
            str(tmp_wav),
            "-c:v",
            "libx264",
            "-t",
            "10.0",
            "-c:a",
            "aac",
            "-b:a",
            "256k",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )
    if tmp_wav.exists():
        tmp_wav.unlink()

    return out_path


if __name__ == "__main__":
    p = generate_sample_video()
    print(f"Generated sample video: {p}")
