"""
ロスレス音声抽出・ホワイトノイズ除去・屋外風切り音フィルタ・EQ・マスタリング・曲分割の単体/結合テスト
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from sound_remover.core.audio_data import (
    AudioSession,
    ProcessingConfig,
    SongChapter,
    TimeRegion,
)
from sound_remover.core.ffmpeg_manager import FFmpegManager
from sound_remover.core.pipeline import AudioPipeline
from sound_remover.processors.eq_filter import WindBandFilter
from sound_remover.processors.extractor import LosslessExtractor
from sound_remover.processors.mastering import MasteringProcessor
from sound_remover.processors.noise_reducer import SpectralNoiseReducer
from sound_remover.processors.vocal_remover import apply_crossfade_splice


def _generate_synthetic_windband_audio(sr: int = 48000, duration_sec: float = 4.0) -> np.ndarray:
    """テスト用の吹奏楽模擬ステレオ音声 (48kHz, 2ch) を生成する"""
    t = np.linspace(0.0, duration_sec, int(sr * duration_sec), endpoint=False, dtype=np.float32)
    # Bb メジャーコード (Bb2=116.5Hz, D3=146.8Hz, F3=174.6Hz, Bb4=466.16Hz) + 倍音
    chord = (
        0.20 * np.sin(2.0 * np.pi * 116.54 * t)
        + 0.15 * np.sin(2.0 * np.pi * 233.08 * t)
        + 0.15 * np.sin(2.0 * np.pi * 293.66 * t)
        + 0.15 * np.sin(2.0 * np.pi * 349.23 * t)
        + 0.12 * np.sin(2.0 * np.pi * 466.16 * t)
    )
    # 冒頭 0.0〜0.8秒は演奏前の無音（ホワイトノイズのみ）
    envelope = np.ones_like(t)
    envelope[: int(0.8 * sr)] = 0.0

    rng = np.random.default_rng(42)
    white_noise = rng.normal(0.0, 0.015, size=t.shape).astype(np.float32)

    left = chord * envelope + white_noise
    right = np.roll(chord, 12) * envelope + white_noise
    return np.vstack([left, right]).astype(np.float32)


def test_lossless_extractor_video_and_audio_io(tmp_path: Path) -> None:
    sr = 48000
    orig_audio = _generate_synthetic_windband_audio(sr=sr, duration_sec=2.0)
    wav_src = tmp_path / "src_audio.wav"
    LosslessExtractor.save_audio(orig_audio, sr, wav_src, format_type="wav24")
    assert wav_src.exists()

    # テスト用のMP4動画ファイルを作成 (黒画面 + AAC音声)
    mp4_path = tmp_path / "concert_sample.mp4"
    ffmpeg_cmd = FFmpegManager.get_ffmpeg_cmd()
    subprocess.run(
        [
            ffmpeg_cmd,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:d=2.0",
            "-i",
            str(wav_src),
            "-c:v",
            "libx264",
            "-t",
            "2.0",
            "-c:a",
            "aac",
            "-b:a",
            "256k",
            str(mp4_path),
        ],
        check=True,
        capture_output=True,
    )
    assert mp4_path.exists()

    # 1. 完全ストリームコピー抽出 (再エンコードなし)
    extracted_copy, info = LosslessExtractor.extract_stream_copy(mp4_path)
    assert extracted_copy.exists()
    assert extracted_copy.stat().st_size > 1000
    assert info.codec_name == "aac"
    assert info.is_video_container is True

    # 2. ロスレスFLAC抽出
    flac_out = tmp_path / "extracted_lossless.flac"
    LosslessExtractor.extract_lossless_pcm(mp4_path, flac_out, codec="flac")
    assert flac_out.exists()

    # 3. 動画からの直接float32デコード読み込み
    loaded_audio, loaded_sr, _ = LosslessExtractor.load_audio(mp4_path)
    assert loaded_sr == sr
    assert loaded_audio.shape[0] == 2
    assert loaded_audio.shape[1] >= int(sr * 1.8)


def test_spectral_noise_reducer_with_noise_profile() -> None:
    sr = 48000
    audio = _generate_synthetic_windband_audio(sr=sr, duration_sec=3.0)
    reducer = SpectralNoiseReducer()

    # 0.0〜0.7秒のホワイトノイズ見本区間を使ってノイズ除去を実行
    cleaned = reducer.process(
        audio=audio,
        sample_rate=sr,
        strength=0.80,
        stationary=True,
        noise_profile_region=(0.05, 0.75),
        only_regions=False,
    )
    assert cleaned.shape == audio.shape

    # 冒頭のノイズ区間 (0.1〜0.6秒) のRMSエネルギーが確実に減少していることを確認
    s_idx, e_idx = int(0.1 * sr), int(0.6 * sr)
    rms_before = float(np.sqrt(np.mean(np.square(audio[:, s_idx:e_idx]))))
    rms_after = float(np.sqrt(np.mean(np.square(cleaned[:, s_idx:e_idx]))))
    assert rms_after < rms_before * 0.6

    # 演奏区間 (1.2〜2.5秒) の楽器音エネルギーはしっかり維持されていることを確認
    m_s, m_e = int(1.2 * sr), int(2.5 * sr)
    music_rms_before = float(np.sqrt(np.mean(np.square(audio[:, m_s:m_e]))))
    music_rms_after = float(np.sqrt(np.mean(np.square(cleaned[:, m_s:m_e]))))
    assert music_rms_after > music_rms_before * 0.85


def test_wind_band_highpass_and_eq() -> None:
    sr = 48000
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
    # 20Hz の屋外風切り音（超低周波ノイズ） + 440Hz の管楽器A4トーン
    wind_rumble = 0.5 * np.sin(2.0 * np.pi * 20.0 * t)
    concert_a4 = 0.3 * np.sin(2.0 * np.pi * 440.0 * t)
    sig = np.vstack([wind_rumble + concert_a4, wind_rumble + concert_a4]).astype(np.float32)

    filtered = WindBandFilter.process(
        audio=sig,
        sample_rate=sr,
        highpass_enabled=True,
        highpass_cutoff_hz=55.0,
        eq_enabled=True,
        low_db=1.0,
        low_mid_db=-2.0,
        high_mid_db=1.5,
        high_db=2.0,
    )
    assert filtered.shape == sig.shape
    # 風切り音 (20Hz) が大幅に除去されたため全体RMSが低下し、かつ440Hz成分は残っていること
    rms_filtered = float(np.sqrt(np.mean(np.square(filtered))))
    assert 0.15 < rms_filtered < 0.32


def test_mastering_loudness_limiter_and_chapter_split(tmp_path: Path) -> None:
    sr = 48000
    audio = _generate_synthetic_windband_audio(sr=sr, duration_sec=4.0) * 0.2
    # 突発的な強奏ピーク (1.5倍のクリッピング相当) を挿入
    audio[:, sr : sr + 200] = 1.4

    mastered = MasteringProcessor.process(
        audio=audio,
        sample_rate=sr,
        loudness_norm_enabled=True,
        target_lufs=-15.0,
        limiter_enabled=True,
        limiter_ceiling_db=-1.0,
    )
    ceiling_lin = 10.0 ** (-1.0 / 20.0)
    max_peak = float(np.max(np.abs(mastered)))
    # ピークリミッターにより確実に -1.0 dBFS 以下に収まっていること
    assert max_peak <= ceiling_lin + 1e-4

    # 曲分割＆S字コサインフェード書き出しの検証
    session = AudioSession(
        source_path=tmp_path / "concert.mp4",
        original_audio=audio,
        sample_rate=sr,
        processed_audio=mastered,
        song_chapters=[
            SongChapter(title="課題曲マーチ", start_sec=0.2, end_sec=1.8, fade_in_sec=0.3, fade_out_sec=0.4),
            SongChapter(title="自由曲", start_sec=2.0, end_sec=3.8, fade_in_sec=0.2, fade_out_sec=0.5),
        ],
    )
    exported_files = AudioPipeline.export_chapters(
        session=session,
        output_dir=tmp_path / "chapters_out",
        format_type="wav24",
        use_processed=True,
    )
    assert len(exported_files) == 2
    for f in exported_files:
        assert f.exists()


def test_crossfade_splice_continuity() -> None:
    sr = 48000
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
    orig = np.vstack([np.sin(2.0 * np.pi * 440.0 * t)] * 2).astype(np.float32)
    proc_seg = orig[:, int(0.5 * sr) : int(1.5 * sr)] * 0.5

    spliced = apply_crossfade_splice(
        original=orig,
        processed_segment=proc_seg,
        start_sample=int(0.5 * sr),
        end_sample=int(1.5 * sr),
        crossfade_samples=int(0.05 * sr),
    )
    assert spliced.shape == orig.shape
    # 非適用区間 (0.0〜0.4秒) は1ビットの狂いもなく元の音声と完全一致（ビットパーフェクト）していること
    np.testing.assert_array_equal(spliced[:, : int(0.4 * sr)], orig[:, : int(0.4 * sr)])
