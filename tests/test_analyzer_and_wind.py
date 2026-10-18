"""
スマート風切り音除去 (`suppress_wind_noise`) および音響・周波数特性自動診断 (`AudioAnalyzer`) のテスト
"""

from __future__ import annotations

import numpy as np

from sound_remover.processors.analyzer import AudioAnalyzer
from sound_remover.processors.eq_filter import WindBandFilter


def test_smart_wind_suppressor_preserves_bass_and_removes_wind() -> None:
    sr = 48000
    duration = 4.0
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False, dtype=np.float32)

    # 1. 楽器の低音 (110Hz チューバ・ベース基音): 左右同相 (Mid成分)
    bass_tone = 0.25 * np.sin(2.0 * np.pi * 110.0 * t)
    # 2. 管楽器メロディ (880Hz): ステレオ
    melody_l = 0.15 * np.sin(2.0 * np.pi * 880.0 * t)
    melody_r = 0.15 * np.sin(2.0 * np.pi * 880.0 * t + 0.3)

    left = bass_tone + melody_l
    right = bass_tone + melody_r

    # 2.0秒 〜 3.0秒 の1秒間だけ、マイクに直接当たる強い風切り音 (60Hz〜180Hz の左右逆相・非相関乱流) を混入
    w_s, w_e = int(2.0 * sr), int(3.0 * sr)
    t_w = t[w_s:w_e]
    wind_turbulence = 0.35 * np.sin(2.0 * np.pi * 85.0 * t_w) + 0.25 * np.sin(2.0 * np.pi * 140.0 * t_w)
    left[w_s:w_e] += wind_turbulence
    right[w_s:w_e] -= wind_turbulence  # 逆相乱流 (Side成分)

    stereo_noisy = np.vstack([left, right]).astype(np.float32)

    # スマート風切り音除去を実行
    cleaned = WindBandFilter.suppress_wind_noise(
        audio=stereo_noisy,
        sample_rate=sr,
        strength=0.90,
        wind_cutoff_hz=260.0,
    )

    # A. 風が吹いていない通常演奏区間 (0.5秒 〜 1.5秒) では、110Hzのチューバ低音が98%以上無傷で維持されていること
    n_s, n_e = int(0.5 * sr), int(1.5 * sr)
    rms_normal_before = float(np.sqrt(np.mean(np.square(stereo_noisy[:, n_s:n_e]))))
    rms_normal_after = float(np.sqrt(np.mean(np.square(cleaned[:, n_s:n_e]))))
    assert rms_normal_after > rms_normal_before * 0.98

    # B. 風が吹いた区間 (2.1秒 〜 2.9秒) では、Side成分（風の乱流ノイズ）が80%以上大幅に除去されていること
    b_s, b_e = int(2.1 * sr), int(2.9 * sr)
    side_before = 0.5 * (stereo_noisy[0, b_s:b_e] - stereo_noisy[1, b_s:b_e])
    side_after = 0.5 * (cleaned[0, b_s:b_e] - cleaned[1, b_s:b_e])
    rms_side_before = float(np.sqrt(np.mean(np.square(side_before))))
    rms_side_after = float(np.sqrt(np.mean(np.square(side_after))))
    assert rms_side_after < rms_side_before * 0.20


def test_audio_analyzer_detects_wind_and_recommends_settings() -> None:
    sr = 48000
    duration = 5.0
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False, dtype=np.float32)

    rng = np.random.default_rng(7)
    # 0.0〜1.2秒: 静かなホワイトノイズ区間
    noise = rng.normal(0.0, 0.008, size=t.shape).astype(np.float32)
    music = 0.20 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    music[: int(1.2 * sr)] = 0.0

    left = music + noise
    right = music + noise

    # 3.0〜4.5秒: 強い風切り音混入
    w_s, w_e = int(3.0 * sr), int(4.5 * sr)
    wind = 0.30 * np.sin(2.0 * np.pi * 90.0 * t[w_s:w_e]).astype(np.float32)
    left[w_s:w_e] += wind
    right[w_s:w_e] -= wind

    stereo = np.vstack([left, right]).astype(np.float32)
    report = AudioAnalyzer.analyze(stereo, sr)

    assert len(report.band_stats) == 5
    assert len(report.wind_events) >= 1
    assert report.recommended_config.wind_reduction_enabled is True
    assert report.suggested_noise_sample_region is not None
    assert len(report.recommendations_text) >= 2
