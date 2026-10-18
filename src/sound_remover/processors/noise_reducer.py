"""
スペクトルゲーティングによるホワイトノイズ・空調音・定常雑音除去エンジン (`noise_reducer.py`)

特徴:
1. 演奏開始前や曲間の「サーッ」「ゴーッ」というノイズのみの区間（ノイズ見本）を波形から指定し、
   その周波数プロファイルを学習してピンポイントにホワイトノイズを引き算する機能。
2. ノイズ見本を指定しない場合でも、音声全体から定常ノイズフロアを自動推定して除去可能。
3. 吹奏楽の管楽器やシンバルの余韻が「水中にいるような音（ミュージカルノイズ）」にならないよう、
   周波数・時間方向のマスク平滑化 (`freq_mask_smooth_hz=500`, `time_mask_smooth_ms=50`) を最適化。
4. 特定の時間帯（リージョン）のみへのピンポイント適用にも対応。
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import noisereduce as nr
import numpy as np

from sound_remover.core.audio_data import TimeRegion
from sound_remover.processors.vocal_remover import apply_crossfade_splice


class SpectralNoiseReducer:
    """吹奏楽・音楽録音向けに最適化されたホワイトノイズ・環境ノイズ除去クラス"""

    @staticmethod
    def _reduce_segment(
        segment: np.ndarray,
        sample_rate: int,
        strength: float,
        stationary: bool,
        noise_clip: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        (channels, samples) の音声セグメントに対してスペクトルゲーティングを適用する。
        """
        prop_decrease = float(np.clip(strength, 0.0, 1.0))
        if prop_decrease <= 0.001 or segment.shape[1] < 512:
            return segment.copy()

        n_fft = min(2048, max(256, 2 ** int(np.floor(np.log2(segment.shape[1] // 2)))))

        cleaned = nr.reduce_noise(
            y=segment,
            sr=sample_rate,
            stationary=stationary,
            y_noise=noise_clip,
            prop_decrease=prop_decrease,
            n_fft=n_fft,
            freq_mask_smooth_hz=100,
            time_mask_smooth_ms=30,
            n_jobs=1,
        )
        return np.asarray(cleaned, dtype=np.float32)

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        strength: float = 0.75,
        stationary: bool = True,
        noise_profile_region: Optional[Tuple[float, float]] = None,
        only_regions: bool = False,
        regions: Optional[List[TimeRegion]] = None,
        crossfade_ms: float = 60.0,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> np.ndarray:
        """
        ホワイトノイズ・空調音などの環境ノイズを除去する。

        Args:
            audio: (channels, samples) float32 配列
            sample_rate: サンプリングレート (Hz)
            strength: ノイズ減衰率 (0.0 〜 1.0, 楽器保護のため 0.70〜0.85 推奨)
            stationary: 定常ノイズ除去モード (ホワイトノイズ・空調音はTrue推奨)
            noise_profile_region: ノイズ見本区間 (start_sec, end_sec)
            only_regions: Trueの場合、`regions` で指定された区間のみに適用
            regions: ピンポイント適用区間のリスト
            crossfade_ms: 区間境界クロスフェード時間 (ms)
            progress_callback: 進捗コールバック
        """
        if strength <= 0.001:
            return audio.copy()

        total_samples = audio.shape[1]
        duration_sec = total_samples / float(sample_rate)
        crossfade_samples = max(16, int((crossfade_ms / 1000.0) * sample_rate))

        # ノイズ見本区間が指定されている場合は切り出す
        noise_clip: Optional[np.ndarray] = None
        if noise_profile_region is not None:
            ns_sec, ne_sec = noise_profile_region
            ns_idx = max(0, min(int(min(ns_sec, ne_sec) * sample_rate), total_samples))
            ne_idx = max(ns_idx, min(int(max(ns_sec, ne_sec) * sample_rate), total_samples))
            if ne_idx - ns_idx >= int(0.1 * sample_rate):
                noise_clip = audio[:, ns_idx:ne_idx]

        active_regions = [
            r.normalized(duration_sec)
            for r in (regions or [])
            if r.apply_noise_reduction and r.duration > 0.05
        ]

        if only_regions:
            if not active_regions:
                return audio.copy()

            result = audio.copy()
            pad_samples = int(0.25 * sample_rate)

            for idx, reg in enumerate(active_regions):
                if progress_callback:
                    pct = int(60 + (idx / len(active_regions)) * 15)
                    progress_callback(
                        pct,
                        f"ホワイトノイズ除去（ピンポイント区間 {idx + 1}/{len(active_regions)}）を処理中...",
                    )

                s_idx = int(reg.start_sec * sample_rate)
                e_idx = int(reg.end_sec * sample_rate)
                ctx_s = max(0, s_idx - pad_samples)
                ctx_e = min(total_samples, e_idx + pad_samples)

                cleaned_ctx = self._reduce_segment(
                    audio[:, ctx_s:ctx_e],
                    sample_rate=sample_rate,
                    strength=strength,
                    stationary=stationary,
                    noise_clip=noise_clip,
                )
                inner_start = s_idx - ctx_s
                inner_end = inner_start + (e_idx - s_idx)
                cleaned_seg = cleaned_ctx[:, inner_start:inner_end]

                result = apply_crossfade_splice(
                    result,
                    cleaned_seg,
                    s_idx,
                    e_idx,
                    crossfade_samples,
                )

            return result

        if progress_callback:
            mode_desc = "ノイズ見本学習プロファイル" if noise_clip is not None else "自動定常ノイズ推定"
            progress_callback(62, f"ホワイトノイズ・環境ノイズ除去（{mode_desc}）を実行中...")

        return self._reduce_segment(
            audio,
            sample_rate=sample_rate,
            strength=strength,
            stationary=stationary,
            noise_clip=noise_clip,
        )
