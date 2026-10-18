"""
音声データ・ピンポイント適用リージョン・曲分割チャプター・処理設定のデータモデル
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class TimeRegion:
    """ピンポイントで声除去やノイズ除去を適用する時間範囲（秒）"""
    start_sec: float
    end_sec: float
    label: str = "ピンポイント除去区間"
    apply_vocal_removal: bool = True
    apply_noise_reduction: bool = True

    def normalized(self, max_duration: float) -> "TimeRegion":
        """開始・終了時刻を正規化し、有効範囲内に収めたコピーを返す"""
        s = max(0.0, min(self.start_sec, self.end_sec))
        e = min(max_duration, max(self.start_sec, self.end_sec))
        return TimeRegion(
            start_sec=s,
            end_sec=e,
            label=self.label,
            apply_vocal_removal=self.apply_vocal_removal,
            apply_noise_reduction=self.apply_noise_reduction,
        )

    @property
    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass
class SongChapter:
    """1本の長い動画から曲ごとに分割出力するためのチャプター情報"""
    title: str
    start_sec: float
    end_sec: float
    fade_in_sec: float = 0.5
    fade_out_sec: float = 2.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass
class ProcessingConfig:
    """音声加工パイプラインの全設定パラメータ"""
    # 1. AIピンポイント声（話し声・アナウンス・雑音）除去設定
    vocal_removal_enabled: bool = True
    vocal_apply_only_regions: bool = False  # Trueの場合、登録されたTimeRegionのみにAI声除去を適用
    vocal_reduction_ratio: float = 0.85     # 0.0 (除去なし) 〜 1.0 (100%完全除去)
    vocal_band_protection: bool = True      # Trueの場合、声の帯域(110Hz〜6500Hz)のみ引き算し重低音・超高音は100%保護
    demucs_model_name: str = "htdemucs"     # 'htdemucs' または 'htdemucs_ft'

    # 2. ホワイトノイズ・定常環境ノイズ除去設定
    noise_reduction_enabled: bool = True
    noise_apply_only_regions: bool = False  # Trueの場合、登録されたTimeRegionのみにノイズ除去を適用
    noise_reduction_strength: float = 0.75  # prop_decrease (0.0 〜 1.0): 楽器の余韻を守るため0.7〜0.85推奨
    noise_stationary_mode: bool = True      # 定常ノイズ除去モード
    noise_profile_region: Optional[Tuple[float, float]] = None  # ノイズ見本区間 (start_sec, end_sec)

    # 3. 屋外風切り音ローカット＆吹奏楽向け4バンドEQ設定
    highpass_enabled: bool = True
    highpass_cutoff_hz: float = 45.0        # 風切り音・空調振動カット (30Hz〜150Hz)
    eq_enabled: bool = False
    eq_low_db: float = 0.0                  # 80Hz Low-Shelf (チューバ・低音の厚み)
    eq_low_mid_db: float = 0.0              # 280Hz Peaking (体育館・ホールのこもり軽減)
    eq_high_mid_db: float = 0.0             # 2.5kHz Peaking (木管・金管の明瞭度)
    eq_high_db: float = 0.0                 # 8.0kHz High-Shelf (倍音・きらびやかさ・空気感)

    # 4. 吹奏楽向けマスタリング（ラウドネスノーマライズ＆音割れ防止リミッター）設定
    loudness_norm_enabled: bool = True
    target_lufs: float = -15.0              # 吹奏楽のダイナミクスを活かしつつ聴きやすい音量 (-20.0 〜 -11.0 LUFS)
    limiter_enabled: bool = True
    limiter_ceiling_db: float = -1.0        # 最大ピーク上限 (-3.0 〜 -0.1 dBFS)

    # 5. ピンポイント境界クロスフェード時間
    region_crossfade_ms: float = 60.0       # 区間境界のクリックノイズ防止クロスフェード (ms)

    @classmethod
    def preset_outdoor_stage(cls) -> "ProcessingConfig":
        """プリセット1: 屋外ステージ録音（風切り音カット＋話し声除去＋低音補強＋音量最適化）"""
        return cls(
            vocal_removal_enabled=True,
            vocal_apply_only_regions=False,
            vocal_reduction_ratio=0.85,
            vocal_band_protection=True,
            noise_reduction_enabled=True,
            noise_reduction_strength=0.70,
            highpass_enabled=True,
            highpass_cutoff_hz=55.0,
            eq_enabled=True,
            eq_low_db=2.0,
            eq_low_mid_db=-1.0,
            eq_high_mid_db=1.0,
            eq_high_db=1.5,
            loudness_norm_enabled=True,
            target_lufs=-15.0,
            limiter_enabled=True,
            limiter_ceiling_db=-1.0,
        )

    @classmethod
    def preset_indoor_hall(cls) -> "ProcessingConfig":
        """プリセット2: 屋内ホール・体育館録音（空調ホワイトノイズ除去＋こもり解消EQ＋音量最適化）"""
        return cls(
            vocal_removal_enabled=False,
            vocal_apply_only_regions=True,
            vocal_reduction_ratio=0.85,
            vocal_band_protection=True,
            noise_reduction_enabled=True,
            noise_reduction_strength=0.80,
            highpass_enabled=True,
            highpass_cutoff_hz=40.0,
            eq_enabled=True,
            eq_low_db=0.5,
            eq_low_mid_db=-2.5,
            eq_high_mid_db=0.5,
            eq_high_db=1.5,
            loudness_norm_enabled=True,
            target_lufs=-16.0,
            limiter_enabled=True,
            limiter_ceiling_db=-1.0,
        )

    @classmethod
    def preset_pinpoint_voice_only(cls) -> "ProcessingConfig":
        """プリセット3: 話し声・雑音のピンポイント除去のみ（他は100%無加工維持）"""
        return cls(
            vocal_removal_enabled=True,
            vocal_apply_only_regions=True,
            vocal_reduction_ratio=0.90,
            vocal_band_protection=True,
            noise_reduction_enabled=False,
            highpass_enabled=False,
            eq_enabled=False,
            loudness_norm_enabled=False,
            limiter_enabled=False,
        )

    @classmethod
    def preset_white_noise_only(cls) -> "ProcessingConfig":
        """プリセット4: ホワイトノイズ・空調音除去のみ"""
        return cls(
            vocal_removal_enabled=False,
            noise_reduction_enabled=True,
            noise_apply_only_regions=False,
            noise_reduction_strength=0.80,
            highpass_enabled=True,
            highpass_cutoff_hz=40.0,
            eq_enabled=False,
            loudness_norm_enabled=False,
            limiter_enabled=False,
        )


@dataclass
class AudioSession:
    """現在読み込まれている音声・動画セッションの状態"""
    source_path: Path
    original_audio: np.ndarray          # shape: (channels, samples), dtype: float32
    sample_rate: int
    source_codec_name: str = "unknown"
    is_video_source: bool = False
    processed_audio: Optional[np.ndarray] = None
    pinpoint_regions: List[TimeRegion] = field(default_factory=list)
    noise_sample_region: Optional[Tuple[float, float]] = None
    song_chapters: List[SongChapter] = field(default_factory=list)
    config: ProcessingConfig = field(default_factory=ProcessingConfig.preset_outdoor_stage)

    @property
    def num_channels(self) -> int:
        return int(self.original_audio.shape[0])

    @property
    def num_samples(self) -> int:
        return int(self.original_audio.shape[1])

    @property
    def duration_sec(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return float(self.num_samples) / float(self.sample_rate)

    def get_active_audio(self, use_processed: bool = True) -> np.ndarray:
        """再生・書き出し対象の音声配列を返す"""
        if use_processed and self.processed_audio is not None:
            return self.processed_audio
        return self.original_audio
