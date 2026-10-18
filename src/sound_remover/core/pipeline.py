"""
非破壊オーディオ処理パイプライン統合エンジン (`pipeline.py`)

セッション (`AudioSession`) と設定 (`ProcessingConfig`) に基づき、以下の処理を順次実行する:
1. AIピンポイント声（話し声・アナウンス・咳払い）除去 (`DemucsVocalRemover`)
2. ホワイトノイズ・空調音除去 (`SpectralNoiseReducer`)
3. 屋外風切り音ローカット＆吹奏楽4バンドEQ (`WindBandFilter`)
4. ラウドネスノーマライズ＆音割れ防止ピークリミッター (`MasteringProcessor`)
5. 曲分割チャプターの一括書き出し機能 (`export_all_chapters`)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from sound_remover.core.audio_data import AudioSession, ProcessingConfig
from sound_remover.core.device_manager import ComputeDevice, DeviceManager
from sound_remover.processors.eq_filter import WindBandFilter
from sound_remover.processors.extractor import LosslessExtractor
from sound_remover.processors.mastering import MasteringProcessor
from sound_remover.processors.noise_reducer import SpectralNoiseReducer
from sound_remover.processors.vocal_remover import DemucsVocalRemover


class AudioPipeline:
    """音声加工パイプライン統括クラス"""

    def __init__(self, device_manager: Optional[DeviceManager] = None) -> None:
        self.device_manager = device_manager or DeviceManager()
        self.vocal_remover = DemucsVocalRemover(model_name="htdemucs")
        self.noise_reducer = SpectralNoiseReducer()

    def run_pipeline(
        self,
        session: AudioSession,
        config: Optional[ProcessingConfig] = None,
        device: Optional[ComputeDevice] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> np.ndarray:
        """
        `session.original_audio` に対して非破壊で一連の音声加工を実行し、
        `session.processed_audio` に格納して返す。
        """
        cfg = config or session.config
        active_device = device or self.device_manager.current_device
        if not active_device.is_gpu:
            self.device_manager.optimize_cpu_threads()

        audio = session.original_audio.copy()
        sr = session.sample_rate

        if progress_callback:
            progress_callback(5, "音声処理パイプラインを開始しています...")

        # Step 1: AIピンポイント声除去
        if cfg.vocal_removal_enabled and cfg.vocal_reduction_ratio > 0.001:
            self.vocal_remover.model_name = cfg.demucs_model_name
            audio = self.vocal_remover.process(
                audio=audio,
                sample_rate=sr,
                device=active_device,
                reduction_ratio=cfg.vocal_reduction_ratio,
                band_protection=cfg.vocal_band_protection,
                only_regions=cfg.vocal_apply_only_regions,
                regions=session.pinpoint_regions,
                crossfade_ms=cfg.region_crossfade_ms,
                progress_callback=progress_callback,
            )

        # Step 2: ホワイトノイズ・空調音除去
        if cfg.noise_reduction_enabled and cfg.noise_reduction_strength > 0.001:
            noise_prof = cfg.noise_profile_region or session.noise_sample_region
            audio = self.noise_reducer.process(
                audio=audio,
                sample_rate=sr,
                strength=cfg.noise_reduction_strength,
                stationary=cfg.noise_stationary_mode,
                noise_profile_region=noise_prof,
                only_regions=cfg.noise_apply_only_regions,
                regions=session.pinpoint_regions,
                crossfade_ms=cfg.region_crossfade_ms,
                progress_callback=progress_callback,
            )

        # Step 3: 屋外風切り音ローカット・スマート風切り音除去＆吹奏楽4バンドEQ
        if cfg.highpass_enabled or cfg.wind_reduction_enabled or cfg.eq_enabled:
            if progress_callback:
                progress_callback(78, "スマート風切り音除去・ローカット＆吹奏楽イコライザーを適用中...")
            audio = WindBandFilter.process(
                audio=audio,
                sample_rate=sr,
                highpass_enabled=cfg.highpass_enabled,
                highpass_cutoff_hz=cfg.highpass_cutoff_hz,
                wind_reduction_enabled=cfg.wind_reduction_enabled,
                wind_reduction_strength=cfg.wind_reduction_strength,
                wind_cutoff_hz=cfg.wind_cutoff_hz,
                eq_enabled=cfg.eq_enabled,
                low_db=cfg.eq_low_db,
                low_mid_db=cfg.eq_low_mid_db,
                high_mid_db=cfg.eq_high_mid_db,
                high_db=cfg.eq_high_db,
            )

        # Step 4: 吹奏楽向けラウドネスノーマライズ＆音割れ防止ピークリミッター
        if cfg.loudness_norm_enabled or cfg.limiter_enabled:
            if progress_callback:
                progress_callback(88, "吹奏楽向けラウドネスノーマライズ＆音割れ防止リミッターを適用中...")
            audio = MasteringProcessor.process(
                audio=audio,
                sample_rate=sr,
                loudness_norm_enabled=cfg.loudness_norm_enabled,
                target_lufs=cfg.target_lufs,
                limiter_enabled=cfg.limiter_enabled,
                limiter_ceiling_db=cfg.limiter_ceiling_db,
            )

        session.processed_audio = audio.astype(np.float32)
        if progress_callback:
            progress_callback(100, "音声処理が完了しました！「加工後 (Processed)」で聴き比べできます。")

        return session.processed_audio

    @staticmethod
    def export_chapters(
        session: AudioSession,
        output_dir: Path | str,
        format_type: str = "wav24",
        use_processed: bool = True,
    ) -> List[Path]:
        """
        登録されているすべての曲チャプター (`session.song_chapters`) を切り出し、
        フェードイン・フェードアウトを適用して個別ファイルとして一括書き出しする。
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        src_audio = session.get_active_audio(use_processed=use_processed)
        sr = session.sample_rate

        ext_map = {
            "wav24": ".wav",
            "wav16": ".wav",
            "flac": ".flac",
            "m4a": ".m4a",
            "mp3": ".mp3",
        }
        ext = ext_map.get(format_type.lower(), ".wav")
        saved_paths: List[Path] = []

        for idx, ch in enumerate(session.song_chapters, start=1):
            sliced = MasteringProcessor.slice_chapter(src_audio, sr, ch)
            safe_title = re.sub(r'[\\/:*?"<>|]+', "_", ch.title.strip()) or f"Track_{idx:02d}"
            file_name = f"{idx:02d}_{safe_title}{ext}"
            target_path = out_dir / file_name
            saved = LosslessExtractor.save_audio(
                audio=sliced,
                sample_rate=sr,
                output_path=target_path,
                format_type=format_type,
            )
            saved_paths.append(saved)

        return saved_paths
