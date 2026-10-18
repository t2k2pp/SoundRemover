"""
吹奏楽向けマスタリング・ラウドネスノーマライズ・ピークリミッター・曲分割フェード処理 (`mastering.py`)

特徴:
1. 吹奏楽特有の広いダイナミックレンジ（ピアニッシモ〜フォルティッシモ）を活かしつつ、
   スマホやPCで聴きやすい音量に揃える ITU-R BS.1770-4 ラウドネス正規化 (`pyloudnorm`)。
2. 打楽器（シンバル・バスドラム）や金管楽器の強奏時に絶対にデジタル音割れ（クリッピング）を
   起こさせない、ルックアヘッド・ソフトエンベロープ・ピークリミッター。
3. 1本の長い発表動画から曲ごとに切り出し、自然なS字コサインカーブでフェードイン・フェードアウトを適用する機能。
"""

from __future__ import annotations

import numpy as np
import pyloudnorm as pyln
import scipy.ndimage as ndimage

from sound_remover.core.audio_data import SongChapter


class MasteringProcessor:
    """音量正規化・音割れ防止リミッター・フェード処理クラス"""

    @staticmethod
    def measure_lufs(audio: np.ndarray, sample_rate: int) -> float:
        """
        音声配列 (channels, samples) の統合ラウドネス (Integrated LUFS) を計測する。
        0.4秒未満の極短音声の場合はRMSベースの推定値を返す。
        """
        duration = audio.shape[1] / float(sample_rate)
        if duration >= 0.45:
            try:
                meter = pyln.Meter(sample_rate)
                # pyloudnorm は (samples, channels) を受け取る
                lufs = float(meter.integrated_loudness(audio.T.astype(np.float64)))
                if np.isfinite(lufs):
                    return lufs
            except Exception:
                pass

        rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64))) + 1e-12))
        return 20.0 * np.log10(rms) - 0.691

    @classmethod
    def apply_peak_limiter(
        cls,
        audio: np.ndarray,
        sample_rate: int,
        ceiling_db: float = -1.0,
    ) -> np.ndarray:
        """
        ルックアヘッド付きエンベロープ・ピークリミッター。
        指定上限 `ceiling_db` (例: -1.0 dBFS) を超える瞬間ピークのみを滑らかに減衰させ、
        吹奏楽のフォルティッシモや打楽器アタックでの音割れ（クリッピング）を防止する。
        """
        ceiling_lin = float(10.0 ** (ceiling_db / 20.0))
        abs_peak = np.max(np.abs(audio), axis=0)  # (samples,)
        max_val = float(np.max(abs_peak)) if abs_peak.size > 0 else 0.0

        if max_val <= ceiling_lin:
            return audio.copy()

        # 必要なゲインリダクション包絡線を計算
        # ルックアヘッド窓 (約 5ms) で急激なアタックを先読み捕捉
        lookahead_samples = max(4, int(0.005 * sample_rate))
        env_peak = ndimage.maximum_filter1d(abs_peak, size=lookahead_samples, mode="nearest")

        # 上限を超えた箇所のみ減衰率 (<= 1.0) を求める
        target_gain = np.where(env_peak > ceiling_lin, ceiling_lin / (env_peak + 1e-12), 1.0)

        # ゲイン変化を滑らかにスムージング (約 15ms の移動平均窓) して波形歪みを防止
        smooth_samples = max(4, int(0.015 * sample_rate))
        smoothed_gain = ndimage.uniform_filter1d(target_gain, size=smooth_samples, mode="nearest")
        # 先読みピーク時点で確実に ceiling_lin 以下になるよう min を取る
        effective_gain = np.minimum(target_gain, smoothed_gain).astype(np.float32)

        limited = audio * effective_gain[np.newaxis, :]
        # 万が一の数値誤差に対する最終安全クリップ
        return np.clip(limited, -ceiling_lin, ceiling_lin).astype(np.float32)

    @classmethod
    def process(
        cls,
        audio: np.ndarray,
        sample_rate: int,
        loudness_norm_enabled: bool = True,
        target_lufs: float = -15.0,
        limiter_enabled: bool = True,
        limiter_ceiling_db: float = -1.0,
    ) -> np.ndarray:
        """
        ラウドネス正規化とピークリミッターを適用する。
        """
        result = audio.astype(np.float32, copy=True)

        if loudness_norm_enabled and result.shape[1] > 0:
            current_lufs = cls.measure_lufs(result, sample_rate)
            if np.isfinite(current_lufs) and current_lufs > -70.0:
                delta_db = float(target_lufs - current_lufs)
                # 無音に近い録音でノイズが異常増幅されないよう最大増幅量を +24dB に制限
                delta_db = float(np.clip(delta_db, -24.0, 24.0))
                gain_lin = float(10.0 ** (delta_db / 20.0))
                result = result * gain_lin

        if limiter_enabled and result.shape[1] > 0:
            result = cls.apply_peak_limiter(result, sample_rate, ceiling_db=limiter_ceiling_db)

        return result

    @staticmethod
    def apply_fade(
        audio: np.ndarray,
        sample_rate: int,
        fade_in_sec: float = 0.0,
        fade_out_sec: float = 0.0,
    ) -> np.ndarray:
        """
        音声配列 (channels, samples) の冒頭と末尾に自然なS字コサイン・フェードイン／フェードアウトを適用する。
        """
        result = audio.copy()
        total_samples = result.shape[1]
        if total_samples <= 1:
            return result

        fade_in_samples = min(total_samples, max(0, int(fade_in_sec * sample_rate)))
        fade_out_samples = min(total_samples, max(0, int(fade_out_sec * sample_rate)))

        if fade_in_samples > 1:
            t_in = np.linspace(0.0, np.pi, fade_in_samples, dtype=np.float32)
            curve_in = 0.5 * (1.0 - np.cos(t_in))
            result[:, :fade_in_samples] *= curve_in[np.newaxis, :]

        if fade_out_samples > 1:
            t_out = np.linspace(0.0, np.pi, fade_out_samples, dtype=np.float32)
            curve_out = 0.5 * (1.0 + np.cos(t_out))
            result[:, total_samples - fade_out_samples :] *= curve_out[np.newaxis, :]

        return result

    @classmethod
    def slice_chapter(
        cls,
        audio: np.ndarray,
        sample_rate: int,
        chapter: SongChapter,
    ) -> np.ndarray:
        """
        指定された `SongChapter` の時間範囲を切り出し、フェードイン・フェードアウトを適用して返す。
        """
        total_samples = audio.shape[1]
        s_idx = max(0, min(int(chapter.start_sec * sample_rate), total_samples))
        e_idx = max(s_idx, min(int(chapter.end_sec * sample_rate), total_samples))
        sliced = audio[:, s_idx:e_idx].copy()
        return cls.apply_fade(
            sliced,
            sample_rate=sample_rate,
            fade_in_sec=chapter.fade_in_sec,
            fade_out_sec=chapter.fade_out_sec,
        )
