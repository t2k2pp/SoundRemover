"""
屋外風切り音対策・ハイパスフィルタ＆吹奏楽向け4バンドイコライザー (`eq_filter.py`)

特徴:
1. 屋外ステージでの風切り音（吹かれ）や空調・足音振動を抑えるバターワース・ハイパスフィルタ (30Hz〜150Hz)。
   吹奏楽の最低音（チューバ・コントラバス・バスドラムの約35〜40Hz）を損なわないよう精密なカットオフ調整が可能。
2. 吹奏楽の録音補正に特化した4バンド・パラメトリック/シェルビングEQ:
   - Low (80Hz Low-Shelf): 屋外で抜けやすい低音楽器の厚み・温かみを補強
   - Low-Mid (280Hz Peaking, Q=1.2): 体育館や反響の強いホールでの「モコモコしたこもり」を解消
   - High-Mid (2500Hz Peaking, Q=1.0): 木管・金管メロディの明瞭度・輪郭を調整
   - High (8000Hz High-Shelf): 金管の華やかな倍音・シンバルやフルートの空気感を補強
"""

from __future__ import annotations

import numpy as np
import scipy.signal as signal


class WindBandFilter:
    """吹奏楽録音向けハイパスフィルタ＆4バンドイコライザー"""

    @staticmethod
    def _design_biquad_lowshelf(freq_hz: float, sample_rate: int, gain_db: float, q: float = 0.707) -> np.ndarray:
        """Robert Bristow-Johnson Audio EQ Cookbook に基づく Low-Shelf フィルタの SOS 係数を生成"""
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * np.pi * freq_hz / float(sample_rate)
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2.0 * q)
        sqrt_A_2_alpha = 2.0 * np.sqrt(A) * alpha

        b0 = A * ((A + 1.0) - (A - 1.0) * cos_w0 + sqrt_A_2_alpha)
        b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0)
        b2 = A * ((A + 1.0) - (A - 1.0) * cos_w0 - sqrt_A_2_alpha)
        a0 = (A + 1.0) + (A - 1.0) * cos_w0 + sqrt_A_2_alpha
        a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w0)
        a2 = (A + 1.0) + (A - 1.0) * cos_w0 - sqrt_A_2_alpha

        return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)

    @staticmethod
    def _design_biquad_highshelf(freq_hz: float, sample_rate: int, gain_db: float, q: float = 0.707) -> np.ndarray:
        """Audio EQ Cookbook に基づく High-Shelf フィルタの SOS 係数を生成"""
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * np.pi * freq_hz / float(sample_rate)
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2.0 * q)
        sqrt_A_2_alpha = 2.0 * np.sqrt(A) * alpha

        b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + sqrt_A_2_alpha)
        b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
        b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - sqrt_A_2_alpha)
        a0 = (A + 1.0) - (A - 1.0) * cos_w0 + sqrt_A_2_alpha
        a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
        a2 = (A + 1.0) - (A - 1.0) * cos_w0 - sqrt_A_2_alpha

        return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)

    @staticmethod
    def _design_biquad_peaking(freq_hz: float, sample_rate: int, gain_db: float, q: float = 1.0) -> np.ndarray:
        """Audio EQ Cookbook に基づく Peaking EQ フィルタの SOS 係数を生成"""
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * np.pi * freq_hz / float(sample_rate)
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2.0 * q)

        b0 = 1.0 + alpha * A
        b1 = -2.0 * cos_w0
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha / A

        return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)

    @classmethod
    def process(
        cls,
        audio: np.ndarray,
        sample_rate: int,
        highpass_enabled: bool = True,
        highpass_cutoff_hz: float = 45.0,
        eq_enabled: bool = False,
        low_db: float = 0.0,
        low_mid_db: float = 0.0,
        high_mid_db: float = 0.0,
        high_db: float = 0.0,
    ) -> np.ndarray:
        """
        ハイパスフィルタおよび4バンドイコライザーを適用する。
        """
        if audio.shape[1] < 64:
            return audio.copy()

        result = audio.astype(np.float64, copy=True)
        nyquist = 0.5 * sample_rate

        # 1. 屋外風切り音・低周波振動ローカット（ゼロ位相 2次×前後双方向＝実質4次ハイパス）
        if highpass_enabled and 15.0 <= highpass_cutoff_hz < nyquist * 0.8:
            sos_hp = signal.butter(2, highpass_cutoff_hz / nyquist, btype="highpass", output="sos")
            result = signal.sosfiltfilt(sos_hp, result, axis=1)

        # 2. 吹奏楽向け4バンドEQ
        if eq_enabled:
            sos_list = []
            if abs(low_db) > 0.05 and 80.0 < nyquist * 0.9:
                sos_list.append(cls._design_biquad_lowshelf(80.0, sample_rate, low_db, q=0.707))
            if abs(low_mid_db) > 0.05 and 280.0 < nyquist * 0.9:
                sos_list.append(cls._design_biquad_peaking(280.0, sample_rate, low_mid_db, q=1.2))
            if abs(high_mid_db) > 0.05 and 2500.0 < nyquist * 0.9:
                sos_list.append(cls._design_biquad_peaking(2500.0, sample_rate, high_mid_db, q=1.0))
            if abs(high_db) > 0.05 and 8000.0 < nyquist * 0.9:
                sos_list.append(cls._design_biquad_highshelf(8000.0, sample_rate, high_db, q=0.707))

            if sos_list:
                full_sos = np.vstack(sos_list)
                result = signal.sosfilt(full_sos, result, axis=1)

        return result.astype(np.float32)
