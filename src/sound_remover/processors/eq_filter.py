"""
屋外風切り音対策（スマート風切り音サプレッサー）・ハイパスフィルタ＆吹奏楽向け4バンドイコライザー (`eq_filter.py`)

特徴:
1. 屋外ステージでの風切り音（吹かれ）や空調・足音振動を抑えるバターワース・ハイパスフィルタ (25Hz〜150Hz)。
   吹奏楽の最低音（チューバ・コントラバス・バスドラムの約35〜40Hz）を損なわないよう精密なカットオフ調整が可能。
2. スマート風切り音除去 (`suppress_wind_noise`):
   - ステレオ録音において、楽器の低音は左右同相 (Mid成分) で届くのに対し、マイク穴に直接当たる
     風切り音（吹かれ）は左右非相関・逆相 (Side成分) が激増する物理特性を利用。
   - 260Hz以下のSide乱流成分を常時カットし、さらに風センサー (`|Side| / |Mid|` 比) が風を検知した瞬間だけ
     Mid成分の低域乱流ノイズを動的に減衰させることで、チューバやベースの低音を一切痩せさせずに風の音だけを除去。
3. 吹奏楽の録音補正に特化した4バンド・パラメトリック/シェルビングEQ:
   - Low (80Hz Low-Shelf): 屋外で抜けやすい低音楽器の厚み・温かみを補強
   - Low-Mid (280Hz Peaking, Q=1.2): 体育館や反響の強いホールでの「モコモコしたこもり」を解消
   - High-Mid (2500Hz Peaking, Q=1.0): 木管・金管メロディの明瞭度・輪郭を調整
   - High (8000Hz High-Shelf): 金管の華やかな倍音・シンバルやフルートの空気感を補強
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal


class WindBandFilter:
    """吹奏楽録音向けハイパスフィルタ・スマート風切り音除去＆4バンドイコライザー"""

    @staticmethod
    def _design_biquad_lowshelf(
        freq_hz: float, sample_rate: int, gain_db: float, q: float = 0.707
    ) -> np.ndarray:
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
    def _design_biquad_highshelf(
        freq_hz: float, sample_rate: int, gain_db: float, q: float = 0.707
    ) -> np.ndarray:
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
    def _design_biquad_peaking(
        freq_hz: float, sample_rate: int, gain_db: float, q: float = 1.0
    ) -> np.ndarray:
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
    def suppress_wind_noise(
        cls,
        audio: np.ndarray,
        sample_rate: int,
        strength: float = 0.85,
        wind_cutoff_hz: float = 260.0,
    ) -> np.ndarray:
        """
        スマート風切り音除去（M/S乱流カット ＋ 風センサー連動・動的吹かれサプレッサー）。
        チューバやベース等の同相低音（Mid成分）を温存したまま、マイクに直接当たる風の乱流ノイズを除去する。
        """
        alpha = float(np.clip(strength, 0.0, 1.0))
        if alpha <= 0.001 or audio.shape[1] < 128:
            return audio.copy()

        nyquist = 0.5 * sample_rate
        cutoff = float(np.clip(wind_cutoff_hz, 100.0, min(400.0, nyquist * 0.8)))
        sos_lp = signal.butter(2, cutoff / nyquist, btype="lowpass", output="sos")

        win_samples = max(16, int(0.06 * sample_rate))     # 60ms RMS検知窓
        smooth_samples = max(32, int(0.12 * sample_rate))  # 120ms ゲイン平滑化窓

        if audio.shape[0] >= 2:
            left = audio[0].astype(np.float64)
            right = audio[1].astype(np.float64)
            mid = 0.5 * (left + right)
            side = 0.5 * (left - right)

            mid_low = signal.sosfiltfilt(sos_lp, mid)
            side_low = signal.sosfiltfilt(sos_lp, side)

            # 1. 低域のSide成分（左右逆相の風乱流）を常時カット
            side_cleaned = side - alpha * side_low

            # 2. |Side_low| / |Mid_low| 比から風吹かれ瞬間を検知し、その瞬間だけ Mid_low も動的に抑制
            env_side = np.sqrt(ndimage.uniform_filter1d(np.square(side_low), size=win_samples, mode="nearest") + 1e-12)
            env_mid = np.sqrt(ndimage.uniform_filter1d(np.square(mid_low), size=win_samples, mode="nearest") + 1e-12)

            wind_ratio = env_side / (env_mid + 1e-7)
            # 通常演奏時は wind_ratio < 0.22、風吹かれ時は 0.30〜1.20+ に上昇
            excess = np.clip((wind_ratio - 0.22) / 0.38, 0.0, 1.0)
            level_gate = np.clip((env_side - 0.0008) / 0.004, 0.0, 1.0)

            raw_suppress = excess * level_gate * (alpha * 0.90)
            smooth_suppress = ndimage.uniform_filter1d(raw_suppress, size=smooth_samples, mode="nearest")

            mid_cleaned = mid - smooth_suppress * mid_low

            out = audio.copy()
            out[0] = (mid_cleaned + side_cleaned).astype(np.float32)
            out[1] = (mid_cleaned - side_cleaned).astype(np.float32)
            return out

        # モノラル音声の場合: 超低域(20-120Hz)の突発的膨張を検知して動的抑制
        mono = audio[0].astype(np.float64)
        mono_low = signal.sosfiltfilt(sos_lp, mono)
        env_low = np.sqrt(ndimage.uniform_filter1d(np.square(mono_low), size=win_samples, mode="nearest") + 1e-12)
        median_low = float(np.median(env_low)) + 1e-6
        surge = np.clip((env_low / median_low - 2.0) / 3.0, 0.0, 1.0) * (alpha * 0.80)
        smooth_surge = ndimage.uniform_filter1d(surge, size=smooth_samples, mode="nearest")
        cleaned_mono = (mono - smooth_surge * mono_low).astype(np.float32)
        return cleaned_mono[np.newaxis, :]

    @classmethod
    def process(
        cls,
        audio: np.ndarray,
        sample_rate: int,
        highpass_enabled: bool = True,
        highpass_cutoff_hz: float = 45.0,
        wind_reduction_enabled: bool = False,
        wind_reduction_strength: float = 0.85,
        wind_cutoff_hz: float = 260.0,
        eq_enabled: bool = False,
        low_db: float = 0.0,
        low_mid_db: float = 0.0,
        high_mid_db: float = 0.0,
        high_db: float = 0.0,
    ) -> np.ndarray:
        """
        ハイパスフィルタ、スマート風切り音除去、および4バンドイコライザーを適用する。
        """
        if audio.shape[1] < 64:
            return audio.copy()

        work = audio.astype(np.float32, copy=True)

        # 1. スマート風切り音除去（M/S乱流カット＋動的吹かれサプレッサー）
        if wind_reduction_enabled and wind_reduction_strength > 0.001:
            work = cls.suppress_wind_noise(
                work,
                sample_rate=sample_rate,
                strength=wind_reduction_strength,
                wind_cutoff_hz=wind_cutoff_hz,
            )

        result = work.astype(np.float64, copy=True)
        nyquist = 0.5 * sample_rate

        # 2. 屋外風切り音・低周波振動ローカット（ゼロ位相 2次×前後双方向＝実質4次ハイパス）
        if highpass_enabled and 15.0 <= highpass_cutoff_hz < nyquist * 0.8:
            sos_hp = signal.butter(2, highpass_cutoff_hz / nyquist, btype="highpass", output="sos")
            result = signal.sosfiltfilt(sos_hp, result, axis=1)

        # 3. 吹奏楽向け4バンドEQ
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
