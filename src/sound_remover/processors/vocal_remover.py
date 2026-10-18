"""
AIピンポイント声（話し声・アナウンス・咳払い）除去エンジン (`vocal_remover.py`)

特徴:
1. Hybrid Transformer Demucs (`htdemucs`) を用いて音源から人の声成分 (`vocals`) を高精度に推論。
2. 3ステム合算ではなく「残差引き算方式 (`Original - alpha * Vocals`)」を採用することで、
   AIの再合成による楽器の倍音劣化や位相ボケを防ぎ、元の楽器音質を極限まで保持。
3. 「楽器保護帯域制限モード (`vocal_band_protection`)」により、人の声が存在する帯域
   (110Hz〜6500Hz) のみ引き算し、チューバ・大太鼓の重低音やシンバル・管楽器の超高域空気感は100%無加工維持。
4. 特定の時間帯（リージョン）のみにピンポイント適用し、前後をコサインクロスフェードで滑らかに結合する機能。
5. マルチGPU (CUDA / ROCm / MPS / DirectML) 対応およびCPUへの自動フォールバック安全機構。
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import scipy.signal as signal
import torch

from sound_remover.core.audio_data import TimeRegion
from sound_remover.core.device_manager import ComputeDevice


def apply_crossfade_splice(
    original: np.ndarray,
    processed_segment: np.ndarray,
    start_sample: int,
    end_sample: int,
    crossfade_samples: int,
) -> np.ndarray:
    """
    元の音声配列 `original` (channels, samples) の [start_sample:end_sample] 区間に、
    `processed_segment` をコサインカーブ（ハン窓）クロスフェードで滑らかに埋め込む。
    """
    result = original.copy()
    total_samples = original.shape[1]
    start_sample = max(0, min(start_sample, total_samples))
    end_sample = max(start_sample, min(end_sample, total_samples))
    seg_len = end_sample - start_sample

    if seg_len <= 0 or processed_segment.shape[1] < seg_len:
        return result

    proc_slice = processed_segment[:, :seg_len]
    orig_slice = original[:, start_sample:end_sample]

    fade_len = min(crossfade_samples, seg_len // 2)
    if fade_len <= 1:
        result[:, start_sample:end_sample] = proc_slice
        return result

    # 0.0 -> 1.0 の滑らかな S字コサインカーブ重み配列を作成
    weights = np.ones(seg_len, dtype=np.float32)
    t_in = np.linspace(0.0, np.pi, fade_len, dtype=np.float32)
    fade_in_curve = 0.5 * (1.0 - np.cos(t_in))
    fade_out_curve = fade_in_curve[::-1]

    weights[:fade_len] = fade_in_curve
    weights[-fade_len:] = fade_out_curve

    w_2d = weights[np.newaxis, :]
    result[:, start_sample:end_sample] = orig_slice * (1.0 - w_2d) + proc_slice * w_2d
    return result


class DemucsVocalRemover:
    """Demucsを用いた吹奏楽特化型・ピンポイント声除去クラス"""

    def __init__(self, model_name: str = "htdemucs") -> None:
        self.model_name = model_name
        self._model = None
        self._loaded_model_name: Optional[str] = None

    def _ensure_model(self, model_name: str):
        """Demucsモデルを遅延ロードしてキャッシュする"""
        if self._model is not None and self._loaded_model_name == model_name:
            return self._model

        from demucs.pretrained import get_model

        model = get_model(model_name)
        model.eval()
        self._model = model
        self._loaded_model_name = model_name
        return self._model

    @staticmethod
    def _bandpass_vocal_stem(vocal_audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        抽出された声成分 (vocals) に対し、人の声の主要帯域 (110Hz 〜 6500Hz) のみを残す
        ゼロ位相バンドパスフィルタを適用する。
        これにより、引き算時に110Hz以下の重低音（チューバ・バスドラム）や
        6500Hz以上の超高域倍音（シンバル・フルート空気感）が一切削られないように保護する。
        """
        nyquist = 0.5 * sample_rate
        low_hz = 110.0
        high_hz = min(6500.0, nyquist * 0.92)
        if low_hz >= high_hz:
            return vocal_audio

        sos = signal.butter(
            4,
            [low_hz / nyquist, high_hz / nyquist],
            btype="bandpass",
            output="sos",
        )
        # 短すぎる信号の場合はそのまま返す
        if vocal_audio.shape[1] < 64:
            return vocal_audio
        return signal.sosfiltfilt(sos, vocal_audio, axis=1).astype(np.float32)

    def _infer_vocal_stem(
        self,
        audio_segment: np.ndarray,
        sample_rate: int,
        device: ComputeDevice,
    ) -> np.ndarray:
        """
        指定された音声配列 (channels, samples) から Demucs で vocals ステムを推論して返す。
        モデルのサンプルレート (44100Hz) と異なる場合は内部で高精度リサンプリングを行う。
        """
        from demucs.apply import apply_model

        model = self._ensure_model(self.model_name)
        model_sr = int(getattr(model, "samplerate", 44100))
        model_channels = int(getattr(model, "audio_channels", 2))

        orig_channels, orig_samples = audio_segment.shape

        # チャンネル数をモデル仕様 (通常2chステレオ) に合わせる
        if orig_channels == 1 and model_channels == 2:
            work_audio = np.repeat(audio_segment, 2, axis=0)
        elif orig_channels > model_channels:
            work_audio = audio_segment[:model_channels, :]
        else:
            work_audio = audio_segment.copy()

        # サンプルレート変換 (例: 48000Hz -> 44100Hz)
        if sample_rate != model_sr:
            gcd = np.gcd(sample_rate, model_sr)
            up = model_sr // gcd
            down = sample_rate // gcd
            work_audio = signal.resample_poly(work_audio, up, down, axis=1).astype(np.float32)

        # 正規化（Demucs推論安定化のため）
        ref_mean = work_audio.mean(axis=0)
        mean_val = float(ref_mean.mean())
        std_val = float(ref_mean.std())
        if std_val < 1e-6:
            std_val = 1.0

        norm_audio = (work_audio - mean_val) / std_val
        wav_tensor = torch.from_numpy(norm_audio).unsqueeze(0).float()  # (1, C, T)

        # デバイス上で推論（失敗時は自動的にCPUへフォールバック）
        target_dev_str = device.torch_device_str
        try:
            torch_dev = torch.device(target_dev_str)
            model.to(torch_dev)
            with torch.no_grad():
                sources = apply_model(
                    model,
                    wav_tensor.to(torch_dev),
                    device=torch_dev,
                    shifts=1,
                    split=True,
                    overlap=0.25,
                    progress=False,
                )
        except Exception:
            # DirectMLやMPS等で未対応オペレータに遭遇した場合は安全にCPUへフォールバック
            cpu_dev = torch.device("cpu")
            model.to(cpu_dev)
            with torch.no_grad():
                sources = apply_model(
                    model,
                    wav_tensor.to(cpu_dev),
                    device=cpu_dev,
                    shifts=1,
                    split=True,
                    overlap=0.25,
                    progress=False,
                )

        # sources shape: (1, num_sources, C, T)
        source_names = list(getattr(model, "sources", ["drums", "bass", "other", "vocals"]))
        vocal_idx = source_names.index("vocals") if "vocals" in source_names else len(source_names) - 1

        vocal_tensor = sources[0, vocal_idx].cpu().numpy()
        vocal_est = vocal_tensor * std_val + mean_val

        # 元のサンプルレートへ戻す
        if sample_rate != model_sr:
            gcd = np.gcd(model_sr, sample_rate)
            up = sample_rate // gcd
            down = model_sr // gcd
            vocal_est = signal.resample_poly(vocal_est, up, down, axis=1).astype(np.float32)

        # サンプル長を厳密に元の長さに揃える
        if vocal_est.shape[1] > orig_samples:
            vocal_est = vocal_est[:, :orig_samples]
        elif vocal_est.shape[1] < orig_samples:
            pad_width = orig_samples - vocal_est.shape[1]
            vocal_est = np.pad(vocal_est, ((0, 0), (0, pad_width)), mode="constant")

        # 元のチャンネル数に戻す
        if orig_channels == 1 and vocal_est.shape[0] == 2:
            vocal_est = vocal_est.mean(axis=0, keepdims=True)

        return vocal_est.astype(np.float32)

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        device: ComputeDevice,
        reduction_ratio: float = 0.85,
        band_protection: bool = True,
        only_regions: bool = False,
        regions: Optional[List[TimeRegion]] = None,
        crossfade_ms: float = 60.0,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> np.ndarray:
        """
        音声配列に対してAIピンポイント声除去を実行する。

        Args:
            audio: (channels, samples) の float32 音声配列
            sample_rate: サンプリングレート (Hz)
            device: 使用する演算デバイス (CUDA / MPS / ROCm / DirectML / CPU)
            reduction_ratio: 声の除去強度 (0.0 〜 1.0)
            band_protection: 楽器保護帯域制限 (110Hz〜6500Hzのみ引き算)
            only_regions: Trueの場合、`regions` で指定された区間のみに声除去を適用
            regions: ピンポイント適用区間のリスト
            crossfade_ms: 区間境界のクロスフェード時間 (ミリ秒)
            progress_callback: 進捗通知コールバック (percent, message)
        """
        alpha = float(np.clip(reduction_ratio, 0.0, 1.0))
        if alpha <= 0.001:
            return audio.copy()

        total_samples = audio.shape[1]
        duration_sec = total_samples / float(sample_rate)
        crossfade_samples = max(16, int((crossfade_ms / 1000.0) * sample_rate))

        # リージョン指定モードかつ有効なリージョンがある場合
        active_regions = [
            r.normalized(duration_sec)
            for r in (regions or [])
            if r.apply_vocal_removal and r.duration > 0.05
        ]

        if only_regions:
            if not active_regions:
                # リージョン限定モードで対象区間が1つも無い場合は無加工で返す
                return audio.copy()

            result = audio.copy()
            pad_sec = 0.5  # Demucs推論端の境界劣化を防ぐための前後マージン秒数
            pad_samples = int(pad_sec * sample_rate)

            for idx, reg in enumerate(active_regions):
                if progress_callback:
                    pct = int(15 + (idx / len(active_regions)) * 40)
                    progress_callback(
                        pct,
                        f"AI声除去（ピンポイント区間 {idx + 1}/{len(active_regions)}: "
                        f"{reg.start_sec:.1f}秒〜{reg.end_sec:.1f}秒）を処理中...",
                    )

                s_idx = int(reg.start_sec * sample_rate)
                e_idx = int(reg.end_sec * sample_rate)

                ctx_s = max(0, s_idx - pad_samples)
                ctx_e = min(total_samples, e_idx + pad_samples)
                ctx_audio = audio[:, ctx_s:ctx_e]

                vocal_ctx = self._infer_vocal_stem(ctx_audio, sample_rate, device)
                if band_protection:
                    vocal_ctx = self._bandpass_vocal_stem(vocal_ctx, sample_rate)

                cleaned_ctx = ctx_audio - alpha * vocal_ctx
                # マージンを除いた本来必要なスライスを取り出す
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

        # 全体適用モード
        if progress_callback:
            progress_callback(20, f"AI音源分離モデル ({self.model_name}) で声成分を検出中 ({device.backend_type})...")

        vocal_stem = self._infer_vocal_stem(audio, sample_rate, device)
        if band_protection:
            if progress_callback:
                progress_callback(50, "楽器保護フィルタ（重低音・超高域倍音の完全保護）を適用中...")
            vocal_stem = self._bandpass_vocal_stem(vocal_stem, sample_rate)

        return (audio - alpha * vocal_stem).astype(np.float32)
