"""
音声ファイルの周波数特性・風切り音（マイク吹かれ）・ホワイトノイズ自動診断エンジン (`analyzer.py`)

読み込んだ動画・音声ファイル（または選択範囲）の周波数エネルギー分布、左右マイクの位相相関、
風切り音の発生時間帯、ホワイトノイズ見本に適した静寂区間を自動解析し、
ユーザーに分かりやすい診断レポートと最適なフィルタ推奨設定 (`ProcessingConfig`) を提案する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import scipy.signal as signal

from sound_remover.core.audio_data import ProcessingConfig
from sound_remover.processors.mastering import MasteringProcessor


@dataclass
class BandStat:
    """周波数帯域ごとのエネルギー・ステレオ相関診断結果"""
    name: str
    low_hz: float
    high_hz: float
    rms_db: float
    lr_correlation: float
    comment: str


@dataclass
class WindBurstEvent:
    """検出された風切り音（マイク吹かれ）発生区間"""
    start_sec: float
    end_sec: float
    peak_side_db: float
    max_side_mid_ratio: float
    severity: str  # "強", "中", "軽"


@dataclass
class VoiceCandidateEvent:
    """検出された人の声・話し声・咳払い・アナウンス候補区間"""
    start_sec: float
    end_sec: float
    voice_db: float
    confidence: float  # 0.0 〜 1.0
    detection_type: str  # "近接話し声・咳払い候補", "曲間・弱奏時の声候補", "AI声成分検出"
    reason: str


@dataclass
class AudioDiagnosisReport:
    """音響・周波数特性の総合診断レポート"""
    duration_sec: float
    sample_rate: int
    channels: int
    integrated_lufs: float
    peak_dbfs: float
    band_stats: List[BandStat] = field(default_factory=list)
    wind_events: List[WindBurstEvent] = field(default_factory=list)
    wind_severity_summary: str = "なし"
    voice_events: List[VoiceCandidateEvent] = field(default_factory=list)
    voice_summary: str = "なし"
    suggested_noise_sample_region: Optional[Tuple[float, float]] = None
    noise_floor_db: float = -70.0
    recommendations_text: List[str] = field(default_factory=list)
    recommended_config: ProcessingConfig = field(default_factory=ProcessingConfig.preset_outdoor_stage)


class AudioAnalyzer:
    """音声データの周波数特性・風切り音・人の声候補・ホワイトノイズを診断し、最適設定を提案するクラス"""


    @staticmethod
    def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
        sx = float(np.std(x))
        sy = float(np.std(y))
        if sx < 1e-9 or sy < 1e-9:
            return 1.0
        c = float(np.corrcoef(x, y)[0, 1])
        return c if np.isfinite(c) else 1.0

    @classmethod
    def analyze(
        cls,
        audio: np.ndarray,
        sample_rate: int,
        base_config: Optional[ProcessingConfig] = None,
    ) -> AudioDiagnosisReport:
        """
        音声配列 (channels, samples) を解析し、`AudioDiagnosisReport` を生成して返す。
        """
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        channels, total_samples = audio.shape
        duration_sec = total_samples / float(sample_rate) if sample_rate > 0 else 0.0
        nyquist = 0.5 * sample_rate

        # 高速解析のため、長尺音声でもサンプルレートが48kHz超の場合はそのまま、フィルタ演算を効率化する
        lufs = MasteringProcessor.measure_lufs(audio, sample_rate)
        max_abs = float(np.max(np.abs(audio))) if total_samples > 0 else 0.0
        peak_dbfs = 20.0 * np.log10(max_abs + 1e-12)

        # 1. 5帯域の周波数エネルギー＆L/R相関解析
        band_defs = [
            ("重低音・振動 (20〜60 Hz)", 20.0, 60.0),
            ("低音・風吹かれ帯域 (60〜180 Hz)", 60.0, 180.0),
            ("中低域・こもり帯域 (180〜400 Hz)", 180.0, 400.0),
            ("中域・管楽器主旋律 (400〜2.5k Hz)", 400.0, 2500.0),
            ("高域・倍音＆空気感 (2.5k〜10k Hz)", 2500.0, min(10000.0, nyquist * 0.92)),
        ]

        band_stats: List[BandStat] = []
        band_db_map = {}

        for b_name, f_lo, f_hi in band_defs:
            if f_lo >= f_hi or total_samples < 128:
                continue
            sos = signal.butter(3, [f_lo / nyquist, f_hi / nyquist], btype="bandpass", output="sos")
            filt = signal.sosfiltfilt(sos, audio, axis=1)
            rms_val = float(np.sqrt(np.mean(np.square(filt, dtype=np.float64)) + 1e-12))
            rms_db = 20.0 * np.log10(rms_val)
            corr = cls._safe_corrcoef(filt[0], filt[1]) if channels >= 2 else 1.0
            band_db_map[f_lo] = (rms_db, corr)

            comment = "良好"
            if f_lo == 20.0:
                if corr < 0.70 or rms_db > -34.0:
                    comment = "⚠ 風吹かれ・振動ノイズ混入あり"
                else:
                    comment = "クリア"
            elif f_lo == 60.0:
                if corr < 0.80:
                    comment = "⚠ ステレオ乱流（風切り音）混入"
                elif rms_db < -30.0:
                    comment = "やや低音控えめ（補強推奨）"
                else:
                    comment = "豊かな低音バランス"
            elif f_lo == 180.0:
                comment = "中低域の厚み"
            elif f_lo == 400.0:
                comment = "主旋律・声の主要帯域"
            elif f_lo == 2500.0:
                comment = "倍音・きらびやかさ"

            band_stats.append(
                BandStat(
                    name=b_name,
                    low_hz=f_lo,
                    high_hz=f_hi,
                    rms_db=rms_db,
                    lr_correlation=corr,
                    comment=comment,
                )
            )

        # 2. 風切り音（マイク吹かれ：30〜280Hz の Side/Mid 乱流比率）スキャン
        wind_events: List[WindBurstEvent] = []
        if total_samples >= int(0.5 * sample_rate) and channels >= 2:
            sos_w = signal.butter(3, [30.0 / nyquist, min(280.0, nyquist * 0.8) / nyquist], btype="bandpass", output="sos")
            w_sig = signal.sosfiltfilt(sos_w, audio, axis=1)
            mid_w = 0.5 * (w_sig[0] + w_sig[1])
            side_w = 0.5 * (w_sig[0] - w_sig[1])

            step = max(1, sample_rate // 2)  # 0.5秒単位
            raw_hits: List[Tuple[float, float, float, float]] = []

            for i in range(0, total_samples - step + 1, step):
                m_rms = float(np.sqrt(np.mean(np.square(mid_w[i : i + step], dtype=np.float64)) + 1e-12))
                s_rms = float(np.sqrt(np.mean(np.square(side_w[i : i + step], dtype=np.float64)) + 1e-12))
                ratio = s_rms / m_rms
                s_db = 20.0 * np.log10(s_rms)
                if ratio > 0.34 and s_db > -42.0:
                    t_s = i / float(sample_rate)
                    t_e = min(duration_sec, (i + step) / float(sample_rate))
                    raw_hits.append((t_s, t_e, s_db, ratio))

            # 連続または1.0秒以内に近接する風切り音区間を1つのイベントにマージする
            for h_s, h_e, h_db, h_ratio in raw_hits:
                if wind_events and (h_s - wind_events[-1].end_sec) <= 1.0:
                    prev = wind_events[-1]
                    prev.end_sec = h_e
                    prev.peak_side_db = max(prev.peak_side_db, h_db)
                    prev.max_side_mid_ratio = max(prev.max_side_mid_ratio, h_ratio)
                    if prev.max_side_mid_ratio >= 0.70 or prev.peak_side_db >= -28.0:
                        prev.severity = "強"
                    elif prev.max_side_mid_ratio >= 0.48 or prev.peak_side_db >= -34.0:
                        prev.severity = "中"
                else:
                    sev = "軽"
                    if h_ratio >= 0.70 or h_db >= -28.0:
                        sev = "強"
                    elif h_ratio >= 0.48 or h_db >= -34.0:
                        sev = "中"
                    wind_events.append(
                        WindBurstEvent(
                            start_sec=h_s,
                            end_sec=h_e,
                            peak_side_db=h_db,
                            max_side_mid_ratio=h_ratio,
                            severity=sev,
                        )
                    )

        total_wind_sec = sum(ev.end_sec - ev.start_sec for ev in wind_events)
        strong_count = sum(1 for ev in wind_events if ev.severity in {"強", "中"})
        if not wind_events:
            wind_summary = "なし（風切り音は検出されませんでした）"
        elif strong_count >= 3 or total_wind_sec >= 8.0:
            wind_summary = f"重度（計 {len(wind_events)} 区間 / 合計 {total_wind_sec:.1f} 秒の風吹かれを検出）"
        elif strong_count >= 1 or total_wind_sec >= 2.5:
            wind_summary = f"中度（計 {len(wind_events)} 区間 / 合計 {total_wind_sec:.1f} 秒の風吹かれを検出）"
        else:
            wind_summary = f"軽度（計 {len(wind_events)} 区間 / 合計 {total_wind_sec:.1f} 秒の微弱な風を検出）"

        # 3. 人の声・話し声・咳払い・曲間アナウンス候補区間の高速スキャン
        voice_events = cls.scan_voice_candidates_fast(audio, sample_rate)
        total_voice_sec = sum(ev.end_sec - ev.start_sec for ev in voice_events)
        if not voice_events:
            voice_summary = "なし（顕著な近接話し声・咳払い候補は検出されませんでした）"
        else:
            voice_summary = f"計 {len(voice_events)} 区間（合計 {total_voice_sec:.1f} 秒の声・咳払い・近接雑音候補を検出）"

        # 4. ホワイトノイズ見本（最も静かで定常的な1.0秒区間）の自動探索
        suggested_noise_region: Optional[Tuple[float, float]] = None
        noise_floor_db = -65.0
        win_1s = sample_rate
        if total_samples >= win_1s * 2:
            mono = audio.mean(axis=0)
            step_q = max(1, sample_rate // 2)
            candidates: List[Tuple[float, float, float]] = []
            for i in range(0, total_samples - win_1s, step_q):
                seg = mono[i : i + win_1s]
                rms = float(np.sqrt(np.mean(np.square(seg, dtype=np.float64)) + 1e-12))
                db = 20.0 * np.log10(rms)
                # 完全デジタル無音(-85dB以下)は除外し、サーッという環境音区間(-78dB〜-32dB)を探す
                if -78.0 < db < -32.0:
                    # 前半・後半の音量差が小さい（定常的な）区間を優先
                    r1 = float(np.sqrt(np.mean(np.square(seg[: win_1s // 2])) + 1e-12))
                    r2 = float(np.sqrt(np.mean(np.square(seg[win_1s // 2 :])) + 1e-12))
                    steadiness_penalty = abs(20.0 * np.log10(r1 / r2))
                    score = db + steadiness_penalty * 2.0
                    candidates.append((score, db, i / float(sample_rate)))

            if candidates:
                candidates.sort(key=lambda x: x[0])
                _, best_db, best_start = candidates[0]
                noise_floor_db = best_db
                suggested_noise_region = (best_start, min(duration_sec, best_start + 1.2))

        # 5. 診断に基づく推奨設定 (`ProcessingConfig`) とアドバイス文章の生成
        rec_cfg = ProcessingConfig.preset_outdoor_stage() if base_config is None else ProcessingConfig(**base_config.__dict__)
        recommendations: List[str] = []

        if wind_events:
            rec_cfg.wind_reduction_enabled = True
            rec_cfg.highpass_enabled = True
            if strong_count >= 2 or total_wind_sec >= 5.0:
                rec_cfg.wind_reduction_strength = 0.90
                rec_cfg.wind_cutoff_hz = 280.0
                rec_cfg.highpass_cutoff_hz = 60.0
                recommendations.append(
                    f"🌪 【風切り音対策】スマホ/カメラのマイクに風が当たっている区間が {len(wind_events)} 箇所（合計 {total_wind_sec:.1f}秒）見つかりました。"
                    "『スマート風切り音除去（強度 90% / 上限 280Hz）』と『ローカット 60Hz』を有効にすることで、"
                    "チューバやベースの低音を削らずにボフボフ・ゴォーという風の音だけを狙い撃ちで除去します。"
                )
            else:
                rec_cfg.wind_reduction_strength = 0.75
                rec_cfg.wind_cutoff_hz = 240.0
                rec_cfg.highpass_cutoff_hz = 48.0
                recommendations.append(
                    f"🌪 【風切り音対策】軽微な風の混入が {len(wind_events)} 箇所見つかりました。"
                    "『スマート風切り音除去（強度 75% / 上限 240Hz）』を推奨します。"
                )
        else:
            rec_cfg.wind_reduction_enabled = False
            rec_cfg.highpass_cutoff_hz = 40.0
            recommendations.append(
                "✅ 【風切り音診断】マイク吹かれ（風切り音）は検出されませんでした。ローカットは最低限の 40Hz で楽器の重低音をフルに活かせます。"
            )

        if voice_events:
            recommendations.append(
                f"🗣 【人の声・咳払い候補検出】近接話し声・咳払い・曲間声の疑いがある区間が {len(voice_events)} 箇所（合計 {total_voice_sec:.1f}秒）検出されました。"
                "上の『🗣 検出された声・咳払い候補区間』をダブルクリックして試聴し、不要な声であれば『➕ ピンポイント除去リストへ追加』を押すだけでその時間帯の声だけを狙い撃ちで消去できます。"
                "さらに演奏と重なった微細な声も探したい場合は『🤖 AIで声区間を精密スキャン』を実行してください。"
            )
        else:
            recommendations.append(
                "🗣 【人の声・咳払い診断】高速空間スキャンでは顕著な近接私語や曲間アナウンスは検出されませんでした（大音量の演奏中に紛れた声も探す場合は『🤖 AIで声区間を精密スキャン』ボタンを押してください）。"
            )

        # 中低域のこもり (180-400Hz) vs 主旋律 (400-2500Hz) のバランス診断
        low_mid_db = band_db_map.get(180.0, (-25.0, 1.0))[0]
        mid_db = band_db_map.get(400.0, (-25.0, 1.0))[0]
        bass_db = band_db_map.get(60.0, (-25.0, 1.0))[0]
        high_db = band_db_map.get(2500.0, (-35.0, 1.0))[0]

        rec_cfg.eq_enabled = True
        if low_mid_db - mid_db > -0.5:
            # 180-400Hzが主旋律と同等以上に強い＝こもり気味、または風の中低域かぶり
            rec_cfg.eq_low_mid_db = -2.5
            recommendations.append(
                "🎺 【こもり解消EQ】中低域（180〜400Hz）のエネルギーが強く、少し音がこもって聴こえやすいバランスです。"
                "『Low-Mid 280Hz を -2.5 dB』に下げて管楽器の輪郭をスッキリさせる設定を推奨します。"
            )
        else:
            rec_cfg.eq_low_mid_db = -1.0

        if mid_db - bass_db > 5.0:
            rec_cfg.eq_low_db = 2.5
            recommendations.append(
                "🎶 【低音補強EQ】主旋律に対して低音楽器（60〜180Hz）が控えめなため、『Low 80Hz を +2.5 dB』でチューバ・ベースの土台を補強します。"
            )
        else:
            rec_cfg.eq_low_db = 1.0

        if mid_db - high_db > 8.5:
            rec_cfg.eq_high_mid_db = 1.0
            rec_cfg.eq_high_db = 2.0
            recommendations.append(
                "✨ 【倍音・空気感EQ】高域（2.5kHz以上）がやや減衰しているため、『High 8.0kHz を +2.0 dB』で金管の華やかさとシンバルの空気感を補います。"
            )

        if suggested_noise_region is not None:
            ns, ne = suggested_noise_region
            rec_cfg.noise_profile_region = suggested_noise_region
            rec_cfg.noise_reduction_enabled = True
            rec_cfg.noise_reduction_strength = 0.70
            recommendations.append(
                f"🌫 【ホワイトノイズ見本】{ns:.1f}秒〜{ne:.1f}秒（ノイズレベル {noise_floor_db:.1f} dB）が最も静かな環境音区間として検出されました。"
                "この区間をホワイトノイズ学習見本として自動セットできます。"
            )

        recommendations.append(
            f"🎚 【音量・音割れ診断】現在のラウドネスは {lufs:.1f} LUFS（最大ピーク {peak_dbfs:.1f} dBFS）です。"
            "吹奏楽の強弱（ダイナミクス）を活かしつつ音割れを防ぐ『目標 -15.0 LUFS ＋ リミッター上限 -1.0 dBFS』を推奨します。"
        )

        return AudioDiagnosisReport(
            duration_sec=duration_sec,
            sample_rate=sample_rate,
            channels=channels,
            integrated_lufs=lufs,
            peak_dbfs=peak_dbfs,
            band_stats=band_stats,
            wind_events=wind_events,
            wind_severity_summary=wind_summary,
            voice_events=voice_events,
            voice_summary=voice_summary,
            suggested_noise_sample_region=suggested_noise_region,
            noise_floor_db=noise_floor_db,
            recommendations_text=recommendations,
            recommended_config=rec_cfg,
        )

    @classmethod
    def scan_voice_candidates_fast(
        cls,
        audio: np.ndarray,
        sample_rate: int,
    ) -> List[VoiceCandidateEvent]:
        """
        スマート風切り音検出の空間・音響解析を発展させ、1秒以内で「人の話し声・咳払い・曲間アナウンス」の
        疑いがある時間帯を自動ピックアップする（高速音響・空間特徴スキャン）。

        検出ロジック:
        1. 声のフォルマント＆子音帯域 (300〜3800Hz) の Mid/Side 比率の局所突出（カメラ近傍での私語・咳払いは左右マイクのバランス・位相がステージ演奏より偏る）
        2. 短時間サブフレーム (20ms単位) の包絡線変調度 (CV = std/mean) およびクレストファクタ（人の話し声特有の3〜9Hz音節リズムや咳の突発アタック）
        3. 低音楽器帯域 (60〜220Hz) が静かな曲間・弱奏時に、声帯域 (300〜3800Hz) が突出する現象
        """
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        channels, total_samples = audio.shape
        win = int(0.5 * sample_rate)
        hop = max(1, int(0.25 * sample_rate))
        if total_samples < win or sample_rate <= 0:
            return []

        nyquist = 0.5 * sample_rate
        duration_sec = total_samples / float(sample_rate)

        # 声の主要フォルマント・子音帯域 (300Hz 〜 3800Hz)
        v_hi = min(3800.0, nyquist * 0.90)
        if v_hi <= 320.0:
            return []
        sos_v = signal.butter(3, [300.0 / nyquist, v_hi / nyquist], btype="bandpass", output="sos")
        v_sig = signal.sosfiltfilt(sos_v, audio, axis=1)

        # 低音楽器リファレンス帯域 (60Hz 〜 220Hz)
        sos_b = signal.butter(3, [60.0 / nyquist, min(220.0, nyquist * 0.5) / nyquist], btype="bandpass", output="sos")
        b_sig = signal.sosfiltfilt(sos_b, audio, axis=1)

        if channels >= 2:
            mid_v = 0.5 * (v_sig[0] + v_sig[1])
            side_v = 0.5 * (v_sig[0] - v_sig[1])
            side_b = 0.5 * (b_sig[0] - b_sig[1])
        else:
            mid_v = v_sig[0]
            side_v = np.zeros_like(mid_v)
            side_b = np.zeros_like(mid_v)

        b_mono = b_sig.mean(axis=0)
        v_energy_trace = np.mean(np.square(v_sig, dtype=np.float32), axis=0)


        sub_len = max(1, int(0.02 * sample_rate))  # 20ms サブフレーム（音節変調・咳払いアタック計測用）
        window_records = []

        for start_idx in range(0, total_samples - win + 1, hop):
            end_idx = start_idx + win
            v_seg_sq = v_energy_trace[start_idx:end_idx]
            v_rms = float(np.sqrt(np.mean(v_seg_sq, dtype=np.float64) + 1e-12))
            v_db = 20.0 * np.log10(v_rms)

            b_rms = float(np.sqrt(np.mean(np.square(b_mono[start_idx:end_idx], dtype=np.float64)) + 1e-12))
            b_db = 20.0 * np.log10(b_rms)

            m_rms = float(np.sqrt(np.mean(np.square(mid_v[start_idx:end_idx], dtype=np.float64)) + 1e-12))
            s_rms = float(np.sqrt(np.mean(np.square(side_v[start_idx:end_idx], dtype=np.float64)) + 1e-12))
            sm_ratio = s_rms / (m_rms + 1e-9)

            sb_rms = float(np.sqrt(np.mean(np.square(side_b[start_idx:end_idx], dtype=np.float64)) + 1e-12))
            sb_db = 20.0 * np.log10(sb_rms)

            # 20msサブフレームごとの振幅包絡線から音節変調度 (CV) とクレストファクタを算出
            n_subs = win // sub_len
            if n_subs >= 4:
                trimmed = v_seg_sq[: n_subs * sub_len].reshape(n_subs, sub_len)
                sub_env = np.sqrt(np.mean(trimmed, axis=1) + 1e-12)
                mean_env = float(np.mean(sub_env))
                mod_cv = float(np.std(sub_env)) / (mean_env + 1e-9)
                crest = float(np.max(sub_env)) / (mean_env + 1e-9)
            else:
                mod_cv = 0.0
                crest = 1.0

            window_records.append(
                (start_idx / float(sample_rate), end_idx / float(sample_rate), v_db, b_db, sm_ratio, sb_db, mod_cv, crest)
            )

        if not window_records:
            return []

        active_sm = [r[4] for r in window_records if r[2] > -50.0]
        active_mod = [r[6] for r in window_records if r[2] > -50.0]
        median_sm = float(np.median(active_sm)) if active_sm else 0.25
        median_mod = float(np.median(active_mod)) if active_mod else 0.20

        raw_candidates: List[VoiceCandidateEvent] = []
        for t_s, t_e, v_db, b_db, sm_ratio, sb_db, mod_cv, crest in window_records:
            # 無音に近い区間や、低域の風吹かれ乱流(sb_db)が声帯域より圧倒的に強い風切り音区間は除外
            if v_db < -46.0 or sb_db > v_db + 3.0:
                continue

            # パターンA: 曲間・弱奏時のアナウンス・話し声・咳払い（低音楽器が静かで、声帯域が突出し音節変動がある）
            is_quiet_speech = (
                b_db < -35.0
                and (v_db - b_db) >= 6.0
                and (mod_cv >= 0.26 or crest >= 1.75)
            )

            # パターンB: カメラ近傍での話し声・ささやき・咳払い（声帯域のステレオ近接偏り＋音節リズム/突発アタック）
            is_nearfield_chatter = (
                channels >= 2
                and sm_ratio >= max(0.32, median_sm * 1.35)
                and (mod_cv >= 0.30 or crest >= 1.85)
                and (v_db - b_db) >= -5.0
            )

            # パターンC: 強い音節変調・咳払いアタック（モノラル録音や正面近接の声にも対応）
            is_strong_syllabic = (
                mod_cv >= max(0.46, median_mod * 1.60)
                and (v_db - b_db) >= -2.0
                and (channels == 1 or sm_ratio >= median_sm * 1.05)
            )

            if is_quiet_speech or is_nearfield_chatter or is_strong_syllabic:
                if is_quiet_speech:
                    dtype = "曲間・弱奏時の声候補"
                    reason = f"静寂時の声帯域突出(+{v_db - b_db:.1f}dB) / 音節変動:{mod_cv:.2f}"
                    conf = min(0.95, 0.55 + (v_db - b_db) * 0.02 + mod_cv * 0.3)
                elif is_nearfield_chatter:
                    dtype = "近接話し声・咳払い候補"
                    reason = f"近接ステレオ偏り(S/M比:{sm_ratio:.2f}) / 音節変動:{mod_cv:.2f}"
                    conf = min(0.95, 0.50 + (sm_ratio / max(0.1, median_sm) - 1.0) * 0.25 + mod_cv * 0.3)
                else:
                    dtype = "話し声・咳払い候補"
                    reason = f"音節リズム・突発変動(CV:{mod_cv:.2f}, Peak比:{crest:.1f})"
                    conf = min(0.90, 0.45 + mod_cv * 0.4)

                raw_candidates.append(
                    VoiceCandidateEvent(
                        start_sec=max(0.0, t_s - 0.05),
                        end_sec=min(duration_sec, t_e + 0.05),
                        voice_db=v_db,
                        confidence=float(np.clip(conf, 0.30, 0.98)),
                        detection_type=dtype,
                        reason=reason,
                    )
                )

        return cls._merge_voice_events(raw_candidates, merge_gap_sec=0.6)

    @classmethod
    def scan_voice_candidates_ai(
        cls,
        audio: np.ndarray,
        sample_rate: int,
        vocal_stem: np.ndarray,
    ) -> List[VoiceCandidateEvent]:
        """
        AI (Demucs) が分離した `vocal_stem` の短時間エネルギーと対全体比率、および音節変調度を解析し、
        大音量の吹奏楽演奏中に重なって混入した話し声・掛け声・アナウンスも含めて高精度に検出する。
        """
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        if vocal_stem.ndim == 1:
            vocal_stem = vocal_stem[np.newaxis, :]

        total_samples = min(audio.shape[1], vocal_stem.shape[1])
        win = int(0.5 * sample_rate)
        hop = max(1, int(0.25 * sample_rate))
        if total_samples < win or sample_rate <= 0:
            return []

        duration_sec = total_samples / float(sample_rate)
        nyquist = 0.5 * sample_rate

        # 人の声の基音〜フォルマント帯域 (180Hz 〜 5200Hz) に絞って評価
        v_hi = min(5200.0, nyquist * 0.90)
        if v_hi > 200.0:
            sos_v = signal.butter(3, [180.0 / nyquist, v_hi / nyquist], btype="bandpass", output="sos")
            v_filt = signal.sosfiltfilt(sos_v, vocal_stem[:, :total_samples], axis=1)
        else:
            v_filt = vocal_stem[:, :total_samples]

        v_sq = np.mean(np.square(v_filt, dtype=np.float32), axis=0)
        tot_sq = np.mean(np.square(audio[:, :total_samples], dtype=np.float32), axis=0)
        sub_len = max(1, int(0.02 * sample_rate))

        raw_candidates: List[VoiceCandidateEvent] = []
        for start_idx in range(0, total_samples - win + 1, hop):
            end_idx = start_idx + win
            v_rms = float(np.sqrt(np.mean(v_sq[start_idx:end_idx], dtype=np.float64) + 1e-12))
            v_db = 20.0 * np.log10(v_rms)
            if v_db < -46.0:
                continue

            t_rms = float(np.sqrt(np.mean(tot_sq[start_idx:end_idx], dtype=np.float64) + 1e-12))
            v_ratio = v_rms / (t_rms + 1e-9)

            n_subs = win // sub_len
            if n_subs >= 4:
                trimmed = v_sq[start_idx : start_idx + n_subs * sub_len].reshape(n_subs, sub_len)
                sub_env = np.sqrt(np.mean(trimmed, axis=1) + 1e-12)
                mod_cv = float(np.std(sub_env)) / (float(np.mean(sub_env)) + 1e-9)
            else:
                mod_cv = 0.0

            # AI Vocal成分が明確に存在し、管楽器の持続音漏れではない（比率が高い、または音節変動がある）区間を検出
            if v_ratio >= 0.18 or (v_ratio >= 0.10 and mod_cv >= 0.30):
                conf = float(np.clip(0.50 + v_ratio * 0.8 + mod_cv * 0.25, 0.45, 0.99))
                raw_candidates.append(
                    VoiceCandidateEvent(
                        start_sec=max(0.0, start_idx / float(sample_rate) - 0.05),
                        end_sec=min(duration_sec, end_idx / float(sample_rate) + 0.05),
                        voice_db=v_db,
                        confidence=conf,
                        detection_type="AI声成分検出(Demucs)",
                        reason=f"AI声成分レベル:{v_db:.1f}dB / 声成分比率:{v_ratio * 100:.0f}%",
                    )
                )

        return cls._merge_voice_events(raw_candidates, merge_gap_sec=0.6)

    @staticmethod
    def _merge_voice_events(
        events: List[VoiceCandidateEvent],
        merge_gap_sec: float = 0.6,
    ) -> List[VoiceCandidateEvent]:
        """連続または近接する声候補区間を1つのイベントにマージする"""
        if not events:
            return []

        merged: List[VoiceCandidateEvent] = []
        for ev in sorted(events, key=lambda x: x.start_sec):
            if merged and (ev.start_sec - merged[-1].end_sec) <= merge_gap_sec:
                prev = merged[-1]
                prev.end_sec = max(prev.end_sec, ev.end_sec)
                if ev.confidence > prev.confidence:
                    prev.confidence = ev.confidence
                    prev.detection_type = ev.detection_type
                    prev.reason = ev.reason
                prev.voice_db = max(prev.voice_db, ev.voice_db)
            else:
                merged.append(
                    VoiceCandidateEvent(
                        start_sec=ev.start_sec,
                        end_sec=ev.end_sec,
                        voice_db=ev.voice_db,
                        confidence=ev.confidence,
                        detection_type=ev.detection_type,
                        reason=ev.reason,
                    )
                )
        return merged

