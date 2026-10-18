"""
SoundRemover メインGUIウィンドウ (`main_window.py`)

動画・音声の読み込み（ドラッグ＆ドロップ対応）、完全ロスレス抽出、
音響・周波数特性＆風切り音自動診断アシスタント、スマート風切り音除去（M/S乱流サプレッサー）、
AIピンポイント声除去、ホワイトノイズ除去（ノイズ見本学習）、屋外風切り音ローカット＆吹奏楽4バンドEQ、
ラウドネスノーマライズ・リミッター、曲分割一括書き出し、A/B聴き比べを1画面に統合する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from sound_remover.core.audio_data import (
    AudioSession,
    ProcessingConfig,
    SongChapter,
    TimeRegion,
)
from sound_remover.core.device_manager import DeviceManager
from sound_remover.core.pipeline import AudioPipeline
from sound_remover.processors.analyzer import (
    AudioAnalyzer,
    AudioDiagnosisReport,
    VoiceCandidateEvent,
)
from sound_remover.processors.extractor import LosslessExtractor
from sound_remover.processors.mastering import MasteringProcessor
from sound_remover.ui.player_widget import AudioPreviewPlayer
from sound_remover.ui.theme import STUDIO_DARK_STYLESHEET
from sound_remover.ui.waveform_widget import WaveformWidget, format_time_mmss


class PipelineWorker(QThread):
    """GUIをフリーズさせずにバックグラウンドでAI・音声処理を実行するワーカースレッド"""

    progress_updated = Signal(int, str)
    finished_success = Signal(object)  # np.ndarray
    failed_error = Signal(str)

    def __init__(
        self,
        pipeline: AudioPipeline,
        session: AudioSession,
        config: ProcessingConfig,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.pipeline = pipeline
        self.session = session
        self.config = config

    def run(self) -> None:
        try:
            result = self.pipeline.run_pipeline(
                session=self.session,
                config=self.config,
                progress_callback=lambda pct, msg: self.progress_updated.emit(pct, msg),
            )
            self.finished_success.emit(result)
        except Exception as exc:
            self.failed_error.emit(str(exc))


class VoiceScanWorker(QThread):
    """バックグラウンドで AI (Demucs) による人の声・話し声混入区間スキャンを実行するワーカースレッド"""

    progress_updated = Signal(int, str)
    finished_success = Signal(object, float)  # (List[VoiceCandidateEvent], offset_sec)
    failed_error = Signal(str)

    def __init__(
        self,
        pipeline: AudioPipeline,
        audio: np.ndarray,
        sample_rate: int,
        offset_sec: float = 0.0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.pipeline = pipeline
        self.audio = audio
        self.sample_rate = sample_rate
        self.offset_sec = offset_sec

    def run(self) -> None:
        try:
            dev = self.pipeline.device_manager.get_selected_device()
            events = self.pipeline.vocal_remover.scan_vocal_events(
                audio=self.audio,
                sample_rate=self.sample_rate,
                device=dev,
                progress_callback=lambda pct, msg: self.progress_updated.emit(pct, msg),
            )
            self.finished_success.emit(events, self.offset_sec)
        except Exception as exc:
            self.failed_error.emit(str(exc))


class MainWindow(QMainWindow):
    """SoundRemover メインアプリケーションウィンドウ"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(
            "SoundRemover - 吹奏楽・発表会動画 ロスレス音声抽出＆ピンポイント声・風・ノイズ除去スタジオ"
        )
        self.resize(1420, 880)
        self.setAcceptDrops(True)
        self.setStyleSheet(STUDIO_DARK_STYLESHEET)

        self.device_manager = DeviceManager()
        self.pipeline = AudioPipeline(device_manager=self.device_manager)
        self.session: Optional[AudioSession] = None
        self.latest_diagnosis: Optional[AudioDiagnosisReport] = None
        self._worker: Optional[PipelineWorker] = None
        self._voice_scan_worker: Optional[VoiceScanWorker] = None

        self._setup_ui()
        self._apply_config_to_ui(ProcessingConfig.preset_outdoor_stage())


    def _setup_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        # 1. 上部ツールバー（ファイル開く・完全ロスレス抽出・音響診断・プリセット・演算デバイス選択）
        top_bar = QHBoxLayout()

        self.btn_open_file = QPushButton("📂 動画・音声ファイルを開く...")
        self.btn_open_file.setObjectName("primaryButton")
        self.btn_open_file.clicked.connect(self.on_open_file_dialog)

        self.btn_lossless_extract = QPushButton("⚡ 完全ロスレス抽出（再エンコードなし）")
        self.btn_lossless_extract.setObjectName("successButton")
        self.btn_lossless_extract.setToolTip(
            "動画内の元音声ストリームを一切再圧縮・加工せずに100%無劣化のまま別ファイルとして即座に抜き出します"
        )
        self.btn_lossless_extract.setEnabled(False)
        self.btn_lossless_extract.clicked.connect(self.on_direct_lossless_extract)

        self.btn_diagnose_quick = QPushButton("🔍 周波数・風切り音自動診断")
        self.btn_diagnose_quick.setObjectName("accentButton")
        self.btn_diagnose_quick.setEnabled(False)
        self.btn_diagnose_quick.clicked.connect(self.on_quick_diagnose_clicked)

        top_bar.addWidget(self.btn_open_file)
        top_bar.addWidget(self.btn_lossless_extract)
        top_bar.addWidget(self.btn_diagnose_quick)
        top_bar.addSpacing(12)

        lbl_preset = QLabel("🎵 プリセット:")
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(
            [
                "屋外ステージ標準（スマート風切り音除去＋声除去＋低音補強＋音量最適化）",
                "屋内ホール・体育館標準（空調ホワイトノイズ除去＋こもり解消EQ＋音量最適化）",
                "話し声ピンポイント除去のみ（他は100%無加工維持）",
                "ホワイトノイズ・空調音除去のみ",
            ]
        )
        self.combo_preset.currentIndexChanged.connect(self.on_preset_changed)

        top_bar.addWidget(lbl_preset)
        top_bar.addWidget(self.combo_preset)
        top_bar.addStretch()

        lbl_dev = QLabel("🖥 演算デバイス:")
        self.combo_device = QComboBox()
        self._populate_device_combo()
        self.combo_device.currentIndexChanged.connect(self.on_device_changed)

        top_bar.addWidget(lbl_dev)
        top_bar.addWidget(self.combo_device)
        root_layout.addLayout(top_bar)

        # 2. 中央メインスプリッター（左: 波形＆A/Bプレイヤー / 右: 機能設定タブ）
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左ペイン
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(8)

        self.lbl_file_info = QLabel(
            "💡 動画ファイル (.mp4, .mov, .mts, .mkv) または音声ファイルをここにドラッグ＆ドロップするか、「動画・音声ファイルを開く」を押してください。"
        )
        self.lbl_file_info.setStyleSheet(
            "background-color: #1f2430; padding: 8px 12px; border-radius: 5px; color: #9ecbff;"
        )
        left_layout.addWidget(self.lbl_file_info)

        self.waveform_widget = WaveformWidget()
        self.waveform_widget.region_changed.connect(self.on_waveform_region_changed)
        self.waveform_widget.seek_requested.connect(self.on_waveform_seek_requested)
        self.waveform_widget.add_pinpoint_requested.connect(self.on_add_pinpoint_region)
        self.waveform_widget.set_noise_sample_requested.connect(self.on_set_noise_sample_region)
        self.waveform_widget.add_chapter_requested.connect(self.on_add_song_chapter)
        left_layout.addWidget(self.waveform_widget, stretch=1)

        self.player_widget = AudioPreviewPlayer()
        self.player_widget.position_sec_changed.connect(self.waveform_widget.set_playhead_sec)
        self.player_widget.ab_mode_changed.connect(self.waveform_widget.set_display_mode)
        left_layout.addWidget(self.player_widget)

        splitter.addWidget(left_pane)

        # 右ペイン（設定タブ＋実行・保存ボタン）
        right_pane = QWidget()
        right_pane.setMinimumWidth(480)
        right_pane.setMaximumWidth(580)
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(8)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_tab_diagnostics(), "🔍 音響診断・推奨")
        self.tabs.addTab(self._build_tab_vocal_removal(), "🗣 声・ピンポイント")
        self.tabs.addTab(self._build_tab_noise_and_eq(), "🌬 風・ノイズ・EQ")
        self.tabs.addTab(self._build_tab_mastering(), "🎚 音量・マスタリング")
        self.tabs.addTab(self._build_tab_chapters(), "✂️ 曲分割・出力")
        right_layout.addWidget(self.tabs, stretch=1)

        # 下部アクションボタン（処理実行＆ファイル書き出し）
        exec_group = QGroupBox("🚀 加工プレビュー生成・ファイル書き出し")
        exec_layout = QVBoxLayout(exec_group)

        self.btn_run_pipeline = QPushButton("✨ 音声加工を実行（A/B聴き比べプレビュー生成）")
        self.btn_run_pipeline.setObjectName("primaryButton")
        self.btn_run_pipeline.setEnabled(False)
        self.btn_run_pipeline.clicked.connect(self.on_run_pipeline)

        export_row = QHBoxLayout()
        self.combo_export_format = QComboBox()
        self.combo_export_format.addItem("24-bit WAV (非圧縮・吹奏楽マスター推奨)", "wav24")
        self.combo_export_format.addItem("FLAC (ロスレス可逆圧縮・高音質保存用)", "flac")
        self.combo_export_format.addItem("16-bit WAV (CD標準フォーマット)", "wav16")
        self.combo_export_format.addItem("AAC / M4A (320kbps・スマホ配布用)", "m4a")
        self.combo_export_format.addItem("MP3 (320kbps・汎用)", "mp3")

        self.btn_export_audio = QPushButton("💾 音声をファイルに保存...")
        self.btn_export_audio.setObjectName("successButton")
        self.btn_export_audio.setEnabled(False)
        self.btn_export_audio.clicked.connect(self.on_export_full_audio)

        export_row.addWidget(self.combo_export_format, stretch=1)
        export_row.addWidget(self.btn_export_audio)

        exec_layout.addWidget(self.btn_run_pipeline)
        exec_layout.addLayout(export_row)
        right_layout.addWidget(exec_group)

        splitter.addWidget(right_pane)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root_layout.addWidget(splitter, stretch=1)

        # 3. 下部ステータスバー＆プログレスバー
        status_row = QHBoxLayout()
        self.lbl_status = QLabel("準備完了")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(320)

        status_row.addWidget(self.lbl_status, stretch=1)
        status_row.addWidget(self.progress_bar)
        root_layout.addLayout(status_row)

    def _populate_device_combo(self) -> None:
        self.combo_device.clear()
        devices = self.device_manager.refresh_devices()
        best = devices[0]
        self.combo_device.addItem(f"自動選択 (推奨: {best.display_name})", "auto")
        for dev in devices:
            self.combo_device.addItem(dev.display_name, dev.device_id)

    def _build_tab_diagnostics(self) -> QWidget:
        """🔍 音響・周波数特性＆風切り音自動診断アシスタントタブ"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(6)

        btn_row = QHBoxLayout()
        self.btn_diag_full = QPushButton("🔄 トラック全体を診断")
        self.btn_diag_sel = QPushButton("🎯 現在の選択範囲だけを診断")
        self.btn_diag_full.clicked.connect(lambda: self.run_audio_diagnosis(use_selection=False))
        self.btn_diag_sel.clicked.connect(lambda: self.run_audio_diagnosis(use_selection=True))
        btn_row.addWidget(self.btn_diag_full)
        btn_row.addWidget(self.btn_diag_sel)
        layout.addLayout(btn_row)

        # 1. 5帯域周波数バランス＆ステレオ相関
        grp_bands = QGroupBox("📊 周波数帯域別エネルギー＆ステレオ位相相関 (L/R Corr)")
        b_layout = QVBoxLayout(grp_bands)
        self.list_diag_bands = QListWidget()
        self.list_diag_bands.setMaximumHeight(105)
        b_layout.addWidget(self.list_diag_bands)
        layout.addWidget(grp_bands)

        # 2. 検出された風切り音（マイク吹かれ）区間一覧
        grp_wind = QGroupBox("🌪 検出された風切り音（マイク吹かれ）区間（ダブルクリックでその秒数へジャンプ）")
        w_layout = QVBoxLayout(grp_wind)
        self.lbl_wind_summary = QLabel("ファイルを開くと自動で風切り音や周波数バランスを診断します。")
        self.lbl_wind_summary.setStyleSheet("color: #fbbf24; font-weight: bold;")
        w_layout.addWidget(self.lbl_wind_summary)

        self.list_diag_wind = QListWidget()
        self.list_diag_wind.setMaximumHeight(88)
        self.list_diag_wind.itemDoubleClicked.connect(self.on_wind_event_double_clicked)
        w_layout.addWidget(self.list_diag_wind)

        self.btn_add_wind_to_pinpoint = QPushButton("➕ 検出された風切り音区間をピンポイント除去リストへ一括追加")
        self.btn_add_wind_to_pinpoint.clicked.connect(self.on_add_wind_events_to_pinpoint)
        w_layout.addWidget(self.btn_add_wind_to_pinpoint)
        layout.addWidget(grp_wind)

        # 3. 検出された人の声・話し声・咳払い・曲間アナウンス候補区間一覧
        grp_voice = QGroupBox("🗣 検出された人の声・咳払い・近接雑音候補（ダブルクリックでジャンプ＆試聴）")
        v_layout = QVBoxLayout(grp_voice)
        self.lbl_voice_summary = QLabel("ファイルを開くと自動で近接話し声・咳払い・曲間声の候補をスキャンします。")
        self.lbl_voice_summary.setStyleSheet("color: #38bdf8; font-weight: bold;")
        v_layout.addWidget(self.lbl_voice_summary)

        self.list_diag_voice = QListWidget()
        self.list_diag_voice.setMaximumHeight(98)
        self.list_diag_voice.itemDoubleClicked.connect(self.on_voice_event_double_clicked)
        v_layout.addWidget(self.list_diag_voice)

        voice_btn_row = QHBoxLayout()
        self.btn_scan_voice_ai = QPushButton("🤖 AI (Demucs) で声区間を精密スキャン")
        self.btn_scan_voice_ai.setToolTip(
            "大音量の演奏中に紛れ込んだ話し声や掛け声までAIで高精度に探します（GPUで数秒〜十数秒）"
        )
        self.btn_scan_voice_ai.clicked.connect(self.on_run_ai_voice_scan)

        self.btn_add_sel_voice_to_pinpoint = QPushButton("➕ 選択した声候補を追加")
        self.btn_add_sel_voice_to_pinpoint.clicked.connect(self.on_add_selected_voice_event_to_pinpoint)

        self.btn_add_all_voice_to_pinpoint = QPushButton("➕ 全声候補を一括追加")
        self.btn_add_all_voice_to_pinpoint.clicked.connect(self.on_add_all_voice_events_to_pinpoint)

        voice_btn_row.addWidget(self.btn_scan_voice_ai)
        voice_btn_row.addWidget(self.btn_add_sel_voice_to_pinpoint)
        voice_btn_row.addWidget(self.btn_add_all_voice_to_pinpoint)
        v_layout.addLayout(voice_btn_row)
        layout.addWidget(grp_voice)

        # 4. 診断アドバイス＆ワンクリック推奨設定反映
        grp_rec = QGroupBox("💡 診断結果からのフィルタ設定アドバイス")
        r_layout = QVBoxLayout(grp_rec)
        self.txt_diag_recommendations = QTextEdit()
        self.txt_diag_recommendations.setReadOnly(True)
        self.txt_diag_recommendations.setStyleSheet("background-color: #14161b; border: 1px solid #2d323e;")
        r_layout.addWidget(self.txt_diag_recommendations, stretch=1)

        self.btn_apply_recommended = QPushButton("⚡ 診断結果の推奨フィルタ設定をワンクリックで反映する")
        self.btn_apply_recommended.setObjectName("accentButton")
        self.btn_apply_recommended.setEnabled(False)
        self.btn_apply_recommended.clicked.connect(self.on_apply_recommended_diagnosis)
        r_layout.addWidget(self.btn_apply_recommended)
        layout.addWidget(grp_rec, stretch=1)

        return tab

    def _build_tab_vocal_removal(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        grp_ai = QGroupBox("🤖 AI音源分離 (Demucs) による話し声・雑音除去")
        form = QFormLayout(grp_ai)

        self.chk_vocal_enable = QCheckBox("AI声除去（話し声・アナウンス・咳払い除去）を有効にする")
        self.chk_vocal_enable.setChecked(True)
        form.addRow(self.chk_vocal_enable)

        self.combo_vocal_scope = QComboBox()
        self.combo_vocal_scope.addItem("トラック全体に適用する", False)
        self.combo_vocal_scope.addItem("登録したピンポイント区間のみに適用（他は100%無加工維持）", True)
        form.addRow("適用範囲:", self.combo_vocal_scope)

        ratio_row = QHBoxLayout()
        self.slider_vocal_ratio = QSlider(Qt.Orientation.Horizontal)
        self.slider_vocal_ratio.setRange(0, 100)
        self.slider_vocal_ratio.setValue(85)
        self.lbl_vocal_ratio_val = QLabel("85%")
        self.lbl_vocal_ratio_val.setFixedWidth(45)
        self.slider_vocal_ratio.valueChanged.connect(
            lambda v: self.lbl_vocal_ratio_val.setText(f"{v}%")
        )
        ratio_row.addWidget(self.slider_vocal_ratio)
        ratio_row.addWidget(self.lbl_vocal_ratio_val)
        form.addRow("声の除去強度:", ratio_row)

        self.chk_vocal_band_protect = QCheckBox(
            "楽器保護帯域制限（110Hz以下の重低音・6.5kHz以上の超高域倍音を100%完全保護）"
        )
        self.chk_vocal_band_protect.setChecked(True)
        form.addRow(self.chk_vocal_band_protect)

        self.spin_crossfade_ms = QDoubleSpinBox()
        self.spin_crossfade_ms.setRange(10.0, 500.0)
        self.spin_crossfade_ms.setValue(60.0)
        self.spin_crossfade_ms.setSuffix(" ms")
        form.addRow("区間境界クロスフェード:", self.spin_crossfade_ms)

        layout.addWidget(grp_ai)

        # ピンポイント適用区間リスト管理
        grp_regions = QGroupBox("📋 ピンポイント除去対象区間リスト（波形下の「➕」ボタンまたは自動検出で追加）")
        reg_layout = QVBoxLayout(grp_regions)

        lbl_hint = QLabel(
            "※観客の話し声や咳払いが入った数秒間だけをピンポイント登録すると、\n"
            "　その時間帯だけ声を除去し、他の演奏区間を100%無劣化に保てます。"
        )
        lbl_hint.setStyleSheet("color: #9aa2b1; font-size: 12px;")
        reg_layout.addWidget(lbl_hint)

        btn_jump_voice_scan = QPushButton("🔍 声・咳払い混入候補の自動検出リストを見る / AI精密スキャン（診断タブへ）")
        btn_jump_voice_scan.setObjectName("accentButton")
        btn_jump_voice_scan.clicked.connect(lambda: self.tabs.setCurrentIndex(0))
        reg_layout.addWidget(btn_jump_voice_scan)

        self.list_pinpoint_regions = QListWidget()
        self.list_pinpoint_regions.itemDoubleClicked.connect(self.on_pinpoint_item_double_clicked)
        reg_layout.addWidget(self.list_pinpoint_regions)

        reg_btn_row = QHBoxLayout()
        btn_del_reg = QPushButton("🗑 選択した区間を削除")
        btn_clear_reg = QPushButton("すべてクリア")
        btn_del_reg.clicked.connect(self.on_delete_selected_pinpoint)
        btn_clear_reg.clicked.connect(self.on_clear_pinpoint_regions)
        reg_btn_row.addWidget(btn_del_reg)
        reg_btn_row.addWidget(btn_clear_reg)
        reg_layout.addLayout(reg_btn_row)

        layout.addWidget(grp_regions, stretch=1)
        return tab


    def _build_tab_noise_and_eq(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(6)

        # 1. スマート風切り音除去（屋外マイク吹かれサプレッサー）
        grp_wind = QGroupBox("🌪 スマート風切り音除去（屋外マイク吹かれ・ボフボフ音サプレッサー）")
        wind_form = QFormLayout(grp_wind)

        self.chk_wind_enable = QCheckBox(
            "スマート風切り音除去を有効にする（チューバ・ベースの低音を保ち、風の乱流だけ除去）"
        )
        self.chk_wind_enable.setChecked(True)
        wind_form.addRow(self.chk_wind_enable)

        wind_str_row = QHBoxLayout()
        self.slider_wind_strength = QSlider(Qt.Orientation.Horizontal)
        self.slider_wind_strength.setRange(0, 100)
        self.slider_wind_strength.setValue(85)
        self.lbl_wind_strength_val = QLabel("85%")
        self.lbl_wind_strength_val.setFixedWidth(45)
        self.slider_wind_strength.valueChanged.connect(
            lambda v: self.lbl_wind_strength_val.setText(f"{v}%")
        )
        wind_str_row.addWidget(self.slider_wind_strength)
        wind_str_row.addWidget(self.lbl_wind_strength_val)
        wind_form.addRow("風切り音抑制強度:", wind_str_row)

        self.spin_wind_cutoff = QDoubleSpinBox()
        self.spin_wind_cutoff.setRange(120.0, 360.0)
        self.spin_wind_cutoff.setSingleStep(10.0)
        self.spin_wind_cutoff.setValue(260.0)
        self.spin_wind_cutoff.setSuffix(" Hz (風乱流監視・M/S抑制上限)")
        wind_form.addRow("風乱流監視上限周波数:", self.spin_wind_cutoff)

        layout.addWidget(grp_wind)

        # 2. ホワイトノイズ・空調音除去
        grp_nr = QGroupBox("🌫 ホワイトノイズ・空調音除去 (Spectral Gating)")
        nr_form = QFormLayout(grp_nr)

        self.chk_nr_enable = QCheckBox("ホワイトノイズ・定常ノイズ除去を有効にする")
        self.chk_nr_enable.setChecked(True)
        nr_form.addRow(self.chk_nr_enable)

        self.combo_nr_scope = QComboBox()
        self.combo_nr_scope.addItem("トラック全体に適用する", False)
        self.combo_nr_scope.addItem("登録したピンポイント区間のみに適用", True)
        nr_form.addRow("適用範囲:", self.combo_nr_scope)

        nr_str_row = QHBoxLayout()
        self.slider_nr_strength = QSlider(Qt.Orientation.Horizontal)
        self.slider_nr_strength.setRange(0, 100)
        self.slider_nr_strength.setValue(75)
        self.lbl_nr_strength_val = QLabel("75%")
        self.lbl_nr_strength_val.setFixedWidth(45)
        self.slider_nr_strength.valueChanged.connect(
            lambda v: self.lbl_nr_strength_val.setText(f"{v}%")
        )
        nr_str_row.addWidget(self.slider_nr_strength)
        nr_str_row.addWidget(self.lbl_nr_strength_val)
        nr_form.addRow("ノイズ減衰強度:", nr_str_row)

        noise_prof_row = QHBoxLayout()
        self.lbl_noise_profile = QLabel("未登録（自動推定モード）")
        self.lbl_noise_profile.setStyleSheet("color: #c084fc;")
        btn_clear_np = QPushButton("クリア")
        btn_clear_np.clicked.connect(self.on_clear_noise_sample)
        noise_prof_row.addWidget(self.lbl_noise_profile, stretch=1)
        noise_prof_row.addWidget(btn_clear_np)
        nr_form.addRow("ノイズ見本区間:", noise_prof_row)

        layout.addWidget(grp_nr)

        # 3. 屋外超低域ローカット＆吹奏楽4バンドEQ
        grp_eq = QGroupBox("🎺 超低域ローカット＆吹奏楽向け4バンドEQ")
        eq_form = QFormLayout(grp_eq)

        self.chk_hp_enable = QCheckBox("超低域振動ローカット (High-Pass) を有効にする")
        self.chk_hp_enable.setChecked(True)
        eq_form.addRow(self.chk_hp_enable)

        self.spin_hp_cutoff = QDoubleSpinBox()
        self.spin_hp_cutoff.setRange(25.0, 150.0)
        self.spin_hp_cutoff.setValue(45.0)
        self.spin_hp_cutoff.setSuffix(" Hz")
        eq_form.addRow("ローカット周波数:", self.spin_hp_cutoff)

        self.chk_eq_enable = QCheckBox("吹奏楽トーン補正イコライザー (4-Band EQ) を有効にする")
        self.chk_eq_enable.setChecked(True)
        eq_form.addRow(self.chk_eq_enable)

        self.spin_eq_low = self._create_db_spinbox(2.0)
        self.spin_eq_low_mid = self._create_db_spinbox(-1.0)
        self.spin_eq_high_mid = self._create_db_spinbox(1.0)
        self.spin_eq_high = self._create_db_spinbox(1.5)

        eq_form.addRow("Low 80Hz (チューバ・低音楽器の厚み):", self.spin_eq_low)
        eq_form.addRow("Low-Mid 280Hz (ホール・体育館のこもり軽減):", self.spin_eq_low_mid)
        eq_form.addRow("High-Mid 2.5kHz (木管・金管メロディの明瞭度):", self.spin_eq_high_mid)
        eq_form.addRow("High 8.0kHz (金管の倍音・シンバルの空気感):", self.spin_eq_high)

        btn_reset_eq = QPushButton("EQをフラット (0.0 dB) にリセット")
        btn_reset_eq.clicked.connect(self._reset_eq_flat)
        eq_form.addRow(btn_reset_eq)

        layout.addWidget(grp_eq)
        layout.addStretch()
        return tab

    def _build_tab_mastering(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        grp_master = QGroupBox("🎚 吹奏楽向けラウドネスノーマライズ＆音割れ防止リミッター")
        form = QFormLayout(grp_master)

        lbl_desc = QLabel(
            "吹奏楽はピアニッシモ(極小音)からフォルティッシモ(大音量)まで音量差が広いため、\n"
            "音割れを防止するピークリミッターと組み合わせて全体の聴感音量を揃えます。"
        )
        lbl_desc.setStyleSheet("color: #9aa2b1; font-size: 12px;")
        form.addRow(lbl_desc)

        self.lbl_measured_lufs = QLabel("現在の音声ラウドネス: -- LUFS")
        self.lbl_measured_lufs.setStyleSheet("color: #38bdf8; font-weight: bold;")
        form.addRow(self.lbl_measured_lufs)

        self.chk_loudness_enable = QCheckBox("ラウドネスノーマライズ（聴感音量の自動最適化）を有効にする")
        self.chk_loudness_enable.setChecked(True)
        form.addRow(self.chk_loudness_enable)

        self.spin_target_lufs = QDoubleSpinBox()
        self.spin_target_lufs.setRange(-24.0, -9.0)
        self.spin_target_lufs.setSingleStep(0.5)
        self.spin_target_lufs.setValue(-15.0)
        self.spin_target_lufs.setSuffix(" LUFS (吹奏楽推奨: -16〜-14)")
        form.addRow("目標ラウドネス:", self.spin_target_lufs)

        self.chk_limiter_enable = QCheckBox("ルックアヘッド・ピークリミッター（強奏・打楽器の音割れ防止）")
        self.chk_limiter_enable.setChecked(True)
        form.addRow(self.chk_limiter_enable)

        self.spin_limiter_ceiling = QDoubleSpinBox()
        self.spin_limiter_ceiling.setRange(-6.0, -0.1)
        self.spin_limiter_ceiling.setSingleStep(0.1)
        self.spin_limiter_ceiling.setValue(-1.0)
        self.spin_limiter_ceiling.setSuffix(" dBFS")
        form.addRow("最大ピーク上限:", self.spin_limiter_ceiling)

        layout.addWidget(grp_master)
        layout.addStretch()
        return tab

    def _build_tab_chapters(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        grp_ch = QGroupBox("✂️ 曲ごとの分割書き出し（チャプター管理＆フェード処理）")
        ch_layout = QVBoxLayout(grp_ch)

        lbl_info = QLabel(
            "1本の長い撮影動画から曲ごとに範囲を選択し、波形下の「✂️曲分割チャプターに追加」で\n"
            "登録すると、各曲に自然なフェードイン・フェードアウトを付けて一括保存できます。"
        )
        lbl_info.setStyleSheet("color: #9aa2b1; font-size: 12px;")
        ch_layout.addWidget(lbl_info)

        fade_form = QFormLayout()
        self.spin_fade_in = QDoubleSpinBox()
        self.spin_fade_in.setRange(0.0, 15.0)
        self.spin_fade_in.setSingleStep(0.5)
        self.spin_fade_in.setValue(0.5)
        self.spin_fade_in.setSuffix(" 秒")

        self.spin_fade_out = QDoubleSpinBox()
        self.spin_fade_out.setRange(0.0, 20.0)
        self.spin_fade_out.setSingleStep(0.5)
        self.spin_fade_out.setValue(2.0)
        self.spin_fade_out.setSuffix(" 秒")

        fade_form.addRow("新規追加時のフェードイン時間:", self.spin_fade_in)
        fade_form.addRow("新規追加時のフェードアウト時間:", self.spin_fade_out)
        ch_layout.addLayout(fade_form)

        self.list_chapters = QListWidget()
        self.list_chapters.itemDoubleClicked.connect(self.on_chapter_item_double_clicked)
        ch_layout.addWidget(self.list_chapters, stretch=1)

        ch_btn_row = QHBoxLayout()
        btn_del_ch = QPushButton("🗑 選択曲を削除")
        btn_clear_ch = QPushButton("すべてクリア")
        btn_del_ch.clicked.connect(self.on_delete_selected_chapter)
        btn_clear_ch.clicked.connect(self.on_clear_chapters)
        ch_btn_row.addWidget(btn_del_ch)
        ch_btn_row.addWidget(btn_clear_ch)
        ch_layout.addLayout(ch_btn_row)

        self.btn_export_chapters = QPushButton("📀 登録したすべての曲を分割して一括書き出し...")
        self.btn_export_chapters.setObjectName("successButton")
        self.btn_export_chapters.clicked.connect(self.on_export_all_chapters)
        ch_layout.addWidget(self.btn_export_chapters)

        layout.addWidget(grp_ch, stretch=1)
        return tab

    @staticmethod
    def _create_db_spinbox(default_val: float) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(-12.0, 12.0)
        sb.setSingleStep(0.5)
        sb.setValue(default_val)
        sb.setSuffix(" dB")
        return sb

    def _reset_eq_flat(self) -> None:
        self.spin_eq_low.setValue(0.0)
        self.spin_eq_low_mid.setValue(0.0)
        self.spin_eq_high_mid.setValue(0.0)
        self.spin_eq_high.setValue(0.0)

    def _collect_config_from_ui(self) -> ProcessingConfig:
        """現在のUI上の設定値から `ProcessingConfig` インスタンスを生成する"""
        noise_prof = self.session.noise_sample_region if self.session else None
        return ProcessingConfig(
            vocal_removal_enabled=self.chk_vocal_enable.isChecked(),
            vocal_apply_only_regions=bool(self.combo_vocal_scope.currentData()),
            vocal_reduction_ratio=self.slider_vocal_ratio.value() / 100.0,
            vocal_band_protection=self.chk_vocal_band_protect.isChecked(),
            noise_reduction_enabled=self.chk_nr_enable.isChecked(),
            noise_apply_only_regions=bool(self.combo_nr_scope.currentData()),
            noise_reduction_strength=self.slider_nr_strength.value() / 100.0,
            noise_stationary_mode=True,
            noise_profile_region=noise_prof,
            highpass_enabled=self.chk_hp_enable.isChecked(),
            highpass_cutoff_hz=float(self.spin_hp_cutoff.value()),
            wind_reduction_enabled=self.chk_wind_enable.isChecked(),
            wind_reduction_strength=self.slider_wind_strength.value() / 100.0,
            wind_cutoff_hz=float(self.spin_wind_cutoff.value()),
            eq_enabled=self.chk_eq_enable.isChecked(),
            eq_low_db=float(self.spin_eq_low.value()),
            eq_low_mid_db=float(self.spin_eq_low_mid.value()),
            eq_high_mid_db=float(self.spin_eq_high_mid.value()),
            eq_high_db=float(self.spin_eq_high.value()),
            loudness_norm_enabled=self.chk_loudness_enable.isChecked(),
            target_lufs=float(self.spin_target_lufs.value()),
            limiter_enabled=self.chk_limiter_enable.isChecked(),
            limiter_ceiling_db=float(self.spin_limiter_ceiling.value()),
            region_crossfade_ms=float(self.spin_crossfade_ms.value()),
        )

    def _apply_config_to_ui(self, cfg: ProcessingConfig) -> None:
        """プリセット等の `ProcessingConfig` をUIコントロールへ反映する"""
        self.chk_vocal_enable.setChecked(cfg.vocal_removal_enabled)
        self.combo_vocal_scope.setCurrentIndex(1 if cfg.vocal_apply_only_regions else 0)
        self.slider_vocal_ratio.setValue(int(round(cfg.vocal_reduction_ratio * 100)))
        self.chk_vocal_band_protect.setChecked(cfg.vocal_band_protection)
        self.spin_crossfade_ms.setValue(cfg.region_crossfade_ms)

        self.chk_nr_enable.setChecked(cfg.noise_reduction_enabled)
        self.combo_nr_scope.setCurrentIndex(1 if cfg.noise_apply_only_regions else 0)
        self.slider_nr_strength.setValue(int(round(cfg.noise_reduction_strength * 100)))

        self.chk_wind_enable.setChecked(cfg.wind_reduction_enabled)
        self.slider_wind_strength.setValue(int(round(cfg.wind_reduction_strength * 100)))
        self.spin_wind_cutoff.setValue(cfg.wind_cutoff_hz)

        self.chk_hp_enable.setChecked(cfg.highpass_enabled)
        self.spin_hp_cutoff.setValue(cfg.highpass_cutoff_hz)
        self.chk_eq_enable.setChecked(cfg.eq_enabled)
        self.spin_eq_low.setValue(cfg.eq_low_db)
        self.spin_eq_low_mid.setValue(cfg.eq_low_mid_db)
        self.spin_eq_high_mid.setValue(cfg.eq_high_mid_db)
        self.spin_eq_high.setValue(cfg.eq_high_db)

        self.chk_loudness_enable.setChecked(cfg.loudness_norm_enabled)
        self.spin_target_lufs.setValue(cfg.target_lufs)
        self.chk_limiter_enable.setChecked(cfg.limiter_enabled)
        self.spin_limiter_ceiling.setValue(cfg.limiter_ceiling_db)

    # --- ドラッグ＆ドロップ対応 ---
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            local_path = Path(urls[0].toLocalFile())
            if local_path.is_file():
                self.load_media_file(local_path)

    # --- イベントハンドラ ---
    def on_open_file_dialog(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "動画または音声ファイルを選択",
            "",
            "動画・音声ファイル (*.mp4 *.mov *.mts *.m2ts *.mkv *.avi *.webm *.wav *.flac *.m4a *.aac *.mp3 *.ogg);;すべてのファイル (*.*)",
        )
        if file_path:
            self.load_media_file(Path(file_path))

    def load_media_file(self, path: Path) -> None:
        try:
            self.lbl_status.setText(f"読み込み中: {path.name} ...")
            self.progress_bar.setValue(15)
            from PySide6.QtWidgets import QApplication

            QApplication.processEvents()

            audio, sr, info = LosslessExtractor.load_audio(path)
            self.session = AudioSession(
                source_path=path,
                original_audio=audio,
                sample_rate=sr,
                source_codec_name=info.codec_name,
                is_video_source=info.is_video_container,
            )

            orig_lufs = MasteringProcessor.measure_lufs(audio, sr)
            self.lbl_measured_lufs.setText(f"現在の元音声ラウドネス: {orig_lufs:.1f} LUFS")

            kind_label = "🎬 動画ファイル" if info.is_video_container else "🎵 音声ファイル"
            self.lbl_file_info.setText(
                f"{kind_label}: {path.name}  |  元コーデック: {info.codec_name.upper()}  |  "
                f"{sr} Hz / {audio.shape[0]} ch  |  長さ: {format_time_mmss(self.session.duration_sec)}  |  "
                f"統合ラウドネス: {orig_lufs:.1f} LUFS"
            )

            self.waveform_widget.set_audio_data(audio, sr, processed_audio=None, show_processed=False)
            self.waveform_widget.refresh_overlays([], None, [])
            self.player_widget.load_session_audio(audio, sr, processed_audio=None)

            self.list_pinpoint_regions.clear()
            self.list_chapters.clear()
            self.lbl_noise_profile.setText("未登録（自動推定モード）")

            self.btn_lossless_extract.setEnabled(True)
            self.btn_diagnose_quick.setEnabled(True)
            self.btn_run_pipeline.setEnabled(True)
            self.btn_export_audio.setEnabled(True)

            # 自動的に音響・風切り音・周波数特性を診断して結果を表示する
            self.run_audio_diagnosis(use_selection=False)

            self.progress_bar.setValue(100)
            self.lbl_status.setText(
                f"読み込み＆自動音響診断が完了しました: {path.name} （右側の「🔍 音響診断・推奨」タブをご確認ください）"
            )
        except Exception as exc:
            self.progress_bar.setValue(0)
            self.lbl_status.setText("読み込みエラー")
            QMessageBox.critical(self, "ファイル読み込みエラー", f"ファイルの読み込みに失敗しました:\n{exc}")

    # --- 音響・周波数特性＆風切り音自動診断 ---
    def on_quick_diagnose_clicked(self) -> None:
        self.tabs.setCurrentIndex(0)
        use_sel = self.waveform_widget.has_selection
        self.run_audio_diagnosis(use_selection=use_sel)

    def run_audio_diagnosis(self, use_selection: bool = False) -> None:
        if not self.session:
            return

        sr = self.session.sample_rate
        full_audio = self.session.original_audio
        offset_sec = 0.0

        if use_selection and self.waveform_widget.has_selection:
            s_sec, e_sec = self.waveform_widget.get_selected_region()
            s_idx = max(0, min(int(s_sec * sr), full_audio.shape[1]))
            e_idx = max(s_idx, min(int(e_sec * sr), full_audio.shape[1]))
            if e_idx - s_idx >= int(0.5 * sr):
                target_audio = full_audio[:, s_idx:e_idx]
                offset_sec = s_sec
            else:
                target_audio = full_audio
        else:
            target_audio = full_audio

        report = AudioAnalyzer.analyze(
            audio=target_audio,
            sample_rate=sr,
            base_config=self._collect_config_from_ui(),
        )
        # 選択範囲診断の場合はオフセット秒数を加算
        if offset_sec > 0.0:
            for ev in report.wind_events:
                ev.start_sec += offset_sec
                ev.end_sec += offset_sec
            for vev in report.voice_events:
                vev.start_sec += offset_sec
                vev.end_sec += offset_sec
            if report.suggested_noise_sample_region is not None:
                ns, ne = report.suggested_noise_sample_region
                report.suggested_noise_sample_region = (ns + offset_sec, ne + offset_sec)
                report.recommended_config.noise_profile_region = report.suggested_noise_sample_region

        self.latest_diagnosis = report
        self._populate_diagnosis_ui(report, is_selection=(offset_sec > 0.0))

    def _populate_diagnosis_ui(self, report: AudioDiagnosisReport, is_selection: bool = False) -> None:
        self.list_diag_bands.clear()
        for b in report.band_stats:
            self.list_diag_bands.addItem(
                f"{b.name}: {b.rms_db:5.1f} dB  |  L/R相関: {b.lr_correlation:+.2f}  ➔  {b.comment}"
            )

        scope_str = "【選択範囲の診断】" if is_selection else "【トラック全体の診断】"
        self.lbl_wind_summary.setText(f"{scope_str} 風切り音（吹かれ）判定: {report.wind_severity_summary}")

        self.list_diag_wind.clear()
        if not report.wind_events:
            self.list_diag_wind.addItem("✅ 風切り音（マイク吹かれ）が顕著な区間はありませんでした。")
        else:
            for idx, ev in enumerate(report.wind_events, start=1):
                self.list_diag_wind.addItem(
                    f"{idx:02d}. {format_time_mmss(ev.start_sec)} 〜 {format_time_mmss(ev.end_sec)} "
                    f"[{ev.severity}] (風乱流レベル:{ev.peak_side_db:.1f}dB / 乱流比:{ev.max_side_mid_ratio:.2f})"
                )

        self._populate_voice_events_list(report, scope_str)

        self.txt_diag_recommendations.setPlainText("\n\n".join(report.recommendations_text))
        self.btn_apply_recommended.setEnabled(True)

    def _populate_voice_events_list(self, report: AudioDiagnosisReport, scope_str: str = "") -> None:
        prefix = f"{scope_str} " if scope_str else ""
        self.lbl_voice_summary.setText(f"{prefix}声・咳払い候補判定: {report.voice_summary}")
        self.list_diag_voice.clear()
        if not report.voice_events:
            self.list_diag_voice.addItem(
                "✅ 顕著な近接話し声・咳払い候補はありません（演奏中の声も探す場合は『🤖 AIで声区間を精密スキャン』を押してください）。"
            )
        else:
            for idx, vev in enumerate(report.voice_events, start=1):
                self.list_diag_voice.addItem(
                    f"{idx:02d}. {format_time_mmss(vev.start_sec)} 〜 {format_time_mmss(vev.end_sec)} "
                    f"[{vev.detection_type} / 確度:{vev.confidence * 100:.0f}%] ({vev.reason})"
                )

    def on_wind_event_double_clicked(self, item: QListWidgetItem) -> None:
        if not self.latest_diagnosis or not self.latest_diagnosis.wind_events:
            return
        row = self.list_diag_wind.row(item)
        if 0 <= row < len(self.latest_diagnosis.wind_events):
            ev = self.latest_diagnosis.wind_events[row]
            self.waveform_widget.set_selected_region(ev.start_sec, ev.end_sec)
            self.player_widget.seek_to_sec(ev.start_sec)
            self.lbl_status.setText(
                f"🌪 風切り音検出区間 ({format_time_mmss(ev.start_sec)} 〜 {format_time_mmss(ev.end_sec)}) を選択しました。再生ボタンで確認できます。"
            )

    def on_add_wind_events_to_pinpoint(self) -> None:
        if not self.session or not self.latest_diagnosis or not self.latest_diagnosis.wind_events:
            QMessageBox.information(self, "風切り音区間なし", "追加できる風切り音検出区間がありません。")
            return

        added = 0
        for ev in self.latest_diagnosis.wind_events:
            self.session.pinpoint_regions.append(
                TimeRegion(
                    start_sec=ev.start_sec,
                    end_sec=ev.end_sec,
                    label=f"風切り音[{ev.severity}]",
                )
            )
            added += 1

        self._refresh_region_lists_and_overlays()
        QMessageBox.information(
            self,
            "ピンポイント区間への追加完了",
            f"検出された {added} 箇所の風切り音区間をピンポイント除去リストへ登録しました。",
        )

    def on_voice_event_double_clicked(self, item: QListWidgetItem) -> None:
        if not self.latest_diagnosis or not self.latest_diagnosis.voice_events:
            return
        row = self.list_diag_voice.row(item)
        if 0 <= row < len(self.latest_diagnosis.voice_events):
            vev = self.latest_diagnosis.voice_events[row]
            self.waveform_widget.set_selected_region(vev.start_sec, vev.end_sec)
            self.player_widget.seek_to_sec(vev.start_sec)
            self.lbl_status.setText(
                f"🗣 声・咳払い候補区間 ({format_time_mmss(vev.start_sec)} 〜 {format_time_mmss(vev.end_sec)}) を選択しました。再生ボタンで実際の音を確認できます。"
            )

    def on_run_ai_voice_scan(self) -> None:
        if not self.session:
            return
        if self._voice_scan_worker is not None and self._voice_scan_worker.isRunning():
            return

        sr = self.session.sample_rate
        full_audio = self.session.original_audio
        offset_sec = 0.0

        if self.waveform_widget.has_selection:
            s_sec, e_sec = self.waveform_widget.get_selected_region()
            if e_sec - s_sec >= 1.0:
                reply = QMessageBox.question(
                    self,
                    "AI声スキャンの範囲確認",
                    f"現在波形上で選択されている範囲 ({format_time_mmss(s_sec)} 〜 {format_time_mmss(e_sec)}) だけをAIスキャンしますか？\n"
                    "（「No」を選ぶとトラック全体をAIスキャンします）",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    s_idx = max(0, min(int(s_sec * sr), full_audio.shape[1]))
                    e_idx = max(s_idx, min(int(e_sec * sr), full_audio.shape[1]))
                    target_audio = full_audio[:, s_idx:e_idx]
                    offset_sec = s_sec
                else:
                    target_audio = full_audio
            else:
                target_audio = full_audio
        else:
            target_audio = full_audio

        self.btn_scan_voice_ai.setEnabled(False)
        self.btn_run_pipeline.setEnabled(False)
        self.progress_bar.setValue(10)
        self.lbl_status.setText("🤖 AI (Demucs) で人の声・話し声混入区間を精密スキャン中...")

        self._voice_scan_worker = VoiceScanWorker(
            pipeline=self.pipeline,
            audio=target_audio,
            sample_rate=sr,
            offset_sec=offset_sec,
            parent=self,
        )
        self._voice_scan_worker.progress_updated.connect(self._on_worker_progress)
        self._voice_scan_worker.finished_success.connect(self._on_voice_scan_finished)
        self._voice_scan_worker.failed_error.connect(self._on_voice_scan_failed)
        self._voice_scan_worker.start()

    def _on_voice_scan_finished(self, ai_events: list, offset_sec: float) -> None:
        self.btn_scan_voice_ai.setEnabled(True)
        self.btn_run_pipeline.setEnabled(True)
        self.progress_bar.setValue(100)

        if not self.latest_diagnosis:
            return

        for ev in ai_events:
            ev.start_sec += offset_sec
            ev.end_sec += offset_sec

        # 既存の高速スキャン候補とAIスキャン候補を統合してマージする
        combined = list(self.latest_diagnosis.voice_events) + list(ai_events)
        merged = AudioAnalyzer._merge_voice_events(combined, merge_gap_sec=0.6)
        self.latest_diagnosis.voice_events = merged
        total_sec = sum(e.end_sec - e.start_sec for e in merged)
        if merged:
            self.latest_diagnosis.voice_summary = (
                f"AI精密スキャン完了：計 {len(merged)} 区間（合計 {total_sec:.1f} 秒の声・咳払い候補を検出）"
            )
        else:
            self.latest_diagnosis.voice_summary = "AI精密スキャン完了：人の声の混入区間は検出されませんでした"

        self._populate_voice_events_list(self.latest_diagnosis)
        self.lbl_status.setText(
            f"🤖 AI声精密スキャンが完了しました（計 {len(merged)} 箇所の候補区間）。リストをダブルクリックして試聴できます。"
        )

    def _on_voice_scan_failed(self, err_msg: str) -> None:
        self.btn_scan_voice_ai.setEnabled(True)
        self.btn_run_pipeline.setEnabled(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("AI声スキャンエラー")
        QMessageBox.critical(self, "AI声スキャンエラー", f"AI声スキャン中にエラーが発生しました:\n{err_msg}")

    def on_add_selected_voice_event_to_pinpoint(self) -> None:
        if not self.session or not self.latest_diagnosis or not self.latest_diagnosis.voice_events:
            QMessageBox.information(self, "声候補区間なし", "追加できる声・咳払い候補区間がありません。")
            return

        row = self.list_diag_voice.currentRow()
        if not (0 <= row < len(self.latest_diagnosis.voice_events)):
            QMessageBox.information(
                self,
                "候補を選択してください",
                "リストから追加したい声・咳払い候補区間をクリックして選択してから押してください。\n"
                "（すべて追加する場合は『➕ 全声候補を一括追加』を押してください）",
            )
            return

        vev = self.latest_diagnosis.voice_events[row]
        self.session.pinpoint_regions.append(
            TimeRegion(
                start_sec=vev.start_sec,
                end_sec=vev.end_sec,
                label=f"{vev.detection_type} ({vev.confidence * 100:.0f}%)",
            )
        )
        self._refresh_region_lists_and_overlays()
        self.chk_vocal_enable.setChecked(True)
        self.combo_vocal_scope.setCurrentIndex(1)  # ピンポイント区間のみ適用に自動切替
        self.lbl_status.setText(
            f"🗣 声候補区間 ({format_time_mmss(vev.start_sec)} 〜 {format_time_mmss(vev.end_sec)}) をピンポイント声除去リストへ追加しました！"
        )

    def on_add_all_voice_events_to_pinpoint(self) -> None:
        if not self.session or not self.latest_diagnosis or not self.latest_diagnosis.voice_events:
            QMessageBox.information(self, "声候補区間なし", "追加できる声・咳払い候補区間がありません。")
            return

        added = 0
        for vev in self.latest_diagnosis.voice_events:
            self.session.pinpoint_regions.append(
                TimeRegion(
                    start_sec=vev.start_sec,
                    end_sec=vev.end_sec,
                    label=f"{vev.detection_type} ({vev.confidence * 100:.0f}%)",
                )
            )
            added += 1

        self._refresh_region_lists_and_overlays()
        self.chk_vocal_enable.setChecked(True)
        self.combo_vocal_scope.setCurrentIndex(1)  # ピンポイント区間のみ適用に自動切替
        QMessageBox.information(
            self,
            "ピンポイント声除去リストへ追加完了",
            f"検出された {added} 箇所の声・咳払い候補区間をピンポイント除去リストへ登録し、\n"
            "AI声除去の適用範囲を『登録したピンポイント区間のみに適用（他は100%無加工維持）』に設定しました！",
        )

    def on_apply_recommended_diagnosis(self) -> None:

        if not self.session or not self.latest_diagnosis:
            return
        rec = self.latest_diagnosis.recommended_config
        self._apply_config_to_ui(rec)

        if self.latest_diagnosis.suggested_noise_sample_region is not None:
            ns, ne = self.latest_diagnosis.suggested_noise_sample_region
            self.session.noise_sample_region = (ns, ne)
            self.lbl_noise_profile.setText(f"自動検出 ({format_time_mmss(ns)} 〜 {format_time_mmss(ne)})")
            self._refresh_region_lists_and_overlays()

        self.tabs.setCurrentIndex(2)  # 「🌬 風・ノイズ・EQ」タブを表示して反映結果を見せる
        self.lbl_status.setText("⚡ 音響診断に基づく推奨フィルタ設定（スマート風切り音除去・EQ・ノイズ見本）を反映しました！")
        QMessageBox.information(
            self,
            "推奨設定を反映しました",
            "音響・周波数特性の診断結果に基づき、\n"
            "・スマート風切り音除去（M/S乱流サプレッサー）\n"
            "・ローカット周波数＆吹奏楽4バンドEQ\n"
            "・ホワイトノイズ見本区間\n"
            "を最適値に設定しました！\n\n"
            "右下の「✨ 音声加工を実行」ボタンを押して聴き比べてみてください。",
        )

    def on_direct_lossless_extract(self) -> None:
        if not self.session:
            return
        in_path = self.session.source_path
        info = LosslessExtractor.probe_audio_stream(in_path)
        default_out = in_path.with_name(f"{in_path.stem}_lossless{info.recommended_ext}")

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "完全ロスレス音声抽出の保存先を指定",
            str(default_out),
            f"元ストリーム無劣化抽出 (*{info.recommended_ext});;非圧縮 24-bit WAV (*.wav);;可逆圧縮 FLAC (*.flac)",
        )
        if not save_path:
            return

        target = Path(save_path)
        try:
            self.lbl_status.setText("完全ロスレス音声抽出を実行中...")
            if target.suffix.lower() == info.recommended_ext.lower():
                out_file, _ = LosslessExtractor.extract_stream_copy(in_path, target)
                mode_msg = f"再エンコードなしの完全ストリームコピー ({info.codec_name.upper()})"
            elif target.suffix.lower() == ".flac":
                out_file = LosslessExtractor.extract_lossless_pcm(in_path, target, codec="flac")
                mode_msg = "可逆圧縮 FLAC ロスレス抽出"
            else:
                out_file = LosslessExtractor.extract_lossless_pcm(in_path, target, codec="pcm_s24le")
                mode_msg = "非圧縮 24-bit PCM WAV ロスレス抽出"

            self.lbl_status.setText(f"ロスレス抽出完了: {out_file.name}")
            QMessageBox.information(
                self,
                "完全ロスレス抽出完了",
                f"音声を無劣化で抽出しました！\n\n方式: {mode_msg}\n保存先:\n{out_file}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "抽出エラー", f"ロスレス抽出中にエラーが発生しました:\n{exc}")

    def on_preset_changed(self, index: int) -> None:
        if index == 0:
            self._apply_config_to_ui(ProcessingConfig.preset_outdoor_stage())
        elif index == 1:
            self._apply_config_to_ui(ProcessingConfig.preset_indoor_hall())
        elif index == 2:
            self._apply_config_to_ui(ProcessingConfig.preset_pinpoint_voice_only())
        elif index == 3:
            self._apply_config_to_ui(ProcessingConfig.preset_white_noise_only())

    def on_device_changed(self, _index: int) -> None:
        dev_id = self.combo_device.currentData()
        if dev_id:
            selected = self.device_manager.select_device_by_id(str(dev_id))
            self.lbl_status.setText(f"演算デバイスを切り替えました: {selected.display_name}")

    def on_waveform_region_changed(self, start_sec: float, end_sec: float) -> None:
        self.player_widget.set_loop_region(start_sec, end_sec)

    def on_waveform_seek_requested(self, sec: float) -> None:
        self.player_widget.seek_to_sec(sec)

    def on_add_pinpoint_region(self, start_sec: float, end_sec: float) -> None:
        if not self.session:
            return
        if end_sec - start_sec < 0.1:
            QMessageBox.warning(self, "範囲が短すぎます", "0.1秒以上の範囲を選択してください。")
            return

        reg = TimeRegion(
            start_sec=start_sec,
            end_sec=end_sec,
            label=f"ピンポイント区間 #{len(self.session.pinpoint_regions) + 1}",
        )
        self.session.pinpoint_regions.append(reg)
        self._refresh_region_lists_and_overlays()
        self.combo_vocal_scope.setCurrentIndex(1)
        self.tabs.setCurrentIndex(1)
        self.lbl_status.setText(
            f"ピンポイント除去区間 ({format_time_mmss(start_sec)} 〜 {format_time_mmss(end_sec)}) を追加しました。"
        )

    def on_delete_selected_pinpoint(self) -> None:
        if not self.session:
            return
        row = self.list_pinpoint_regions.currentRow()
        if 0 <= row < len(self.session.pinpoint_regions):
            self.session.pinpoint_regions.pop(row)
            self._refresh_region_lists_and_overlays()

    def on_clear_pinpoint_regions(self) -> None:
        if not self.session:
            return
        self.session.pinpoint_regions.clear()
        self._refresh_region_lists_and_overlays()

    def on_pinpoint_item_double_clicked(self, item: QListWidgetItem) -> None:
        if not self.session:
            return
        row = self.list_pinpoint_regions.row(item)
        if 0 <= row < len(self.session.pinpoint_regions):
            reg = self.session.pinpoint_regions[row]
            self.waveform_widget.set_selected_region(reg.start_sec, reg.end_sec)
            self.player_widget.seek_to_sec(reg.start_sec)

    def on_set_noise_sample_region(self, start_sec: float, end_sec: float) -> None:
        if not self.session:
            return
        if end_sec - start_sec < 0.2:
            QMessageBox.warning(
                self,
                "ノイズ見本が短すぎます",
                "ホワイトノイズ・空調音の特性を正確に学習するため、0.2秒以上（推奨1〜3秒）のノイズ区間を選択してください。",
            )
            return
        self.session.noise_sample_region = (start_sec, end_sec)
        self.lbl_noise_profile.setText(
            f"学習済 ({format_time_mmss(start_sec)} 〜 {format_time_mmss(end_sec)})"
        )
        self._refresh_region_lists_and_overlays()
        self.tabs.setCurrentIndex(2)
        self.lbl_status.setText(
            f"ホワイトノイズ見本区間 ({format_time_mmss(start_sec)} 〜 {format_time_mmss(end_sec)}) を登録しました。"
        )

    def on_clear_noise_sample(self) -> None:
        if not self.session:
            return
        self.session.noise_sample_region = None
        self.lbl_noise_profile.setText("未登録（自動推定モード）")
        self._refresh_region_lists_and_overlays()

    def on_add_song_chapter(self, start_sec: float, end_sec: float) -> None:
        if not self.session:
            return
        if end_sec - start_sec < 1.0:
            QMessageBox.warning(self, "曲の長さが短すぎます", "1秒以上の範囲を選択してください。")
            return

        default_title = f"曲目_{len(self.session.song_chapters) + 1:02d}"
        title, ok = QInputDialog.getText(
            self,
            "曲分割チャプターの追加",
            f"選択範囲 ({format_time_mmss(start_sec)} 〜 {format_time_mmss(end_sec)}) の曲名を入力してください:",
            text=default_title,
        )
        if not ok:
            return

        ch = SongChapter(
            title=title.strip() or default_title,
            start_sec=start_sec,
            end_sec=end_sec,
            fade_in_sec=float(self.spin_fade_in.value()),
            fade_out_sec=float(self.spin_fade_out.value()),
        )
        self.session.song_chapters.append(ch)
        self._refresh_region_lists_and_overlays()
        self.tabs.setCurrentIndex(4)

    def on_delete_selected_chapter(self) -> None:
        if not self.session:
            return
        row = self.list_chapters.currentRow()
        if 0 <= row < len(self.session.song_chapters):
            self.session.song_chapters.pop(row)
            self._refresh_region_lists_and_overlays()

    def on_clear_chapters(self) -> None:
        if not self.session:
            return
        self.session.song_chapters.clear()
        self._refresh_region_lists_and_overlays()

    def on_chapter_item_double_clicked(self, item: QListWidgetItem) -> None:
        if not self.session:
            return
        row = self.list_chapters.row(item)
        if 0 <= row < len(self.session.song_chapters):
            ch = self.session.song_chapters[row]
            self.waveform_widget.set_selected_region(ch.start_sec, ch.end_sec)
            self.player_widget.seek_to_sec(ch.start_sec)

    def _refresh_region_lists_and_overlays(self) -> None:
        if not self.session:
            return

        self.list_pinpoint_regions.clear()
        for idx, reg in enumerate(self.session.pinpoint_regions, start=1):
            self.list_pinpoint_regions.addItem(
                f"{idx}. {format_time_mmss(reg.start_sec)} 〜 {format_time_mmss(reg.end_sec)} "
                f"({reg.duration:.1f}秒) - {reg.label}"
            )

        self.list_chapters.clear()
        for idx, ch in enumerate(self.session.song_chapters, start=1):
            self.list_chapters.addItem(
                f"{idx:02d}. {ch.title} [{format_time_mmss(ch.start_sec)} 〜 {format_time_mmss(ch.end_sec)}] "
                f"(FadeIn:{ch.fade_in_sec:.1f}s / Out:{ch.fade_out_sec:.1f}s)"
            )

        self.waveform_widget.refresh_overlays(
            pinpoint_regions=self.session.pinpoint_regions,
            noise_sample_region=self.session.noise_sample_region,
            chapters=self.session.song_chapters,
        )

    # --- 音声処理パイプラインの非同期実行 ---
    def on_run_pipeline(self) -> None:
        if not self.session:
            return
        if self._worker is not None and self._worker.isRunning():
            return

        cfg = self._collect_config_from_ui()
        self.session.config = cfg

        if cfg.vocal_removal_enabled and cfg.vocal_apply_only_regions and not self.session.pinpoint_regions:
            reply = QMessageBox.question(
                self,
                "ピンポイント区間が未登録です",
                "AI声除去の適用範囲が「登録したピンポイント区間のみ」になっていますが、区間リストが空です。\n"
                "トラック全体にAI声除去を適用してよろしいですか？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.combo_vocal_scope.setCurrentIndex(0)
                cfg.vocal_apply_only_regions = False
            else:
                return

        self.btn_run_pipeline.setEnabled(False)
        self.btn_open_file.setEnabled(False)
        self.progress_bar.setValue(5)

        self._worker = PipelineWorker(
            pipeline=self.pipeline,
            session=self.session,
            config=cfg,
            parent=self,
        )
        self._worker.progress_updated.connect(self._on_worker_progress)
        self._worker.finished_success.connect(self._on_worker_finished)
        self._worker.failed_error.connect(self._on_worker_failed)
        self._worker.start()

    def _on_worker_progress(self, pct: int, msg: str) -> None:
        self.progress_bar.setValue(pct)
        self.lbl_status.setText(msg)

    def _on_worker_finished(self, processed_audio: np.ndarray) -> None:
        self.btn_run_pipeline.setEnabled(True)
        self.btn_open_file.setEnabled(True)
        self.progress_bar.setValue(100)

        if not self.session:
            return

        proc_lufs = MasteringProcessor.measure_lufs(processed_audio, self.session.sample_rate)
        orig_lufs = MasteringProcessor.measure_lufs(self.session.original_audio, self.session.sample_rate)
        self.lbl_measured_lufs.setText(
            f"ラウドネス: 元音声 {orig_lufs:.1f} LUFS ➔ 加工後 {proc_lufs:.1f} LUFS"
        )

        self.waveform_widget.update_processed_waveform(processed_audio, show_processed=True)
        self.player_widget.update_processed_audio(processed_audio, self.session.sample_rate)
        self.lbl_status.setText(
            "✨ 音声加工が完了しました！下部プレイヤーで「🔵元の音声」と「🟢加工後」をワンクリックで聴き比べできます。"
        )

    def _on_worker_failed(self, err_msg: str) -> None:
        self.btn_run_pipeline.setEnabled(True)
        self.btn_open_file.setEnabled(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("処理エラーが発生しました")
        QMessageBox.critical(self, "音声処理エラー", f"処理中にエラーが発生しました:\n{err_msg}")

    # --- ファイル保存・曲分割一括書き出し ---
    def on_export_full_audio(self) -> None:
        if not self.session:
            return

        fmt_code = str(self.combo_export_format.currentData() or "wav24")
        ext_map = {
            "wav24": (".wav", "24-bit WAV (*.wav)"),
            "flac": (".flac", "Lossless FLAC (*.flac)"),
            "wav16": (".wav", "16-bit WAV (*.wav)"),
            "m4a": (".m4a", "AAC M4A (*.m4a)"),
            "mp3": (".mp3", "MP3 (*.mp3)"),
        }
        ext, filter_str = ext_map.get(fmt_code, (".wav", "WAV (*.wav)"))

        suffix_tag = "_processed" if self.session.processed_audio is not None else "_extracted"
        default_path = self.session.source_path.with_name(
            f"{self.session.source_path.stem}{suffix_tag}{ext}"
        )

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "音声ファイルの保存先を指定",
            str(default_path),
            filter_str,
        )
        if not save_path:
            return

        try:
            target_audio = self.session.get_active_audio(use_processed=True)
            saved = LosslessExtractor.save_audio(
                audio=target_audio,
                sample_rate=self.session.sample_rate,
                output_path=Path(save_path),
                format_type=fmt_code,
            )
            self.lbl_status.setText(f"保存完了: {saved.name}")
            QMessageBox.information(self, "保存完了", f"音声ファイルを保存しました:\n{saved}")
        except Exception as exc:
            QMessageBox.critical(self, "保存エラー", f"ファイルの保存に失敗しました:\n{exc}")

    def on_export_all_chapters(self) -> None:
        if not self.session:
            return
        if not self.session.song_chapters:
            QMessageBox.warning(
                self,
                "曲チャプターが未登録です",
                "先に波形上で各曲の範囲を選択し、「✂️選択範囲を曲分割チャプターに追加」ボタンで曲を登録してください。",
            )
            return

        out_dir = QFileDialog.getExistingDirectory(
            self,
            "分割した曲ファイルを保存するフォルダを選択",
            str(self.session.source_path.parent),
        )
        if not out_dir:
            return

        fmt_code = str(self.combo_export_format.currentData() or "wav24")
        try:
            saved_files = AudioPipeline.export_chapters(
                session=self.session,
                output_dir=Path(out_dir),
                format_type=fmt_code,
                use_processed=True,
            )
            file_list_str = "\n".join(f"・{p.name}" for p in saved_files)
            QMessageBox.information(
                self,
                "曲分割一括書き出し完了",
                f"{len(saved_files)} 曲の書き出しが完了しました！\n\n保存先フォルダ: {out_dir}\n\n{file_list_str}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "分割書き出しエラー", f"曲分割書き出しに失敗しました:\n{exc}")
