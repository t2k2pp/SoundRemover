"""
インタラクティブ波形・リージョン選択・マーカー表示ウィジェット (`waveform_widget.py`)

PyQtGraph を使用し、長時間の演奏録音でも軽快な拡大・縮小・範囲選択・再生ヘッド追従を実現する。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from sound_remover.core.audio_data import SongChapter, TimeRegion


def format_time_mmss(seconds: float) -> str:
    """秒数を MM:SS.ss 形式の文字列に変換する"""
    seconds = max(0.0, float(seconds))
    mins = int(seconds // 60)
    secs = seconds - mins * 60
    return f"{mins:02d}:{secs:05.2f}"


class WaveformWidget(QWidget):
    """波形表示・時間範囲選択・ピンポイント領域可視化ウィジェット"""

    region_changed = Signal(float, float)          # (start_sec, end_sec)
    seek_requested = Signal(float)                 # playhead_sec
    add_pinpoint_requested = Signal(float, float)  # (start_sec, end_sec)
    set_noise_sample_requested = Signal(float, float)
    add_chapter_requested = Signal(float, float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._duration_sec: float = 0.0
        self._sample_rate: int = 48000
        self._overlay_items: List[pg.GraphicsObject] = []

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # 上部ヘッダー（現在の選択範囲時刻表示＆ズームボタン）
        header_layout = QHBoxLayout()
        self.lbl_selection_info = QLabel("選択範囲: 00:00.00 〜 00:00.00 (長さ: 0.00秒)")
        self.lbl_selection_info.setStyleSheet("color: #9ecbff; font-weight: bold;")
        header_layout.addWidget(self.lbl_selection_info)
        header_layout.addStretch()

        self.btn_zoom_in = QPushButton("🔍 ズーム＋")
        self.btn_zoom_out = QPushButton("🔍 ズーム－")
        self.btn_zoom_all = QPushButton("全体表示")

        self.btn_zoom_in.clicked.connect(lambda: self._zoom(0.5))
        self.btn_zoom_out.clicked.connect(lambda: self._zoom(2.0))
        self.btn_zoom_all.clicked.connect(self.reset_zoom)

        header_layout.addWidget(self.btn_zoom_in)
        header_layout.addWidget(self.btn_zoom_out)
        header_layout.addWidget(self.btn_zoom_all)
        layout.addLayout(header_layout)

        # PyQtGraph 波形プロットエリア
        pg.setConfigOptions(antialias=False)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#111318")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setLabel("bottom", "時間 (秒)")
        self.plot_widget.setLabel("left", "振幅")
        self.plot_widget.setYRange(-1.05, 1.05, padding=0)
        self.plot_widget.setMouseEnabled(x=True, y=False)

        # 波形曲線 (Original: シアンブルー / Processed: エメラルドグリーン)
        self.curve_original = self.plot_widget.plot(
            pen=pg.mkPen(color=(56, 189, 248, 210), width=1),
            name="元の音声 (Original)",
        )
        self.curve_processed = self.plot_widget.plot(
            pen=pg.mkPen(color=(16, 185, 129, 210), width=1),
            name="加工後 (Processed)",
        )
        self.curve_processed.setVisible(False)

        # 選択範囲 (LinearRegionItem)
        self.selection_region = pg.LinearRegionItem(
            values=(0.0, 5.0),
            orientation="vertical",
            brush=pg.mkBrush(59, 130, 246, 55),
            hoverBrush=pg.mkBrush(59, 130, 246, 85),
            pen=pg.mkPen(color=(96, 165, 250), width=2),
        )
        self.selection_region.setZValue(10)
        self.selection_region.sigRegionChanged.connect(self._on_region_changed)
        self.plot_widget.addItem(self.selection_region)

        # 再生ヘッド (赤黄色の縦線カーソル)
        self.playhead_line = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            movable=True,
            pen=pg.mkPen(color=(250, 204, 21), width=2),
        )
        self.playhead_line.setZValue(20)
        self.playhead_line.sigPositionChangeFinished.connect(self._on_playhead_dragged)
        self.plot_widget.addItem(self.playhead_line)

        layout.addWidget(self.plot_widget, stretch=1)

        # 下部クイック範囲登録ボタンバー
        action_layout = QHBoxLayout()
        self.btn_add_pinpoint = QPushButton("➕ 選択範囲を「声/ノイズのピンポイント除去区間」に追加")
        self.btn_add_pinpoint.setObjectName("accentButton")
        self.btn_add_pinpoint.setToolTip(
            "波形上の青い選択範囲（観客の声や咳払いが入った数秒間など）をピンポイント除去リストに追加します"
        )

        self.btn_set_noise_sample = QPushButton("🎯 選択範囲を「ホワイトノイズ見本」に登録")
        self.btn_set_noise_sample.setToolTip(
            "演奏が始まる前などの「サーッというノイズだけの区間」を選択して登録すると、そのノイズ特性を学習して引き算します"
        )

        self.btn_add_chapter = QPushButton("✂️ 選択範囲を「曲分割チャプター」に追加")
        self.btn_add_chapter.setToolTip(
            "長い動画から1曲分の開始〜終了範囲を選択して、分割書き出しリストに登録します"
        )

        self.btn_add_pinpoint.clicked.connect(self._emit_add_pinpoint)
        self.btn_set_noise_sample.clicked.connect(self._emit_set_noise_sample)
        self.btn_add_chapter.clicked.connect(self._emit_add_chapter)

        action_layout.addWidget(self.btn_add_pinpoint)
        action_layout.addWidget(self.btn_set_noise_sample)
        action_layout.addWidget(self.btn_add_chapter)
        layout.addLayout(action_layout)

    @staticmethod
    def _downsample_for_display(audio: np.ndarray, sample_rate: int, max_points: int = 40000) -> Tuple[np.ndarray, np.ndarray]:
        """
        長時間の音声でも瞬時に描画できるよう、ピーク包絡線（最大・最小）を保持したままダウンサンプリングする。
        """
        mono = audio.mean(axis=0) if audio.ndim == 2 else audio
        n_samples = mono.shape[0]
        if n_samples == 0:
            return np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32)

        if n_samples <= max_points:
            t = np.arange(n_samples, dtype=np.float32) / float(sample_rate)
            return t, mono.astype(np.float32)

        n_buckets = max_points // 2
        bucket_size = n_samples // n_buckets
        trimmed = mono[: n_buckets * bucket_size].reshape(n_buckets, bucket_size)
        mins = trimmed.min(axis=1)
        maxs = trimmed.max(axis=1)

        y_interleaved = np.empty(n_buckets * 2, dtype=np.float32)
        y_interleaved[0::2] = mins
        y_interleaved[1::2] = maxs

        t_interleaved = np.linspace(0.0, n_samples / float(sample_rate), n_buckets * 2, dtype=np.float32)
        return t_interleaved, y_interleaved

    def set_audio_data(
        self,
        original_audio: np.ndarray,
        sample_rate: int,
        processed_audio: Optional[np.ndarray] = None,
        show_processed: bool = False,
    ) -> None:
        """波形データをセットして描画を更新する"""
        self._sample_rate = sample_rate
        self._duration_sec = original_audio.shape[1] / float(sample_rate)

        t_orig, y_orig = self._downsample_for_display(original_audio, sample_rate)
        self.curve_original.setData(t_orig, y_orig)

        if processed_audio is not None:
            t_proc, y_proc = self._downsample_for_display(processed_audio, sample_rate)
            self.curve_processed.setData(t_proc, y_proc)
            self.curve_processed.setVisible(show_processed)
            self.curve_original.setVisible(not show_processed)
        else:
            self.curve_processed.clear()
            self.curve_processed.setVisible(False)
            self.curve_original.setVisible(True)

        self.selection_region.setBounds((0.0, max(0.1, self._duration_sec)))
        default_end = min(self._duration_sec, max(1.0, self._duration_sec * 0.2))
        self.selection_region.setRegion((0.0, default_end))
        self.reset_zoom()

    def update_processed_waveform(self, processed_audio: Optional[np.ndarray], show_processed: bool = True) -> None:
        """加工後音声のみを更新して表示を切り替える"""
        if processed_audio is not None:
            t_proc, y_proc = self._downsample_for_display(processed_audio, self._sample_rate)
            self.curve_processed.setData(t_proc, y_proc)
            self.set_display_mode(show_processed=show_processed)
        else:
            self.curve_processed.clear()
            self.set_display_mode(show_processed=False)

    def set_display_mode(self, show_processed: bool) -> None:
        """Original波形とProcessed波形の表示を切り替える"""
        has_proc = self.curve_processed.xData is not None and len(self.curve_processed.xData) > 0
        if show_processed and has_proc:
            self.curve_processed.setVisible(True)
            self.curve_original.setVisible(False)
        else:
            self.curve_processed.setVisible(False)
            self.curve_original.setVisible(True)

    def refresh_overlays(
        self,
        pinpoint_regions: List[TimeRegion],
        noise_sample_region: Optional[Tuple[float, float]],
        chapters: List[SongChapter],
    ) -> None:
        """ピンポイント適用区間・ノイズ見本区間・曲分割マーカーのオーバーレイを再描画する"""
        for item in self._overlay_items:
            self.plot_widget.removeItem(item)
        self._overlay_items.clear()

        # 1. ピンポイント除去区間（オレンジ〜赤の半透明帯）
        for reg in pinpoint_regions:
            item = pg.LinearRegionItem(
                values=(reg.start_sec, reg.end_sec),
                orientation="vertical",
                movable=False,
                brush=pg.mkBrush(249, 115, 22, 45),
                pen=pg.mkPen(color=(251, 146, 60, 180), width=1, style=Qt.PenStyle.DashLine),
            )
            item.setZValue(5)
            self.plot_widget.addItem(item)
            self._overlay_items.append(item)

        # 2. ホワイトノイズ見本区間（紫の半透明帯）
        if noise_sample_region is not None:
            ns, ne = noise_sample_region
            n_item = pg.LinearRegionItem(
                values=(ns, ne),
                orientation="vertical",
                movable=False,
                brush=pg.mkBrush(168, 85, 247, 50),
                pen=pg.mkPen(color=(192, 132, 252, 200), width=1, style=Qt.PenStyle.DotLine),
            )
            n_item.setZValue(6)
            self.plot_widget.addItem(n_item)
            self._overlay_items.append(n_item)

        # 3. 曲分割チャプター境界線（グリーンの破線）
        for ch in chapters:
            line_s = pg.InfiniteLine(
                pos=ch.start_sec,
                angle=90,
                movable=False,
                pen=pg.mkPen(color=(52, 211, 153, 180), width=1, style=Qt.PenStyle.DashDotLine),
            )
            self.plot_widget.addItem(line_s)
            self._overlay_items.append(line_s)

    def get_selected_region(self) -> Tuple[float, float]:
        """現在の選択範囲 (start_sec, end_sec) を返す"""
        r = self.selection_region.getRegion()
        s = max(0.0, min(float(r[0]), float(r[1])))
        e = min(max(0.1, self._duration_sec), max(float(r[0]), float(r[1])))
        return s, e

    def set_selected_region(self, start_sec: float, end_sec: float) -> None:
        """選択範囲を指定位置へ移動する"""
        self.selection_region.setRegion((start_sec, end_sec))

    def set_playhead_sec(self, sec: float) -> None:
        """再生ヘッドの位置を更新する"""
        self.playhead_line.blockSignals(True)
        self.playhead_line.setPos(max(0.0, min(sec, self._duration_sec)))
        self.playhead_line.blockSignals(False)

    def reset_zoom(self) -> None:
        """波形全体が表示されるようにズームをリセットする"""
        self.plot_widget.setXRange(0.0, max(1.0, self._duration_sec), padding=0.01)
        self.plot_widget.setYRange(-1.05, 1.05, padding=0)

    def _zoom(self, factor: float) -> None:
        view_range = self.plot_widget.viewRange()[0]
        center = 0.5 * (view_range[0] + view_range[1])
        half_width = max(0.2, 0.5 * (view_range[1] - view_range[0]) * factor)
        new_start = max(0.0, center - half_width)
        new_end = min(max(1.0, self._duration_sec), center + half_width)
        self.plot_widget.setXRange(new_start, new_end, padding=0)

    def _on_region_changed(self) -> None:
        s, e = self.get_selected_region()
        dur = max(0.0, e - s)
        self.lbl_selection_info.setText(
            f"選択範囲: {format_time_mmss(s)} 〜 {format_time_mmss(e)} (長さ: {dur:.2f}秒)"
        )
        self.region_changed.emit(s, e)

    def _on_playhead_dragged(self) -> None:
        pos = float(self.playhead_line.value())
        pos = max(0.0, min(pos, self._duration_sec))
        self.seek_requested.emit(pos)

    def _emit_add_pinpoint(self) -> None:
        s, e = self.get_selected_region()
        self.add_pinpoint_requested.emit(s, e)

    def _emit_set_noise_sample(self) -> None:
        s, e = self.get_selected_region()
        self.set_noise_sample_requested.emit(s, e)

    def _emit_add_chapter(self) -> None:
        s, e = self.get_selected_region()
        self.add_chapter_requested.emit(s, e)
