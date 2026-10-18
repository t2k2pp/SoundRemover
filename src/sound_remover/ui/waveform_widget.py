"""
インタラクティブ波形・リージョン選択・マーカー表示ウィジェット (`waveform_widget.py`)

PyQtGraph を使用し、長時間の演奏録音でも軽快な拡大・縮小・範囲選択・再生ヘッド追従を実現する。

範囲選択の操作方法:
1. 初期状態（ファイル読み込み直後）は「範囲未選択」状態。
2. 波形上の任意の場所を「マウス左ドラッグ」すると、その区間を直接新しい選択範囲として指定・変更可能。
3. 波形上を「左クリック（ドラッグなし）」すると、その位置に再生ヘッド（黄色い線）がジャンプしてシーク可能。
4. 既存の青い選択範囲の左右の境界線をドラッグして範囲を微調整可能。
5. 波形上部の「[現在位置を開始(In)に]」「[現在位置を終了(Out)に]」ボタン、または秒数スピンボックスでも指定可能。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
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


class InteractiveWaveformPlotWidget(pg.PlotWidget):
    """
    マウス左ドラッグで直接範囲選択（Audacity等の音声編集ソフトと同様の操作感）、
    左クリック単発で再生位置シークを行うカスタム PlotWidget。
    """

    drag_selection_updated = Signal(float, float)  # (start_sec, end_sec)
    click_seek_requested = Signal(float)           # clicked_sec

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._is_dragging_new_region: bool = False
        self._drag_start_sec: float = 0.0
        self._press_pixel_x: int = 0
        self._selection_region_ref: Optional[pg.LinearRegionItem] = None

    def set_selection_region_ref(self, region_item: pg.LinearRegionItem) -> None:
        self._selection_region_ref = region_item

    def _scene_pos_to_sec(self, scene_pos: QPointF) -> float:
        vb = self.getPlotItem().getViewBox()
        view_pt = vb.mapSceneToView(scene_pos)
        return float(view_pt.x())

    def _is_near_existing_region_edge(self, scene_pos: QPointF, threshold_px: float = 12.0) -> bool:
        """既に表示されている選択範囲の左右境界線の近くをクリックしたかどうかを判定"""
        reg = self._selection_region_ref
        if reg is None or not reg.isVisible():
            return False

        vb = self.getPlotItem().getViewBox()
        r0, r1 = reg.getRegion()
        p0_scene = vb.mapViewToScene(QPointF(r0, 0.0))
        p1_scene = vb.mapViewToScene(QPointF(r1, 0.0))
        dx0 = abs(scene_pos.x() - p0_scene.x())
        dx1 = abs(scene_pos.x() - p1_scene.x())
        return min(dx0, dx1) <= threshold_px

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(ev.pos())
            # 既存の選択枠の端（境界線）を掴んだ場合は、標準の境界線ドラッグに任せる
            if self._is_near_existing_region_edge(scene_pos):
                super().mousePressEvent(ev)
                return

            self._is_dragging_new_region = True
            self._press_pixel_x = ev.pos().x()
            self._drag_start_sec = self._scene_pos_to_sec(scene_pos)
            ev.accept()
            return

        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        if self._is_dragging_new_region:
            dx_px = abs(ev.pos().x() - self._press_pixel_x)
            if dx_px > 4:
                scene_pos = self.mapToScene(ev.pos())
                cur_sec = self._scene_pos_to_sec(scene_pos)
                s = min(self._drag_start_sec, cur_sec)
                e = max(self._drag_start_sec, cur_sec)
                self.drag_selection_updated.emit(s, e)
            ev.accept()
            return

        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        if ev.button() == Qt.MouseButton.LeftButton and self._is_dragging_new_region:
            self._is_dragging_new_region = False
            dx_px = abs(ev.pos().x() - self._press_pixel_x)
            scene_pos = self.mapToScene(ev.pos())
            cur_sec = self._scene_pos_to_sec(scene_pos)

            if dx_px <= 4:
                # ドラッグではなくクリックのみの場合は、その秒数へ再生ヘッドを移動（シーク）
                self.click_seek_requested.emit(cur_sec)
            else:
                s = min(self._drag_start_sec, cur_sec)
                e = max(self._drag_start_sec, cur_sec)
                self.drag_selection_updated.emit(s, e)

            ev.accept()
            return

        super().mouseReleaseEvent(ev)


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
        self._has_selection: bool = False
        self._updating_spinboxes: bool = False
        self._overlay_items: List[pg.GraphicsObject] = []

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # 1段目ヘッダー: 選択状態ラベル ＆ ズーム・スクロール操作ボタン
        header_layout = QHBoxLayout()
        self.lbl_selection_info = QLabel(
            "選択範囲: 未選択 （波形上をマウスで左ドラッグして範囲選択 / クリックで再生位置移動）"
        )
        self.lbl_selection_info.setStyleSheet("color: #9ecbff; font-weight: bold;")
        header_layout.addWidget(self.lbl_selection_info)
        header_layout.addStretch()

        self.btn_pan_left = QPushButton("◀ 左へ")
        self.btn_pan_right = QPushButton("右へ ▶")
        self.btn_zoom_in = QPushButton("🔍 ズーム＋")
        self.btn_zoom_out = QPushButton("🔍 ズーム－")
        self.btn_zoom_all = QPushButton("全体表示")

        self.btn_pan_left.clicked.connect(lambda: self._pan(-0.25))
        self.btn_pan_right.clicked.connect(lambda: self._pan(0.25))
        self.btn_zoom_in.clicked.connect(lambda: self._zoom(0.5))
        self.btn_zoom_out.clicked.connect(lambda: self._zoom(2.0))
        self.btn_zoom_all.clicked.connect(self.reset_zoom)

        header_layout.addWidget(self.btn_pan_left)
        header_layout.addWidget(self.btn_pan_right)
        header_layout.addWidget(self.btn_zoom_in)
        header_layout.addWidget(self.btn_zoom_out)
        header_layout.addWidget(self.btn_zoom_all)
        layout.addLayout(header_layout)

        # 2段目バー: 開始(In) / 終了(Out) 秒数スピンボックス ＆ 現在再生位置からのワンクリックセット ＆ 選択解除
        range_bar = QHBoxLayout()

        self.btn_set_in = QPushButton("[ 現在の再生位置を開始点(In)に")
        self.btn_set_in.setToolTip("再生中に押すと、現在の黄色い再生バーの位置を選択範囲の「開始点」にします")
        self.btn_set_in.clicked.connect(self.set_selection_in_at_playhead)

        lbl_start = QLabel("開始:")
        self.spin_sel_start = QDoubleSpinBox()
        self.spin_sel_start.setRange(0.0, 99999.0)
        self.spin_sel_start.setDecimals(2)
        self.spin_sel_start.setSingleStep(0.1)
        self.spin_sel_start.setSuffix(" 秒")
        self.spin_sel_start.valueChanged.connect(self._on_spinbox_range_edited)

        lbl_end = QLabel("終了:")
        self.spin_sel_end = QDoubleSpinBox()
        self.spin_sel_end.setRange(0.0, 99999.0)
        self.spin_sel_end.setDecimals(2)
        self.spin_sel_end.setSingleStep(0.1)
        self.spin_sel_end.setSuffix(" 秒")
        self.spin_sel_end.valueChanged.connect(self._on_spinbox_range_edited)

        self.btn_set_out = QPushButton("現在の再生位置を終了点(Out)に ]")
        self.btn_set_out.setToolTip("再生中に押すと、現在の黄色い再生バーの位置を選択範囲の「終了点」にします")
        self.btn_set_out.clicked.connect(self.set_selection_out_at_playhead)

        self.btn_clear_selection = QPushButton("✖ 選択解除")
        self.btn_clear_selection.clicked.connect(self.clear_selection)

        range_bar.addWidget(self.btn_set_in)
        range_bar.addWidget(lbl_start)
        range_bar.addWidget(self.spin_sel_start)
        range_bar.addWidget(lbl_end)
        range_bar.addWidget(self.spin_sel_end)
        range_bar.addWidget(self.btn_set_out)
        range_bar.addWidget(self.btn_clear_selection)
        range_bar.addStretch()
        layout.addLayout(range_bar)

        # PyQtGraph カスタム波形プロットエリア
        pg.setConfigOptions(antialias=False)
        self.plot_widget = InteractiveWaveformPlotWidget()
        self.plot_widget.setBackground("#111318")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setLabel("bottom", "時間 (秒)")
        self.plot_widget.setLabel("left", "振幅")
        self.plot_widget.setYRange(-1.05, 1.05, padding=0)
        # マウスホイールでのX軸ズームは有効にしつつ、左ドラッグは範囲選択に使用
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

        # 選択範囲 (LinearRegionItem) - 初期状態は非表示 (未選択)
        self.selection_region = pg.LinearRegionItem(
            values=(0.0, 0.0),
            orientation="vertical",
            brush=pg.mkBrush(59, 130, 246, 65),
            hoverBrush=pg.mkBrush(59, 130, 246, 95),
            pen=pg.mkPen(color=(96, 165, 250), width=2),
        )
        self.selection_region.setZValue(25)
        self.selection_region.setVisible(False)
        self.selection_region.sigRegionChanged.connect(self._on_region_item_changed)
        self.plot_widget.addItem(self.selection_region)
        self.plot_widget.set_selection_region_ref(self.selection_region)

        # 再生ヘッド (黄色の縦線カーソル)
        self.playhead_line = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            movable=True,
            pen=pg.mkPen(color=(250, 204, 21), width=2),
        )
        self.playhead_line.setZValue(15)
        self.playhead_line.sigPositionChangeFinished.connect(self._on_playhead_dragged)
        self.plot_widget.addItem(self.playhead_line)

        self.plot_widget.drag_selection_updated.connect(self._on_mouse_drag_selection)
        self.plot_widget.click_seek_requested.connect(self._on_mouse_click_seek)

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
    def _downsample_for_display(
        audio: np.ndarray, sample_rate: int, max_points: int = 40000
    ) -> Tuple[np.ndarray, np.ndarray]:
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

        t_interleaved = np.linspace(
            0.0, n_samples / float(sample_rate), n_buckets * 2, dtype=np.float32
        )
        return t_interleaved, y_interleaved

    def set_audio_data(
        self,
        original_audio: np.ndarray,
        sample_rate: int,
        processed_audio: Optional[np.ndarray] = None,
        show_processed: bool = False,
    ) -> None:
        """波形データをセットして描画を更新する（初期状態は範囲未選択）"""
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

        max_dur = max(0.1, self._duration_sec)
        self.selection_region.setBounds((0.0, max_dur))
        self.spin_sel_start.setMaximum(max_dur)
        self.spin_sel_end.setMaximum(max_dur)

        # 初期状態は「範囲未選択」にする（勝手に冒頭を選択しない）
        self.clear_selection()
        self.set_playhead_sec(0.0)
        self.reset_zoom()

    def clear_selection(self) -> None:
        """選択範囲を解除して非表示にする"""
        self._has_selection = False
        self.selection_region.blockSignals(True)
        self.selection_region.setVisible(False)
        self.selection_region.setRegion((0.0, 0.0))
        self.selection_region.blockSignals(False)

        self._updating_spinboxes = True
        self.spin_sel_start.setValue(0.0)
        self.spin_sel_end.setValue(0.0)
        self._updating_spinboxes = False

        self.lbl_selection_info.setText(
            "選択範囲: 未選択 （波形上をマウスで左ドラッグして範囲選択 / クリックで再生位置移動）"
        )
        self.region_changed.emit(0.0, self._duration_sec)

    @property
    def has_selection(self) -> bool:
        """現在ユーザーが範囲を選択しているかどうかを返す"""
        if not self._has_selection or not self.selection_region.isVisible():
            return False
        s, e = self.get_selected_region()
        return (e - s) >= 0.02

    def update_processed_waveform(
        self, processed_audio: Optional[np.ndarray], show_processed: bool = True
    ) -> None:
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
        if not self._has_selection:
            return 0.0, 0.0
        r = self.selection_region.getRegion()
        s = max(0.0, min(float(r[0]), float(r[1])))
        e = min(max(0.0, self._duration_sec), max(float(r[0]), float(r[1])))
        return s, e

    def set_selected_region(self, start_sec: float, end_sec: float) -> None:
        """選択範囲を指定位置へ表示・移動する"""
        max_dur = max(0.0, self._duration_sec)
        s = max(0.0, min(start_sec, end_sec, max_dur))
        e = max(s, min(max(start_sec, end_sec), max_dur))

        self._has_selection = True
        self.selection_region.setVisible(True)
        self.selection_region.setRegion((s, e))
        self._sync_ui_from_region(s, e)

    def set_selection_in_at_playhead(self) -> None:
        """現在の再生ヘッド位置を選択範囲の開始点(In)にセットする"""
        if self._duration_sec <= 0:
            return
        cur_pos = float(self.playhead_line.value())
        cur_pos = max(0.0, min(cur_pos, self._duration_sec))

        if self.has_selection:
            _, prev_end = self.get_selected_region()
            new_end = prev_end if prev_end > cur_pos + 0.05 else min(self._duration_sec, cur_pos + 5.0)
        else:
            new_end = min(self._duration_sec, cur_pos + 5.0)

        self.set_selected_region(cur_pos, new_end)

    def set_selection_out_at_playhead(self) -> None:
        """現在の再生ヘッド位置を選択範囲の終了点(Out)にセットする"""
        if self._duration_sec <= 0:
            return
        cur_pos = float(self.playhead_line.value())
        cur_pos = max(0.0, min(cur_pos, self._duration_sec))

        if self.has_selection:
            prev_start, _ = self.get_selected_region()
            new_start = prev_start if prev_start < cur_pos - 0.05 else max(0.0, cur_pos - 5.0)
        else:
            new_start = max(0.0, cur_pos - 5.0)

        self.set_selected_region(new_start, cur_pos)

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

    def _pan(self, step_ratio: float) -> None:
        view_range = self.plot_widget.viewRange()[0]
        width = max(0.2, view_range[1] - view_range[0])
        shift = width * step_ratio
        new_start = max(0.0, min(view_range[0] + shift, max(0.0, self._duration_sec - width)))
        new_end = min(max(1.0, self._duration_sec), new_start + width)
        self.plot_widget.setXRange(new_start, new_end, padding=0)

    def _on_mouse_drag_selection(self, start_sec: float, end_sec: float) -> None:
        """波形上のマウス左ドラッグによる新規範囲選択・更新"""
        if self._duration_sec <= 0:
            return
        self.set_selected_region(start_sec, end_sec)

    def _on_mouse_click_seek(self, clicked_sec: float) -> None:
        """波形上の単発クリックによる再生ヘッド移動"""
        if self._duration_sec <= 0:
            return
        pos = max(0.0, min(clicked_sec, self._duration_sec))
        self.set_playhead_sec(pos)
        self.seek_requested.emit(pos)

    def _on_region_item_changed(self) -> None:
        if not self._has_selection or not self.selection_region.isVisible():
            return
        s, e = self.get_selected_region()
        self._sync_ui_from_region(s, e)

    def _on_spinbox_range_edited(self) -> None:
        if self._updating_spinboxes or self._duration_sec <= 0:
            return
        s = float(self.spin_sel_start.value())
        e = float(self.spin_sel_end.value())
        if e > s + 0.01:
            self.set_selected_region(s, e)

    def _sync_ui_from_region(self, s: float, e: float) -> None:
        dur = max(0.0, e - s)
        self.lbl_selection_info.setText(
            f"選択範囲: {format_time_mmss(s)} 〜 {format_time_mmss(e)} (長さ: {dur:.2f}秒)"
        )
        self._updating_spinboxes = True
        self.spin_sel_start.setValue(s)
        self.spin_sel_end.setValue(e)
        self._updating_spinboxes = False
        self.region_changed.emit(s, e)

    def _on_playhead_dragged(self) -> None:
        pos = float(self.playhead_line.value())
        pos = max(0.0, min(pos, self._duration_sec))
        self.seek_requested.emit(pos)

    def _require_selection_or_warn(self) -> Optional[Tuple[float, float]]:
        if not self.has_selection:
            QMessageBox.information(
                self,
                "範囲が選択されていません",
                "先に波形の上をマウスで左ドラッグするか、\n"
                "「[ 現在の再生位置を開始点(In)に」と「現在の再生位置を終了点(Out)に ]」ボタンで\n"
                "対象の時間範囲を選択してください。",
            )
            return None
        return self.get_selected_region()

    def _emit_add_pinpoint(self) -> None:
        sel = self._require_selection_or_warn()
        if sel is not None:
            self.add_pinpoint_requested.emit(sel[0], sel[1])

    def _emit_set_noise_sample(self) -> None:
        sel = self._require_selection_or_warn()
        if sel is not None:
            self.set_noise_sample_requested.emit(sel[0], sel[1])

    def _emit_add_chapter(self) -> None:
        sel = self._require_selection_or_warn()
        if sel is not None:
            self.add_chapter_requested.emit(sel[0], sel[1])
