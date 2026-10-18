"""
Original / Processed ワンクリックA/B聴き比べリアルタイムオーディオプレイヤー (`player_widget.py`)

再生位置を維持したまま「元の音声 (Original)」と「加工後 (Processed)」を瞬時に切り替え、
また選択した範囲のみのループ再生に対応することで、声やノイズが消えて楽器の音色が保たれているかを
耳で容易に確認できるようにする。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
     QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from sound_remover.ui.waveform_widget import format_time_mmss


class AudioPreviewPlayer(QGroupBox):
    """Original / Processed A/B比較対応オーディオプレイヤーウィジェット"""

    position_sec_changed = Signal(float)
    ab_mode_changed = Signal(bool)  # True: Processed, False: Original

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("🎧 リアルタイムA/B聴き比べプレイヤー", parent)

        self._temp_dir = tempfile.TemporaryDirectory(prefix="sound_remover_preview_")
        self._orig_wav_path: Optional[Path] = None
        self._proc_wav_path: Optional[Path] = None
        self._has_processed: bool = False
        self._is_playing_processed: bool = False
        self._duration_sec: float = 0.0
        self._loop_region: Tuple[float, float] = (0.0, 5.0)
        self._file_counter: int = 0

        self._audio_output = QAudioOutput(self)
        self._audio_output.setVolume(0.9)
        self._player = QMediaPlayer(self)
        self._player.setAudioOutput(self._audio_output)

        self._player.positionChanged.connect(self._on_position_changed)
        self._player.playbackStateChanged.connect(self._on_state_changed)

        self._setup_ui()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(8)

        # 上段: 再生ボタン・停止・選択範囲ループ・A/B聴き比べ切替
        top_row = QHBoxLayout()

        self.btn_play_pause = QPushButton("▶ 再生")
        self.btn_play_pause.setObjectName("primaryButton")
        self.btn_play_pause.setMinimumWidth(110)
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)

        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.clicked.connect(self.stop)

        self.btn_play_region = QPushButton("🎯 選択範囲の先頭から再生")
        self.btn_play_region.clicked.connect(self.play_from_region_start)

        self.chk_loop_region = QCheckBox("🔁 選択範囲をループ再生")
        self.chk_loop_region.setChecked(False)

        top_row.addWidget(self.btn_play_pause)
        top_row.addWidget(self.btn_stop)
        top_row.addWidget(self.btn_play_region)
        top_row.addWidget(self.chk_loop_region)
        top_row.addSpacing(20)

        # A/B 比較ラジオボタン
        lbl_ab = QLabel("聴き比べ切替:")
        lbl_ab.setStyleSheet("font-weight: bold; color: #fbbf24;")
        self.rb_original = QRadioButton("🔵 元の音声 (Original)")
        self.rb_processed = QRadioButton("🟢 加工後 (Processed)")
        self.rb_original.setChecked(True)
        self.rb_processed.setEnabled(False)

        self._ab_group = QButtonGroup(self)
        self._ab_group.addButton(self.rb_original, 0)
        self._ab_group.addButton(self.rb_processed, 1)
        self._ab_group.idToggled.connect(self._on_ab_toggled)

        top_row.addWidget(lbl_ab)
        top_row.addWidget(self.rb_original)
        top_row.addWidget(self.rb_processed)
        top_row.addStretch()

        # 音量スライダー
        lbl_vol = QLabel("🔊 音量:")
        self.slider_vol = QSlider(Qt.Orientation.Horizontal)
        self.slider_vol.setRange(0, 100)
        self.slider_vol.setValue(90)
        self.slider_vol.setFixedWidth(100)
        self.slider_vol.valueChanged.connect(lambda v: self._audio_output.setVolume(v / 100.0))

        top_row.addWidget(lbl_vol)
        top_row.addWidget(self.slider_vol)
        main_layout.addLayout(top_row)

        # 下段: 現在時刻表示とシークスライダー
        bottom_row = QHBoxLayout()
        self.lbl_time = QLabel("00:00.00 / 00:00.00")
        self.lbl_time.setMinimumWidth(145)
        self.lbl_time.setStyleSheet("font-family: Consolas, monospace; font-size: 13px; color: #e2e8f0;")

        self.slider_seek = QSlider(Qt.Orientation.Horizontal)
        self.slider_seek.setRange(0, 1000)
        self.slider_seek.sliderMoved.connect(self._on_seek_slider_moved)

        bottom_row.addWidget(self.lbl_time)
        bottom_row.addWidget(self.slider_seek, stretch=1)
        main_layout.addLayout(bottom_row)

    def load_session_audio(
        self,
        original_audio: np.ndarray,
        sample_rate: int,
        processed_audio: Optional[np.ndarray] = None,
    ) -> None:
        """セッションの元音声および加工後音声をプレビュープレイヤーにセットする"""
        self.stop()
        self._file_counter += 1
        self._duration_sec = original_audio.shape[1] / float(sample_rate)

        orig_path = Path(self._temp_dir.name) / f"orig_{self._file_counter}.wav"
        sf.write(
            str(orig_path),
            np.clip(original_audio.T, -1.0, 1.0),
            sample_rate,
            subtype="PCM_16",
        )
        self._orig_wav_path = orig_path

        if processed_audio is not None:
            proc_path = Path(self._temp_dir.name) / f"proc_{self._file_counter}.wav"
            sf.write(
                str(proc_path),
                np.clip(processed_audio.T, -1.0, 1.0),
                sample_rate,
                subtype="PCM_16",
            )
            self._proc_wav_path = proc_path
            self._has_processed = True
            self.rb_processed.setEnabled(True)
            self.rb_processed.setChecked(True)
            self._set_media_source(proc_path, 0)
        else:
            self._proc_wav_path = None
            self._has_processed = False
            self.rb_processed.setEnabled(False)
            self.rb_original.setChecked(True)
            self._set_media_source(orig_path, 0)

        self._update_time_label(0.0)

    def update_processed_audio(self, processed_audio: np.ndarray, sample_rate: int) -> None:
        """加工完了時に加工後音声を更新し、自動的に「加工後 (Processed)」モードへ切り替える"""
        was_playing = self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        current_pos_ms = self._player.position()

        self._file_counter += 1
        proc_path = Path(self._temp_dir.name) / f"proc_{self._file_counter}.wav"
        sf.write(
            str(proc_path),
            np.clip(processed_audio.T, -1.0, 1.0),
            sample_rate,
            subtype="PCM_16",
        )
        self._proc_wav_path = proc_path
        self._has_processed = True
        self.rb_processed.setEnabled(True)

        if self.rb_processed.isChecked():
            self._set_media_source(proc_path, current_pos_ms, auto_play=was_playing)
        else:
            self.rb_processed.setChecked(True)

    def set_loop_region(self, start_sec: float, end_sec: float) -> None:
        """選択範囲の開始・終了秒数を更新する"""
        self._loop_region = (start_sec, end_sec)

    def seek_to_sec(self, sec: float) -> None:
        """指定秒数にシークする"""
        ms = int(max(0.0, min(sec, self._duration_sec)) * 1000.0)
        self._player.setPosition(ms)
        self._update_time_label(sec)

    def play_from_region_start(self) -> None:
        """選択範囲の先頭から再生を開始する"""
        s_sec, _ = self._loop_region
        self.seek_to_sec(s_sec)
        self._player.play()

    def toggle_play_pause(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            if self.chk_loop_region.isChecked():
                pos_sec = self._player.position() / 1000.0
                s_sec, e_sec = self._loop_region
                if pos_sec < s_sec or pos_sec >= e_sec:
                    self.seek_to_sec(s_sec)
            self._player.play()

    def stop(self) -> None:
        self._player.stop()
        self._update_time_label(0.0)
        self.position_sec_changed.emit(0.0)

    def _set_media_source(self, wav_path: Path, restore_pos_ms: int = 0, auto_play: bool = False) -> None:
        self._player.setSource(QUrl.fromLocalFile(str(wav_path)))
        if restore_pos_ms > 0:
            self._player.setPosition(restore_pos_ms)
        if auto_play:
            self._player.play()

    def _on_ab_toggled(self, button_id: int, checked: bool) -> None:
        if not checked:
            return
        use_processed = button_id == 1
        self._is_playing_processed = use_processed
        was_playing = self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        current_pos_ms = self._player.position()

        target_path = self._proc_wav_path if (use_processed and self._proc_wav_path) else self._orig_wav_path
        if target_path is not None and target_path.exists():
            self._set_media_source(target_path, restore_pos_ms=current_pos_ms, auto_play=was_playing)

        self.ab_mode_changed.emit(use_processed)

    def _on_position_changed(self, pos_ms: int) -> None:
        pos_sec = pos_ms / 1000.0
        if self.chk_loop_region.isChecked():
            s_sec, e_sec = self._loop_region
            if e_sec > s_sec + 0.1 and pos_sec >= e_sec:
                self._player.setPosition(int(s_sec * 1000.0))
                pos_sec = s_sec

        self._update_time_label(pos_sec)
        if self._duration_sec > 0 and not self.slider_seek.isSliderDown():
            ratio = int((pos_sec / self._duration_sec) * 1000.0)
            self.slider_seek.blockSignals(True)
            self.slider_seek.setValue(max(0, min(1000, ratio)))
            self.slider_seek.blockSignals(False)

        self.position_sec_changed.emit(pos_sec)

    def _on_seek_slider_moved(self, val: int) -> None:
        if self._duration_sec <= 0:
            return
        target_sec = (val / 1000.0) * self._duration_sec
        self.seek_to_sec(target_sec)
        self.position_sec_changed.emit(target_sec)

    def _on_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.btn_play_pause.setText("⏸ 一時停止")
        else:
            self.btn_play_pause.setText("▶ 再生")

    def _update_time_label(self, current_sec: float) -> None:
        self.lbl_time.setText(
            f"{format_time_mmss(current_sec)} / {format_time_mmss(self._duration_sec)}"
        )
