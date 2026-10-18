"""
WaveformWidget の範囲選択UI動作テスト (`test_ui_waveform.py`)
- ファイル読み込み直後は「未選択（非表示）」であること
- 任意の範囲への変更・In/Outボタン・選択解除が正しく機能すること
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PySide6.QtWidgets import QApplication

from sound_remover.ui.waveform_widget import WaveformWidget


def _ensure_qapp() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app  # type: ignore[return-value]


def test_waveform_initial_no_selection_and_region_change() -> None:
    _ensure_qapp()
    w = WaveformWidget()

    sr = 48000
    dummy_audio = np.zeros((2, sr * 10), dtype=np.float32)  # 10秒の音声
    w.set_audio_data(dummy_audio, sr)

    # 1. 読み込み直後は未選択状態であること
    assert w.has_selection is False
    assert w.selection_region.isVisible() is False
    assert w.get_selected_region() == (0.0, 0.0)

    # 2. マウスドラッグまたは直接指定で任意の範囲 (2.5秒 〜 6.0秒) を選択・変更できること
    w._on_mouse_drag_selection(2.5, 6.0)
    assert w.has_selection is True
    assert w.selection_region.isVisible() is True
    s, e = w.get_selected_region()
    assert abs(s - 2.5) < 1e-3
    assert abs(e - 6.0) < 1e-3

    # 3. 別の範囲 (7.0秒 〜 9.2秒) へ自由に変更できること
    w._on_mouse_drag_selection(7.0, 9.2)
    s2, e2 = w.get_selected_region()
    assert abs(s2 - 7.0) < 1e-3
    assert abs(e2 - 9.2) < 1e-3

    # 4. 再生ヘッド位置からの [In] / [Out] ボタンで範囲変更できること
    w.set_playhead_sec(1.5)
    w.set_selection_in_at_playhead()
    w.set_playhead_sec(4.5)
    w.set_selection_out_at_playhead()
    s3, e3 = w.get_selected_region()
    assert abs(s3 - 1.5) < 1e-3
    assert abs(e3 - 4.5) < 1e-3

    # 5. 選択解除ボタンで未選択に戻ること
    w.clear_selection()
    assert w.has_selection is False
    assert w.selection_region.isVisible() is False
