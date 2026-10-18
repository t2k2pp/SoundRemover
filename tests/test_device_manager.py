"""
マルチGPU・CPUデバイス検出・切替ロジックおよび起動スクリプトの単体テスト (`test_device_manager.py`)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from sound_remover.core.device_manager import DeviceManager
from sound_remover.core.ffmpeg_manager import FFmpegManager


def test_device_manager_enumeration_and_selection() -> None:
    dm = DeviceManager()
    devices = dm.available_devices
    assert len(devices) >= 1
    # 末尾には必ずCPUフォールバックが存在すること
    assert devices[-1].device_id == "cpu"
    assert devices[-1].backend_type == "CPU"
    assert not devices[-1].is_gpu

    # 'auto' 選択時は先頭（最優先デバイス）が選ばれること
    auto_dev = dm.select_device_by_id("auto")
    assert auto_dev == devices[0]

    # 'cpu' 明示選択時はCPUに切り替わること
    cpu_dev = dm.select_device_by_id("cpu")
    assert cpu_dev.device_id == "cpu"

    # 未知のIDが指定された場合もクラッシュせずCPUへフォールバックすること
    fallback_dev = dm.select_device_by_id("non_existent_accelerator_99")
    assert fallback_dev.device_id == "cpu"


def test_ffmpeg_manager_resolves_binary() -> None:
    ffmpeg_path, _ = FFmpegManager.ensure_ffmpeg()
    assert ffmpeg_path is not None
    assert len(ffmpeg_path) > 0


def test_run_windows_bat_execution() -> None:
    """Windowsバッチファイル (run_windows.bat) が文字化けや構文エラーなく正常実行できることを検証"""
    if sys.platform != "win32":
        return

    bat_path = Path(__file__).resolve().parent.parent / "run_windows.bat"
    assert bat_path.exists()

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"

    res = subprocess.run(
        ["cmd.exe", "/c", str(bat_path), "--check"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    assert res.returncode == 0, f"run_windows.bat failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    assert "Self-check OK" in res.stdout
