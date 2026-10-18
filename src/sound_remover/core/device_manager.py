"""
ハードウェア・アクセラレータ（GPU / CPU）検出・管理モジュール

対応環境:
- NVIDIA GPU (Windows / Linux): CUDA (`cuda:0`, ...)
- AMD GPU on Linux (Strix Halo Ryzen AI Max+ 395 / Radeon): ROCm (`cuda:0` via HIP)
- Apple Silicon (M1 / M2 / M3 / M4 Mac): MPS (`mps`)
- Windows AMD / Intel 内蔵・外付けGPU: DirectML (`privateuseone:0` via torch-directml)
- 全OS共通フォールバック: マルチスレッドCPU (`cpu`)
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass(frozen=True)
class ComputeDevice:
    """利用可能な演算デバイスの情報"""
    device_id: str          # 'cuda:0', 'mps', 'dml:0', 'cpu' 等
    backend_type: str       # 'CUDA', 'ROCm', 'MPS', 'DirectML', 'CPU'
    display_name: str       # UI表示用ラベル
    torch_device_str: str   # PyTorch / Demucs に渡すデバイス文字列
    is_gpu: bool            # GPUアクセラレータかどうか


class DeviceManager:
    """実行環境のGPU/CPUを自動検出し、最適なデバイスを選択・管理するクラス"""

    def __init__(self) -> None:
        self._devices: List[ComputeDevice] = []
        self._selected_device: Optional[ComputeDevice] = None
        self.refresh_devices()

    def refresh_devices(self) -> List[ComputeDevice]:
        """システム上で利用可能なすべての演算デバイスをスキャンする"""
        devices: List[ComputeDevice] = []

        # 1. CUDA / ROCm (HIP) の検出
        if torch.cuda.is_available():
            is_rocm = getattr(torch.version, "hip", None) is not None
            backend_label = "ROCm" if is_rocm else "CUDA"
            count = torch.cuda.device_count()
            for idx in range(count):
                try:
                    gpu_name = torch.cuda.get_device_name(idx)
                except Exception:
                    gpu_name = f"{backend_label} GPU #{idx}"
                dev_id = f"cuda:{idx}"
                devices.append(
                    ComputeDevice(
                        device_id=dev_id,
                        backend_type=backend_label,
                        display_name=f"{backend_label}: {gpu_name} ({dev_id})",
                        torch_device_str=dev_id,
                        is_gpu=True,
                    )
                )

        # 2. Apple Silicon MPS (macOS M1/M2/M3/M4) の検出
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            cpu_brand = platform.processor() or "Apple Silicon"
            devices.append(
                ComputeDevice(
                    device_id="mps",
                    backend_type="MPS",
                    display_name=f"MPS (Apple Metal GPU - {cpu_brand})",
                    torch_device_str="mps",
                    is_gpu=True,
                )
            )

        # 3. Windows DirectML (AMD Radeon / Intel Arc / Iris) の検出
        try:
            import torch_directml  # type: ignore

            if torch_directml.is_available():
                dml_count = torch_directml.device_count()
                for idx in range(dml_count):
                    dml_name = torch_directml.device_name(idx)
                    dml_dev = str(torch_directml.device(idx))
                    devices.append(
                        ComputeDevice(
                            device_id=f"dml:{idx}",
                            backend_type="DirectML",
                            display_name=f"DirectML: {dml_name}",
                            torch_device_str=dml_dev,
                            is_gpu=True,
                        )
                    )
        except ImportError:
            pass

        # 4. CPU（常に利用可能なフォールバック）
        cpu_cores = os.cpu_count() or 4
        cpu_name = platform.processor() or platform.machine() or "Generic CPU"
        devices.append(
            ComputeDevice(
                device_id="cpu",
                backend_type="CPU",
                display_name=f"CPU マルチスレッド ({cpu_cores}論理コア - {cpu_name})",
                torch_device_str="cpu",
                is_gpu=False,
            )
        )

        self._devices = devices
        if self._selected_device is None or all(
            d.device_id != self._selected_device.device_id for d in devices
        ):
            # デフォルトは最優先のGPU（リスト先頭）、無ければCPU
            self._selected_device = devices[0]

        return list(self._devices)

    @property
    def available_devices(self) -> List[ComputeDevice]:
        """検出されたデバイス一覧を返す"""
        return list(self._devices)

    @property
    def current_device(self) -> ComputeDevice:
        """現在選択されているデバイスを返す"""
        if self._selected_device is None:
            self.refresh_devices()
        assert self._selected_device is not None
        return self._selected_device

    def select_device_by_id(self, device_id: str) -> ComputeDevice:
        """
        指定した device_id ('auto', 'cuda:0', 'mps', 'dml:0', 'cpu' 等) に切り替える。
        'auto' または該当デバイスが存在しない場合は最優先デバイスを選択する。
        """
        if device_id == "auto":
            self._selected_device = self._devices[0]
            return self._selected_device

        for dev in self._devices:
            if dev.device_id == device_id:
                self._selected_device = dev
                return dev

        # 完全一致しない場合、前方一致（例: 'cuda' -> 'cuda:0'）を試みる
        for dev in self._devices:
            if dev.device_id.startswith(device_id):
                self._selected_device = dev
                return dev

        # フォールバックとしてCPUを選択
        self._selected_device = self._devices[-1]
        return self._selected_device

    def optimize_cpu_threads(self) -> None:
        """CPU推論時のスレッド数を最適化する"""
        cores = os.cpu_count() or 4
        optimal_threads = max(1, min(cores, 16))
        try:
            torch.set_num_threads(optimal_threads)
        except Exception:
            pass
