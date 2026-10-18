# SoundRemover 作業引き継ぎ書 (Handover Document)

最終更新: 2026-10-18 (全ToDo 1〜5 実装・テスト完了)

---

## 1. プロジェクト概要・目的
- **目的**: 中学生の吹奏楽部の発表動画（屋外ステージ・ホール・体育館など）からロスレス（無劣化）で音声を抽出し、演奏音質（楽器の倍音やダイナミクス）を落とさずに「観客の話し声・アナウンス・咳払い」「ホワイトノイズ・空調音」「屋外の風切り音」をピンポイントで除去・音量調整・曲分割できるデスクトップGUIアプリを作成する。
- **作業ディレクトリ**: `C:\Users\osia3\GitProjects\movieseparator`
- **リモートリポジトリ**: `https://github.com/t2k2pp/SoundRemover.git` (`main` ブランチ)

---

## 2. 厳守すべき開発ルール・ユーザー要件
1. **設計とToDo管理**: 設計書を `Doc/` 配下に残し、`ToDo.md` および本引き継ぎ書 (`Doc/04_Handover_and_Status.md`) をステップごとに更新する。
2. **こまめなPush**: 各ToDoの区切りごとにリモートリポジトリ (`origin/main`) にコミット・Pushする。
3. **Gitコマンドの制約**: Gitコマンドを `&&` で繋がない（1行ずつ改行して実行する）。
4. **コマンド実行時の説明**: コマンドを実行する際は、日本語で「何のために何の許可を貰おうとしているのか」を明示する。
5. **コード削除・変更時の注意**: 意図しない処理の削除がないか確認し、用途不明のコードを勝手に削除しない。楽な方法への勝手な変更や妥協をしない。
6. **マルチ環境・配布対応**:
   - 開発機(Windows): `NVIDIA GeForce RTX 3060 Laptop GPU` + `AMD Radeon(TM) Graphics`
   - 他所有機: `M4 Mac mini` (MPS)、`AMD AI Max+ 395 Strix Halo` (Linux ROCm)
   - 他の吹奏楽部の保護者への配布も見据え、環境に合わせて `CUDA` / `DirectML` / `MPS` / `ROCm` / `CPU` を自動選択・手動切替できるようにする。

---

## 3. 現在の進捗状況（全基本実装完了）

### 完了済み
- **[完了] ToDo 1: 設計書およびToDoリストの作成・初回Push** (Commit: `b0e5f7c`)
  - `Doc/01_Architecture_and_Philosophy.md`, `Doc/02_Audio_Processing_Design.md`, `Doc/03_UI_and_Distribution_Design.md`, `ToDo.md`, `.gitignore`
- **[完了] ToDo 2: プロジェクト基盤・マルチGPUデバイス管理・FFmpeg自動解決・ロスレス音声抽出エンジンの実装** (Commit: `fea7257`)
  - `pyproject.toml`, `uv.lock`, `.venv`, `README.md`
  - `src/sound_remover/core/device_manager.py`, `src/sound_remover/core/ffmpeg_manager.py`, `src/sound_remover/core/audio_data.py`
  - `src/sound_remover/processors/extractor.py`
- **[完了] ToDo 3: 音声加工・AI処理エンジン群の実装** (Commit: `d6aefd3`)
  - `src/sound_remover/processors/vocal_remover.py`, `src/sound_remover/processors/noise_reducer.py`, `src/sound_remover/processors/eq_filter.py`, `src/sound_remover/processors/mastering.py`, `src/sound_remover/core/pipeline.py`
- **[完了] ToDo 4: デスクトップGUI (`PySide6` + `PyQtGraph`) の実装** (Commit: `fbe66bd`)
  - `src/sound_remover/ui/theme.py`, `src/sound_remover/ui/waveform_widget.py`, `src/sound_remover/ui/player_widget.py`, `src/sound_remover/ui/main_window.py`, `src/sound_remover/main.py`
- **[完了] ToDo 5: 自動テスト・サンプル検証用音源生成スクリプト・起動スクリプト整備と最終動作確認**
  - `tests/test_device_manager.py`, `tests/test_processors.py` (`pytest` 全8テストPASS確認：`run_windows.bat --check` の実起動テスト含む)
  - `scripts/generate_sample_video.py` (屋外吹奏楽模擬サンプル動画生成スクリプト)
  - `run_windows.bat`, `run_unix.sh` (ワンクリック起動スクリプト：`cmd.exe` のCP932文字化け・改行ズレ防止のため純ASCII化済)

---

## 4. 今後の拡張・実サンプル動画での調整時の引き継ぎ事項

1. **実物の吹奏楽動画サンプルでのチューニング**
   - ユーザーの実撮影動画サンプルを用いて、声の除去強度（デフォルト85%）、楽器保護帯域（現在 `110Hz〜6500Hz`）、および屋外ステージ用ローカット周波数を微調整する場合は、`src/sound_remover/core/audio_data.py` のプリセット値または `src/sound_remover/processors/vocal_remover.py` の `_bandpass_vocal_stem` を調整する。
2. **アプリの起動・テスト実行コマンド**
   - GUI起動 (Windows): `run_windows.bat` をダブルクリック、または `& "C:\Users\osia3\.local\bin\uv.exe" run sound-remover`
   - テスト実行: `& "C:\Users\osia3\.local\bin\uv.exe" run pytest -v`
   - サンプル動画生成: `& "C:\Users\osia3\.local\bin\uv.exe" run python scripts/generate_sample_video.py`
