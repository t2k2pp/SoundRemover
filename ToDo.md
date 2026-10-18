# SoundRemover 実装ToDoリスト

本ToDoリストに従い段階的に実装を進め、各ToDoステップの完了ごとにリモートリポジトリ (`https://github.com/t2k2pp/SoundRemover.git` の `main` ブランチ) へコミット・Pushを行います。

---

## ToDo一覧

- [x] **ToDo 1: 設計書 (`Doc/` 配下) および `ToDo.md` の作成とGitリポジトリ初期化・初回Push**
  - [x] `Doc/01_Architecture_and_Philosophy.md`（全体思想・アーキテクチャ設計書）の作成
  - [x] `Doc/02_Audio_Processing_Design.md`（音声処理・AIエンジン詳細設計書）の作成
  - [x] `Doc/03_UI_and_Distribution_Design.md`（GUI画面・マルチ環境配布設計書）の作成
  - [x] `ToDo.md` および `.gitignore` の作成、Git初期化・リモート登録・`main` ブランチへのPush

- [x] **ToDo 2: プロジェクト基盤・マルチGPUデバイス管理・FFmpeg自動解決・ロスレス音声抽出エンジンの実装**
  - [x] `pyproject.toml` の作成と `uv` による仮想環境（PyTorch CUDA / DirectML対応、Demucs、PySide6、static-ffmpeg、noisereduce、scipy、pyloudnorm等）の構築
  - [x] `src/sound_remover/core/device_manager.py`（CUDA / DirectML / MPS / ROCm / CPU 自動検出と切替）の実装
  - [x] `src/sound_remover/core/ffmpeg_manager.py`（システムFFmpeg / 内蔵static-ffmpeg 自動検出）の実装
  - [x] `src/sound_remover/core/audio_data.py`（音声バッファ、ピンポイント適用リージョン、曲分割マーカー管理）の実装
  - [x] `src/sound_remover/processors/extractor.py`（動画からの無劣化ストリームコピー抽出 `ffmpeg -vn -acodec copy`、WAV/FLAC抽出、高品位float32読み込み・保存）の実装

- [x] **ToDo 3: 音声加工・AI処理エンジン群（AIピンポイント声除去・ホワイトノイズ除去・屋外風EQ・吹奏楽マスタリング・非破壊パイプライン）の実装**
  - [x] `src/sound_remover/processors/vocal_remover.py`（Demucsによる声成分抽出、残差引き算方式、楽器保護帯域制限、特定時間帯のみのピンポイント適用＋クロスフェード合成）の実装
  - [x] `src/sound_remover/processors/noise_reducer.py`（スペクトルゲーティングによるホワイトノイズ除去、ノイズ見本区間からのプロファイル学習機能）の実装
  - [x] `src/sound_remover/processors/eq_filter.py`（屋外風切り音ローカットフィルタ、吹奏楽向け4バンドEQ）の実装
  - [x] `src/sound_remover/processors/mastering.py`（ラウドネスノーマライズ LUFS、音割れ防止ピークリミッター、曲分割＆フェードイン・アウト書き出し）の実装
  - [x] `src/sound_remover/core/pipeline.py`（各プロセッサを統合し、全体適用およびピンポイント区間適用を行う非破壊パイプライン）の実装

- [x] **ToDo 4: デスクトップGUI（波形ビューア・範囲選択・A/B聴き比べプレイヤー・各種設定タブ・プリセット）の実装**
  - [x] `src/sound_remover/ui/theme.py`（視認性の高いダークスタジオテーマ）の実装
  - [x] `src/sound_remover/ui/waveform_widget.py`（PyQtGraphによる高速波形描画、ドラッグ範囲選択、ピンポイント適用区間・曲分割マーカーの可視化）の実装
  - [x] `src/sound_remover/ui/player_widget.py`（Original / Processed を再生位置を維持したままワンクリックで聴き比べできるオーディオプレイヤー、ループ再生）の実装
  - [x] `src/sound_remover/ui/main_window.py` および `src/sound_remover/main.py`（全機能を統合したメインウィンドウ、非同期ワーカースレッド処理、プリセット切替）の実装

- [x] **ToDo 5: 自動テスト・サンプル検証用音源生成スクリプト・起動スクリプト・READMEの整備と最終動作確認**
  - [x] `tests/test_device_manager.py` および `tests/test_processors.py`（ロスレス抽出、声除去、ホワイトノイズ除去、EQ、マスタリング、曲分割の単体・結合テスト）の作成と全テスト実行
  - [x] 吹奏楽模擬テスト動画/音声生成スクリプトによる一連の動作確認とGUI起動確認
  - [x] `run_windows.bat`、`run_unix.sh`、`README.md`（他の保護者への配布手順・環境別セットアップ手順含む）の作成と最終Push

- [x] **ToDo 6: 波形ビューアの範囲選択UI・操作性の改善（実ファイル読み込み時のフィードバック対応）**
  - [x] ファイル読み込み直後は「範囲未選択（選択ボックス非表示）」にする
  - [x] 波形上の任意の場所を「マウス左ドラッグ」することで直接新しい選択範囲を作成・変更できるようにする（クリックのみの場合はその位置へ再生ヘッド移動）
  - [x] 再生位置連動の「`[ 開始点(In)`」「`終了点(Out) ]`」ボタン、開始・終了秒数の数値スピンボックス、「✖ 選択解除」ボタンを追加する
  - [x] 単体テストの追加・実行確認と引き継ぎ書更新・リモートリポジトリへのPush
