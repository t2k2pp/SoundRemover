#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -x ".venv/bin/python" ]; then
    echo "[SoundRemover] 仮想環境からアプリを起動しています..."
    exec ".venv/bin/python" -m sound_remover.main "$@"
fi

if command -v uv >/dev/null 2>&1; then
    echo "[SoundRemover] uv で環境を準備して起動しています..."
    exec uv run sound-remover "$@"
fi

echo "[エラー] .venv または uv が見つかりませんでした。先に uv sync を実行してください。"
exit 1
