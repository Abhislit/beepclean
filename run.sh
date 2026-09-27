#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
unset PYTHONPATH
VENV=".venv"
PY="$VENV/bin/python"
PORT="${PORT:-8000}"

find_python() {
  for candidate in python3.14 python3.13 python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done
  echo "No python3 found on PATH." >&2
  exit 1
}

if [ ! -x "$PY" ]; then
  echo "First run: creating the virtual environment (this takes a minute)..."
  if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  command -v uv >/dev/null 2>&1 || { export PATH="$HOME/.local/bin:$PATH"; }
  uv venv --python "$(find_python)" >/dev/null
  uv pip install --python "$PY" -r requirements.txt
  echo "Environment ready."
fi

echo
echo "  BeepClean is starting on http://localhost:$PORT"
echo "  Press Ctrl+C to stop."
echo
exec "$PY" -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
