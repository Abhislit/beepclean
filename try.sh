#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
unset PYTHONPATH
if [ ! -x .venv/bin/python ]; then
  echo "No environment yet. Run ./run.sh once first." >&2
  exit 1
fi
exec .venv/bin/python -m app.cli "$@"
