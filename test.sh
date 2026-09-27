#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
unset PYTHONPATH
exec .venv/bin/python -m pytest "$@"
