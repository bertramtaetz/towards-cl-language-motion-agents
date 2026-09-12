#!/usr/bin/env bash
# Install only into this repository's .venv; never reuse source environments.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ ! -x .venv/bin/python ]]; then
  python3.12 -m venv .venv
fi
.venv/bin/python -c 'import sys; from pathlib import Path; assert Path(sys.prefix).resolve() == Path(".venv").resolve(); assert sys.version_info[:2] == (3, 12)'
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt ./third_party/nlg-eval
.venv/bin/python -m pip check