#!/usr/bin/env bash
# Publication-scoped replacement for the original all-method orchestration.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON:-$ROOT/.venv/bin/python}" "$ROOT/scripts/run_benchmark.py" "$@"