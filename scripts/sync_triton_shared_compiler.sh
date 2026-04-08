#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO_ROOT/backend/compiler.py"

if [[ ! -f "$SRC" ]]; then
  echo "[error] source not found: $SRC" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
TARGET="$($PYTHON_BIN - <<'PY'
import triton.backends.triton_shared.compiler as c
print(c.__file__)
PY
)"

if [[ -z "$TARGET" || ! -f "$TARGET" ]]; then
  echo "[error] target compiler.py not found via $PYTHON_BIN" >&2
  exit 1
fi

BACKUP="${TARGET}.bak.$(date +%Y%m%d%H%M%S)"
cp "$TARGET" "$BACKUP"
cp "$SRC" "$TARGET"

echo "[ok] synced compiler.py"
echo "  source: $SRC"
echo "  target: $TARGET"
echo "  backup: $BACKUP"

$PYTHON_BIN - <<'PY'
import inspect
import triton.backends.triton_shared.compiler as c
src = inspect.getsource(c)
print("[check] has env key:", "TRITON_RISCV_STRUCTURED_LDST_MODE" in src)
print("[check] has tensor-first pass option:", "structured-ldst-mode=tensor-first-vector-cpu" in src)
PY
