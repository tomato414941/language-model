#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$project_root/.tools/uv-cache}"
if [[ ! -x "$project_root/.tools/uv" ]]; then
  installer="$(mktemp)"
  trap 'rm -f "$installer"' EXIT
  curl --fail --silent --show-error --location \
    https://astral.sh/uv/0.10.12/install.sh --output "$installer"
  UV_INSTALL_DIR="$project_root/.tools" UV_NO_MODIFY_PATH=1 sh "$installer"
fi
"$project_root/.tools/uv" sync --frozen --extra train --extra hub --extra eval
# The lock pins the CUDA 13 build of PyTorch. On a host whose driver predates
# CUDA 13, use the same PyTorch version built for CUDA 12.6 instead.
if ! "$project_root/.tools/uv" run --no-sync python -c \
  'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  version="$("$project_root/.tools/uv" run --no-sync python -c \
    'import torch; print(torch.__version__.split("+")[0])')"
  "$project_root/.tools/uv" pip install --reinstall-package torch \
    --index-url https://download.pytorch.org/whl/cu126 "torch==$version"
fi
"$project_root/.tools/uv" run --no-sync python - <<'PY'
import json
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU is not available")
print(json.dumps({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
}))
PY
