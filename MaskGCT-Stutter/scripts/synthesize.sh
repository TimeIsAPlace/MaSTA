#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/MaskGCT"
exec "${PYTHON:-python}" infer_fill_mask.py "$@"
