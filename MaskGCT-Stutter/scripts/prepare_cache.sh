#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KIND="${1:?Usage: prepare_cache.sh stutter|aishell1 [dataset arguments]}"
shift
cd "$ROOT/MaskGCT"
case "$KIND" in
  stutter) exec "${PYTHON:-python}" dataset.py prepare-cache "$@" ;;
  aishell1) exec "${PYTHON:-python}" dataset_aishell1.py prepare-cache "$@" ;;
  *) echo "Unknown dataset: $KIND" >&2; exit 2 ;;
esac
