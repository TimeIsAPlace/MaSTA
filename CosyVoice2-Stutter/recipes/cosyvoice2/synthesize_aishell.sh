#!/usr/bin/env bash
# Research snapshot: distributed/adapted by CosyVoice2-Stutter; see NOTICE for provenance.
# Synthesize the first 50,000 AISHELL-Stutter transcripts with a complete,
# inference-ready fine-tuned CosyVoice2 checkpoint.  prompt_wav is selected
# automatically from another recording of the same speaker.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: bash $0 [MODEL_DIR [AUDIO_ROOT [OUTPUT_DIR]]]"
  echo "AUDIO_ROOT contains AISHELL-1/. Default: CosyVoice2-0.5B data data/aishell_stutter_synth_50k"
  echo "For --dry_run, --num_lines and other options, use the Python entry point."
  exit 0
fi
if [[ $# -gt 3 ]]; then
  echo "Expected at most three positional arguments. Use --help." >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# Defaults assume the command is run for this repository layout. AUDIO_ROOT is
# the parent directory of AISHELL-1, not AISHELL-1 itself.
MODEL_DIR=${1:-CosyVoice2-0.5B}
AUDIO_ROOT=${2:-data}
OUTPUT_DIR=${3:-data/aishell_stutter_synth_50k}

export PYTHONPATH="${ROOT_DIR}/third_party/Matcha-TTS:${ROOT_DIR}:${PYTHONPATH:-}"

python recipes/cosyvoice2/synthesize_aishell.py \
  --model_dir "${MODEL_DIR}" \
  --input_file aishell_stutter.txt \
  --audio_root "${AUDIO_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_lines 50000 \
  --instruct '<|stutter|><|endofprompt|>' \
  --fp16
