#!/usr/bin/env bash
# Research snapshot: distributed/adapted by CosyVoice2-Stutter; see NOTICE for provenance.
# Resynthesize a CosyVoice2 training set with a complete checkpoint directory.

set -euo pipefail

model_dir=
data_dir=data/instruct_cosyvoice2/train
output_dir=data/instruct_cosyvoice2_resynth/train
spk2prompt=
instruct='<|stutter|><|endofprompt|>'
select_instruct_token='<|stutter|>'
cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-0}
prompt_target_seconds=5.0
prompt_min_seconds=2.0
prompt_max_seconds=10.0
seed=1986
limit=-1
fp16=false
overwrite=false

print_usage() {
  cat <<EOF
Usage:
  bash recipes/cosyvoice2/synthesize_dataset.sh \\
    --model_dir path/to/complete-cosyvoice2-checkpoint

Options:
  --model_dir DIR            Complete inference-ready CosyVoice2 checkpoint directory. Required.
  --data_dir DIR             Input Kaldi training directory. Default: data/instruct_cosyvoice2/train
  --output_dir DIR           Synthesized Kaldi directory. Default: data/instruct_cosyvoice2_resynth/train
  --spk2prompt FILE          Optional speaker-to-reference-WAV mapping.
  --instruct TEXT            Instruction used for every utterance. Default: <|stutter|><|endofprompt|>
  --select_instruct_token T  Only synthesize source rows containing T. Default: <|stutter|>
  --cuda_visible_devices STR GPU list. Default: current CUDA_VISIBLE_DEVICES or 0
  --prompt_target_seconds N  Preferred prompt duration. Default: 5.0
  --prompt_min_seconds N     Minimum preferred prompt duration. Default: 2.0
  --prompt_max_seconds N     Maximum preferred prompt duration. Default: 10.0
  --seed N                   Base random seed. Default: 1986
  --limit N                  Only synthesize first N utterances; -1 means all.
  --fp16                     Use fp16 inference on CUDA.
  --overwrite                Regenerate existing output WAV files.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_dir) model_dir="$2"; shift 2 ;;
    --data_dir) data_dir="$2"; shift 2 ;;
    --output_dir) output_dir="$2"; shift 2 ;;
    --spk2prompt) spk2prompt="$2"; shift 2 ;;
    --instruct) instruct="$2"; shift 2 ;;
    --select_instruct_token) select_instruct_token="$2"; shift 2 ;;
    --cuda_visible_devices) cuda_visible_devices="$2"; shift 2 ;;
    --prompt_target_seconds) prompt_target_seconds="$2"; shift 2 ;;
    --prompt_min_seconds) prompt_min_seconds="$2"; shift 2 ;;
    --prompt_max_seconds) prompt_max_seconds="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --limit) limit="$2"; shift 2 ;;
    --fp16) fp16=true; shift ;;
    --overwrite) overwrite=true; shift ;;
    -h|--help) print_usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; print_usage; exit 1 ;;
  esac
done

if [[ -z "${model_dir}" ]]; then
  echo "--model_dir is required." >&2
  print_usage
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/third_party/Matcha-TTS:${ROOT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${cuda_visible_devices}"

cmd=(
  python recipes/cosyvoice2/synthesize_dataset.py
  --model_dir "${model_dir}"
  --data_dir "${data_dir}"
  --output_dir "${output_dir}"
  --instruct "${instruct}"
  --select_instruct_token "${select_instruct_token}"
  --prompt_target_seconds "${prompt_target_seconds}"
  --prompt_min_seconds "${prompt_min_seconds}"
  --prompt_max_seconds "${prompt_max_seconds}"
  --seed "${seed}"
  --limit "${limit}"
)

if [[ -n "${spk2prompt}" ]]; then
  cmd+=(--spk2prompt "${spk2prompt}")
fi
if [[ "${fp16}" == true ]]; then
  cmd+=(--fp16)
fi
if [[ "${overwrite}" == true ]]; then
  cmd+=(--overwrite)
fi

"${cmd[@]}"
