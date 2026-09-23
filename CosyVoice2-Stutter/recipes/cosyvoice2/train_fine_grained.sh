#!/usr/bin/env bash
# Research snapshot: distributed/adapted by CosyVoice2-Stutter; see NOTICE for provenance.
# Fine-grained instruction tuning: learn inline stutter event types/positions.
# This recipe does not add a separate normal/stutter mode instruction.
#
# Run from the CosyVoice repository root:
#   bash recipes/cosyvoice2/train_fine_grained.sh \
#     --annotation_dir data/Annotation \
#     --audio_dir data/Audio \
#     --pretrained_model_dir CosyVoice2-0.5B
#
# Expected annotation format:
#   start_seconds<TAB>end_seconds<TAB>annotated_text
#
# The script preserves your annotation tokens in the training text, for example:
#   [block], [phone_rep], <rep>...</rep>, <prolong>...</prolong>,
#   <fill>...</fill>, [noise]

set -euo pipefail

stage=0
stop_stage=5

annotation_dir=data/Annotation
audio_dir=data/Audio
pretrained_model_dir=CosyVoice2-0.5B
work_dir=data/stutter_cosyvoice2
exp_dir=exp/stutter_cosyvoice2
tensorboard_dir=tensorboard/stutter_cosyvoice2

config=configs/cosyvoice2.yaml
train_engine=torch_ddp
cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-0}
dist_backend=nccl
rdzv_port=12345
num_workers=2
prefetch=100
num_processes=8
num_utts_per_parquet=1000
dev_every=20
sample_rate=24000

print_usage() {
  cat <<EOF
Usage:
  bash recipes/cosyvoice2/train_fine_grained.sh [options]

Options:
  --annotation_dir DIR        Marked txt root. Default: data/Annotation
  --audio_dir DIR             Full recording audio root. Default: data/Audio
  --pretrained_model_dir DIR  CosyVoice2 model dir. Default: CosyVoice2-0.5B
  --work_dir DIR              Data/output working dir. Default: data/stutter_cosyvoice2
  --exp_dir DIR               Training checkpoint dir. Default: exp/stutter_cosyvoice2
  --tensorboard_dir DIR       Tensorboard dir. Default: tensorboard/stutter_cosyvoice2
  --config FILE               CosyVoice2 yaml. Default: configs/cosyvoice2.yaml
  --stage N                   Start stage. Default: 0
  --stop_stage N              Stop stage. Default: 5
  --dev_every N               Put every Nth utterance into dev. Default: 20
  --num_processes N           Parquet writer processes. Default: 8
  --num_workers N             Training dataloader workers. Default: 2
  --prefetch N                Training dataloader prefetch. Default: 100
  --train_engine NAME         torch_ddp or deepspeed. Default: torch_ddp
  --cuda_visible_devices STR  GPU list. Default: current CUDA_VISIBLE_DEVICES or 0
  --rdzv_port N               torchrun rendezvous port. Default: 12345
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --annotation_dir) annotation_dir="$2"; shift 2 ;;
    --audio_dir) audio_dir="$2"; shift 2 ;;
    --pretrained_model_dir) pretrained_model_dir="$2"; shift 2 ;;
    --work_dir) work_dir="$2"; shift 2 ;;
    --exp_dir) exp_dir="$2"; shift 2 ;;
    --tensorboard_dir) tensorboard_dir="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --stage) stage="$2"; shift 2 ;;
    --stop_stage) stop_stage="$2"; shift 2 ;;
    --dev_every) dev_every="$2"; shift 2 ;;
    --num_processes) num_processes="$2"; shift 2 ;;
    --num_workers) num_workers="$2"; shift 2 ;;
    --prefetch) prefetch="$2"; shift 2 ;;
    --train_engine) train_engine="$2"; shift 2 ;;
    --cuda_visible_devices) cuda_visible_devices="$2"; shift 2 ;;
    --rdzv_port) rdzv_port="$2"; shift 2 ;;
    -h|--help) print_usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; print_usage; exit 1 ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/third_party/Matcha-TTS:${ROOT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${cuda_visible_devices}"

annotation_dir="$(python -c "import os; print(os.path.abspath('${annotation_dir}'))")"
audio_dir="$(python -c "import os; print(os.path.abspath('${audio_dir}'))")"
pretrained_model_dir="$(python -c "import os; print(os.path.abspath('${pretrained_model_dir}'))")"
work_dir="$(python -c "import os; print(os.path.abspath('${work_dir}'))")"
exp_dir="$(python -c "import os; print(os.path.abspath('${exp_dir}'))")"
tensorboard_dir="$(python -c "import os; print(os.path.abspath('${tensorboard_dir}'))")"
config="$(python -c "import os; print(os.path.abspath('${config}'))")"

if [[ ! -d "${annotation_dir}" ]]; then
  echo "annotation_dir not found: ${annotation_dir}" >&2
  exit 1
fi
if [[ ! -d "${audio_dir}" ]]; then
  echo "audio_dir not found: ${audio_dir}" >&2
  exit 1
fi
if [[ ! -f "${pretrained_model_dir}/llm.pt" ]]; then
  echo "llm checkpoint not found: ${pretrained_model_dir}/llm.pt" >&2
  exit 1
fi
if [[ ! -f "${pretrained_model_dir}/speech_tokenizer_v2.onnx" ]]; then
  echo "speech tokenizer not found: ${pretrained_model_dir}/speech_tokenizer_v2.onnx" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required to cut full recordings into utterance wavs." >&2
  exit 1
fi

if [[ ${stage} -le 0 && ${stop_stage} -ge 0 ]]; then
  echo "Stage 0: prepare segmented wav.scp/text/utt2spk/spk2utt from marked txt"
  export STUTTER_ANNOTATION_DIR="${annotation_dir}"
  export STUTTER_AUDIO_DIR="${audio_dir}"
  export STUTTER_WORK_DIR="${work_dir}"
  export STUTTER_DEV_EVERY="${dev_every}"
  export STUTTER_SAMPLE_RATE="${sample_rate}"
  python - <<'PY'
import os
import subprocess
from pathlib import Path

annotation_dir = Path(os.environ["STUTTER_ANNOTATION_DIR"])
audio_dir = Path(os.environ["STUTTER_AUDIO_DIR"])
work_dir = Path(os.environ["STUTTER_WORK_DIR"])
dev_every = int(os.environ["STUTTER_DEV_EVERY"])
sample_rate = int(os.environ["STUTTER_SAMPLE_RATE"])

segment_dir = work_dir / "segments"
all_dir = work_dir / "all"
train_dir = work_dir / "train"
dev_dir = work_dir / "dev"
for d in [segment_dir, all_dir, train_dir, dev_dir]:
    d.mkdir(parents=True, exist_ok=True)

def find_audio(rel_txt):
    rel_no_ext = rel_txt.with_suffix("")
    speaker_id = rel_txt.parts[0] if len(rel_txt.parts) > 1 else rel_txt.stem
    candidates = []
    for ext in [".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"]:
        candidates.append(audio_dir / rel_no_ext.with_suffix(ext))
        candidates.append(audio_dir / (rel_no_ext.name + ext))
        candidates.append(audio_dir / (speaker_id + ext))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None

def write_kaldi_dir(target_dir, rows):
    with (target_dir / "wav.scp").open("w", encoding="utf-8") as wavscp, \
         (target_dir / "text").open("w", encoding="utf-8") as textf, \
         (target_dir / "utt2spk").open("w", encoding="utf-8") as utt2spk:
        for utt, wav, spk, text in rows:
            wavscp.write(f"{utt} {wav}\n")
            textf.write(f"{utt} {text}\n")
            utt2spk.write(f"{utt} {spk}\n")
    spk2utts = {}
    for utt, _, spk, _ in rows:
        spk2utts.setdefault(spk, []).append(utt)
    with (target_dir / "spk2utt").open("w", encoding="utf-8") as spk2utt:
        for spk in sorted(spk2utts):
            spk2utt.write(f"{spk} {' '.join(spk2utts[spk])}\n")

rows = []
missing_audio = []
bad_lines = []

for txt in sorted(annotation_dir.rglob("*.txt")):
    rel = txt.relative_to(annotation_dir)
    audio = find_audio(rel)
    if audio is None:
        missing_audio.append(str(rel))
        continue
    spk = rel.parts[0] if len(rel.parts) > 1 else rel.stem
    base = rel.with_suffix("").as_posix().replace("/", "_")
    with txt.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.rstrip("\n")
            parts = line.split("\t", 2)
            if len(parts) != 3:
                bad_lines.append(f"{rel}:{line_idx}: not 3 tab-separated fields")
                continue
            try:
                start = float(parts[0])
                end = float(parts[1])
            except ValueError:
                bad_lines.append(f"{rel}:{line_idx}: invalid time")
                continue
            text = parts[2].strip()
            if not text or end <= start:
                continue
            utt = f"{base}_{line_idx:06d}"
            wav = segment_dir / f"{utt}.wav"
            if not wav.exists():
                cmd = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(audio),
                    "-ss", f"{start:.6f}",
                    "-to", f"{end:.6f}",
                    "-ac", "1",
                    "-ar", str(sample_rate),
                    str(wav),
                ]
                subprocess.run(cmd, check=True)
            rows.append((utt, str(wav.resolve()), spk, text))

if missing_audio:
    msg = "\n".join(missing_audio[:20])
    raise SystemExit(f"Missing matching audio for {len(missing_audio)} txt files. Examples:\n{msg}")
if bad_lines:
    msg = "\n".join(bad_lines[:20])
    raise SystemExit(f"Bad annotation lines: {len(bad_lines)}. Examples:\n{msg}")
if not rows:
    raise SystemExit("No utterances prepared.")

train_rows, dev_rows = [], []
for idx, row in enumerate(rows, 1):
    if dev_every > 0 and idx % dev_every == 0:
        dev_rows.append(row)
    else:
        train_rows.append(row)
if not dev_rows:
    dev_rows = train_rows[-max(1, len(train_rows) // 20):]
    train_rows = train_rows[:-len(dev_rows)]

write_kaldi_dir(all_dir, rows)
write_kaldi_dir(train_dir, train_rows)
write_kaldi_dir(dev_dir, dev_rows)
print(f"prepared utterances: all={len(rows)} train={len(train_rows)} dev={len(dev_rows)}")
print(f"data root: {work_dir}")
PY
fi

if [[ ${stage} -le 1 && ${stop_stage} -ge 1 ]]; then
  echo "Stage 1: extract campplus speaker embeddings"
  for x in train dev; do
    python tools/extract_embedding.py \
      --dir "${work_dir}/${x}" \
      --onnx_path "${pretrained_model_dir}/campplus.onnx"
  done
fi

if [[ ${stage} -le 2 && ${stop_stage} -ge 2 ]]; then
  echo "Stage 2: extract CosyVoice2 speech tokens"
  for x in train dev; do
    python tools/extract_speech_token.py \
      --dir "${work_dir}/${x}" \
      --onnx_path "${pretrained_model_dir}/speech_tokenizer_v2.onnx"
  done
fi

if [[ ${stage} -le 3 && ${stop_stage} -ge 3 ]]; then
  echo "Stage 3: make parquet data lists"
  for x in train dev; do
    mkdir -p "${work_dir}/${x}/parquet"
    python tools/make_parquet_list.py \
      --num_utts_per_parquet "${num_utts_per_parquet}" \
      --num_processes "${num_processes}" \
      --src_dir "${work_dir}/${x}" \
      --des_dir "${work_dir}/${x}/parquet"
  done
fi

if [[ ${stage} -le 4 && ${stop_stage} -ge 4 ]]; then
  echo "Stage 4: check tokenizer size against Qwen embedding size"
  export STUTTER_PRETRAINED_MODEL_DIR="${pretrained_model_dir}"
  python - <<'PY'
import os
from cosyvoice.tokenizer.tokenizer import CosyVoice2Tokenizer
from transformers import Qwen2ForCausalLM

model_dir = os.environ["STUTTER_PRETRAINED_MODEL_DIR"]
qwen_dir = os.path.join(model_dir, "CosyVoice-BlankEN")
tokenizer = CosyVoice2Tokenizer(qwen_dir).tokenizer
model = Qwen2ForCausalLM.from_pretrained(qwen_dir)
tok_size = len(tokenizer)
emb_size = model.model.embed_tokens.num_embeddings
print(f"tokenizer_size={tok_size} embedding_size={emb_size}")
if tok_size > emb_size:
    raise SystemExit(
        "Tokenizer is larger than Qwen embedding. Add resize_token_embeddings "
        "before training/inference or reduce added tokens."
    )
PY
fi

if [[ ${stage} -le 5 && ${stop_stage} -ge 5 ]]; then
  echo "Stage 5: fine-tune CosyVoice2 llm only"
  num_gpus=$(python - <<'PY'
import os
v = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
print(len([x for x in v.split(",") if x.strip()]))
PY
)
  job_id=$((RANDOM + 10000))
  mkdir -p "${exp_dir}/llm/${train_engine}" "${tensorboard_dir}/llm/${train_engine}"
  torchrun --nnodes=1 --nproc_per_node="${num_gpus}" \
    --rdzv_id="${job_id}" --rdzv_backend="c10d" --rdzv_endpoint="localhost:${rdzv_port}" \
    cosyvoice/bin/train.py \
      --train_engine "${train_engine}" \
      --config "${config}" \
      --train_data "${work_dir}/train/parquet/data.list" \
      --cv_data "${work_dir}/dev/parquet/data.list" \
      --qwen_pretrain_path "${pretrained_model_dir}/CosyVoice-BlankEN" \
      --onnx_path "${pretrained_model_dir}" \
      --model llm \
      --checkpoint "${pretrained_model_dir}/llm.pt" \
      --model_dir "${exp_dir}/llm/${train_engine}" \
      --tensorboard_dir "${tensorboard_dir}/llm/${train_engine}" \
      --ddp.dist_backend "${dist_backend}" \
      --num_workers "${num_workers}" \
      --prefetch "${prefetch}" \
      --pin_memory \
      --use_amp \
      --deepspeed_config configs/deepspeed_stage2.json \
      --deepspeed.save_states model+optimizer
fi

echo "Done. Fine-tuned checkpoints are under: ${exp_dir}/llm/${train_engine}"
