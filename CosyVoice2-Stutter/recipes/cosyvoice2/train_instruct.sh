#!/usr/bin/env bash
# Research snapshot: distributed/adapted by CosyVoice2-Stutter; see NOTICE for provenance.
# Instruct tuning: learn normal/stutter mode instructions with mixed speech.
# Stutter target text retains fine-grained inline event instructions.
#
# Run from the CosyVoice repository root:
#   bash recipes/cosyvoice2/train_instruct.sh \
#     --annotation_dir data/Annotation \
#     --audio_dir data/Audio \
#     --aishell3_dir data/AISHELL-3 \
#     --pretrained_model_dir CosyVoice2-0.5B
#
# Expected stutter annotation format:
#   start_seconds<TAB>end_seconds<TAB>annotated_text
#
# Stutter text keeps its tags. AISHELL-3 text is used as clean text.
# Both types receive explicit control tags and share the same train/dev
# preparation pipeline.

set -euo pipefail

stage=0
stop_stage=5

annotation_dir=data/Annotation
audio_dir=data/Audio
aishell3_dir=data/AISHELL-3
aishell3_content=data/AISHELL-3/train/content.txt
aishell3_fraction=0.75
pretrained_model_dir=CosyVoice2-0.5B
work_dir=data/instruct_cosyvoice2
exp_dir=exp/instruct_cosyvoice2
tensorboard_dir=tensorboard/instruct_cosyvoice2

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
  bash recipes/cosyvoice2/train_instruct.sh [options]

Options:
  --annotation_dir DIR        Stutter marked txt root. Default: data/Annotation
  --audio_dir DIR             Stutter full recording audio root. Default: data/Audio
  --aishell3_dir DIR          AISHELL-3 root. Default: data/AISHELL-3
  --aishell3_content FILE     AISHELL-3 content.txt. Default: auto-detect
  --aishell3_fraction FLOAT   Fraction of AISHELL-3 content.txt to use. Default: 0.75
  --pretrained_model_dir DIR  CosyVoice2 model dir. Default: CosyVoice2-0.5B
  --work_dir DIR              Data/output working dir. Default: data/instruct_cosyvoice2
  --exp_dir DIR               Training checkpoint dir. Default: exp/instruct_cosyvoice2
  --tensorboard_dir DIR       Tensorboard dir. Default: tensorboard/instruct_cosyvoice2
  --config FILE               CosyVoice2 yaml. Default: configs/cosyvoice2.yaml
  --stage N                   Start stage. Default: 0
  --stop_stage N              Stop stage. Default: 5
  --num_processes N           Parquet writer processes. Default: 8
  --num_workers N             Training dataloader workers. Default: 2
  --prefetch N                Training dataloader prefetch. Default: 100
  --dev_every N               Put every Nth utterance into dev inside each data type. Default: 20
  --train_engine NAME         torch_ddp or deepspeed. Default: torch_ddp
  --cuda_visible_devices STR  GPU list. Default: current CUDA_VISIBLE_DEVICES or 0
  --rdzv_port N               torchrun rendezvous port. Default: 12345
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --annotation_dir) annotation_dir="$2"; shift 2 ;;
    --audio_dir) audio_dir="$2"; shift 2 ;;
    --aishell3_dir) aishell3_dir="$2"; shift 2 ;;
    --aishell3_content) aishell3_content="$2"; shift 2 ;;
    --aishell3_fraction) aishell3_fraction="$2"; shift 2 ;;
    --pretrained_model_dir) pretrained_model_dir="$2"; shift 2 ;;
    --work_dir) work_dir="$2"; shift 2 ;;
    --exp_dir) exp_dir="$2"; shift 2 ;;
    --tensorboard_dir) tensorboard_dir="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --stage) stage="$2"; shift 2 ;;
    --stop_stage) stop_stage="$2"; shift 2 ;;
    --num_processes) num_processes="$2"; shift 2 ;;
    --num_workers) num_workers="$2"; shift 2 ;;
    --prefetch) prefetch="$2"; shift 2 ;;
    --dev_every) dev_every="$2"; shift 2 ;;
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
aishell3_dir="$(python -c "import os; print(os.path.abspath('${aishell3_dir}'))")"
if [[ -n "${aishell3_content}" ]]; then
  aishell3_content="$(python -c "import os; print(os.path.abspath('${aishell3_content}'))")"
fi
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
if [[ ! -d "${aishell3_dir}" ]]; then
  echo "aishell3_dir not found: ${aishell3_dir}" >&2
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
  echo "ffmpeg is required to cut full stutter recordings into utterance wavs." >&2
  exit 1
fi

if [[ ${stage} -le 0 && ${stop_stage} -ge 0 ]]; then
  echo "Stage 0: prepare normal/stutter wav.scp/text/utt2spk/spk2utt/instruct"
  export INSTRUCT_ANNOTATION_DIR="${annotation_dir}"
  export INSTRUCT_AUDIO_DIR="${audio_dir}"
  export INSTRUCT_AISHELL3_DIR="${aishell3_dir}"
  export INSTRUCT_AISHELL3_CONTENT="${aishell3_content}"
  export INSTRUCT_AISHELL3_FRACTION="${aishell3_fraction}"
  export INSTRUCT_WORK_DIR="${work_dir}"
  export INSTRUCT_DEV_EVERY="${dev_every}"
  export INSTRUCT_SAMPLE_RATE="${sample_rate}"
  python - <<'PY'
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

annotation_dir = Path(os.environ["INSTRUCT_ANNOTATION_DIR"])
audio_dir = Path(os.environ["INSTRUCT_AUDIO_DIR"])
aishell3_dir = Path(os.environ["INSTRUCT_AISHELL3_DIR"])
aishell3_content_env = os.environ["INSTRUCT_AISHELL3_CONTENT"]
aishell3_content = Path(aishell3_content_env) if aishell3_content_env else None
aishell3_fraction = float(os.environ["INSTRUCT_AISHELL3_FRACTION"])
work_dir = Path(os.environ["INSTRUCT_WORK_DIR"])
dev_every = int(os.environ["INSTRUCT_DEV_EVERY"])
sample_rate = int(os.environ["INSTRUCT_SAMPLE_RATE"])

if not (0 < aishell3_fraction <= 1):
    raise SystemExit(f"aishell3_fraction must be in (0, 1], got {aishell3_fraction}")

segment_dir = work_dir / "segments"
all_dir = work_dir / "all"
train_dir = work_dir / "train"
dev_dir = work_dir / "dev"
for directory in [segment_dir, all_dir, train_dir, dev_dir]:
    directory.mkdir(parents=True, exist_ok=True)

normal_instruction = "<|normal|><|endofprompt|>"
stutter_instruction = "<|stutter|><|endofprompt|>"

def instruction(kind):
    if kind == "normal":
        return normal_instruction
    if kind == "stutter":
        return stutter_instruction
    raise ValueError(kind)

space_re = re.compile(r"\s+")
comma_like = "\uFF0C,\u3001\uFF1B;\uFF1A:"
end_like = "\u3002\uFF01\uFF1F.!?"
leading_punct_re = re.compile(rf"^[{comma_like}{end_like}\s]+")
trailing_comma_re = re.compile(rf"[{comma_like}\s]+$")
comma_before_end_re = re.compile(rf"[{comma_like}]+(?=[{end_like}])")
comma_run_re = re.compile(rf"[{comma_like}]{{2,}}")
end_run_patterns = [
    (re.compile(r"\u3002{2,}"), "\u3002"),
    (re.compile(r"\uFF01{2,}"), "\uFF01"),
    (re.compile(r"\uFF1F{2,}"), "\uFF1F"),
    (re.compile(r"!{2,}"), "!"),
    (re.compile(r"\?{2,}"), "?"),
]

def cleanup_punctuation(text):
    text = space_re.sub("", text)
    previous = None
    while previous != text:
        previous = text
        text = comma_before_end_re.sub("", text)
        text = comma_run_re.sub("\uFF0C", text)
        for pattern, repl in end_run_patterns:
            text = pattern.sub(repl, text)
        text = leading_punct_re.sub("", text)
        text = trailing_comma_re.sub("", text)
    return text

def read_lines(path):
    raw = path.read_bytes()
    for encoding in ["utf-8-sig", "utf-8", "gb18030"]:
        try:
            return raw.decode(encoding).splitlines()
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore").splitlines()

def find_stutter_audio(spk):
    for suffix in [".wav", ".flac", ".mp3", ".m4a"]:
        direct = audio_dir / f"{spk}{suffix}"
        if direct.exists():
            return direct
    matches = sorted(audio_dir.rglob(f"{spk}.*"))
    return matches[0] if matches else None

def resolve_aishell3_content():
    candidates = []
    if aishell3_content is not None:
        candidates.append(aishell3_content)
    candidates.extend([
        aishell3_dir / "content.txt",
        aishell3_dir / "train" / "content.txt",
        aishell3_dir.parent / "content.txt",
        Path.cwd() / "content.txt",
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit("AISHELL-3 content.txt not found. Use --aishell3_content FILE.")

def resolve_aishell3_wav(wav_part):
    wav_path = Path(wav_part)
    if wav_path.is_absolute():
        return wav_path if wav_path.exists() else None
    candidates = [
        aishell3_dir / wav_path,
        aishell3_dir.parent / wav_path,
        Path.cwd() / wav_path,
    ]
    return next((candidate for candidate in candidates if candidate.exists()), None)

def build_normal_rows():
    content_file = resolve_aishell3_content()
    parsed = []
    for line in read_lines(content_file):
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            wav_part, text = line.split("\t", 1)
        else:
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            wav_part, text = parts
        wav_path = resolve_aishell3_wav(wav_part.strip())
        clean_text = cleanup_punctuation(text.strip())
        if wav_path is not None and clean_text:
            parsed.append((wav_path, clean_text))
    keep = int(len(parsed) * aishell3_fraction)
    keep = max(1, keep) if parsed else 0
    rows = []
    for index, (wav_path, text) in enumerate(parsed[:keep], 1):
        utt = f"normal_{wav_path.stem}_{index:06d}"
        spk = f"normal_{wav_path.parent.name}"
        rows.append((utt, str(wav_path.resolve()), spk, text, instruction("normal"), "normal"))
    return rows

def build_stutter_rows():
    rows = []
    missing_audio = []
    bad_lines = []
    for txt in sorted(annotation_dir.rglob("*.txt")):
        rel = txt.relative_to(annotation_dir)
        spk = rel.parts[0] if len(rel.parts) > 1 else rel.stem
        audio = find_stutter_audio(spk)
        if audio is None:
            missing_audio.append(str(rel))
            continue
        base = rel.with_suffix("").as_posix().replace("/", "_")
        for line_idx, line in enumerate(read_lines(txt), 1):
            raw = line.rstrip("\n")
            parts = raw.split("\t", 2)
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
            utt = f"stutter_{base}_{line_idx:06d}"
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
            rows.append((utt, str(wav.resolve()), f"stutter_{spk}", text, instruction("stutter"), "stutter"))
    if missing_audio:
        msg = "\n".join(missing_audio[:20])
        raise SystemExit(f"Missing matching stutter audio for {len(missing_audio)} txt files. Examples:\n{msg}")
    if bad_lines:
        msg = "\n".join(bad_lines[:20])
        raise SystemExit(f"Bad stutter annotation lines: {len(bad_lines)}. Examples:\n{msg}")
    return rows

def split_rows(rows):
    train_rows, dev_rows = [], []
    for idx, row in enumerate(rows, 1):
        if dev_every > 0 and idx % dev_every == 0:
            dev_rows.append(row)
        else:
            train_rows.append(row)
    if not dev_rows and len(train_rows) > 1:
        dev_size = max(1, len(train_rows) // 20)
        dev_rows = train_rows[-dev_size:]
        train_rows = train_rows[:-dev_size]
    return train_rows, dev_rows

def write_kaldi_dir(target_dir, rows):
    with (target_dir / "wav.scp").open("w", encoding="utf-8") as wavscp, \
         (target_dir / "text").open("w", encoding="utf-8") as textf, \
         (target_dir / "utt2spk").open("w", encoding="utf-8") as utt2spk, \
         (target_dir / "instruct").open("w", encoding="utf-8") as instructf:
        for utt, wav, spk, text, instruct, _kind in rows:
            wavscp.write(f"{utt} {wav}\n")
            textf.write(f"{utt} {text}\n")
            utt2spk.write(f"{utt} {spk}\n")
            instructf.write(f"{utt} {instruct}\n")
    spk2utts = {}
    for utt, _wav, spk, _text, _instruct, _kind in rows:
        spk2utts.setdefault(spk, []).append(utt)
    with (target_dir / "spk2utt").open("w", encoding="utf-8") as spk2utt:
        for spk in sorted(spk2utts):
            spk2utt.write(f"{spk} {' '.join(spk2utts[spk])}\n")

stutter_rows = build_stutter_rows()
normal_rows = build_normal_rows()

if not stutter_rows:
    raise SystemExit("No stutter utterances prepared.")
if not normal_rows:
    raise SystemExit("No normal AISHELL-3 utterances prepared.")

all_rows = stutter_rows + normal_rows
train_rows, dev_rows = [], []
for typed_rows in [stutter_rows, normal_rows]:
    typed_train, typed_dev = split_rows(typed_rows)
    train_rows.extend(typed_train)
    dev_rows.extend(typed_dev)

write_kaldi_dir(all_dir, all_rows)
write_kaldi_dir(train_dir, train_rows)
write_kaldi_dir(dev_dir, dev_rows)

print(f"prepared utterances: all={len(all_rows)} train={len(train_rows)} dev={len(dev_rows)}")
print(f"AISHELL-3 fraction used: {aishell3_fraction:.4f}")
for name, rows in [("all", all_rows), ("train", train_rows), ("dev", dev_rows)]:
    counts = Counter(row[5] for row in rows)
    print(
        f"{name}: normal={counts.get('normal', 0)} "
        f"stutter={counts.get('stutter', 0)}"
    )
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
  export INSTRUCT_PRETRAINED_MODEL_DIR="${pretrained_model_dir}"
  python - <<'PY'
import os
from cosyvoice.tokenizer.tokenizer import CosyVoice2Tokenizer
from transformers import Qwen2ForCausalLM

model_dir = os.environ["INSTRUCT_PRETRAINED_MODEL_DIR"]
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
