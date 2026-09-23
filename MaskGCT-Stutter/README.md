# MaskGCT-Stutter

Tag-conditioned Mandarin stuttering speech synthesis with a three-stage
MaskGCT fine-tuning pipeline. This repository contains the MaskGCT branch only:
token-cache preparation, training, event-region infilling, and waveform synthesis.
No private speech, pretrained weights, ASR experiments, or CosyVoice code is bundled.

See [中文使用说明](README_zh.md) for the full workflow and
[data specification](docs/DATA.md) for manifest/cache fields.

## Three Training Stages

| Stage | Data | Objective | Entry point | Initialization |
| --- | --- | --- | --- | --- |
| 1. Tag-conditioned T2S adaptation | Real stuttering speech | Learn speech/text-tag conditioning using the original T2S masking objective | `MaskGCT/train.py` | Pretrained T2S |
| 2. General speech infilling | AISHELL-1 fluent speech | Reconstruct randomly selected character-aligned regions (20-40% of characters) | `MaskGCT/train_fill_mask.py` | Selected stage-1 weights |
| 3. Stuttering-event infilling | Real stuttering speech | Reconstruct event-region semantic tokens using event-tagged text and unmasked context | `MaskGCT/train_fill_mask_stutter.py` | Selected stage-2 weights |

These are **three fine-tuning stages**, not the upstream T2S/S2A generation stages.
Stage 1 operates on utterance-level examples but computes loss on the model's
sampled token mask, not literally every frame. Stages 2 and 3 use cached masks.
The original S2A and acoustic decoder then turn predicted semantics into audio.
`train_s2a.py` is optional, not a fourth required stage.

## Setup

Linux, Python 3.10, NVIDIA CUDA GPU(s), and `espeak-ng` are required for training.
The dependency pins reflect the source project's older Transformers API;
do not upgrade Transformers independently without compatibility testing.

```bash
conda create -n maskgct-stutter python=3.10 -y
conda activate maskgct-stutter
sudo apt-get install espeak-ng
python -m pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch/torchaudio pair if your platform needs a specific
wheel index. Full GPU training has not been re-run as part of repository packaging.

Obtain the official [Amphion repository](https://github.com/open-mmlab/Amphion)
with its MaskGCT G2P resources and `wav2vec2bert_stats.pt` (Git LFS content, not
pointer files). Then run:

```bash
python scripts/setup_assets.py --amphion-root ../Amphion
```

This copies only required external assets and downloads
[MaskGCT weights](https://huggingface.co/amphion/MaskGCT) and
[w2v-BERT](https://huggingface.co/facebook/w2v-bert-2.0).
Existing local weights can instead be placed according to [ASSETS.md](docs/ASSETS.md).

## Prepare Caches

Provide your own licensed audio and character-level TextGrid alignments. Alignment
models are deliberately not vendored. Example manifests describe the schema only;
the example audio does not exist. Keep event tags in the source transcript.

Run from this repository root. **Cache/synthesis shell scripts change into
`MaskGCT/`; their relative arguments are relative to that directory.**

```bash
bash scripts/prepare_cache.sh stutter --jsonl ../data/stutter/train.jsonl --out-dir ../data/cache/stutter/train
bash scripts/prepare_cache.sh stutter --jsonl ../data/stutter/dev.jsonl --out-dir ../data/cache/stutter/dev
bash scripts/prepare_cache.sh aishell1 --jsonl ../data/aishell1/train.jsonl --out-dir ../data/cache/aishell1/train
bash scripts/prepare_cache.sh aishell1 --jsonl ../data/aishell1/dev.jsonl --out-dir ../data/cache/aishell1/dev
```

## Train and Synthesize

Training wrapper paths are relative to the caller's working directory (repository
root below). Hyperparameters come from `MaskGCT/config/train.json`; model structure
comes from `MaskGCT/config/maskgct.json`. Existing batch size and gradient
accumulation are preserved. Set the desired training budget in your config.
Use `python`, **not torchrun**: trainers spawn one process per visible GPU themselves.

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/stage1_t2s.sh \
  --train-manifest data/cache/stutter/train/manifest.jsonl \
  --valid-manifest data/cache/stutter/dev/manifest.jsonl \
  --init-checkpoint MaskGCT/MaskGCT_model/t2s_model/model.safetensors

# Replace STAGE1_SELECTED and STAGE2_SELECTED with actual saved checkpoint paths.
CUDA_VISIBLE_DEVICES=0,1 bash scripts/stage2_fill_mask.sh \
  --train-manifest data/cache/aishell1/train/manifest.jsonl \
  --valid-manifest data/cache/aishell1/dev/manifest.jsonl \
  --init-checkpoint STAGE1_SELECTED

CUDA_VISIBLE_DEVICES=0,1 bash scripts/stage3_stutter.sh \
  --train-manifest data/cache/stutter/train/manifest.jsonl \
  --valid-manifest data/cache/stutter/dev/manifest.jsonl \
  --init-checkpoint STAGE2_SELECTED

bash scripts/synthesize.sh \
  --manifest ../data/cache/stutter/dev/manifest.jsonl \
  --t2s-ckpt ../outputs/stage3/SELECTED.safetensors \
  --out-dir ../outputs/synthesis --save-semantic
```

Each training stage writes to `outputs/stageN/`. Select a checkpoint using
validation loss and held-out listening tests; this release does not silently pick
the most recently modified file. Checkpoints contain model weights only, not a
full optimizer/scaler/RNG state. Reusing weights is **not exact training resume**.
Use a new `--output-dir` for a continuation experiment. `--dry-run` prints the
resolved command/config without loading models or requiring GPU dependencies.

Inference fills the cached semantic mask, then generates acoustic tokens and audio.
It needs a source cache, not just arbitrary tagged text. Unmasked semantic tokens
provide context; the decoded waveform outside a mask is not guaranteed to be
sample-identical to the source recording.

## Additional Components

- `dataset_aishell_stutter.py`: duration-based event injection into AISHELL-1 caches;
  requires aligned manifests, tagged transcripts, and your duration statistics.
- `infer_full_t2s.py`: utterance-level generation baseline after stage 1.
- `train_s2a.py`: optional acoustic-token model adaptation.
- `dataset_aishell_stutter.py` also exports timestamps directly from applied
  event edits, keeping event text aligned with the generated regions.

## Checks and Release Scope

```bash
python scripts/check_repository.py
python -m unittest discover -s tests
```

Checks cover syntax, packaging exclusions, and stage-launch contracts, not GPU
quality or model convergence. No paper results are claimed by these checks.

## Attribution and Responsible Use

Based on [Amphion / MaskGCT](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct).
Upstream copyright notices are preserved. Code is distributed under MIT;
pretrained MaskGCT weights have separate **CC BY-NC 4.0** terms. Dataset, speaker,
G2P-resource, and other model rights are not granted by this code license.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Use speech only with appropriate permission, label generated speech, and do not
impersonate speakers or infer clinical diagnoses from synthetic stuttering.
This release is a research implementation, not a clinical assessment tool.
