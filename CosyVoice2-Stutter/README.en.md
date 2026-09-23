# CosyVoice2-Stutter

A research fork of [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) for controllable Chinese stutter speech synthesis.

[中文完整文档](README.md)

## Scope

- Fine-grained instruction tuning: inline event tags specify where and what stutter event to generate (`train_fine_grained.sh`).
- Instruct tuning: a separate `<|normal|>` / `<|stutter|>` prompt selects utterance-level mode (`train_instruct.sh`); inline event tags may still be used.
- Batch synthesis from a complete fine-tuned checkpoint directory.
- Per-utterance conditioning on a different recording of the same speaker.

Code and fictional format examples only. No audio datasets, real transcripts, pretrained weights, fine-tuned weights, or benchmark claims are included. Speaker conditioning aims to preserve speaker identity; exact timbre is not guaranteed.

## Setup

Use Linux, Python 3.10, ffmpeg and a CUDA-capable NVIDIA GPU for training/inference.

```bash
conda create -n cosyvoice2-stutter python=3.10 -y
conda activate cosyvoice2-stutter
python -m pip install "setuptools==80.9.0" wheel
python -m pip install "torch==2.3.1" "torchaudio==2.3.1" --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD:$PWD/third_party/Matcha-TTS:${PYTHONPATH:-}"
python scripts/check_environment.py
```

The two processes are distinct. To run them sequentially, package the fine-grained
training result as a complete model directory and explicitly pass it as
`--pretrained_model_dir` to `recipes/cosyvoice2/train_instruct.sh`.
The scripts do not automatically chain checkpoints.

Download the official [CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) model separately to initialize training. Stutter synthesis requires weights trained for this fork's tags and matching tokenizer/configuration. See the Chinese [training guide](docs/training.md).

## Synthesis

Input rows: `relative/audio.wav<TAB>annotated text`. Each target must have a different same-speaker recording available in the selected rows, or through an explicit `--spk2prompt` map.

```bash
CUDA_VISIBLE_DEVICES=0 bash recipes/cosyvoice2/synthesize_aishell.sh \
  CosyVoice2-0.5B-instruct data data/aishell_stutter_synth_50k
```

This reads `aishell_stutter.txt` and synthesizes the first 50,000 nonempty records. `data` must contain `AISHELL-1/`. The Python entry point supports other limits, `--fp16`, and a dependency-light `--dry_run`.

Inline conversion: `/b → [block]`, `/r → [phone_rep]`, `字/p → <prolong>字</prolong>`, `嗯/i → <fill>嗯</fill>`, `[词] → <rep>词</rep>`.

## Validation and license

```bash
python -m unittest discover -s tests -v
python scripts/check_release.py
```

CI tests parsing, prompt selection, source syntax and package contents only. GPU training and synthesis have not been validated as part of this release preparation. This is not an official FunAudioLLM release.

Apache-2.0 for this repository, with retained upstream copyright notices; vendored Matcha-TTS components retain their MIT licenses. Dataset/model licenses are separate. See [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
