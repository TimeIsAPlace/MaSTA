#!/usr/bin/env python3
# Research snapshot: distributed/adapted by CosyVoice2-Stutter; see NOTICE for provenance.
"""Resynthesize a Kaldi-style training set with a complete CosyVoice2 checkpoint.

The original utterance audio is never overwritten.  Each utterance is generated
with a prompt selected from the same speaker so that speaker timbre is retained
as closely as CosyVoice2 zero-shot speaker conditioning allows.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / 'third_party' / 'Matcha-TTS')]

import torch
import torchaudio

from cosyvoice.cli.cosyvoice import CosyVoice2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resynthesize every training utterance with a complete CosyVoice2 checkpoint directory."
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Complete inference-ready CosyVoice2 checkpoint directory.",
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Kaldi-style input directory containing wav.scp, text, utt2spk and instruct.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="New Kaldi-style directory for synthesized WAV files and metadata.",
    )
    parser.add_argument(
        "--spk2prompt",
        default="",
        help="Optional mapping file: speaker<space>prompt_wav. Overrides automatic prompt selection.",
    )
    parser.add_argument(
        "--instruct",
        default="<|stutter|><|endofprompt|>",
        help="Instruction applied to every synthesized utterance.",
    )
    parser.add_argument(
        "--select_instruct_token",
        default="<|stutter|>",
        help="Only synthesize source rows whose instruct field contains this token.",
    )
    parser.add_argument("--prompt_target_seconds", type=float, default=5.0)
    parser.add_argument("--prompt_min_seconds", type=float, default=2.0)
    parser.add_argument("--prompt_max_seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1986)
    parser.add_argument("--limit", type=int, default=-1, help="Only synthesize the first N utterances; -1 means all.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 inference on CUDA.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate WAV files that already exist.")
    return parser.parse_args()


def read_mapping(path: Path, required: bool = True) -> Dict[str, str]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return {}
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_number}: expected key and value")
            key, value = parts
            if key in mapping:
                raise ValueError(f"{path}:{line_number}: duplicate key {key!r}")
            mapping[key] = value
    return mapping


def resolve_audio_path(value: str, data_dir: Path) -> Path:
    if value.rstrip().endswith("|"):
        raise ValueError(f"wav.scp command entries are not supported: {value}")
    path = Path(value)
    if not path.is_absolute():
        path = data_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def audio_duration(path: Path) -> float:
    try:
        info = torchaudio.info(str(path))
        if info.sample_rate > 0 and info.num_frames > 0:
            return info.num_frames / info.sample_rate
    except Exception:
        pass
    waveform, sample_rate = torchaudio.load(str(path))
    return waveform.shape[1] / sample_rate


def prompt_score(
    duration: float,
    target_seconds: float,
    min_seconds: float,
    max_seconds: float,
) -> Tuple[int, float]:
    in_range = min_seconds <= duration <= max_seconds
    return (0 if in_range else 1, abs(duration - target_seconds))


def select_prompt_candidates(
    utts: Iterable[str],
    wavs: Dict[str, Path],
    target_seconds: float,
    min_seconds: float,
    max_seconds: float,
) -> List[str]:
    scored = []
    for utt in utts:
        duration = audio_duration(wavs[utt])
        scored.append((prompt_score(duration, target_seconds, min_seconds, max_seconds), utt))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [utt for _score, utt in scored]


def set_utterance_seed(seed: int, index: int) -> None:
    utterance_seed = seed + index
    random.seed(utterance_seed)
    torch.manual_seed(utterance_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(utterance_seed)


def safe_output_name(utt: str) -> str:
    if any(char in utt for char in ("/", "\\")) or utt in {".", ".."}:
        raise ValueError(f"utterance id cannot be used as a filename: {utt!r}")
    return f"{utt}.wav"


def write_kaldi_metadata(
    output_dir: Path,
    rows: List[Tuple[str, Path, str, str, str]],
) -> None:
    with (output_dir / "wav.scp").open("w", encoding="utf-8") as wav_scp, \
         (output_dir / "text").open("w", encoding="utf-8") as text_file, \
         (output_dir / "utt2spk").open("w", encoding="utf-8") as utt2spk_file, \
         (output_dir / "instruct").open("w", encoding="utf-8") as instruct_file:
        for utt, wav, speaker, text, instruct in rows:
            wav_scp.write(f"{utt} {wav.resolve()}\n")
            text_file.write(f"{utt} {text}\n")
            utt2spk_file.write(f"{utt} {speaker}\n")
            instruct_file.write(f"{utt} {instruct}\n")

    spk2utt: Dict[str, List[str]] = defaultdict(list)
    for utt, _wav, speaker, _text, _instruct in rows:
        spk2utt[speaker].append(utt)
    with (output_dir / "spk2utt").open("w", encoding="utf-8") as stream:
        for speaker in sorted(spk2utt):
            stream.write(f"{speaker} {' '.join(spk2utt[speaker])}\n")


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    wav_dir = output_dir / "wavs"

    if output_dir == data_dir:
        raise ValueError("output_dir must differ from data_dir; original training metadata will not be overwritten")
    if not args.instruct.endswith("<|endofprompt|>"):
        raise ValueError("instruct must end with <|endofprompt|>")
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    required_model_files = [
        "cosyvoice2.yaml",
        "llm.pt",
        "flow.pt",
        "hift.pt",
        "campplus.onnx",
        "speech_tokenizer_v2.onnx",
    ]
    missing_model_files = [name for name in required_model_files if not (model_dir / name).is_file()]
    if not (model_dir / "CosyVoice-BlankEN").is_dir():
        missing_model_files.append("CosyVoice-BlankEN/")
    if missing_model_files:
        raise FileNotFoundError(
            f"Incomplete CosyVoice2 checkpoint directory {model_dir}; missing: "
            + ", ".join(missing_model_files)
        )
    if args.prompt_min_seconds <= 0 or args.prompt_max_seconds < args.prompt_min_seconds:
        raise ValueError("invalid prompt duration range")

    raw_wavs = read_mapping(data_dir / "wav.scp")
    texts = read_mapping(data_dir / "text")
    utt2spk = read_mapping(data_dir / "utt2spk")
    source_instructs = read_mapping(data_dir / "instruct")
    source_utterances = sorted(raw_wavs)
    for name, mapping in (("text", texts), ("utt2spk", utt2spk), ("instruct", source_instructs)):
        missing = [utt for utt in source_utterances if utt not in mapping]
        if missing:
            raise KeyError(f"{name} is missing {len(missing)} utterances; examples: {missing[:10]}")
    utterances = sorted(
        utt for utt in source_utterances
        if args.select_instruct_token in source_instructs[utt]
    )
    if args.limit > 0:
        utterances = utterances[:args.limit]
    if not utterances:
        raise SystemExit(f"No utterances matched source instruct token {args.select_instruct_token!r}.")
    print(
        f"Selected {len(utterances)}/{len(source_utterances)} utterances "
        f"containing {args.select_instruct_token!r}."
    )

    wavs = {utt: resolve_audio_path(raw_wavs[utt], data_dir) for utt in raw_wavs}
    speaker_utts: Dict[str, List[str]] = defaultdict(list)
    for utt, speaker in utt2spk.items():
        if utt in wavs:
            speaker_utts[speaker].append(utt)

    external_prompts: Dict[str, Path] = {}
    if args.spk2prompt:
        prompt_mapping = read_mapping(Path(args.spk2prompt).resolve())
        external_prompts = {
            speaker: resolve_audio_path(value, Path(args.spk2prompt).resolve().parent)
            for speaker, value in prompt_mapping.items()
        }

    ranked_prompts: Dict[str, List[str]] = {}
    for speaker in sorted({utt2spk[utt] for utt in utterances}):
        if speaker in external_prompts:
            continue
        candidates = speaker_utts.get(speaker, [])
        if not candidates:
            raise ValueError(f"No prompt candidates for speaker {speaker!r}")
        ranked_prompts[speaker] = select_prompt_candidates(
            candidates,
            wavs,
            args.prompt_target_seconds,
            args.prompt_min_seconds,
            args.prompt_max_seconds,
        )

    # Validate prompts before allocating/loading the inference model.
    for utt in utterances:
        speaker = utt2spk[utt]
        if speaker in external_prompts:
            if external_prompts[speaker].samefile(wavs[utt]):
                raise ValueError(f"External prompt is the target recording for {utt}")
        elif not any(not wavs[candidate].samefile(wavs[utt])
                     for candidate in ranked_prompts[speaker]):
            raise ValueError(f"No different same-speaker recording for {utt}; provide --spk2prompt")

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading complete CosyVoice2 checkpoint: {model_dir}")
    model = CosyVoice2(str(model_dir), fp16=args.fp16)

    rows: List[Tuple[str, Path, str, str, str]] = []
    manifest_rows = []

    for index, utt in enumerate(utterances, 1):
        speaker = utt2spk[utt]
        if speaker in external_prompts:
            prompt_utt = None
            prompt_wav = external_prompts[speaker]
        else:
            candidates = ranked_prompts[speaker]
            prompt_utt = next(candidate for candidate in candidates
                              if not wavs[candidate].samefile(wavs[utt]))
            prompt_wav = wavs[prompt_utt]

        output_wav = wav_dir / safe_output_name(utt)
        if args.overwrite or not output_wav.is_file() or output_wav.stat().st_size == 0:
            set_utterance_seed(args.seed, index)
            pieces = []
            for result in model.inference_instruct2(
                texts[utt],
                args.instruct,
                str(prompt_wav),
                stream=False,
                text_frontend=False,
            ):
                pieces.append(result["tts_speech"].detach().cpu())
            if not pieces:
                raise RuntimeError(f"Model produced no audio for {utt}")
            speech = torch.cat(pieces, dim=1)
            temporary_wav = output_wav.with_suffix(".tmp.wav")
            torchaudio.save(str(temporary_wav), speech, model.sample_rate)
            os.replace(temporary_wav, output_wav)

        rows.append((utt, output_wav, speaker, texts[utt], args.instruct))
        manifest_rows.append({
            "utt": utt,
            "speaker": speaker,
            "text": texts[utt],
            "source_instruct": source_instructs[utt],
            "instruct": args.instruct,
            "source_wav": str(wavs[utt]),
            "prompt_utt": prompt_utt,
            "prompt_wav": str(prompt_wav),
            "output_wav": str(output_wav.resolve()),
        })
        if index % 10 == 0 or index == len(utterances):
            print(f"resynthesized {index}/{len(utterances)}")

    write_kaldi_metadata(output_dir, rows)
    with (output_dir / "resynthesis_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in manifest_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    missing_outputs = [str(wav) for _utt, wav, _speaker, _text, _instruct in rows if not wav.is_file() or wav.stat().st_size == 0]
    if missing_outputs:
        raise RuntimeError(f"Missing or empty synthesized WAV files: {missing_outputs[:10]}")
    if len(rows) != len(utterances):
        raise RuntimeError(f"Output count mismatch: expected {len(utterances)}, got {len(rows)}")
    print(f"Done. Synthesized utterances: {len(rows)}")
    print(f"Output Kaldi data directory: {output_dir}")


if __name__ == "__main__":
    main()
