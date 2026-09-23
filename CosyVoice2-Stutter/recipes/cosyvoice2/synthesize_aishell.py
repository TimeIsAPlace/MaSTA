#!/usr/bin/env python3
# Research snapshot: distributed/adapted by CosyVoice2-Stutter; see NOTICE for provenance.
"""Synthesize tagged AISHELL-Stutter transcripts with a complete CosyVoice2 model.

The input file has rows of the form ``relative/wav/path.wav<TAB>transcript``.
Only the first N rows are synthesized.  Each output uses a different recording
from the same speaker as ``prompt_wav``; the target utterance is never used as
its own prompt.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

# Executing this file by path makes Python use recipes/cosyvoice2 as
# sys.path[0]. Add the repository and Matcha-TTS explicitly so local imports do
# not depend on the caller's current directory or PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT, REPO_ROOT / "third_party" / "Matcha-TTS"):
    import_root_string = str(import_root)
    if import_root_string not in sys.path:
        sys.path.insert(0, import_root_string)

FILLER_CHARS = frozenset("嗯呃啊唔额哎")
EVENT_MARKERS = {"b", "p", "r", "i"}


@dataclass(frozen=True)
class SourceRow:
    index: int
    relative_wav: Path
    source_text: str
    speaker: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True, help="Complete inference-ready CosyVoice2 checkpoint directory.")
    parser.add_argument("--input_file", default="aishell_stutter.txt", help="AISHELL-Stutter transcript file.")
    parser.add_argument(
        "--audio_root",
        required=True,
        help="Directory containing the AISHELL-1 directory. Each first-column path is resolved below it.",
    )
    parser.add_argument("--output_dir", required=True, help="New directory for generated audio and metadata.")
    parser.add_argument("--num_lines", type=int, default=50000, help="Number of input rows to synthesize, in file order.")
    parser.add_argument("--instruct", default="<|stutter|><|endofprompt|>")
    parser.add_argument("--spk2prompt", default="", help="Optional 'speaker prompt_wav' mapping; prompts must differ from targets.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 inference on CUDA.")
    parser.add_argument("--seed", type=int, default=1986)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing WAVs.")
    parser.add_argument("--dry_run", action="store_true", help="Validate paths and write conversion_preview.jsonl without loading the model.")
    return parser.parse_args()


def require_model_dir(model_dir: Path) -> None:
    files = ["cosyvoice2.yaml", "llm.pt", "flow.pt", "hift.pt", "campplus.onnx", "speech_tokenizer_v2.onnx"]
    missing = [name for name in files if not (model_dir / name).is_file()]
    if not (model_dir / "CosyVoice-BlankEN").is_dir():
        missing.append("CosyVoice-BlankEN/")
    if missing:
        raise FileNotFoundError(f"Incomplete CosyVoice2 checkpoint {model_dir}; missing: {', '.join(missing)}")


def read_rows(path: Path, num_lines: int) -> List[SourceRow]:
    if num_lines <= 0:
        raise ValueError("num_lines must be positive")
    rows: List[SourceRow] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            if len(rows) == num_lines:
                break
            line = raw.rstrip("\r\n")
            if not line:
                continue
            try:
                wav_value, text = line.split("\t", maxsplit=1)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: expected WAV path and transcript separated by a tab") from error
            relative_wav = Path(wav_value)
            # AISHELL-1/wav/train/S0002/BAC...wav -> S0002
            if len(relative_wav.parts) < 2 or not relative_wav.suffix.lower() == ".wav":
                raise ValueError(f"{path}:{line_number}: invalid WAV path {wav_value!r}")
            rows.append(SourceRow(len(rows) + 1, relative_wav, text, relative_wav.parent.name))
    if len(rows) != num_lines:
        raise ValueError(f"{path} contains only {len(rows)} usable rows, fewer than --num_lines {num_lines}")
    return rows


def parse_tagged_units(text: str) -> List[str]:
    """Convert AISHELL-Stutter annotation markers to CosyVoice2 stutter tags."""
    units: List[str] = []

    def preceding_spoken_unit() -> int:
        """Find the previous text unit, ignoring preceding event-only markers.

        Some source rows have combined annotations such as ``个/r/p``.  In
        that case /p still modifies 个 rather than the [phone_rep] tag.
        """
        for index in range(len(units) - 1, -1, -1):
            if units[index] not in {"[block]", "[phone_rep]"}:
                return index
        raise ValueError(f"event has no preceding spoken unit: {text!r}")

    cursor = 0
    while cursor < len(text):
        char = text[cursor]
        if char == "[":
            depth, end = 1, cursor + 1
            while end < len(text) and depth:
                if text[end] == "[":
                    depth += 1
                elif text[end] == "]":
                    depth -= 1
                end += 1
            if depth:
                raise ValueError(f"unclosed '[' in transcript: {text!r}")
            units.append(f"<rep>{''.join(parse_tagged_units(text[cursor + 1:end - 1]))}</rep>")
            cursor = end
            continue
        if char == "]":
            raise ValueError(f"unmatched ']' in transcript: {text!r}")
        if char == "/":
            if cursor + 1 >= len(text) or text[cursor + 1] not in EVENT_MARKERS:
                raise ValueError(f"unknown slash annotation near {text[max(0, cursor - 8):cursor + 8]!r}")
            event = text[cursor + 1]
            if event == "b":
                units.append("[block]")
            elif event == "r":
                units.append("[phone_rep]")
            elif event == "p":
                previous = preceding_spoken_unit()
                units[previous] = f"<prolong>{units[previous]}</prolong>"
            else:  # /i: preceding filler/interjection
                previous = preceding_spoken_unit()
                if units[previous] not in FILLER_CHARS:
                    raise ValueError(f"/i must follow a one-character filler {sorted(FILLER_CHARS)}: {text!r}")
                units[previous] = f"<fill>{units[previous]}</fill>"
            cursor += 2
            continue
        # '*' is an AISHELL transcription-side overlap/noise marker, not speech.
        if char != "*":
            units.append(char)
        cursor += 1
    return units


def convert_transcript(text: str) -> str:
    return "".join(parse_tagged_units(text))


def read_prompt_mapping(path: Path, audio_root: Path) -> Dict[str, Path]:
    prompts: Dict[str, Path] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            row = raw.strip()
            if not row:
                continue
            parts = row.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_number}: expected 'speaker prompt_wav'")
            speaker, value = parts
            prompt = Path(value)
            if not prompt.is_absolute():
                prompt = audio_root / prompt
            prompt = prompt.resolve()
            if not prompt.is_file():
                raise FileNotFoundError(f"{path}:{line_number}: prompt not found: {prompt}")
            prompts[speaker] = prompt
    return prompts


def choose_prompts(rows: Sequence[SourceRow], audio_root: Path, external: Dict[str, Path]) -> Dict[int, Path]:
    by_speaker: Dict[str, List[SourceRow]] = defaultdict(list)
    for row in rows:
        by_speaker[row.speaker].append(row)
    result: Dict[int, Path] = {}
    for speaker, speaker_rows in by_speaker.items():
        source_paths = [(row, (audio_root / row.relative_wav).resolve()) for row in speaker_rows]
        for row, target in source_paths:
            if not target.is_file():
                raise FileNotFoundError(f"input audio not found: {target}")
            if speaker in external:
                prompt = external[speaker]
                if prompt.samefile(target):
                    raise ValueError(f"external prompt for {speaker} equals target {target}; choose another recording")
            else:
                prompt = next((candidate for other, candidate in source_paths
                               if candidate != target and not candidate.samefile(target)), None)
                if prompt is None:
                    raise ValueError(f"speaker {speaker} has only one selected record; cannot use a different same-speaker prompt")
            result[row.index] = prompt
    return result


def set_seed(seed: int, index: int) -> None:
    import torch

    random.seed(seed + index)
    torch.manual_seed(seed + index)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + index)


def save_waveform(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    import torchaudio

    waveform = waveform.detach().cpu()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    temporary_path = path.with_suffix(".tmp.wav")
    torchaudio.save(str(temporary_path), waveform, sample_rate)
    temporary_path.replace(path)


def write_metadata(output_dir: Path, records: Sequence[dict]) -> None:
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as manifest, \
         (output_dir / "wav.scp").open("w", encoding="utf-8") as wav_scp, \
         (output_dir / "text").open("w", encoding="utf-8") as text_file, \
         (output_dir / "utt2spk").open("w", encoding="utf-8") as utt2spk, \
         (output_dir / "instruct").open("w", encoding="utf-8") as instruct:
        by_speaker: Dict[str, List[str]] = defaultdict(list)
        for record in records:
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            wav_scp.write(f"{record['utt']} {record['output_wav']}\n")
            text_file.write(f"{record['utt']} {record['tagged_text']}\n")
            utt2spk.write(f"{record['utt']} {record['speaker']}\n")
            instruct.write(f"{record['utt']} {record['instruct']}\n")
            by_speaker[record["speaker"]].append(record["utt"])
    with (output_dir / "spk2utt").open("w", encoding="utf-8") as stream:
        for speaker in sorted(by_speaker):
            stream.write(f"{speaker} {' '.join(by_speaker[speaker])}\n")


def main() -> None:
    args = parse_args()
    if not args.instruct.endswith("<|endofprompt|>"):
        raise ValueError("--instruct must end with <|endofprompt|>")
    model_dir = Path(args.model_dir).resolve()
    input_file = Path(args.input_file).resolve()
    audio_root = Path(args.audio_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_file.is_file() or not audio_root.is_dir():
        raise FileNotFoundError("--input_file must exist and --audio_root must be a directory")
    if not args.dry_run:
        require_model_dir(model_dir)
    rows = read_rows(input_file, args.num_lines)
    external = read_prompt_mapping(Path(args.spk2prompt).resolve(), audio_root) if args.spk2prompt else {}
    prompts = choose_prompts(rows, audio_root, external)
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for row in rows:
        tagged_text = convert_transcript(row.source_text)
        output_wav = (wav_dir / f"aishell_stutter_{row.index:06d}.wav").resolve()
        records.append({
            "utt": f"aishell_stutter_{row.index:06d}", "speaker": row.speaker,
            "source_wav": str((audio_root / row.relative_wav).resolve()), "prompt_wav": str(prompts[row.index]),
            "output_wav": str(output_wav), "source_text": row.source_text,
            "tagged_text": tagged_text, "instruct": args.instruct,
        })
    with (output_dir / "conversion_preview.jsonl").open("w", encoding="utf-8") as stream:
        for record in records[:100]:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    if args.dry_run:
        print(f"Validated {len(records)} rows. Preview: {output_dir / 'conversion_preview.jsonl'}")
        return

    print(f"Loading complete CosyVoice2 checkpoint: {model_dir}")
    import torch
    from cosyvoice.cli.cosyvoice import CosyVoice2

    if args.fp16 and not torch.cuda.is_available():
        raise ValueError("--fp16 requires CUDA")
    model = CosyVoice2(str(model_dir), fp16=args.fp16)
    for record in records:
        output_wav = Path(record["output_wav"])
        if output_wav.is_file() and not args.overwrite:
            continue
        set_seed(args.seed, int(record["utt"].rsplit("_", 1)[1]))
        result = model.inference_instruct2(
            record["tagged_text"], record["instruct"], record["prompt_wav"], stream=False, text_frontend=False
        )
        chunks = list(result)
        if not chunks:
            raise RuntimeError(f"CosyVoice returned no audio for {record['utt']}")
        waveform = torch.cat([chunk["tts_speech"] for chunk in chunks], dim=1)
        save_waveform(output_wav, waveform, model.sample_rate)
        if int(record["utt"].rsplit("_", 1)[1]) % 100 == 0:
            print(f"Synthesized {record['utt']}")
    write_metadata(output_dir, records)
    print(f"Done: {len(records)} utterances written under {output_dir}")


if __name__ == "__main__":
    main()
