#!/usr/bin/env python3
"""AISHELL-1 semantic-token cache with random semantic-aligned text masks.

This follows MaskGCT/dataset.py's cache format as closely as possible, except
that AISHELL-1 cache does not extract or save acoustic codec tokens.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

MASKGCT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MASKGCT_ROOT.parent
if str(MASKGCT_ROOT) not in sys.path:
    sys.path.insert(0, str(MASKGCT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import MaskGCTSemanticExtractor, as_long_tensor
from g2p.g2p_generation import chn_eng_g2p

PHONE_PAD_ID = 1023
SEMANTIC_PAD_ID = 0
TEXTGRID_SPECIAL_LABELS = {"", "NULL", "sil", "sp", "spn", "<eps>", "<sil>", "silence"}


def read_jsonl(path: str | Path, skip_invalid: bool = False) -> List[Dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                if skip_invalid:
                    print(f"skip invalid jsonl line {line_no} in {path}", flush=True)
                    continue
                raise
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            records.append(value)
    return records


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def append_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def safe_filename(name: Any, default: str = "sample") -> str:
    name = str(name or default)
    name = re.sub(r"[^0-9A-Za-z_.-]+", "_", name)
    return name.strip("._") or default


def resolve_manifest_path(path: str | Path, manifest_path: str | Path) -> str:
    path = Path(path)
    if path.is_absolute():
        return str(path)

    manifest_path = Path(manifest_path)
    candidates = [
        manifest_path.parent / path,
        PROJECT_ROOT / path,
        Path.cwd() / path,
        MASKGCT_ROOT / path,
    ]
    parts = path.parts
    if "data" in parts:
        data_index = parts.index("data")
        candidates.append(PROJECT_ROOT.joinpath(*parts[data_index:]))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[1])


def manifest_sample_path(path: str | Path) -> str:
    path = Path(path)
    if path.is_absolute():
        path = Path(os.path.relpath(path, Path.cwd()))
    return path.as_posix()


def resolve_sample_path(path: str | Path, cache_dir: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path

    cache_dir = Path(cache_dir)
    candidates = [
        path,
        cache_dir / path,
        PROJECT_ROOT / path,
        MASKGCT_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return cache_dir / path


def is_alignable_text_char(char: str) -> bool:
    if not char or char.isspace():
        return False
    category = unicodedata.category(char)
    return not (category.startswith("P") or category.startswith("S"))


def clean_raw_text_for_g2p(text: Any) -> str:
    return str(text or "").strip()


def strip_textgrid_label(label: str) -> str:
    return (label or "").strip().strip('"')


def parse_textgrid_words_and_duration(textgrid_path: str | Path) -> tuple[float, List[Dict[str, Any]]]:
    path = Path(textgrid_path)
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    duration = 0.0
    in_words = False
    intervals = []
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if duration <= 0.0 and line.startswith("xmax"):
            try:
                duration = float(line.split("=", 1)[1].strip())
            except ValueError:
                pass
        if line.startswith("name"):
            in_words = strip_textgrid_label(line.split("=", 1)[1]) == "words"
            idx += 1
            continue
        if in_words and line.startswith("item ["):
            break
        if in_words and line.startswith("intervals ["):
            xmin = xmax = None
            label = ""
            idx += 1
            while idx < len(lines) and not lines[idx].strip().startswith("intervals ["):
                cur = lines[idx].strip()
                if cur.startswith("xmin"):
                    xmin = float(cur.split("=", 1)[1].strip())
                elif cur.startswith("xmax"):
                    xmax = float(cur.split("=", 1)[1].strip())
                elif cur.startswith("text"):
                    label = strip_textgrid_label(cur.split("=", 1)[1].strip())
                    break
                idx += 1
            if xmin is not None and xmax is not None:
                intervals.append({"start": xmin, "end": xmax, "text": label})
            continue
        idx += 1
    words = [
        item
        for item in intervals
        if str(item["text"]).strip() not in TEXTGRID_SPECIAL_LABELS
        and str(item["text"]).strip().lower() not in TEXTGRID_SPECIAL_LABELS
        and float(item["end"]) > float(item["start"])
    ]
    return duration, words


def infer_textgrid_path(record: Dict[str, Any], jsonl_path: str | Path, textgrid_field: str) -> Optional[str]:
    for field in (textgrid_field, "textgrid_path", "alignment_path", "TextGrid"):
        if field and record.get(field):
            resolved = resolve_manifest_path(record[field], jsonl_path)
            if Path(resolved).exists():
                return resolved
    utt_id = str(record.get("utt_id") or "")
    if utt_id:
        candidate = Path(jsonl_path).resolve().parent / "alignments" / f"{utt_id}.TextGrid"
        if candidate.exists():
            return str(candidate)
    return None


def mask_time_span(mask: torch.Tensor, start_s: float, end_s: float, duration_s: float) -> None:
    token_len = int(mask.numel())
    if token_len <= 0 or duration_s <= 0 or end_s <= start_s:
        return
    start_s = max(0.0, min(duration_s, float(start_s)))
    end_s = max(start_s, min(duration_s, float(end_s)))
    left = min(token_len, max(0, math.floor(start_s / duration_s * token_len)))
    right = min(token_len, max(left, math.ceil(end_s / duration_s * token_len)))
    if end_s > start_s and right == left:
        if left == token_len:
            left = token_len - 1
        right = min(token_len, left + 1)
    mask[left:right] = 1


def build_random_text_mask_from_textgrid(
    raw_text: Any,
    textgrid_path: str | Path | None,
    token_len: int,
    duration_s: float = 0.0,
    min_mask_ratio: float = 0.2,
    max_mask_ratio: float = 0.4,
) -> torch.Tensor:
    mask = torch.zeros(max(int(token_len), 0), dtype=torch.long)
    if token_len <= 0:
        return mask
    if not textgrid_path or not Path(textgrid_path).exists():
        mask[:] = 1
        return mask

    parsed_duration, word_intervals = parse_textgrid_words_and_duration(textgrid_path)
    if duration_s <= 0 and parsed_duration > 0:
        duration_s = parsed_duration
    if duration_s <= 0:
        duration_s = max(float(token_len) / 50.0, 1e-6)
    if not word_intervals:
        mask[:] = 1
        return mask

    text_chars = [char for char in str(raw_text or "") if is_alignable_text_char(char)]
    reference_count = len(text_chars) or len(word_intervals)
    ratio = random.uniform(float(min_mask_ratio), float(max_mask_ratio))
    sample_count = max(1, int(math.ceil(reference_count * ratio)))
    sample_count = min(sample_count, len(word_intervals))

    for interval in random.sample(word_intervals, sample_count):
        mask_time_span(mask, float(interval["start"]), float(interval["end"]), duration_s)
    if mask.sum().item() == 0:
        mask[:] = 1
    return mask


class AISHELL1SemanticTokenDataset(Dataset):
    """AISHELL-1 dataset matching MaskGCTAudioTokenDataset minus acoustic tokens."""

    def __init__(
        self,
        jsonl_path: str | Path,
        wav_field: str = "audio_path",
        utt_field: str = "utt_id",
        raw_text_field: str = "original_text",
        textgrid_field: str = "textgrid_path",
        semantic_extractor: Optional[Any] = None,
        semantic_extractor_kwargs: Optional[Dict[str, Any]] = None,
        min_mask_ratio: float = 0.2,
        max_mask_ratio: float = 0.4,
        skip_invalid_json: bool = False,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.records = read_jsonl(self.jsonl_path, skip_invalid=skip_invalid_json)
        self.wav_field = wav_field
        self.utt_field = utt_field
        self.raw_text_field = raw_text_field
        self.textgrid_field = textgrid_field
        self.min_mask_ratio = float(min_mask_ratio)
        self.max_mask_ratio = float(max_mask_ratio)
        self.semantic_extractor = semantic_extractor or MaskGCTSemanticExtractor(
            **(semantic_extractor_kwargs or {})
        )

    def __len__(self) -> int:
        return len(self.records)

    def _record_id(self, record: Dict[str, Any], index: int) -> str:
        return str(
            record.get(self.utt_field)
            or record.get("utt_id")
            or record.get("utt")
            or record.get("id")
            or index
        )

    def _raw_wav_path(self, record: Dict[str, Any]) -> Any:
        candidate_fields = [self.wav_field, "audio_path", "wav", "path", "audio"]
        for field in candidate_fields:
            if field and record.get(field):
                return record[field]
        raise KeyError(
            f"Record {record.get(self.utt_field) or record.get('utt_id')} has no audio path. "
            f"Tried fields: {candidate_fields}"
        )

    def _extractor_record(self, record: Dict[str, Any], index: int) -> Dict[str, Any]:
        wav_path = self._raw_wav_path(record)
        extractor_record = dict(record)
        extractor_record["utt_id"] = self._record_id(record, index)
        extractor_record["wav"] = resolve_manifest_path(wav_path, self.jsonl_path)
        return extractor_record

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        utt_id = self._record_id(record, index)
        wav_path = self._raw_wav_path(record)
        extractor_record = self._extractor_record(record, index)

        semantic_tokens = as_long_tensor(
            self.semantic_extractor(extractor_record), "semantic_extractor(record)"
        )
        raw_text = str(record.get(self.raw_text_field, record.get("original_text", "")))
        textgrid_path = infer_textgrid_path(record, self.jsonl_path, self.textgrid_field)
        mask_sequence = build_random_text_mask_from_textgrid(
            raw_text=raw_text,
            textgrid_path=textgrid_path,
            token_len=int(semantic_tokens.numel()),
            duration_s=float(record.get("duration") or 0.0),
            min_mask_ratio=self.min_mask_ratio,
            max_mask_ratio=self.max_mask_ratio,
        )
        g2p_text = clean_raw_text_for_g2p(raw_text)
        phoneme_sequence, token_ids = chn_eng_g2p(g2p_text)
        phone_ids = torch.tensor([int(idx) for idx in token_ids], dtype=torch.long)
        sample_record = {
            "utt_id": utt_id,
            "wav": str(wav_path),
            "raw_text": raw_text,
        }

        return {
            **sample_record,
            "g2p_text": g2p_text,
            "phoneme_sequence": phoneme_sequence,
            "phone_ids": phone_ids,
            "phone_len": torch.tensor(phone_ids.numel(), dtype=torch.long),
            "mask_sequence": mask_sequence,
            "mask_len": torch.tensor(mask_sequence.numel(), dtype=torch.long),
            "mask_sum": torch.tensor(int(mask_sequence.sum().item()), dtype=torch.long),
            "semantic_tokens": semantic_tokens,
            "semantic_len": torch.tensor(semantic_tokens.numel(), dtype=torch.long),
            "record": sample_record,
        }


def save_aishell1_semantic_token_cache(
    jsonl_path: str | Path,
    out_dir: str | Path,
    wav_field: str = "audio_path",
    utt_field: str = "utt_id",
    raw_text_field: str = "original_text",
    textgrid_field: str = "textgrid_path",
    semantic_extractor_kwargs: Optional[Dict[str, Any]] = None,
    min_mask_ratio: float = 0.2,
    max_mask_ratio: float = 0.4,
    overwrite: bool = False,
    resume: bool = True,
    skip_invalid_json: bool = False,
) -> Path:
    out_dir = Path(out_dir)
    sample_dir = out_dir / "samples"
    manifest_path = out_dir / "manifest.jsonl"

    if manifest_path.exists() and not overwrite and not resume:
        raise FileExistsError(
            f"{manifest_path} already exists. Pass overwrite=True/--overwrite to rebuild "
            "or keep resume=True to reuse completed samples."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    required_sample_keys = {
        "semantic_tokens",
        "phone_ids",
        "mask_sequence",
        "mask_len",
        "mask_sum",
    }

    def has_complete_token_cache(item: Dict[str, Any]) -> bool:
        return required_sample_keys.issubset(item.keys()) and "acoustic_tokens" not in item

    dataset = AISHELL1SemanticTokenDataset(
        jsonl_path=jsonl_path,
        wav_field=wav_field,
        utt_field=utt_field,
        raw_text_field=raw_text_field,
        textgrid_field=textgrid_field,
        semantic_extractor_kwargs=semantic_extractor_kwargs,
        min_mask_ratio=min_mask_ratio,
        max_mask_ratio=max_mask_ratio,
        skip_invalid_json=skip_invalid_json,
    )

    def build_manifest_record(index: int, item: Dict[str, Any], sample_path: Path) -> Dict[str, Any]:
        return {
            "utt_id": item["utt_id"],
            "wav": item["wav"],
            "raw_text": item["raw_text"],
            "g2p_text": item["g2p_text"],
            "phoneme_sequence": item["phoneme_sequence"],
            "sample_path": manifest_sample_path(sample_path),
            "phone_len": int(item["phone_len"]),
            "mask_len": int(item["mask_len"]),
            "mask_sum": int(item["mask_sum"]),
            "semantic_len": int(item["semantic_len"]),
        }

    completed_records: Dict[int, Dict[str, Any]] = {}
    if resume and not overwrite and manifest_path.exists():
        for record in read_jsonl(manifest_path, skip_invalid=True):
            record_index = int(record["index"])
            record_sample_path = resolve_sample_path(record["sample_path"], out_dir)
            if record_sample_path.exists() and "mask_sum" in record and "mask_len" in record:
                record["sample_path"] = manifest_sample_path(record_sample_path)
                completed_records[record_index] = record
        write_jsonl(manifest_path, [completed_records[index] for index in sorted(completed_records)])
    else:
        write_jsonl(manifest_path, [])

    for index in range(len(dataset)):
        raw_record = dataset.records[index]
        utt_id = dataset._record_id(raw_record, index)
        sample_name = f"{index:08d}_{safe_filename(utt_id)}.pt"
        sample_path = sample_dir / sample_name

        if resume and not overwrite and sample_path.exists():
            if index in completed_records:
                print(f"[{index + 1}/{len(dataset)}] reused {utt_id} from {sample_path.name}", flush=True)
                continue
            try:
                item = torch.load(sample_path, map_location="cpu")
            except Exception as exc:
                print(
                    f"[{index + 1}/{len(dataset)}] invalid cached sample {sample_path.name}; "
                    f"recomputing ({exc})",
                    flush=True,
                )
            else:
                if not has_complete_token_cache(item):
                    print(f"[{index + 1}/{len(dataset)}] incomplete cached sample {sample_path.name}; recomputing", flush=True)
                else:
                    append_jsonl(manifest_path, build_manifest_record(index, item, sample_path))
                    print(f"[{index + 1}/{len(dataset)}] reused {utt_id} from {sample_path.name}", flush=True)
                    continue

        item = dataset[index]
        torch.save(item, sample_path)
        append_jsonl(manifest_path, build_manifest_record(index, item, sample_path))
        print(
            f"[{index + 1}/{len(dataset)}] saved {item['utt_id']} "
            f"semantic_len={int(item['semantic_len'])} "
            f"phone_len={int(item['phone_len'])} "
            f"mask_sum={int(item['mask_sum'])}",
            flush=True,
        )

    return manifest_path


class AISHELL1CachedSemanticTokenDataset(Dataset):
    """Dataset for locally cached AISHELL-1 semantic-token samples."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self.cache_dir = self.manifest_path.parent
        self.records = read_jsonl(self.manifest_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        sample_path = resolve_sample_path(record["sample_path"], self.cache_dir)
        item = torch.load(sample_path, map_location="cpu")
        item["cache_record"] = record
        return item


def pad_1d(sequences: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    if not sequences:
        return torch.empty(0, dtype=torch.long)
    max_len = max(seq.numel() for seq in sequences)
    out = torch.full((len(sequences), max_len), int(pad_value), dtype=torch.long)
    for idx, seq in enumerate(sequences):
        out[idx, : seq.numel()] = seq.long()
    return out


class AISHELL1SemanticTokenCollate:
    def __init__(
        self,
        phone_pad_id: int = PHONE_PAD_ID,
        semantic_pad_id: int = SEMANTIC_PAD_ID,
    ):
        self.phone_pad_id = int(phone_pad_id)
        self.semantic_pad_id = int(semantic_pad_id)

    def __call__(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        phone_ids = [item["phone_ids"] for item in items]
        semantic_tokens = [item["semantic_tokens"] for item in items]
        mask_sequence = [item["mask_sequence"] for item in items]
        phone_lens = torch.tensor([seq.numel() for seq in phone_ids], dtype=torch.long)
        semantic_lens = torch.tensor([seq.numel() for seq in semantic_tokens], dtype=torch.long)
        mask_lens = torch.tensor([seq.numel() for seq in mask_sequence], dtype=torch.long)

        phone_ids_pad = pad_1d(phone_ids, self.phone_pad_id)
        semantic_tokens_pad = pad_1d(semantic_tokens, self.semantic_pad_id)
        mask_sequence_pad = pad_1d(mask_sequence, 0)

        phone_mask = (
            torch.arange(phone_ids_pad.size(1)).unsqueeze(0) < phone_lens.unsqueeze(1)
            if phone_ids_pad.numel()
            else torch.empty(0, dtype=torch.bool)
        )
        semantic_mask = (
            torch.arange(semantic_tokens_pad.size(1)).unsqueeze(0) < semantic_lens.unsqueeze(1)
            if semantic_tokens_pad.numel()
            else torch.empty(0, dtype=torch.bool)
        )
        mask_valid = (
            torch.arange(mask_sequence_pad.size(1)).unsqueeze(0) < mask_lens.unsqueeze(1)
            if mask_sequence_pad.numel()
            else torch.empty(0, dtype=torch.bool)
        )

        return {
            "utt_id": [item["utt_id"] for item in items],
            "wav": [item["wav"] for item in items],
            "raw_text": [item["raw_text"] for item in items],
            "g2p_text": [item["g2p_text"] for item in items],
            "phoneme_sequence": [item["phoneme_sequence"] for item in items],
            "phone_ids": phone_ids_pad,
            "phone_lens": phone_lens,
            "phone_mask": phone_mask,
            "mask_sequence": mask_sequence_pad,
            "mask_lens": mask_lens,
            "mask_valid": mask_valid,
            "semantic_tokens": semantic_tokens_pad,
            "semantic_lens": semantic_lens,
            "semantic_mask": semantic_mask,
            "records": [item["record"] for item in items],
        }


def aishell1_semantic_token_collate(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return AISHELL1SemanticTokenCollate()(items)


def split_manifest(data_root: str | Path, split: str) -> Path:
    split_dir = Path(data_root) / split
    aligned = split_dir / "manifest_alignment.jsonl"
    if aligned.exists():
        return aligned
    return split_dir / "manifest.jsonl"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AISHELL-1 semantic-token dataset utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-cache",
        help="Precompute semantic tokens and random semantic-aligned text masks for AISHELL-1.",
    )
    prepare.add_argument("--data-root", default="../data/AISHELL-1")
    prepare.add_argument("--split", choices=["train", "dev", "test", "all"], default="all")
    prepare.add_argument("--jsonl", default=None, help="Input JSONL manifest. Overrides --data-root/--split when set.")
    prepare.add_argument("--out-dir", default="../data/AISHELL-1/semantic_mask_cache")
    prepare.add_argument("--wav-field", default="audio_path")
    prepare.add_argument("--utt-field", default="utt_id")
    prepare.add_argument("--raw-text-field", default="original_text")
    prepare.add_argument("--textgrid-field", default="textgrid_path")
    prepare.add_argument("--device", default="cuda")
    prepare.add_argument("--hidden-state-layer", type=int, default=17)
    prepare.add_argument("--semantic-sample-rate", type=int, default=16000)
    prepare.add_argument("--cfg-path", default=str(MASKGCT_ROOT / "config" / "maskgct.json"))
    prepare.add_argument("--w2v-bert-path", default=str(MASKGCT_ROOT / "MaskGCT_model" / "w2v_bert"))
    prepare.add_argument("--stats-path", default=str(MASKGCT_ROOT / "ckpt" / "wav2vec2bert_stats.pt"))
    prepare.add_argument(
        "--semantic-codec-ckpt",
        default=str(MASKGCT_ROOT / "MaskGCT_model" / "semantic_codec" / "model.safetensors"),
    )
    prepare.add_argument("--min-mask-ratio", type=float, default=0.2)
    prepare.add_argument("--max-mask-ratio", type=float, default=0.4)
    prepare.add_argument("--overwrite", action="store_true")
    prepare.add_argument("--no-resume", action="store_true")
    prepare.add_argument("--skip-invalid-json", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "prepare-cache":
        semantic_extractor_kwargs = {
            "device": args.device,
            "cfg_path": args.cfg_path,
            "w2v_bert_path": args.w2v_bert_path,
            "stats_path": args.stats_path,
            "semantic_codec_ckpt": args.semantic_codec_ckpt,
            "hidden_state_layer": args.hidden_state_layer,
            "sample_rate": args.semantic_sample_rate,
        }
        splits = ["train", "dev", "test"] if args.split == "all" and args.jsonl is None else [args.split]
        for split in splits:
            jsonl_path = Path(args.jsonl) if args.jsonl else split_manifest(args.data_root, split)
            out_dir = Path(args.out_dir) / split if args.jsonl is None else Path(args.out_dir)
            if not jsonl_path.exists():
                raise FileNotFoundError(f"manifest not found: {jsonl_path}")
            manifest_path = save_aishell1_semantic_token_cache(
                jsonl_path=jsonl_path,
                out_dir=out_dir,
                wav_field=args.wav_field,
                utt_field=args.utt_field,
                raw_text_field=args.raw_text_field,
                textgrid_field=args.textgrid_field,
                semantic_extractor_kwargs=semantic_extractor_kwargs,
                min_mask_ratio=args.min_mask_ratio,
                max_mask_ratio=args.max_mask_ratio,
                overwrite=args.overwrite,
                resume=not args.no_resume,
                skip_invalid_json=args.skip_invalid_json,
            )
            print(f"cache_manifest={manifest_path}")


if __name__ == "__main__":
    main()

