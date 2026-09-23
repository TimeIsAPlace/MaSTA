#!/usr/bin/env python3
"""Build MaskGCT caches for synthetic AISHELL-1 stutter annotations.

The fluent AISHELL waveform supplies the unmasked semantic context and the
12-layer acoustic prompt. Annotated events expand or replace positions on the
semantic time axis; every newly allocated event position is masked for T2S
fill-mask generation.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
import random
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


MASKGCT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MASKGCT_ROOT.parent
for search_path in (MASKGCT_ROOT, PROJECT_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from dataset import (  # noqa: E402
    MaskGCTAcousticExtractor,
    MaskGCTSemanticExtractor,
    as_long_tensor,
    clean_raw_text_for_g2p,
    manifest_sample_path,
    parse_textgrid_words,
    read_jsonl,
    resolve_manifest_path,
    resolve_sample_path,
    safe_filename,
    write_jsonl,
)
from g2p.g2p_generation import chn_eng_g2p  # noqa: E402


TOKEN_RATE_HZ = 50.0
SEMANTIC_MASK_VALUE = 0
ANGLE_SPAN = re.compile(r"<[^>]*>")
EVENT_TYPES = {"b", "i", "p", "r"}


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def is_alignable_char(char: str) -> bool:
    if not char or char.isspace() or char in "[]<>/":
        return False
    category = unicodedata.category(char)
    return not (category.startswith("P") or category.startswith("S"))


def parse_marked_chars(fragment: str) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    index = 0
    while index < len(fragment):
        char = fragment[index]
        if char == "/" and index + 1 < len(fragment) and fragment[index + 1].lower() in EVENT_TYPES:
            index += 2
            continue
        if not is_alignable_char(char):
            index += 1
            continue
        index += 1
        markers: List[str] = []
        while index + 1 < len(fragment) and fragment[index] == "/" and fragment[index + 1].lower() in EVENT_TYPES:
            markers.append(fragment[index + 1].lower())
            index += 2
        units.append({"char": char, "events": markers})
    return units


def parse_annotation(annotation: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return alignable base units and insertion events anchored between units."""
    text = ANGLE_SPAN.sub("", str(annotation or ""))
    base_units: List[Dict[str, Any]] = []
    insertions: List[Dict[str, Any]] = []
    index = 0
    while index < len(text):
        if text[index] == "[":
            end = text.find("]", index + 1)
            if end < 0:
                index += 1
                continue
            bracket_units = parse_marked_chars(text[index + 1 : end])
            if bracket_units:
                insertions.append(
                    {
                        "boundary": len(base_units),
                        "kind": "repeat",
                        "text": "".join(unit["char"] for unit in bracket_units),
                        "units": bracket_units,
                    }
                )
            index = end + 1
            continue
        char = text[index]
        if not is_alignable_char(char):
            index += 1
            continue
        index += 1
        markers: List[str] = []
        while index + 1 < len(text) and text[index] == "/" and text[index + 1].lower() in EVENT_TYPES:
            markers.append(text[index + 1].lower())
            index += 2
        if "i" in markers:
            insertions.append(
                {
                    "boundary": len(base_units),
                    "kind": "i",
                    "text": char,
                    "units": [{"char": char, "events": markers}],
                }
            )
        else:
            base_units.append({"char": char, "events": markers})
    return base_units, insertions


def align_units_to_words(
    base_units: Sequence[Dict[str, Any]], words: Sequence[Dict[str, Any]]
) -> List[Optional[Dict[str, Any]]]:
    source = [str(unit["char"]) for unit in base_units]
    target = [str(word["text"]) for word in words]
    table: List[Optional[Dict[str, Any]]] = [None] * len(source)
    matcher = difflib.SequenceMatcher(a=source, b=target, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                table[i1 + offset] = words[j1 + offset]
        elif tag == "replace":
            for offset in range(min(i2 - i1, j2 - j1)):
                table[i1 + offset] = words[j1 + offset]
    return table


class DurationSampler:
    def __init__(self, summary_path: str | Path, mode: str, seed: int):
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        self.events = summary["events"]
        self.mode = mode
        self.seed = int(seed)

    def sample(self, event_type: str, key: str) -> float:
        stats = self.events[event_type]
        if self.mode == "p25":
            return float(stats["p25_s"])
        if self.mode == "median":
            return float(stats["median_s"])
        if self.mode == "p75":
            return float(stats["p75_s"])
        digest = hashlib.sha256(f"{self.seed}:{key}:{event_type}".encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        return rng.triangular(
            float(stats["p25_s"]),
            float(stats["p75_s"]),
            float(stats["median_s"]),
        )


def seconds_to_left(seconds: float, token_len: int) -> int:
    return max(0, min(token_len, math.floor(float(seconds) * TOKEN_RATE_HZ)))


def seconds_to_right(seconds: float, token_len: int) -> int:
    return max(0, min(token_len, math.ceil(float(seconds) * TOKEN_RATE_HZ)))


def nearest_previous(table: Sequence[Optional[Dict[str, Any]]], boundary: int) -> Optional[Dict[str, Any]]:
    for index in range(min(boundary, len(table)) - 1, -1, -1):
        if table[index] is not None:
            return table[index]
    return None


def nearest_next(table: Sequence[Optional[Dict[str, Any]]], boundary: int) -> Optional[Dict[str, Any]]:
    for index in range(max(0, boundary), len(table)):
        if table[index] is not None:
            return table[index]
    return None


def event_duration(
    event_types: Sequence[str], sampler: DurationSampler, key: str, fallback: float
) -> float:
    sampled = [
        sampler.sample(event, f"{key}:{sample_index}")
        for sample_index, event in enumerate(event_types)
        if event in EVENT_TYPES
    ]
    if not sampled:
        return fallback
    # Each distribution measures "character + event". For stacked events the
    # same character must only be counted once.
    combined = sum(sampled) - max(0, len(sampled) - 1) * fallback
    return max(fallback, combined)


def previous_repeat_durations(
    insertion_units: Sequence[Dict[str, Any]],
    base_units: Sequence[Dict[str, Any]],
    table: Sequence[Optional[Dict[str, Any]]],
    boundary: int,
) -> Optional[List[float]]:
    """Copy durations from the nearest matching preceding text pattern.

    Exact matches are preferred. A shorter preceding unit may also be tiled for
    annotations such as ``我[我我]`` or ``你好[你好你好]``.
    """
    inserted = "".join(str(unit["char"]) for unit in insertion_units)
    preceding = "".join(str(unit["char"]) for unit in base_units[:boundary])
    if not inserted or not preceding:
        return None

    max_pattern_len = min(len(inserted), len(preceding))
    for pattern_len in range(max_pattern_len, 0, -1):
        pattern = inserted[:pattern_len]
        if any(char != pattern[index % pattern_len] for index, char in enumerate(inserted)):
            continue
        search_end = len(preceding)
        while search_end >= pattern_len:
            start = preceding.rfind(pattern, 0, search_end)
            if start < 0:
                break
            intervals = table[start : start + pattern_len]
            if len(intervals) == pattern_len and all(interval is not None for interval in intervals):
                pattern_durations = [
                    float(interval["end"]) - float(interval["start"])
                    for interval in intervals
                    if interval is not None
                ]
                if all(duration > 0 for duration in pattern_durations):
                    return [
                        pattern_durations[index % pattern_len]
                        for index in range(len(insertion_units))
                    ]
            search_end = start
    return None


def build_event_edits(
    annotation: str,
    words: Sequence[Dict[str, Any]],
    token_len: int,
    sampler: DurationSampler,
    utt_id: str,
    repeat_duration_scale: float = 1.2,
    min_repeat_mask_tokens: int = 20,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    base_units, insertions = parse_annotation(annotation)
    table = align_units_to_words(base_units, words)
    aligned_widths = [float(word["end"]) - float(word["start"]) for word in table if word is not None]
    natural_char_s = sorted(aligned_widths)[len(aligned_widths) // 2] if aligned_widths else 0.30
    edits: List[Dict[str, Any]] = []

    for index, (unit, interval) in enumerate(zip(base_units, table)):
        explicit = [event for event in unit["events"] if event in {"p", "b", "r"}]
        if not explicit:
            continue
        if interval is None:
            raise ValueError(f"cannot locate marked base character {unit['char']!r} at unit {index}")
        aligned_char_s = max(
            1.0 / TOKEN_RATE_HZ,
            float(interval["end"]) - float(interval["start"]),
        )
        duration_s = event_duration(
            explicit, sampler, f"{utt_id}:base:{index}", aligned_char_s
        )
        left = seconds_to_left(float(interval["start"]), token_len)
        right = seconds_to_right(float(interval["end"]), token_len)
        edits.append(
            {
                "left": left,
                "right": max(left + 1, right),
                "target_len": max(1, math.ceil(duration_s * TOKEN_RATE_HZ)),
                "kind": "replace",
                "text": unit["char"],
                "events": explicit,
                "duration_s": duration_s,
            }
        )

    for insertion_index, insertion in enumerate(insertions):
        boundary = int(insertion["boundary"])
        previous = nearest_previous(table, boundary)
        following = nearest_next(table, boundary)
        left = seconds_to_left(float(previous["end"]), token_len) if previous else 0
        right = seconds_to_right(float(following["start"]), token_len) if following else token_len
        right = max(left, right)
        units = insertion["units"]
        if insertion["kind"] == "i":
            event_types = [event for event in units[0]["events"] if event in EVENT_TYPES]
            duration_s = event_duration(
                event_types,
                sampler,
                f"{utt_id}:insert:{insertion_index}",
                natural_char_s,
            )
            copied_durations = None
        else:
            duration_s = 0.0
            event_types = ["repeat"]
            copied_durations = previous_repeat_durations(
                units, base_units, table, boundary
            )
            for unit_index, unit in enumerate(units):
                explicit = [event for event in unit["events"] if event in EVENT_TYPES]
                natural_duration = (
                    copied_durations[unit_index]
                    if copied_durations is not None
                    else natural_char_s
                )
                unit_duration = event_duration(
                    explicit,
                    sampler,
                    f"{utt_id}:repeat:{insertion_index}:{unit_index}",
                    natural_duration,
                )
                # Repeated speech is normally slower than the fluent source.
                # Stretch the character component once without scaling the
                # independently sampled event component.
                unit_duration += (repeat_duration_scale - 1.0) * natural_duration
                duration_s += unit_duration
                event_types.extend(explicit)
        target_len = max(1, math.ceil(duration_s * TOKEN_RATE_HZ))
        repeat_padding = 0
        if insertion["kind"] == "repeat" and target_len < min_repeat_mask_tokens:
            repeat_padding = int(min_repeat_mask_tokens) - target_len
            target_len = int(min_repeat_mask_tokens)
        insertion_edit = {
            "left": left,
            "right": right,
            "target_len": target_len,
            "kind": "insert" if right == left else "replace_gap",
            "text": insertion["text"],
            "events": list(dict.fromkeys(event_types)),
            "duration_s": duration_s,
            "repeat_duration_scale": (
                repeat_duration_scale if insertion["kind"] == "repeat" else 1.0
            ),
            "duration_source": (
                "previous_text"
                if copied_durations is not None
                else "event_distribution" if insertion["kind"] == "i" else "utterance_median"
            ),
        }
        if repeat_padding > 0:
            insertion_edit["minimum_repeat_mask_padding_tokens"] = repeat_padding
            insertion_edit["effective_duration_s"] = target_len / TOKEN_RATE_HZ
        edits.append(insertion_edit)

    if not edits:
        raise ValueError("annotation contains no usable stutter event")
    diagnostics = {
        "base_char_count": len(base_units),
        "aligned_base_char_count": sum(interval is not None for interval in table),
        "alignment_ratio": sum(interval is not None for interval in table) / max(len(base_units), 1),
        "natural_char_s": natural_char_s,
        "minimum_repeat_mask_tokens": int(min_repeat_mask_tokens),
    }
    return edits, diagnostics


def apply_event_edits(
    semantic_tokens: torch.Tensor, edits: Sequence[Dict[str, Any]]
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    tokens = semantic_tokens.detach().cpu().long()
    mask = torch.zeros(tokens.numel(), dtype=torch.long)
    applied: List[Dict[str, Any]] = []
    for edit in sorted(edits, key=lambda item: (int(item["left"]), int(item["right"])), reverse=True):
        left = max(0, min(tokens.numel(), int(edit["left"])))
        right = max(left, min(tokens.numel(), int(edit["right"])))
        target_len = max(1, int(edit["target_len"]))
        replacement = torch.full((target_len,), SEMANTIC_MASK_VALUE, dtype=torch.long)
        replacement_mask = torch.ones(target_len, dtype=torch.long)
        tokens = torch.cat((tokens[:left], replacement, tokens[right:]))
        mask = torch.cat((mask[:left], replacement_mask, mask[right:]))
        applied_edit = dict(edit)
        applied_edit.update({"source_left": left, "source_right": right, "mask_token_count": target_len})
        applied.append(applied_edit)

    # Convert source-token positions to positions on the edited semantic
    # timeline. MaskGCT semantic tokens run at TOKEN_RATE_HZ (50 Hz), so these
    # positions can also be exposed as timestamps for the generated speech.
    ordered = list(reversed(applied))
    shift = 0
    for edit in ordered:
        source_left = int(edit["source_left"])
        source_right = int(edit["source_right"])
        target_len = int(edit["mask_token_count"])
        target_left = source_left + shift
        target_right = target_left + target_len
        edit.update(
            {
                "target_left": target_left,
                "target_right": target_right,
                "source_start_s": source_left / TOKEN_RATE_HZ,
                "source_end_s": source_right / TOKEN_RATE_HZ,
                "target_start_s": target_left / TOKEN_RATE_HZ,
                "target_end_s": target_right / TOKEN_RATE_HZ,
            }
        )
        shift += target_len - (source_right - source_left)
    return tokens, mask, ordered


def load_stutter_annotations(path: str | Path) -> Dict[str, Dict[str, str]]:
    annotations: Dict[str, Dict[str, str]] = {}
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_no} must contain wav path and annotated text")
            wav, text = parts
            utt_id = Path(wav).stem
            if utt_id in annotations:
                raise ValueError(f"duplicate utt_id {utt_id} in {path}")
            annotations[utt_id] = {"wav": wav, "text": text}
    return annotations


class AISHELLStutterTokenDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        stutter_text_path: str | Path,
        duration_summary_path: str | Path,
        duration_mode: str = "triangular",
        seed: int = 1234,
        semantic_extractor: Optional[Any] = None,
        acoustic_extractor: Optional[Any] = None,
        semantic_extractor_kwargs: Optional[Dict[str, Any]] = None,
        acoustic_extractor_kwargs: Optional[Dict[str, Any]] = None,
        expected_acoustic_quantizers: int = 12,
        min_alignment_ratio: float = 0.8,
        repeat_duration_scale: float = 1.2,
        min_repeat_mask_tokens: int = 20,
    ):
        self.manifest_path = Path(manifest_path)
        annotations = load_stutter_annotations(stutter_text_path)
        self.records = [record for record in read_jsonl(self.manifest_path) if str(record["utt_id"]) in annotations]
        for record in self.records:
            record["stutter_text"] = annotations[str(record["utt_id"])]["text"]
        self.sampler = DurationSampler(duration_summary_path, duration_mode, seed)
        self.semantic_extractor = semantic_extractor or MaskGCTSemanticExtractor(
            **(semantic_extractor_kwargs or {})
        )
        self.acoustic_extractor = acoustic_extractor or MaskGCTAcousticExtractor(
            **(acoustic_extractor_kwargs or {})
        )
        self.expected_acoustic_quantizers = int(expected_acoustic_quantizers)
        self.min_alignment_ratio = float(min_alignment_ratio)
        if repeat_duration_scale < 1.0:
            raise ValueError("repeat_duration_scale must be at least 1.0")
        self.repeat_duration_scale = float(repeat_duration_scale)
        if min_repeat_mask_tokens < 1:
            raise ValueError("min_repeat_mask_tokens must be at least 1")
        self.min_repeat_mask_tokens = int(min_repeat_mask_tokens)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        utt_id = str(record["utt_id"])
        wav_path = str(record["audio_path"])
        extractor_record = dict(record)
        extractor_record["wav"] = resolve_manifest_path(wav_path, self.manifest_path)
        original_semantic = as_long_tensor(self.semantic_extractor(extractor_record), "semantic_tokens")
        acoustic_tokens = torch.as_tensor(self.acoustic_extractor(extractor_record), dtype=torch.long).cpu()
        if acoustic_tokens.ndim != 2:
            raise ValueError(f"acoustic tokens for {utt_id} must be [time, quantizer]")
        if self.expected_acoustic_quantizers and acoustic_tokens.size(1) != self.expected_acoustic_quantizers:
            raise ValueError(
                f"expected {self.expected_acoustic_quantizers} acoustic quantizers for {utt_id}, "
                f"got {acoustic_tokens.size(1)}"
            )
        textgrid_path = resolve_manifest_path(record["textgrid_path"], self.manifest_path)
        words = parse_textgrid_words(textgrid_path)
        annotation = str(record["stutter_text"])
        edits, diagnostics = build_event_edits(
            annotation,
            words,
            original_semantic.numel(),
            self.sampler,
            utt_id,
            repeat_duration_scale=self.repeat_duration_scale,
            min_repeat_mask_tokens=self.min_repeat_mask_tokens,
        )
        if diagnostics["alignment_ratio"] < self.min_alignment_ratio:
            raise ValueError(
                f"alignment ratio {diagnostics['alignment_ratio']:.3f} is below "
                f"{self.min_alignment_ratio:.3f} for {utt_id}"
            )
        semantic_tokens, mask_sequence, applied_edits = apply_event_edits(original_semantic, edits)
        # Keep /b, /p, /i and /r as explicit T2S conditioning tokens.
        g2p_text = clean_raw_text_for_g2p(annotation)
        phoneme_sequence, token_ids = chn_eng_g2p(g2p_text)
        phone_ids = torch.tensor([int(token_id) for token_id in token_ids], dtype=torch.long)
        sample_record = {
            "utt_id": utt_id,
            "wav": wav_path,
            "raw_text": annotation,
            "original_text": str(record.get("original_text") or ""),
            "audio_duration_s": float(record.get("duration") or 0.0),
        }
        return {
            **sample_record,
            "g2p_text": g2p_text,
            "phoneme_sequence": phoneme_sequence,
            "phone_ids": phone_ids,
            "phone_len": torch.tensor(phone_ids.numel(), dtype=torch.long),
            "mask_sequence": mask_sequence,
            "mask_len": torch.tensor(mask_sequence.numel(), dtype=torch.long),
            "mask_sum": torch.tensor(int(mask_sequence.sum()), dtype=torch.long),
            "semantic_tokens": semantic_tokens,
            "semantic_len": torch.tensor(semantic_tokens.numel(), dtype=torch.long),
            "original_semantic_len": torch.tensor(original_semantic.numel(), dtype=torch.long),
            "acoustic_tokens": acoustic_tokens,
            "acoustic_len": torch.tensor(acoustic_tokens.size(0), dtype=torch.long),
            "num_quantizer": torch.tensor(acoustic_tokens.size(1), dtype=torch.long),
            "event_edits": applied_edits,
            "alignment_diagnostics": diagnostics,
            "record": sample_record,
        }


def build_timestamp_record(split: str, item: Dict[str, Any]) -> Dict[str, Any]:
    raw_text = str(item["raw_text"])
    # Applied edits can omit unaligned events or reorder them by position.
    # Their own text is the source of truth for generated timestamp regions.
    event_texts = [str(edit.get("text") or "") for edit in item["event_edits"]]
    events = []
    for event_index, edit in enumerate(item["event_edits"]):
        event_text = event_texts[event_index]
        events.append(
            {
                "text": event_text,
                "start_s": round(float(edit["target_start_s"]), 6),
                "end_s": round(float(edit["target_end_s"]), 6),
                "duration_s": round(
                    float(edit["target_end_s"]) - float(edit["target_start_s"]),
                    6,
                ),
            }
        )
    utt_id = str(item["utt_id"])
    return {
        "utt_id": utt_id,
        "split": split,
        "raw_text": raw_text,
        "event_texts": event_texts,
        "stutter_regions": events,
        "audio_duration_s": round(int(item["semantic_len"]) / TOKEN_RATE_HZ, 6),
    }


def save_cache(
    dataset: AISHELLStutterTokenDataset,
    out_dir: str | Path,
    overwrite: bool,
    resume: bool,
    split: str,
    timestamp_name: str = "timestamps.jsonl",
) -> Path:
    out_dir = Path(out_dir)
    sample_dir = out_dir / "samples"
    manifest_path = out_dir / "manifest.jsonl"
    failure_path = out_dir / "failures.jsonl"
    timestamp_path = out_dir / timestamp_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    completed: Dict[int, Dict[str, Any]] = {}
    timestamps: Dict[str, Dict[str, Any]] = {}
    failures: List[Dict[str, Any]] = []
    if resume and not overwrite and manifest_path.exists():
        for record in read_jsonl(manifest_path, skip_invalid=True):
            sample_path = resolve_sample_path(record["sample_path"], out_dir)
            if sample_path.exists():
                completed[int(record["index"])] = record
    if resume and not overwrite and timestamp_path.exists():
        for record in read_jsonl(timestamp_path, skip_invalid=True):
            if all(
                key in record
                for key in (
                    "utt_id",
                    "split",
                    "raw_text",
                    "event_texts",
                    "stutter_regions",
                    "audio_duration_s",
                )
            ):
                timestamps[str(record["utt_id"])] = record
    if overwrite:
        completed.clear()
        timestamps.clear()
    write_jsonl(manifest_path, [completed[key] for key in sorted(completed)])
    write_jsonl(failure_path, [])
    write_jsonl(timestamp_path, [timestamps[key] for key in sorted(timestamps)])

    for index in range(len(dataset)):
        record = dataset.records[index]
        utt_id = str(record["utt_id"])
        if index in completed:
            if utt_id not in timestamps:
                sample_path = resolve_sample_path(completed[index]["sample_path"], out_dir)
                cached_item = torch.load(sample_path, map_location="cpu", weights_only=False)
                timestamp_record = build_timestamp_record(split, cached_item)
                append_jsonl(timestamp_path, timestamp_record)
                timestamps[utt_id] = timestamp_record
            print(f"[{index + 1}/{len(dataset)}] reused {utt_id}", flush=True)
            continue
        sample_path = sample_dir / f"{index:08d}_{safe_filename(utt_id)}.pt"
        try:
            item = dataset[index]
            torch.save(item, sample_path)
        except Exception as exc:
            failure = {"index": index, "utt_id": utt_id, "error": repr(exc)}
            failures.append(failure)
            append_jsonl(failure_path, failure)
            print(f"[{index + 1}/{len(dataset)}] failed {utt_id}: {exc}", file=sys.stderr, flush=True)
            continue
        try:
            portable_sample_path = sample_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        except ValueError:
            portable_sample_path = manifest_sample_path(sample_path)
        cache_record = {
            "index": index,
            "utt_id": utt_id,
            "wav": item["wav"],
            "raw_text": item["raw_text"],
            "original_text": item["original_text"],
            "g2p_text": item["g2p_text"],
            "phoneme_sequence": item["phoneme_sequence"],
            "sample_path": portable_sample_path,
            "phone_len": int(item["phone_len"]),
            "mask_len": int(item["mask_len"]),
            "mask_sum": int(item["mask_sum"]),
            "semantic_len": int(item["semantic_len"]),
            "original_semantic_len": int(item["original_semantic_len"]),
            "acoustic_len": int(item["acoustic_len"]),
            "num_quantizer": int(item["num_quantizer"]),
            "event_count": len(item["event_edits"]),
        }
        append_jsonl(manifest_path, cache_record)
        timestamp_record = build_timestamp_record(split, item)
        append_jsonl(timestamp_path, timestamp_record)
        timestamps[utt_id] = timestamp_record
        print(
            f"[{index + 1}/{len(dataset)}] saved {utt_id} "
            f"semantic={int(item['original_semantic_len'])}->{int(item['semantic_len'])} "
            f"mask_sum={int(item['mask_sum'])} events={len(item['event_edits'])}",
            flush=True,
        )
    print(
        f"cache_manifest={manifest_path} timestamps={timestamp_path} "
        f"failures={len(failures)}",
        flush=True,
    )
    return manifest_path


def split_manifest(data_root: Path, split: str, name: str) -> Path:
    return data_root / split / name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/AISHELL-1")
    parser.add_argument("--split", choices=("train", "dev", "test", "all"), default="all")
    parser.add_argument("--manifest-name", default="manifest_qwen3_alignment.jsonl")
    parser.add_argument("--stutter-text", default="aishell_stutter.txt")
    parser.add_argument("--duration-summary", default="duration_summary.json")
    parser.add_argument("--duration-mode", choices=("triangular", "p25", "median", "p75"), default="triangular")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir", default="data/AISHELL-1/stutter_maskgct_cache")
    parser.add_argument(
        "--timestamp-name",
        default="timestamps.jsonl",
        help="JSONL filename written inside each split cache directory.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cfg-path", default=str(MASKGCT_ROOT / "config" / "maskgct.json"))
    parser.add_argument("--w2v-bert-path", default=str(MASKGCT_ROOT / "MaskGCT_model" / "w2v_bert"))
    parser.add_argument("--stats-path", default=str(MASKGCT_ROOT / "ckpt" / "wav2vec2bert_stats.pt"))
    parser.add_argument("--semantic-codec-ckpt", default=str(MASKGCT_ROOT / "MaskGCT_model" / "semantic_codec" / "model.safetensors"))
    parser.add_argument("--codec-encoder-ckpt", default=str(MASKGCT_ROOT / "MaskGCT_model" / "acoustic_codec" / "model.safetensors"))
    parser.add_argument("--codec-decoder-ckpt", default=str(MASKGCT_ROOT / "MaskGCT_model" / "acoustic_codec" / "model_1.safetensors"))
    parser.add_argument("--expected-acoustic-quantizers", type=int, default=12)
    parser.add_argument("--min-alignment-ratio", type=float, default=0.8)
    parser.add_argument(
        "--repeat-duration-scale",
        type=float,
        default=1.2,
        help="Duration multiplier for the character component inside [...].",
    )
    parser.add_argument(
        "--min-repeat-mask-tokens",
        type=int,
        default=20,
        help="Pad each [...] repeat region to at least this many semantic tokens.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    def from_project(path: str | Path) -> Path:
        value = Path(path)
        return value if value.is_absolute() else PROJECT_ROOT / value

    data_root = from_project(args.data_root)
    stutter_text_path = from_project(args.stutter_text)
    duration_summary_path = from_project(args.duration_summary)
    out_root = from_project(args.out_dir)
    semantic_kwargs = {
        "device": args.device,
        "cfg_path": args.cfg_path,
        "w2v_bert_path": args.w2v_bert_path,
        "stats_path": args.stats_path,
        "semantic_codec_ckpt": args.semantic_codec_ckpt,
        "hidden_state_layer": 17,
        "sample_rate": 16000,
    }
    acoustic_kwargs = {
        "device": args.device,
        "cfg_path": args.cfg_path,
        "codec_encoder_ckpt": args.codec_encoder_ckpt,
        "codec_decoder_ckpt": args.codec_decoder_ckpt,
        "sample_rate": 24000,
        "n_quantizers": None,
    }
    semantic_extractor = MaskGCTSemanticExtractor(**semantic_kwargs)
    acoustic_extractor = MaskGCTAcousticExtractor(**acoustic_kwargs)
    splits = ("train", "dev", "test") if args.split == "all" else (args.split,)
    for split in splits:
        manifest = split_manifest(data_root, split, args.manifest_name)
        if not manifest.exists():
            raise FileNotFoundError(f"alignment manifest not found: {manifest}")
        dataset = AISHELLStutterTokenDataset(
            manifest_path=manifest,
            stutter_text_path=stutter_text_path,
            duration_summary_path=duration_summary_path,
            duration_mode=args.duration_mode,
            seed=args.seed,
            semantic_extractor=semantic_extractor,
            acoustic_extractor=acoustic_extractor,
            expected_acoustic_quantizers=args.expected_acoustic_quantizers,
            min_alignment_ratio=args.min_alignment_ratio,
            repeat_duration_scale=args.repeat_duration_scale,
            min_repeat_mask_tokens=args.min_repeat_mask_tokens,
        )
        print(f"[{split}] matched annotated samples={len(dataset)}", flush=True)
        save_cache(
            dataset,
            out_root / split,
            args.overwrite,
            not args.no_resume,
            split=split,
            timestamp_name=args.timestamp_name,
        )


if __name__ == "__main__":
    main()
