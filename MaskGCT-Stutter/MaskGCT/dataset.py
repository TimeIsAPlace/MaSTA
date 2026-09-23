import json
import argparse
import difflib
import math
import unicodedata
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MASKGCT_ROOT = Path(__file__).resolve().parent
if str(MASKGCT_ROOT) not in sys.path:
    sys.path.insert(0, str(MASKGCT_ROOT))

from g2p.g2p_generation import chn_eng_g2p
PHONE_PAD_ID = 1023
SEMANTIC_PAD_ID = 0
ACOUSTIC_PAD_ID = 0


def read_jsonl(path: str | Path, skip_invalid: bool = False) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    if skip_invalid:
                        print(f"skip invalid jsonl line {line_no} in {path}", flush=True)
                        continue
                    raise
    return records


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def append_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def safe_filename(name: Any, default: str = "sample") -> str:
    name = str(name or default)
    name = re.sub(r"[^0-9A-Za-z_.-]+", "_", name)
    return name.strip("._") or default


def as_long_tensor(value: Any, field_name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().long()
    elif isinstance(value, (list, tuple)):
        tensor = torch.tensor(value, dtype=torch.long)
    else:
        raise TypeError(f"{field_name} must be a tensor/list/tuple, got {type(value).__name__}")
    if tensor.ndim > 1:
        tensor = tensor.squeeze()
    if tensor.ndim != 1:
        raise ValueError(f"{field_name} must be 1-D after squeeze, got shape {tuple(tensor.shape)}")
    return tensor


def resize_binary_mask(mask: torch.Tensor, target_len: int) -> torch.Tensor:
    if target_len <= 0:
        return torch.zeros(0, dtype=torch.long)
    mask = mask.long()
    if mask.numel() == target_len:
        return mask
    if mask.numel() == 0:
        return torch.zeros(target_len, dtype=torch.long)
    positions = torch.linspace(0, mask.numel() - 1, target_len)
    indices = positions.round().long().clamp(0, mask.numel() - 1)
    return mask.index_select(0, indices)


def resolve_manifest_path(path: str | Path, manifest_path: str | Path) -> str:
    path = Path(path)
    if path.is_absolute():
        return str(path)

    manifest_path = Path(manifest_path)
    candidates = [
        manifest_path.parent / path,
        PROJECT_ROOT / path,
        Path.cwd() / path,
    ]
    parts = path.parts
    if "Stutter-data" in parts:
        stutter_index = parts.index("Stutter-data")
        candidates.append(PROJECT_ROOT.joinpath(*parts[stutter_index:]))
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


ANGLE_SPAN = re.compile(r"<[^>]*>")
STUTTER_MARKER = re.compile(r"/[bipr]", flags=re.IGNORECASE)
TEXTGRID_SPECIAL_LABELS = {"", "NULL", "sil", "sp", "spn"}
TOKEN_RATE_HZ = 50.0


def is_alignable_text_char(char: str) -> bool:
    if not char or char.isspace():
        return False
    if char in "[]<>/":
        return False
    category = unicodedata.category(char)
    return not (category.startswith("P") or category.startswith("S"))


def strip_textgrid_label(label: str) -> str:
    return (label or "").strip().strip('"')


def parse_textgrid_words(textgrid_path: str | Path) -> List[Dict[str, Any]]:
    path = Path(textgrid_path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    in_words = False
    intervals = []
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
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
    return [item for item in intervals if item["text"] not in TEXTGRID_SPECIAL_LABELS]


def raw_text_units_for_stutter_mask(raw_text: Any) -> List[Dict[str, Any]]:
    text = ANGLE_SPAN.sub("", str(raw_text or ""))
    units: List[Dict[str, Any]] = []
    idx = 0
    while idx < len(text):
        char = text[idx]
        if char == "[":
            end = text.find("]", idx + 1)
            if end < 0:
                idx += 1
                continue
            bracket_chars = [c for c in text[idx + 1 : end] if is_alignable_text_char(c)]
            if bracket_chars:
                max_overlap = min(len(units), len(bracket_chars))
                for overlap in range(max_overlap, 0, -1):
                    prev_chars = [unit["char"] for unit in units[-overlap:]]
                    if prev_chars == bracket_chars[:overlap]:
                        for unit in units[-overlap:]:
                            unit["event"] = True
                        break
                for c in bracket_chars:
                    units.append({"char": c, "event": True})
            idx = end + 1
            continue
        if char == "/" and idx + 1 < len(text) and text[idx : idx + 2].lower() in {"/b", "/i", "/p", "/r"}:
            for unit in reversed(units):
                if is_alignable_text_char(unit["char"]):
                    unit["event"] = True
                    break
            idx += 2
            continue
        if is_alignable_text_char(char):
            marker = text[idx + 1 : idx + 3].lower()
            is_marked = marker in {"/b", "/i", "/p", "/r"}
            units.append({"char": char, "event": is_marked})
            idx += 3 if is_marked else 1
            continue
        idx += 1
    return units


def build_unit_time_table(
    units: List[Dict[str, Any]],
    word_intervals: List[Dict[str, Any]],
) -> List[Optional[Dict[str, Any]]]:
    source = [str(unit["char"]) for unit in units]
    target = [str(interval["text"]) for interval in word_intervals]
    table: List[Optional[Dict[str, Any]]] = [None] * len(source)
    matcher = difflib.SequenceMatcher(a=source, b=target, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        source_len = i2 - i1
        target_len = j2 - j1
        if tag == "equal":
            for offset in range(source_len):
                table[i1 + offset] = word_intervals[j1 + offset]
        elif tag == "replace":
            for offset in range(min(source_len, target_len)):
                table[i1 + offset] = word_intervals[j1 + offset]
    return table


def non_event_aligned_intervals(
    units: List[Dict[str, Any]],
    unit_time_table: List[Optional[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    return [
        interval
        for unit, interval in zip(units, unit_time_table)
        if not unit["event"] and interval is not None
    ]

def stutter_event_spans(units: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    spans = []
    idx = 0
    while idx < len(units):
        if not units[idx]["event"]:
            idx += 1
            continue
        start = idx
        while idx + 1 < len(units) and units[idx + 1]["event"]:
            idx += 1
        spans.append((start, idx))
        idx += 1
    return spans


def mask_time_span(mask: torch.Tensor, start_s: float, end_s: float) -> None:
    token_len = int(mask.numel())
    start_idx = max(0, min(token_len, int(float(start_s) * TOKEN_RATE_HZ)))
    end_idx = max(start_idx, min(token_len, int(math.ceil(float(end_s) * TOKEN_RATE_HZ))))
    mask[start_idx:end_idx] = 1


def build_stutter_mask_from_textgrid(raw_text: Any, textgrid_path: str | Path, token_len: int) -> torch.Tensor:
    mask = torch.zeros(max(int(token_len), 0), dtype=torch.long)
    if token_len <= 0:
        return mask
    if not textgrid_path:
        mask[:] = 1
        return mask

    path = Path(textgrid_path)
    if not path.exists():
        mask[:] = 1
        return mask

    word_intervals = parse_textgrid_words(path)
    if not word_intervals:
        mask[:] = 1
        return mask

    units = raw_text_units_for_stutter_mask(raw_text)
    unit_time_table = build_unit_time_table(units, word_intervals)
    aligned_non_events = non_event_aligned_intervals(units, unit_time_table)

    if any(unit["event"] for unit in units) and not aligned_non_events:
        mask[:] = 1
        return mask

    if not any(unit["event"] for unit in units):
        aligned_intervals = aligned_non_events or word_intervals
        ratio = random.uniform(0.1, 0.3)
        sample_count = max(1, min(len(aligned_intervals), int(math.ceil(len(aligned_intervals) * ratio))))
        for interval in random.sample(aligned_intervals, sample_count):
            mask_time_span(mask, float(interval["start"]), float(interval["end"]))
        return mask

    for start_unit, end_unit in stutter_event_spans(units):
        prev_interval = None
        next_interval = None
        for idx in range(start_unit - 1, -1, -1):
            interval = unit_time_table[idx]
            if not units[idx]["event"] and interval is not None:
                prev_interval = interval
                break
        for idx in range(end_unit + 1, len(units)):
            interval = unit_time_table[idx]
            if not units[idx]["event"] and interval is not None:
                next_interval = interval
                break
        if prev_interval is None and next_interval is None:
            continue
        start_s = float(prev_interval["end"]) if prev_interval is not None else 0.0
        end_s = float(next_interval["start"]) if next_interval is not None else float(token_len) / TOKEN_RATE_HZ
        if end_s <= start_s:
            continue
        mask_time_span(mask, start_s, end_s)
    if mask.sum().item() == 0:
        mask[:] = 1
    return mask


def clean_raw_text_for_g2p(text: Any) -> str:
    text = str(text or "")
    text = ANGLE_SPAN.sub("", text)
    text = text.replace("[", "").replace("]", "")
    return text.strip()



class MaskGCTSemanticExtractor:
    """Extract MaskGCT semantic codec tokens from wav records.

    The extractor expects each JSONL record to contain ``wav``. It always loads
    the whole audio file and uses MaskGCT's Wav2Vec2-BERT semantic model plus
    semantic codec and returns a 1-D LongTensor semantic code sequence.
    """

    def __init__(
        self,
        device: str | torch.device = "cuda",
        cfg_path: str | Path = MASKGCT_ROOT / "config" / "maskgct.json",
        w2v_bert_path: str | Path = MASKGCT_ROOT / "MaskGCT_model" / "w2v_bert",
        stats_path: str | Path = MASKGCT_ROOT / "ckpt" / "wav2vec2bert_stats.pt",
        semantic_codec_ckpt: str | Path = MASKGCT_ROOT
        / "MaskGCT_model"
        / "semantic_codec"
        / "model.safetensors",
        hidden_state_layer: int = 17,
        sample_rate: int = 16000,
    ):
        self.device = torch.device(device)
        self.sample_rate = int(sample_rate)
        self.hidden_state_layer = int(hidden_state_layer)

        maskgct_root = str(MASKGCT_ROOT)
        if maskgct_root not in sys.path:
            sys.path.insert(0, maskgct_root)

        try:
            import librosa
            import safetensors.torch
            from codec.kmeans.repcodec_model import RepCodec
            from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
            from utils.util import load_config
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "MaskGCTSemanticExtractor requires librosa, safetensors, "
                "transformers, and MaskGCT modules."
            ) from e

        self.librosa = librosa
        self.processor = SeamlessM4TFeatureExtractor.from_pretrained(str(w2v_bert_path))
        self.semantic_model = Wav2Vec2BertModel.from_pretrained(str(w2v_bert_path))
        self.semantic_model.eval().to(self.device)

        cfg = load_config(str(cfg_path))
        self.semantic_codec = RepCodec(cfg=cfg.model.semantic_codec)
        self.semantic_codec.eval().to(self.device)
        safetensors.torch.load_model(self.semantic_codec, str(semantic_codec_ckpt))

        stat_mean_var = torch.load(stats_path, map_location=self.device)
        self.semantic_mean = stat_mean_var["mean"].to(self.device)
        self.semantic_std = torch.sqrt(stat_mean_var["var"]).to(self.device)

    def load_audio(self, record: Dict[str, Any]) -> torch.Tensor:
        wav_path = record.get("wav")
        if not wav_path:
            raise KeyError(f"Record {record.get('utt_id')} has no wav path.")
        kwargs = {"sr": self.sample_rate, "mono": True}
        wav, _ = self.librosa.load(wav_path, **kwargs)
        return torch.tensor(wav, dtype=torch.float32)

    @torch.no_grad()
    def __call__(self, record: Dict[str, Any]) -> torch.Tensor:
        wav = self.load_audio(record)
        inputs = self.processor(
            wav.numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        outputs = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = outputs.hidden_states[self.hidden_state_layer]
        feat = (feat - self.semantic_mean.to(feat)) / self.semantic_std.to(feat)
        semantic_code, _ = self.semantic_codec.quantize(feat)
        return semantic_code.squeeze(0).detach().cpu().long()


class MaskGCTAcousticExtractor:
    """Extract MaskGCT acoustic codec tokens from wav records."""

    def __init__(
        self,
        device: str | torch.device = "cuda",
        cfg_path: str | Path = MASKGCT_ROOT / "config" / "maskgct.json",
        codec_encoder_ckpt: str | Path = MASKGCT_ROOT
        / "MaskGCT_model"
        / "acoustic_codec"
        / "model.safetensors",
        codec_decoder_ckpt: str | Path = MASKGCT_ROOT
        / "MaskGCT_model"
        / "acoustic_codec"
        / "model_1.safetensors",
        sample_rate: int = 24000,
        n_quantizers: Optional[int] = None,
    ):
        self.device = torch.device(device)
        self.sample_rate = int(sample_rate)
        self.n_quantizers = n_quantizers

        maskgct_root = str(MASKGCT_ROOT)
        if maskgct_root not in sys.path:
            sys.path.insert(0, maskgct_root)

        try:
            import librosa
            import safetensors.torch
            from codec.amphion_codec.codec import CodecDecoder, CodecEncoder
            from utils.util import load_config
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "MaskGCTAcousticExtractor requires librosa, safetensors, "
                "and MaskGCT acoustic codec modules."
            ) from e

        self.librosa = librosa
        cfg = load_config(str(cfg_path))
        self.codec_encoder = CodecEncoder(cfg=cfg.model.acoustic_codec.encoder)
        self.codec_decoder = CodecDecoder(cfg=cfg.model.acoustic_codec.decoder)
        self.codec_encoder.eval().to(self.device)
        self.codec_decoder.eval().to(self.device)
        safetensors.torch.load_model(self.codec_encoder, str(codec_encoder_ckpt))
        safetensors.torch.load_model(self.codec_decoder, str(codec_decoder_ckpt))

    def load_audio(self, record: Dict[str, Any]) -> torch.Tensor:
        wav_path = record.get("wav")
        if not wav_path:
            raise KeyError(f"Record {record.get('utt_id')} has no wav path.")
        kwargs = {"sr": self.sample_rate, "mono": True}
        wav, _ = self.librosa.load(wav_path, **kwargs)
        return torch.tensor(wav, dtype=torch.float32)

    @torch.no_grad()
    def __call__(self, record: Dict[str, Any]) -> torch.Tensor:
        wav = self.load_audio(record)
        speech = wav.unsqueeze(0).to(self.device)
        vq_emb = self.codec_encoder(speech.unsqueeze(1))
        _, vq = self.codec_decoder.quantize(vq_emb, n_quantizers=self.n_quantizers)
        return vq.permute(1, 2, 0).squeeze(0).detach().cpu().long()


class MaskGCTAudioTokenDataset(Dataset):
    """Single-audio dataset that extracts MaskGCT semantic and acoustic tokens.

    Each manifest record describes one audio file. The returned sample contains
    only the current audio's metadata plus semantic tokens and acoustic codec tokens.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        wav_field: str = "audio_path",
        utt_field: str = "utt_id",
        raw_text_field: str = "raw_text",
        textgrid_field: str = "textgrid_path",
        semantic_extractor: Optional[Callable[[Dict[str, Any]], Sequence[int] | torch.Tensor]] = None,
        acoustic_extractor: Optional[Callable[[Dict[str, Any]], Sequence[Sequence[int]] | torch.Tensor]] = None,
        semantic_extractor_kwargs: Optional[Dict[str, Any]] = None,
        acoustic_extractor_kwargs: Optional[Dict[str, Any]] = None,
        expected_acoustic_quantizers: int = 12,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.records = read_jsonl(self.jsonl_path)
        self.wav_field = wav_field
        self.utt_field = utt_field
        self.raw_text_field = raw_text_field
        self.textgrid_field = textgrid_field
        self.expected_acoustic_quantizers = int(expected_acoustic_quantizers)
        self.semantic_extractor = semantic_extractor or MaskGCTSemanticExtractor(
            **(semantic_extractor_kwargs or {})
        )
        self.acoustic_extractor = acoustic_extractor or MaskGCTAcousticExtractor(
            **(acoustic_extractor_kwargs or {})
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


    def _raw_textgrid_path(self, record: Dict[str, Any]) -> Optional[Any]:
        candidate_fields = [self.textgrid_field, "textgrid_path", "alignment_path", "TextGrid"]
        for field in candidate_fields:
            if field and record.get(field):
                return record[field]
        return None
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
        acoustic_tokens = torch.as_tensor(
            self.acoustic_extractor(extractor_record), dtype=torch.long
        ).detach().cpu()
        if acoustic_tokens.ndim != 2:
            raise ValueError(
                "acoustic_extractor(record) must return [time, num_quantizer], "
                f"got shape {tuple(acoustic_tokens.shape)} for {utt_id}"
            )
        if self.expected_acoustic_quantizers > 0 and acoustic_tokens.size(1) != self.expected_acoustic_quantizers:
            raise ValueError(
                f"Expected {self.expected_acoustic_quantizers} acoustic quantizers, "
                f"got {acoustic_tokens.size(1)} for {utt_id}. "
                "Set expected_acoustic_quantizers=0 to disable this check."
            )

        raw_text = str(record.get(self.raw_text_field, ""))
        textgrid_path = self._raw_textgrid_path(record)
        resolved_textgrid_path = (
            resolve_manifest_path(textgrid_path, self.jsonl_path) if textgrid_path else None
        )
        mask_sequence = build_stutter_mask_from_textgrid(
            raw_text, resolved_textgrid_path, int(semantic_tokens.numel())
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
            "acoustic_tokens": acoustic_tokens,
            "acoustic_len": torch.tensor(acoustic_tokens.size(0), dtype=torch.long),
            "num_quantizer": torch.tensor(acoustic_tokens.size(1), dtype=torch.long),
            "record": sample_record,
        }


def save_maskgct_token_cache(
    jsonl_path: str | Path,
    out_dir: str | Path,
    wav_field: str = "audio_path",
    utt_field: str = "utt_id",
    raw_text_field: str = "raw_text",
    textgrid_field: str = "textgrid_path",
    semantic_extractor_kwargs: Optional[Dict[str, Any]] = None,
    acoustic_extractor_kwargs: Optional[Dict[str, Any]] = None,
    expected_acoustic_quantizers: int = 12,
    overwrite: bool = False,
    resume: bool = True,
) -> Path:
    """Precompute one cache sample per audio.

    It writes:
      - samples/*.pt with semantic_tokens, semantic-aligned mask_sequence, and acoustic_tokens [time, 12]
      - manifest.jsonl as a lightweight index for MaskGCTCachedAudioTokenDataset
    """

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
        "acoustic_tokens",
        "phone_ids",
        "mask_sequence",
        "mask_len",
        "mask_sum",
    }

    def has_complete_token_cache(item: Dict[str, Any]) -> bool:
        return required_sample_keys.issubset(item.keys())
    dataset = MaskGCTAudioTokenDataset(
        jsonl_path=jsonl_path,
        wav_field=wav_field,
        utt_field=utt_field,
        raw_text_field=raw_text_field,
        textgrid_field=textgrid_field,
        semantic_extractor_kwargs=semantic_extractor_kwargs,
        acoustic_extractor_kwargs=acoustic_extractor_kwargs,
        expected_acoustic_quantizers=expected_acoustic_quantizers,
    )

    def build_manifest_record(index: int, item: Dict[str, Any], sample_path: Path) -> Dict[str, Any]:
        return {
            "index": index,
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
            "acoustic_len": int(item["acoustic_len"]),
            "num_quantizer": int(item["num_quantizer"]),
        }

    completed_records: Dict[int, Dict[str, Any]] = {}
    if resume and not overwrite and manifest_path.exists():
        for record in read_jsonl(manifest_path, skip_invalid=True):
            record_index = int(record["index"])
            record_sample_path = resolve_sample_path(record["sample_path"], out_dir)
            if record_sample_path.exists() and "mask_sum" in record and "mask_len" in record:
                record["sample_path"] = manifest_sample_path(record_sample_path)
                completed_records[record_index] = record
        write_jsonl(
            manifest_path,
            [completed_records[index] for index in sorted(completed_records)],
        )
    else:
        write_jsonl(manifest_path, [])

    for index in range(len(dataset)):
        raw_record = dataset.records[index]
        utt_id = dataset._record_id(raw_record, index)
        sample_name = f"{index:08d}_{safe_filename(utt_id)}.pt"
        sample_path = sample_dir / sample_name

        if resume and not overwrite and sample_path.exists():
            if index in completed_records:
                print(
                    f"[{index + 1}/{len(dataset)}] reused {utt_id} from {sample_path.name}",
                    flush=True,
                )
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
                    print(
                        f"[{index + 1}/{len(dataset)}] incomplete cached sample {sample_path.name}; recomputing",
                        flush=True,
                    )
                else:
                    append_jsonl(manifest_path, build_manifest_record(index, item, sample_path))
                    print(
                        f"[{index + 1}/{len(dataset)}] reused {utt_id} from {sample_path.name}",
                        flush=True,
                    )
                    continue

        item = dataset[index]
        torch.save(item, sample_path)
        append_jsonl(manifest_path, build_manifest_record(index, item, sample_path))
        print(
            f"[{index + 1}/{len(dataset)}] saved {item['utt_id']} "
            f"semantic_len={int(item['semantic_len'])} "
            f"acoustic_len={int(item['acoustic_len'])} "
            f"num_quantizer={int(item['num_quantizer'])} "
            f"mask_sum={int(item['mask_sum'])}",
            flush=True,
        )

    return manifest_path


class MaskGCTCachedAudioTokenDataset(Dataset):
    """Dataset for locally cached single-audio MaskGCT token samples."""

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


def pad_2d(sequences: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    if not sequences:
        return torch.empty(0, dtype=torch.long)
    feature_dim = int(sequences[0].size(1))
    max_len = max(seq.size(0) for seq in sequences)
    out = torch.full(
        (len(sequences), max_len, feature_dim),
        int(pad_value),
        dtype=torch.long,
    )
    for idx, seq in enumerate(sequences):
        if seq.ndim != 2:
            raise ValueError(f"pad_2d expects [time, feature], got {tuple(seq.shape)}")
        if int(seq.size(1)) != feature_dim:
            raise ValueError("All 2-D sequences must share the same feature dimension.")
        out[idx, : seq.size(0), :] = seq.long()
    return out


class MaskGCTAudioTokenCollate:
    def __init__(
        self,
        phone_pad_id: int = PHONE_PAD_ID,
        semantic_pad_id: int = SEMANTIC_PAD_ID,
        acoustic_pad_id: int = ACOUSTIC_PAD_ID,
    ):
        self.phone_pad_id = int(phone_pad_id)
        self.semantic_pad_id = int(semantic_pad_id)
        self.acoustic_pad_id = int(acoustic_pad_id)

    def __call__(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        phone_ids = [item["phone_ids"] for item in items]
        semantic_tokens = [item["semantic_tokens"] for item in items]
        acoustic_tokens = [item["acoustic_tokens"] for item in items]
        mask_sequence = [item["mask_sequence"] for item in items]
        phone_lens = torch.tensor([seq.numel() for seq in phone_ids], dtype=torch.long)
        semantic_lens = torch.tensor([seq.numel() for seq in semantic_tokens], dtype=torch.long)
        acoustic_lens = torch.tensor([seq.size(0) for seq in acoustic_tokens], dtype=torch.long)
        mask_lens = torch.tensor([seq.numel() for seq in mask_sequence], dtype=torch.long)

        phone_ids_pad = pad_1d(phone_ids, self.phone_pad_id)
        semantic_tokens_pad = pad_1d(semantic_tokens, self.semantic_pad_id)
        acoustic_tokens_pad = pad_2d(acoustic_tokens, self.acoustic_pad_id)
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
        acoustic_mask = (
            torch.arange(acoustic_tokens_pad.size(1)).unsqueeze(0) < acoustic_lens.unsqueeze(1)
            if acoustic_tokens_pad.numel()
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
            "acoustic_tokens": acoustic_tokens_pad,
            "acoustic_lens": acoustic_lens,
            "acoustic_mask": acoustic_mask,
            "num_quantizers": torch.tensor(
                [int(item["num_quantizer"]) for item in items],
                dtype=torch.long,
            ),
            "records": [item["record"] for item in items],
        }




def maskgct_audio_token_collate(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return MaskGCTAudioTokenCollate()(items)




def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MaskGCT single-audio token dataset utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-cache",
        help="Precompute semantic tokens and 12-layer acoustic tokens for each audio.",
    )
    prepare.add_argument("--jsonl", required=True, help="Input JSONL manifest.")
    prepare.add_argument("--out-dir", required=True, help="Output cache directory.")
    prepare.add_argument("--wav-field", default="audio_path")
    prepare.add_argument("--utt-field", default="utt_id")
    prepare.add_argument("--raw-text-field", default="raw_text")
    prepare.add_argument("--textgrid-field", default="textgrid_path")
    prepare.add_argument("--device", default="cuda")
    prepare.add_argument("--hidden-state-layer", type=int, default=17)
    prepare.add_argument("--semantic-sample-rate", type=int, default=16000)
    prepare.add_argument("--acoustic-sample-rate", type=int, default=24000)
    prepare.add_argument("--cfg-path", default=str(MASKGCT_ROOT / "config" / "maskgct.json"))
    prepare.add_argument("--w2v-bert-path", default=str(MASKGCT_ROOT / "MaskGCT_model" / "w2v_bert"))
    prepare.add_argument("--stats-path", default=str(MASKGCT_ROOT / "ckpt" / "wav2vec2bert_stats.pt"))
    prepare.add_argument(
        "--semantic-codec-ckpt",
        default=str(MASKGCT_ROOT / "MaskGCT_model" / "semantic_codec" / "model.safetensors"),
    )
    prepare.add_argument(
        "--codec-encoder-ckpt",
        default=str(MASKGCT_ROOT / "MaskGCT_model" / "acoustic_codec" / "model.safetensors"),
    )
    prepare.add_argument(
        "--codec-decoder-ckpt",
        default=str(MASKGCT_ROOT / "MaskGCT_model" / "acoustic_codec" / "model_1.safetensors"),
    )
    prepare.add_argument(
        "--expected-acoustic-quantizers",
        type=int,
        default=12,
        help="Expected acoustic quantizer layers. 0 disables the check.",
    )
    prepare.add_argument("--overwrite", action="store_true")
    prepare.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resumable cache preparation and fail if manifest.jsonl already exists.",
    )
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
        acoustic_extractor_kwargs = {
            "device": args.device,
            "cfg_path": args.cfg_path,
            "codec_encoder_ckpt": args.codec_encoder_ckpt,
            "codec_decoder_ckpt": args.codec_decoder_ckpt,
            "sample_rate": args.acoustic_sample_rate,
            "n_quantizers": None,
        }
        manifest_path = save_maskgct_token_cache(
            jsonl_path=args.jsonl,
            out_dir=args.out_dir,
            wav_field=args.wav_field,
            utt_field=args.utt_field,
            raw_text_field=args.raw_text_field,
            textgrid_field=args.textgrid_field,
            semantic_extractor_kwargs=semantic_extractor_kwargs,
            acoustic_extractor_kwargs=acoustic_extractor_kwargs,
            expected_acoustic_quantizers=args.expected_acoustic_quantizers,
            overwrite=args.overwrite,
            resume=not args.no_resume,
        )
        print(f"cache_manifest={manifest_path}")


if __name__ == "__main__":
    main()































