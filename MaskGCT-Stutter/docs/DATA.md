# Data and masks

Use UTF-8 JSONL, one utterance per line. Examples are synthetic schema examples,
not included recordings. Paths may be relative; keep cache preparation and
training under the documented working directory. The original path resolvers
try the current working directory, manifest/cache location, and project root.
Do not rename or relocate caches without checking `sample_path` and audio paths.

## Real stuttering speech

Required fields: `utt_id`, `audio_path`, `raw_text`, `textgrid_path`.
See `examples/stutter_manifest.jsonl`. Each waveform is already a complete segment;
the extractor does not apply a start/end crop.

| Tag | Meaning | Example |
| --- | --- | --- |
| `/b` | Block | 今/b天 |
| `/p` | Prolongation | 今/p天 |
| `/i` | Interjection | 嗯/i今天 |
| `/r` | Sound/syllable repetition | 今/r天 |
| `[...]` | Repeated characters/words | 今天[今天]去 |

Do not apply ASR transcript-cleaning rules before synthesis training. Slash tags
are passed to the event-aware G2P frontend. The existing dataset removes square
brackets but retains their enclosed text for G2P; angle-bracket annotations and
their content are removed. The original tagged transcript remains in `raw_text`.

TextGrid character anchors localize event regions, including relevant repeated
preceding text. Region boundaries are expanded to semantic-token indices using
floor on the left and ceil on the right, clipped to the token sequence. The token
timebase is 50 Hz. `1` means generate/mask; `0` means observed context.
For event-free real-speech samples, the original dataset randomly masks 10-30%
of characters; empty/unusable alignment handling follows `dataset.py`.
Inspect alignment and mask quality before training, especially repeated text.

Each `.pt` cache includes `utt_id`, `wav`, `raw_text`, `g2p_text`,
`phoneme_sequence`, `phone_ids`, `semantic_tokens`, `acoustic_tokens`,
`mask_sequence`, lengths/counts, and `record`.
Semantic tokens and mask are `[T]`; acoustic tokens are `[Ta, 12]`.
`T` and `Ta` can differ slightly. Masks always refer to semantic time, not acoustic
frame count. Load `.pt` files only from trusted sources.

## AISHELL-1 fluent training data

`dataset_aishell1.py` accepts aligned JSONL with `original_text` instead of
`raw_text`, or its existing AISHELL directory reader. Use `--raw-text-field`
to override the field name. It saves semantic tokens and character-aligned random
masks (20-40%), without acoustic tokens. This is the stage-2 cache, not an
artificially stuttered dataset.

## Optional AISHELL-1 stuttering synthesis

`dataset_aishell_stutter.py` is separate from stage-2 training. It consumes aligned
AISHELL manifests, event-tagged text, and measured event duration statistics to
construct time-modified masked caches for synthesis. After installing assets:

```bash
cd MaskGCT
python dataset_aishell_stutter.py --help
```

Its G2P resources still use `MaskGCT/` as the working directory; for actual
execution run from `MaskGCT/` and explicitly supply `--data-root`, `--stutter-text`,
`--duration-summary`, and `--out-dir` relative to that directory.
No patient-derived duration statistics or transcripts are included in this release.

Keep train/dev/test speakers disjoint according to your experiment protocol.
Do not use test recordings to train, choose checkpoints, or estimate durations.
