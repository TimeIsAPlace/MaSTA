# Demo audio sources and publication status

The checked-in listening collection has 20 groups / 60 WAV files. Dataset IDs
and annotated text are recorded in `as70/text.txt` and `aishell1/text.txt`.

| Paths | Description | Source / status |
| --- | --- | --- |
| `aishell1/GT/` | Original fluent reference recordings | [AISHELL-1 / OpenSLR 33](https://www.openslr.org/33/), listed upstream under Apache-2.0 |
| `as70/GT/` | Original stuttered reference recordings | [AS-70 project](https://stammertalk.github.io/interspeech2024-page/); exact source mapping and redistribution authorization must be supplied by the maintainer |
| `*/CosyVoice2/` | Synthesized examples | CosyVoice2-based experiment; exact checkpoint, inference settings and speaker authorization have not been supplied |
| `*/MaskGCT/` | Synthesized examples | MaskGCT-based experiment; exact checkpoint, inference settings and speaker authorization have not been supplied |

The software licenses do not grant rights to these recordings or dataset-derived
text. Before making this repository public, the maintainer must confirm that the
included recordings and derived outputs may be redistributed, retain the applicable
dataset notices, and supply any required attribution/consent. Public availability
of an upstream listening example alone does not establish permission to mirror it.
If permission is unavailable, omit the affected samples and regenerate `data.js`.

Record the following when finalizing this document: source recording IDs (including
the mapping for AS70's short IDs), dataset release and split, authorization basis,
checkpoint revision, prompt source, synthesis parameters and any audio processing.
Do not claim that these details or quality metrics were recovered from the WAV files.

The page labels synthetic outputs separately. It is for research comparison, not
clinical assessment or speaker impersonation.
