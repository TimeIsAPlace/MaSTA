# Third-party sources and licenses

| Component | Included path | Source | License |
| --- | --- | --- | --- |
| CosyVoice research snapshot | `cosyvoice/`, `tools/`, model configs | https://github.com/FunAudioLLM/CosyVoice | Apache-2.0; original file headers retained |
| Matcha-TTS source subset | `third_party/Matcha-TTS/matcha/` | https://github.com/shivammehta25/Matcha-TTS | MIT; see adjacent LICENSE |
| Matcha HiFi-GAN component | `third_party/Matcha-TTS/matcha/hifigan/` | Matcha-TTS source distribution | See `matcha/hifigan/LICENSE` |
| Python dependencies | Downloaded during installation, not vendored | Package registries and upstream projects | Respective package licenses |

The root Apache-2.0 text was retrieved from the official CosyVoice repository:
https://github.com/FunAudioLLM/CosyVoice/blob/main/LICENSE

Existing inline attributions to other projects/authors remain in the copied
source. This document supplements those notices and does not replace them.

## Snapshot provenance

The available project was a filesystem snapshot without Git history. We cannot
claim an exact upstream commit or a complete change list relative to upstream.
The `release-source-manifest.json` records SHA-256 hashes of copied inputs and
their original paths (`source_path`) and current release paths (`path`). Paths do not identify the source machine.
Hashes describe the input snapshot; files subsequently adjusted for publication
may differ. See `docs/release-notes.md` for intentional packaging changes.

## Assets excluded from this release

No model weights, ONNX models, real annotation corpus, audio, generated speech,
or dataset manifests are distributed. The tokenizer vocabulary asset retained
under `cosyvoice/tokenizer/assets/` belongs to the upstream code distribution.
`recipes/sample_data/` contains newly written fictional examples with no audio.

Code licensing does not grant rights to redistribute external datasets or model
weights. Obtain them from their publishers under their own terms.
