# MaSTA · Controllable stutter speech synthesis

[中文](README.md) · [Live demo](https://timeisaplace.github.io/MaSTA/) · [Local preview](demo_page/README.md) · [Publishing](docs/PUBLISHING.md)

This research repository contains two implementations:

- [CosyVoice2-Stutter](CosyVoice2-Stutter/README.en.md): fine-grained inline event instruction tuning and normal/stutter utterance-level instruction tuning, with reference-conditioned synthesis.
- [MaskGCT-Stutter](MaskGCT-Stutter/README.md): three-stage T2S adaptation and masked semantic-token infilling, followed by acoustic decoding. Inference requires prepared caches and masks.

Use separate Python environments: their PyTorch and Transformers versions differ. Follow each model's README from its own directory. Full training/inference targets Linux, Python 3.10 and NVIDIA CUDA GPUs. Training data and model weights are not included; base weights alone do not implement the fine-tuned stutter controls.

The [live listening demo](https://timeisaplace.github.io/MaSTA/) contains 20 groups and 60 audio files, playable directly in your browser without installing a model. For a local preview, run `python -m http.server 8000 --bind 127.0.0.1` from this repository root, then visit <http://localhost:8000/demo_page/>. After pushing page or audio changes, manually run [Deploy demo](https://github.com/TimeIsAPlace/MaSTA/actions/workflows/pages.yml) to update the published site. See the [publishing guide](docs/PUBLISHING.md) and [audio source notes](demo_page/AUDIO_SOURCES.md).

Run `python scripts/check_repository.py` for offline source checks and lightweight tests. These checks do not establish GPU compatibility, speech quality or benchmark reproducibility. See [validation](docs/VALIDATION.md), [contributing](CONTRIBUTING.md), [security](SECURITY.md), and the component-specific [licenses](LICENSE.md).
