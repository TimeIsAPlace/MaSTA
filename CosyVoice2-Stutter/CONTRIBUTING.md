# Contributing

Please discuss significant changes in an issue before implementation. Include the
recipe, Python/CUDA versions, package versions, and a minimal reproducible example.
Do not attach private audio, credentials, checkpoints or unlicensed corpora.

Run from the repository root:

```bash
python -m unittest discover -s tests -v
python scripts/check_release.py
```

For model changes, also describe the GPU experiment and the checkpoint/tokenizer
combination tested. CI does not establish audio quality or model compatibility.
Keep existing special-token ordering stable to avoid silently changing token IDs.
Preserve upstream copyright/license headers. Contributions are submitted under
this repository's license unless otherwise agreed by its maintainers.
