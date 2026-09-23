# Contributing

Use Issues to discuss changes and report reproducible bugs. Include the model
directory, command, OS, Python/CUDA/package versions and a fictional minimal
input. Do not attach private recordings, credentials or unauthorized weights.

Keep model environments separate. Run `python scripts/check_repository.py` from
the root before opening a pull request. Model changes should document the
checkpoint/tokenizer combination and any GPU experiments actually performed.
Do not present source tests as speech-quality evaluation. Keep special-token
ordering stable and retain upstream copyright notices.

Explain the behavior changed, tests run, and remaining limitations in your PR.
Contributions follow the license of the component being changed; see
[LICENSE.md](LICENSE.md). Community conduct is described in
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
