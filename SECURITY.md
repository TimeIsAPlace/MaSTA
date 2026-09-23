# Security policy

Security fixes target the default branch; older snapshots have no maintenance
guarantee. Enable GitHub private vulnerability reporting before public release.
Report a vulnerability via the repository's **Security → Report a vulnerability**
button, with the affected component, reproduction and impact. If that option is
unavailable, use a maintainer's publicly listed private contact channel; do not
publish exploit details or credentials in an issue.

Only load trusted checkpoints and caches: some PyTorch serialization formats can
execute code. The demo is a static listening page, not a production model-serving
service. This repository's lightweight checks are not a full security audit.
