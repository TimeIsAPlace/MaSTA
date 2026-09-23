# Validation scope

## Local result (2026-09-23)

Windows offline checks passed using both Python 3.7.5 and 3.12.14: CosyVoice2 11 tests,
MaskGCT 5 tests, both release checkers, Node.js syntax checks and all shell scripts
via Git Bash. The demo check passed for 20 groups / 60 WAV files. Python 3.7 is
only a host used for these dependency-free checks, not a supported model runtime.
YAML parsing passed for the root workflows, issue/dependency configuration and
citation file. Git's actual upload listing includes 333 source/demo files, all 60
WAVs and the vendored Matcha data modules; generated Python caches are ignored.
Full model dependency installation and audio metadata regeneration were not run.

Run `python scripts/check_repository.py` from the repository root. This checks
Python/JSON syntax, local Markdown links, flattened symbolic links, common secret
patterns, personal paths and unwanted model artifacts; validates demo IDs, paths,
WAV durations and waveform ranges; runs both model checkers and lightweight tests;
and checks JavaScript/Bash syntax when those runtimes are available.

Root CI repeats these checks on Linux and Windows with Python 3.10/3.11.
The first hosted CI execution occurs only after pushing to GitHub.

This does not install full model dependencies, download weights, perform GPU
training/inference, assess speech quality, or establish audio redistribution rights.
Model reproduction requires a separate clean Linux/CUDA run, a licensed dataset,
compatible checkpoints, and a recorded environment/revision/seed.

Packaging fixes include materializing flattened CosyVoice example links, removing
machine-specific example paths, correcting the sample-data README path, and
exporting MaskGCT event timestamps from applied edits rather than a missing helper.
