# Initial source release

This release packages the local CosyVoice2 stutter research project as a separate
repository. The original workspace is not the publication root.

Included: CosyVoice core source, relevant CosyVoice2 configuration, stutter SFT /
fine-grained / instruct recipes, synthesis entry points, data preparation tools, Matcha
source subset and license notices.

Excluded: all datasets, real annotation text, audio, weights, compiled libraries,
cache directories, server/UI deployment extras and personal one-off scripts.

Publication changes:
- distinguished fine-grained event instructions from normal/stutter mode instructions;
- consolidated entry points under recipes/cosyvoice2 and configuration under configs;
- removed the Stutter-specific preference-training recipe and paired-data example;
- retained CosyVoice's original preference-training APIs and implementation unchanged;
- added Chinese and English documentation, fictional examples and CI;
- copied the upstream Apache-2.0 license and retained MIT component licenses;
- retained special-token order (including the unused legacy final token);
- defaulted GPU selection to device 0;
- fixed a trailing shell continuation before fi in the basic fine-tuning recipe;
- made AISHELL --dry_run usable without PyTorch;
- rejected duplicate-path/same-file prompt choices;
- made the Kaldi resynthesis entry reject self-prompts instead of falling back;
- retained compatibility pins for diffusers and setuptools;
- explicitly declared frontend dependencies that were previously implicit.

Per-file research-snapshot notices identify copied code conservatively, because
the local source had no Git history and may already include research changes.
They do not assert that every file differs from upstream.

Validation results for the assembled artifact are in RELEASE_VALIDATION.md.
