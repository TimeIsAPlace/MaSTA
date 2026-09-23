# Vendored Matcha-TTS source

Source: https://github.com/shivammehta25/Matcha-TTS
License: MIT; see LICENSE and matcha/hifigan/LICENSE.

This directory is a source subset of the local research snapshot, included for
CosyVoice's Flow Matching, decoder and HiFi-GAN imports. The upstream standalone
Matcha training CLI, configuration tree, Cython binaries and build setup are not
packaged as a supported application here.

Use the root requirements.txt and the recipe's PYTHONPATH. Do not install this
directory's historical full training requirements separately. No submodule fetch
is required. The retained historical requirements file aligns diffusers and
setuptools with the root compatibility pins.
