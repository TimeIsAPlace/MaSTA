#!/usr/bin/env python3
"""Check actual inference imports, not only the top-level CosyVoice2 class."""
import argparse
import importlib
import importlib.metadata
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party" / "Matcha-TTS")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", help="Optionally load a complete model; requires its weights and runtime.")
    args = parser.parse_args()
    failed = False
    print("Python:", sys.executable, sys.version.split()[0])
    if sys.version_info < (3, 10):
        print("FAIL: use Python 3.10 or later (3.10 is the intended runtime).")
        failed = True
    for name in ["torch", "torchaudio", "diffusers", "huggingface-hub",
                 "setuptools", "transformers", "lightning", "openai-whisper"]:
        try:
            print(name + ":", importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            print("MISSING:", name)
            failed = True
    for name in ["pkg_resources", "diffusers.models.activations",
                 "matcha.models.components.flow_matching", "cosyvoice.cli.cosyvoice"]:
        try:
            module = importlib.import_module(name)
            print("OK import", name, getattr(module, "__file__", ""))
        except Exception as error:
            print("FAIL import", name, type(error).__name__, str(error))
            failed = True
    print("ffmpeg:", shutil.which("ffmpeg") or "MISSING (required for training segmentation)")
    if not shutil.which("ffmpeg"):
        failed = True
    try:
        import torch
        import onnxruntime
        print("Torch CUDA:", torch.version.cuda, "available:", torch.cuda.is_available())
        print("ONNX providers:", onnxruntime.get_available_providers())
    except Exception as error:
        print("FAIL runtime:", error)
        failed = True
    if args.model_dir and not failed:
        from cosyvoice.cli.cosyvoice import CosyVoice2
        try:
            CosyVoice2(str(Path(args.model_dir).resolve()))
            print("OK: complete model loaded (audio generation not tested).")
        except Exception as error:
            print("FAIL model load:", error)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
