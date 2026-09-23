#!/usr/bin/env python3
"""Offline source-release checks. Does not claim to be a comprehensive secret scanner."""
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {".git", "__pycache__", ".pytest_cache", ".venv", "venv"}
FORBIDDEN_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".bin", ".parquet",
    ".wav", ".flac", ".mp3", ".m4a", ".mp4", ".so", ".pyd", ".zip",
}
PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(r"\b(?:ghp_|github_pat_|hf_)[A-Za-z0-9_]{25,}\b"),
    re.compile(r"/data/(?:luyao|nankai)/|C:[\\/]Users[\\/]Lenovo", re.I),
]


def main():
    errors = []
    files = [p for p in ROOT.rglob("*") if p.is_file()
             and not any(part in IGNORED for part in p.relative_to(ROOT).parts)]
    for name in ["LICENSE", "NOTICE", "README.md", "README.en.md", ".gitignore",
                 ".gitattributes", "THIRD_PARTY_NOTICES.md",
                 "third_party/Matcha-TTS/LICENSE", "third_party/Matcha-TTS/matcha/hifigan/LICENSE"]:
        if not (ROOT / name).is_file():
            errors.append("Missing required file: " + name)
    for path in files:
        rel = path.relative_to(ROOT)
        if rel.parts[0] in {"data", "datasets", "exp", "tensorboard", "pretrained_models"}:
            errors.append("Generated/private directory included: " + str(rel))
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name == "aishell_stutter.txt":
            errors.append("Dataset/model/binary artifact included: " + str(rel))
        if path.stat().st_size > 10 * 1024 * 1024:
            errors.append("Unexpected file over 10 MiB: " + str(rel))
        content = path.read_text(encoding="utf-8")
        if any(pattern.search(content) for pattern in PATTERNS):
            errors.append("Possible credential or personal absolute path: " + str(rel))
        if path.suffix in {".py", ".sh", ".yml", ".yaml"} and b"\r\n" in path.read_bytes():
            errors.append("Use LF line endings: " + str(rel))
        if path.suffix == ".py":
            try:
                ast.parse(content, filename=str(rel))
            except SyntaxError as error:
                errors.append(str(error))
        if path.suffix == ".json":
            try:
                json.loads(content)
            except ValueError as error:
                errors.append(str(rel) + ": " + str(error))
        if path.suffix == ".jsonl":
            for index, line in enumerate(content.splitlines(), 1):
                if line.strip():
                    try:
                        json.loads(line)
                    except ValueError as error:
                        errors.append(f"{rel}:{index}: {error}")
    # Required local script/config references in the public shell recipes.
    for recipe in (ROOT / "recipes/cosyvoice2").glob("*.sh"):
        recipe_text = recipe.read_text(encoding="utf-8")
        for index, block in enumerate(re.findall(r"<<'PY'\n(.*?)\nPY\n", recipe_text, re.S), 1):
            try:
                ast.parse(block, filename=f"{recipe.name}:python-block-{index}")
            except SyntaxError as error:
                errors.append(str(error))
        for value in re.findall(r"(?:tools|cosyvoice|recipes|configs|examples)/[\w/.-]+\.(?:py|yaml|json)", recipe_text):
            if not (ROOT / value).is_file():
                errors.append(f"{recipe.name}: missing referenced file {value}")
    if errors:
        print("\n".join(errors))
        return 1
    total = sum(p.stat().st_size for p in files)
    print(f"PASS: {len(files)} release files, {total / 1024 / 1024:.2f} MiB.")
    print("Checked source syntax, local recipe references, formats, licenses and common publication artifacts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
