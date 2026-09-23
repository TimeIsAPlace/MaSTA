"""Dependency-light tests for conversion and reference selection; no GPU/model."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "recipes/cosyvoice2/synthesize_aishell.py"
SPEC = importlib.util.spec_from_file_location("aishell_synthesis", SCRIPT)
synthesis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = synthesis
SPEC.loader.exec_module(synthesis)


class ConversionTests(unittest.TestCase):
    def test_markers(self):
        examples = {
            "市/b": "市[block]",
            "纷/r": "纷[phone_rep]",
            "的/p": "<prolong>的</prolong>",
            "嗯/i": "<fill>嗯</fill>",
            "其中[其中]": "其中<rep>其中</rep>",
            "个/r/p": "<prolong>个</prolong>[phone_rep]",
            "居住[居住/p]": "居住<rep>居<prolong>住</prolong></rep>",
        }
        for source, expected in examples.items():
            with self.subTest(source=source):
                self.assertEqual(synthesis.convert_transcript(source), expected)

    def test_invalid_annotations_fail(self):
        for source in ["文本/x", "未闭合[", "多余]", "/p", "字/i"]:
            with self.subTest(source=source), self.assertRaises(ValueError):
                synthesis.convert_transcript(source)

    def test_plain_text_is_preserved(self):
        self.assertEqual(synthesis.convert_transcript("这是一段示例。"), "这是一段示例。")


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        folder = self.root / "AISHELL-1/wav/train/S_DEMO"
        folder.mkdir(parents=True)
        self.paths = [folder / "a.wav", folder / "b.wav"]
        # Path checks do not decode audio; content is intentionally not a real WAV.
        for path in self.paths:
            path.write_bytes(b"not-real-audio")
        self.rows = [synthesis.SourceRow(i, p.relative_to(self.root), "嗯/i示例", "S_DEMO")
                     for i, p in enumerate(self.paths, 1)]

    def test_different_same_speaker(self):
        prompts = synthesis.choose_prompts(self.rows, self.root, {})
        self.assertEqual(prompts[1], self.paths[1])
        self.assertEqual(prompts[2], self.paths[0])

    def test_single_record_fails(self):
        with self.assertRaises(ValueError):
            synthesis.choose_prompts(self.rows[:1], self.root, {})

    def test_duplicate_path_fails(self):
        duplicate = synthesis.SourceRow(2, self.rows[0].relative_wav, "其他标注", "S_DEMO")
        with self.assertRaises(ValueError):
            synthesis.choose_prompts([self.rows[0], duplicate], self.root, {})

    def test_external_self_prompt_fails(self):
        with self.assertRaises(ValueError):
            synthesis.choose_prompts(self.rows, self.root, {"S_DEMO": self.paths[0]})

    def test_hard_link_self_prompt_fails(self):
        alias = self.root / "alias.wav"
        try:
            os.link(self.paths[0], alias)
        except OSError:
            self.skipTest("hard links unavailable")
        with self.assertRaises(ValueError):
            synthesis.choose_prompts(self.rows, self.root, {"S_DEMO": alias})

    def test_other_speaker_is_not_used(self):
        other = synthesis.SourceRow(2, self.rows[1].relative_wav, "示例", "OTHER")
        with self.assertRaises(ValueError):
            synthesis.choose_prompts([self.rows[0], other], self.root, {})

    def test_dry_run_without_model(self):
        manifest = self.root / "input.txt"
        manifest.write_text("\n".join(f"{row.relative_wav.as_posix()}\t{row.source_text}"
                                      for row in self.rows), encoding="utf-8")
        output = self.root / "preview"
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--model_dir", str(self.root / "missing-model"),
             "--input_file", str(manifest), "--audio_root", str(self.root),
             "--output_dir", str(output), "--num_lines", "2", "--dry_run"],
            cwd=self.root, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        records = [json.loads(line) for line in (output / "conversion_preview.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["tagged_text"], "<fill>嗯</fill>示例")
        self.assertNotEqual(records[0]["source_wav"], records[0]["prompt_wav"])
        self.assertEqual(list((output / "wavs").glob("*.wav")), [])

    def test_input_order_and_count(self):
        manifest = self.root / "input.txt"
        manifest.write_text("\n".join(f"{r.relative_wav.as_posix()}\t{r.source_text}"
                                      for r in self.rows), encoding="utf-8")
        rows = synthesis.read_rows(manifest, 1)
        self.assertEqual(rows[0].relative_wav, self.rows[0].relative_wav)
        with self.assertRaises(ValueError):
            synthesis.read_rows(manifest, 3)


if __name__ == "__main__":
    unittest.main()
