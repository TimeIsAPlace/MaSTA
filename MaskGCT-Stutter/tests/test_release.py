import ast
import json
from pathlib import Path
import subprocess
import sys
from contextlib import nullcontext
import unittest
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_timestamps_follow_applied_edits(self):
        # Isolate this pure metadata function without importing GPU dependencies.
        source = (ROOT / 'MaskGCT/dataset_aishell_stutter.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'build_timestamp_record')
        tree.body = [function]
        scope = {'Dict': Dict, 'Any': Any, 'TOKEN_RATE_HZ': 50.0}
        exec(compile(tree, '<timestamp-export>', 'exec'), scope)
        record = scope['build_timestamp_record']('dev', {
            'utt_id': 'example', 'raw_text': '甲/b乙/p', 'semantic_len': 100,
            'event_edits': [{'text': '乙', 'target_start_s': 0.5, 'target_end_s': 1.0}],
        })
        self.assertEqual(record['event_texts'], ['乙'])
        self.assertEqual(record['stutter_regions'][0]['text'], '乙')
        self.assertEqual(record['stutter_regions'][0]['duration_s'], 0.5)
        self.assertNotIn('from organize_fill_mask_wavs import', source)

    def test_stage_handoff_dry_run(self):
        expected = {1: 'train.py', 2: 'train_fill_mask.py', 3: 'train_fill_mask_stutter.py'}
        with nullcontext(ROOT / 'examples') as temp:
            before = list(Path(temp).iterdir())
            for stage, entry in expected.items():
                result = subprocess.run([
                    sys.executable, str(ROOT / 'scripts/train_stage.py'), '--stage', str(stage),
                    '--train-manifest', 'train.jsonl', '--valid-manifest', 'dev.jsonl',
                    '--init-checkpoint', 'previous.safetensors', '--dry-run'],
                    cwd=temp, check=True, capture_output=True, text=True)
                run = json.loads(result.stdout)
                self.assertEqual(Path(run['command'][1]).name, entry)
                self.assertEqual(run['config']['train']['batch_size'], 2)
                self.assertEqual(run['config']['train']['grad_accumulation_steps'], 8)
                self.assertEqual(run['config']['train']['keep_training_epochs'], -1)
                self.assertEqual(Path(run['config']['train']['init_checkpoint']),
                                 Path(temp) / 'previous.safetensors')
            self.assertEqual(list(Path(temp).iterdir()), before)

    def test_missing_input_fails_before_training(self):
        result = subprocess.run([
            sys.executable, str(ROOT / 'scripts/train_stage.py'), '--stage', '1',
            '--train-manifest', 'absent_train.jsonl', '--valid-manifest', 'absent_dev.jsonl',
            '--init-checkpoint', 'absent.safetensors'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Required file not found', result.stderr)

    def test_all_stages_accept_explicit_checkpoint(self):
        for name in ('train.py', 'train_fill_mask.py', 'train_fill_mask_stutter.py'):
            source = (ROOT / 'MaskGCT' / name).read_text(encoding='utf-8')
            ast.parse(source)
            self.assertIn('"train.init_checkpoint", T2S_MODEL_CKPT', source)

    def test_event_vocabulary_retained(self):
        vocab = json.loads((ROOT / 'MaskGCT/g2p/g2p/vocab.json').read_text(encoding='utf-8'))['vocab']
        for event in ('/b', '/p', '/i', '/r'):
            self.assertIn(event, vocab)


if __name__ == '__main__':
    unittest.main()
