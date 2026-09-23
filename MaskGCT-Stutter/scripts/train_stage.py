"""Launch one training stage with explicit checkpoint handoff."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = {1: 'train.py', 2: 'train_fill_mask.py', 3: 'train_fill_mask_stutter.py'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=int, choices=STAGES, required=True)
    parser.add_argument('--train-manifest', type=Path, required=True)
    parser.add_argument('--valid-manifest', type=Path, required=True)
    parser.add_argument('--init-checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'MaskGCT/config/train.json')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    output = (args.output_dir or ROOT / 'outputs' / ('stage' + str(args.stage))).resolve()
    config['train'].update(init_checkpoint=str(args.init_checkpoint.absolute().resolve()),
                           save_model_path=str(output), keep_training_epochs=-1,
                           keep_training_steps=0)
    config_path = output / 'run_config.json'
    command = [sys.executable, str(ROOT / 'MaskGCT' / STAGES[args.stage]),
               '--config', str(config_path),
               '--train-manifest', str(args.train_manifest.absolute().resolve()),
               '--valid-manifest', str(args.valid_manifest.absolute().resolve())]
    print(json.dumps({'stage': args.stage, 'command': command,
                      'config': config}, indent=2), flush=True)
    if args.dry_run:
        return
    for path in (args.train_manifest, args.valid_manifest, args.init_checkpoint):
        if not path.is_file():
            parser.error('Required file not found: ' + str(path))
    output.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        parser.error('Output directory already contains a run; choose a new --output-dir. '
                     'For weight-only continuation, pass its checkpoint as --init-checkpoint.')
    config_path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    subprocess.run(command, cwd=ROOT / 'MaskGCT', check=True)


if __name__ == '__main__':
    main()
