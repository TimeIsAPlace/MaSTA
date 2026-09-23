"""Static release checks; does not import models or need GPU packages."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    count = 0
    excluded = {'data', 'outputs', '__pycache__', '.git', '.venv', 'MaskGCT_model', 'ckpt', 'sources'}
    for path in ROOT.rglob('*'):
        if not path.is_file() or excluded.intersection(path.relative_to(ROOT).parts):
            continue
        if path.suffix in {'.pt', '.pth', '.safetensors', '.onnx', '.wav', '.flac'}:
            raise ValueError('Unexpected binary in release: ' + str(path))
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
            count += 1
        if path.suffix == '.json':
            json.loads(path.read_text(encoding='utf-8-sig'))
        if path.suffix == '.jsonl':
            for line in path.read_text(encoding='utf-8-sig').splitlines():
                if line.strip():
                    json.loads(line)
    model = ROOT / 'MaskGCT'
    local_packages = {'codec', 'g2p', 'module', 'utils'}
    for path in model.rglob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and not node.level and node.module:
                if node.module.split('.')[0] in local_packages:
                    target = model.joinpath(*node.module.split('.'))
                    if not target.is_dir() and not target.with_suffix('.py').is_file():
                        raise ValueError('Missing local dependency: ' + node.module)
    print('OK: {} Python files, JSON schemas, local imports and release file types.'.format(count))


if __name__ == '__main__':
    main()
