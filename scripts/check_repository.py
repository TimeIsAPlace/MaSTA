"""Offline monorepo checks. No model imports, downloads or GPU required."""
import ast
import json
import math
import re
import shutil
import subprocess
import struct
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {'.git', '__pycache__', '.venv', 'venv', '.pytest_cache', '.cache'}


def check_sources():
    errors = []
    required = ['README.md', 'README.en.md', 'LICENSE.md', 'LICENSES/MIT.txt',
                'CITATION.cff', 'CONTRIBUTING.md', 'SECURITY.md',
                '.github/workflows/ci.yml', '.github/workflows/pages.yml']
    for name in required:
        if not (ROOT / name).is_file():
            errors.append('Missing repository file: ' + name)
    for path in ROOT.rglob('*'):
        if not path.is_file() or IGNORED.intersection(path.relative_to(ROOT).parts):
            continue
        rel = path.relative_to(ROOT).as_posix()
        if path.stat().st_size >= 50 * 1024 * 1024:
            errors.append('Unexpected large file: ' + rel)
        if path.suffix in {'.pt', '.pth', '.ckpt', '.safetensors', '.onnx', '.parquet'}:
            errors.append('Model/cache artifact in source tree: ' + rel)
        if path.suffix == '.wav' and not rel.startswith(('demo_page/as70/', 'demo_page/aishell1/')):
            errors.append('Unexpected audio: ' + rel)
        if path.suffix not in {'.py', '.md', '.json', '.jsonl', '.yml', '.yaml', '.sh', '.js', '.html', '.css', '.txt', '.cff'}:
            continue
        content = path.read_text(encoding='utf-8-sig')
        if re.search(r'\b(?:ghp_|github_pat_|hf_)[A-Za-z0-9_]{25,}\b', content):
            errors.append('Possible credential: ' + rel)
        if re.search(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----', content):
            errors.append('Private key: ' + rel)
        if re.search(r'/mnt/[\w.-]+/data/|[A-Z]:[\\/]Users[\\/](?!Public\b)', content):
            errors.append('Personal absolute path: ' + rel)
        if path.suffix == '.py':
            ast.parse(content, filename=rel)
        if path.suffix == '.json':
            json.loads(content)
        if path.suffix == '.jsonl':
            for line in content.splitlines():
                if line.strip():
                    json.loads(line)
        if path.suffix == '.md':
            for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', content):
                if re.match(r'[a-z]+:', target, re.I) or target.startswith('#'):
                    continue
                target = unquote(target.split('#')[0])
                if target and not (path.parent / target).exists():
                    errors.append('Broken local link in {}: {}'.format(rel, target))
        if content.strip().startswith('../') and '\n' not in content.strip():
            errors.append('Flattened symbolic link: ' + rel)
    return errors


def wav_duration(path):
    """Read PCM/IEEE-float RIFF duration without installing audio packages."""
    size = path.stat().st_size
    with path.open('rb') as stream:
        header = stream.read(12)
        if len(header) != 12 or header[:4] != b'RIFF' or header[8:] != b'WAVE':
            raise ValueError('Not a RIFF WAV: ' + str(path))
        if struct.unpack('<I', header[4:8])[0] + 8 != size:
            raise ValueError('Invalid RIFF size: ' + str(path))
        byte_rate, data_size = None, None
        while stream.tell() < size:
            chunk = stream.read(8)
            if len(chunk) != 8:
                raise ValueError('Truncated WAV chunk: ' + str(path))
            name, length = struct.unpack('<4sI', chunk)
            end = stream.tell() + length
            if end > size:
                raise ValueError('Truncated WAV: ' + str(path))
            if name == b'fmt ':
                if length < 16:
                    raise ValueError('Invalid WAV format: ' + str(path))
                fmt, channels, rate, byte_rate, align, bits = struct.unpack('<HHIIHH', stream.read(16))
                if fmt not in (1, 3) or not channels or not rate or not align or byte_rate != rate * align:
                    raise ValueError('Unsupported WAV encoding: ' + str(path))
            elif name == b'data':
                data_size = length
            stream.seek(end + length % 2)
        if not byte_rate or not data_size:
            raise ValueError('Empty WAV or missing format: ' + str(path))
        return data_size / byte_rate


def check_demo():
    demo = ROOT / 'demo_page'
    source = (demo / 'data.js').read_text(encoding='utf-8-sig').strip()
    prefix = 'window.DEMO_DATA = '
    if not source.startswith(prefix) or not source.endswith(';'):
        raise ValueError('Demo data must be a JSON assignment')
    records = json.loads(source[len(prefix):-1])
    expected = {}
    for dataset in ('as70', 'aishell1'):
        for line in (demo / dataset / 'text.txt').read_text(encoding='utf-8-sig').splitlines():
            if line.strip():
                sample_id, text = line.split(maxsplit=1)
                key = (dataset, sample_id)
                if key in expected or Path(sample_id).name != sample_id:
                    raise ValueError('Duplicate/unsafe sample ID: ' + sample_id)
                expected[key] = text
    seen, audio_paths = set(), set()
    for sample in records:
        key = (sample['dataset'], sample['id'])
        if key in seen or expected.get(key) != sample['text']:
            raise ValueError('Demo data does not match text manifests: ' + str(key))
        seen.add(key)
        for system, folder in [('reference', 'GT'), ('cosyvoice', 'CosyVoice2'), ('maskgct', 'MaskGCT')]:
            meta = sample['audio'][system]
            expected_src = '{}/{}/{}.wav'.format(key[0], folder, key[1])
            if meta['src'] != expected_src:
                raise ValueError('Unexpected demo path: ' + meta['src'])
            path = demo / expected_src
            audio_paths.add(path.resolve())
            duration = wav_duration(path)
            if abs(duration - meta['duration']) > 0.001:
                raise ValueError('Stale/empty audio metadata: ' + expected_src)
            peaks = meta['peaks']
            if not peaks or not all(isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1 for p in peaks):
                raise ValueError('Invalid waveform peaks: ' + expected_src)
    if seen != set(expected) or audio_paths != {p.resolve() for p in demo.rglob('*.wav')}:
        raise ValueError('Missing or unlisted demo recordings')
    html = (demo / 'index.html').read_text(encoding='utf-8')
    for target in re.findall(r'(?:src|href)="([^"#]+)"', html):
        if not re.match(r'[a-z]+:', target, re.I) and not (demo / target).is_file():
            raise ValueError('Broken demo resource: ' + target)
    print('PASS: demo manifests, {} groups / {} audio files, durations and local resources'.format(len(seen), len(audio_paths)), flush=True)


def main():
    errors = check_sources()
    if errors:
        raise ValueError('\n'.join(errors))
    check_demo()
    for project, checker in [('CosyVoice2-Stutter', 'check_release.py'),
                             ('MaskGCT-Stutter', 'check_repository.py')]:
        for args in [['scripts/' + checker], ['-m', 'unittest', 'discover', '-s', 'tests', '-v']]:
            subprocess.run([sys.executable] + args, cwd=ROOT / project, check=True)
    node = shutil.which('node')
    if node:
        for name in ('app.js', 'data.js'):
            subprocess.run([node, '--check', str(ROOT / 'demo_page' / name)], check=True)
    else:
        print('SKIP: Node.js syntax check (Node.js unavailable)')
    # Git Bash is preferable to a possibly unconfigured WSL launcher on Windows.
    bash = Path('C:/Program Files/Git/bin/bash.exe') if sys.platform == 'win32' else None
    bash = str(bash) if bash and bash.is_file() else (shutil.which('bash') if sys.platform != 'win32' else None)
    if bash:
        for project in ('CosyVoice2-Stutter', 'MaskGCT-Stutter'):
            for path in (ROOT / project).rglob('*.sh'):
                subprocess.run([bash, '-n', path.relative_to(ROOT).as_posix()], cwd=ROOT, check=True)
    else:
        print('SKIP: Bash syntax check (Bash unavailable; Linux CI checks it)')
    print('PASS: repository checks; GPU execution and publication permissions are not verified.')


if __name__ == '__main__':
    main()
