"""Install external assets without publishing weights in this repository."""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1] / 'MaskGCT'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--amphion-root', type=Path, required=True,
                        help='An official Amphion checkout with Git LFS assets downloaded.')
    parser.add_argument('--skip-download', action='store_true')
    args = parser.parse_args()
    upstream = args.amphion_root / 'models/tts/maskgct'
    sources = upstream / 'g2p/sources'
    stats = upstream / 'ckpt/wav2vec2bert_stats.pt'
    for path in (sources / 'g2p_chinese_model/poly_bert_model.onnx', stats):
        if not path.is_file():
            parser.error('Missing upstream asset: ' + str(path))
        with path.open('rb') as f:
            if f.read(80).startswith(b'version https://git-lfs.github.com/spec'):
                parser.error('Git LFS pointer instead of asset: ' + str(path))
    shutil.copytree(sources, ROOT / 'g2p/sources', dirs_exist_ok=True)
    (ROOT / 'ckpt').mkdir(exist_ok=True)
    shutil.copy2(stats, ROOT / 'ckpt' / stats.name)
    if not args.skip_download:
        from huggingface_hub import snapshot_download
        snapshot_download('amphion/MaskGCT', local_dir=str(ROOT / 'MaskGCT_model'))
        snapshot_download('facebook/w2v-bert-2.0', local_dir=str(ROOT / 'MaskGCT_model/w2v_bert'))
    print('Assets prepared. Keep the bundled modified g2p code and vocabulary unchanged.')


if __name__ == '__main__':
    main()
