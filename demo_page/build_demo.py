"""Build offline demo metadata from the two local audio collections."""
import json
import soundfile as sf
from pathlib import Path


def audio_metadata(path, root):
    samples, rate = sf.read(str(path), always_2d=True)
    if not len(samples):
        raise ValueError("Empty audio: {}".format(path))
    duration = len(samples) / rate
    samples = abs(samples).max(axis=1)
    width = max(1, (len(samples) + 99) // 100)
    peaks = [float(samples[i:i + width].max())
             for i in range(0, len(samples), width)]
    peak = max(peaks) or 1
    return {"src": path.relative_to(root).as_posix(), "duration": duration,
            "peaks": [round(v / peak, 4) for v in peaks]}


def main():
    root = Path(__file__).resolve().parent
    records = []
    for dataset in ("as70", "aishell1"):
        folder = root / dataset
        seen = set()
        for line in (folder / "text.txt").read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            sample_id, text = line.split(maxsplit=1)
            if sample_id in seen or Path(sample_id).name != sample_id:
                raise ValueError("Invalid or duplicate id: " + sample_id)
            seen.add(sample_id)
            audio = {key: audio_metadata(folder / system / (sample_id + ".wav"), root)
                     for key, system in (("reference", "GT"),
                                         ("cosyvoice", "CosyVoice2"),
                                         ("maskgct", "MaskGCT"))}
            records.append({"id": sample_id, "text": text,
                            "dataset": dataset, "audio": audio})
        print("{}: {} samples".format(dataset, len(seen)))
    (root / "data.js").write_text(
        "window.DEMO_DATA = " + json.dumps(records, ensure_ascii=False) + ";\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
