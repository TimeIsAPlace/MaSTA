import argparse
import json
from pathlib import Path

import librosa
import safetensors.torch
import soundfile as sf
import torch

from codec.amphion_codec.codec import CodecDecoder, CodecEncoder
from maskgct_s2a import MaskGCT_S2A
from maskgct_t2s_fill_mask import MaskGCT_T2S_FillMask
from utils.util import load_config


MASKGCT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MASKGCT_ROOT.parent
DEFAULT_CFG = MASKGCT_ROOT / "config" / "maskgct.json"
DEFAULT_CODEC_ENCODER_CKPT = MASKGCT_ROOT / "MaskGCT_model" / "acoustic_codec" / "model.safetensors"
DEFAULT_CODEC_DECODER_CKPT = MASKGCT_ROOT / "MaskGCT_model" / "acoustic_codec" / "model_1.safetensors"
DEFAULT_S2A_1LAYER_CKPT = (
    MASKGCT_ROOT / "MaskGCT_model" / "s2a_model" / "s2a_model_1layer" / "model.safetensors"
)
DEFAULT_S2A_FULL_CKPT = (
    MASKGCT_ROOT / "MaskGCT_model" / "s2a_model" / "s2a_model_full" / "model.safetensors"
)


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else (Path.cwd() / path)


def resolve_existing_path(path, base_dirs=()):
    path = Path(path)
    if path.is_absolute():
        return path

    candidates = []
    for base_dir in base_dirs:
        candidates.append(Path(base_dir) / path)
    candidates.extend([
        Path.cwd() / path,
        PROJECT_ROOT / path,
        MASKGCT_ROOT / path,
        path,
    ])

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique_candidates.append(candidate)
            seen.add(key)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate
    return unique_candidates[0]


def load_cached_item(manifest_path, manifest_record):
    manifest_dir = Path(manifest_path).parent
    sample_path = resolve_existing_path(manifest_record["sample_path"], base_dirs=(manifest_dir,))
    item = torch.load(sample_path, map_location="cpu")
    item["cache_sample_path"] = str(sample_path)
    return item, sample_path


def load_audio_24k(record, base_dirs=()):
    wav_path = record.get("wav")
    if not wav_path:
        raise KeyError(f"Record {record.get('utt_id')} has no wav path.")
    wav_path = resolve_existing_path(wav_path, base_dirs=base_dirs)
    kwargs = {"sr": 24000, "mono": True}
    if record.get("start") is not None:
        kwargs["offset"] = float(record["start"])
    if record.get("start") is not None and record.get("end") is not None:
        kwargs["duration"] = max(float(record["end"]) - float(record["start"]), 0.0)
    wav, _ = librosa.load(wav_path, **kwargs)
    return torch.tensor(wav, dtype=torch.float32)


def build_t2s_fill_mask_model(cfg, device):
    model = MaskGCT_T2S_FillMask(cfg=cfg)
    model.eval().to(device)
    return model


def build_s2a_model(cfg, device):
    model = MaskGCT_S2A(cfg=cfg)
    model.eval().to(device)
    return model


def build_acoustic_codec(cfg, device):
    encoder = CodecEncoder(cfg=cfg.encoder)
    decoder = CodecDecoder(cfg=cfg.decoder)
    encoder.eval().to(device)
    decoder.eval().to(device)
    return encoder, decoder


@torch.no_grad()
def extract_acoustic_code(codec_encoder, codec_decoder, wav, device):
    speech = wav.unsqueeze(0).to(device)
    vq_emb = codec_encoder(speech.unsqueeze(1))
    _, vq, _, _, _ = codec_decoder.quantizer(vq_emb)
    return vq.permute(1, 2, 0)


@torch.no_grad()
def semantic_to_audio(
    semantic_code,
    acoustic_code,
    s2a_model_1layer,
    s2a_model_full,
    codec_decoder,
    n_timesteps_s2a,
    cfg_s2a,
    rescale_cfg_s2a,
):
    cond = s2a_model_1layer.cond_emb(semantic_code)
    prompt = acoustic_code[:, :0, :]
    predict_1layer = s2a_model_1layer.reverse_diffusion(
        cond=cond,
        prompt=prompt,
        temp=1.5,
        filter_thres=0.98,
        n_timesteps=n_timesteps_s2a[:1],
        cfg=cfg_s2a,
        rescale_cfg=rescale_cfg_s2a,
    )

    cond = s2a_model_full.cond_emb(semantic_code)
    predict_full = s2a_model_full.reverse_diffusion(
        cond=cond,
        prompt=prompt,
        temp=1.5,
        filter_thres=0.98,
        n_timesteps=n_timesteps_s2a,
        cfg=cfg_s2a,
        rescale_cfg=rescale_cfg_s2a,
        gt_code=predict_1layer,
    )

    vq_emb = codec_decoder.vq2emb(predict_full.permute(2, 0, 1), n_quantizers=12)
    audio = codec_decoder(vq_emb)
    return audio[0][0].detach().cpu().numpy()


@torch.no_grad()
def infer_one(
    item,
    t2s_model,
    codec_encoder,
    codec_decoder,
    s2a_model_1layer,
    s2a_model_full,
    device,
    n_timesteps_t2s,
    cfg_t2s,
    rescale_cfg_t2s,
    n_timesteps_s2a,
    cfg_s2a,
    rescale_cfg_s2a,
    path_base_dirs=(),
):
    semantic_tokens = item["semantic_tokens"].long().unsqueeze(0).to(device)
    target_mask = item["mask_sequence"].long().unsqueeze(0).to(device)
    phone_ids = item["phone_ids"].long().unsqueeze(0).to(device)
    semantic_mask = torch.ones_like(semantic_tokens, dtype=torch.bool, device=device)
    phone_mask = torch.ones_like(phone_ids, dtype=torch.bool, device=device)

    filled_semantic = t2s_model.reverse_diffusion_fill_mask(
        semantic_tokens=semantic_tokens,
        target_mask=target_mask,
        phone_id=phone_ids,
        semantic_mask=semantic_mask,
        phone_mask=phone_mask,
        n_timesteps=n_timesteps_t2s,
        cfg=cfg_t2s,
        rescale_cfg=rescale_cfg_t2s,
    )

    wav_24k = load_audio_24k(item["record"], base_dirs=path_base_dirs)
    acoustic_code = extract_acoustic_code(codec_encoder, codec_decoder, wav_24k, device)
    audio = semantic_to_audio(
        filled_semantic,
        acoustic_code,
        s2a_model_1layer,
        s2a_model_full,
        codec_decoder,
        n_timesteps_s2a=n_timesteps_s2a,
        cfg_s2a=cfg_s2a,
        rescale_cfg_s2a=rescale_cfg_s2a,
    )
    return audio, filled_semantic.detach().cpu().squeeze(0), target_mask.detach().cpu().squeeze(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Fill-mask inference on cached MaskGCT T2S data.")
    parser.add_argument("--manifest", required=True, help="Cached manifest.jsonl from dataset.py prepare-cache.")
    parser.add_argument("--t2s-ckpt", required=True, help="Fill-mask T2S checkpoint.")
    parser.add_argument("--out-dir", default="wav/fill_mask_train")
    parser.add_argument("--config", default=str(DEFAULT_CFG))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--utt-id", nargs="*", default=[])
    parser.add_argument("--n-timesteps-t2s", type=int, default=40)
    parser.add_argument("--cfg-t2s", type=float, default=1.0)
    parser.add_argument("--rescale-cfg-t2s", type=float, default=1.0)
    parser.add_argument(
        "--n-timesteps-s2a",
        type=int,
        nargs="+",
        default=[25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    )
    parser.add_argument("--cfg-s2a", type=float, default=2.5)
    parser.add_argument("--rescale-cfg-s2a", type=float, default=0.75)
    parser.add_argument("--codec-encoder-ckpt", default=str(DEFAULT_CODEC_ENCODER_CKPT))
    parser.add_argument("--codec-decoder-ckpt", default=str(DEFAULT_CODEC_DECODER_CKPT))
    parser.add_argument("--s2a-1layer-ckpt", default=str(DEFAULT_S2A_1LAYER_CKPT))
    parser.add_argument("--s2a-full-ckpt", default=str(DEFAULT_S2A_FULL_CKPT))
    parser.add_argument("--save-semantic", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)

    t2s_model = build_t2s_fill_mask_model(cfg.model.t2s_model, device)
    codec_encoder, codec_decoder = build_acoustic_codec(cfg.model.acoustic_codec, device)
    s2a_model_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, device)
    s2a_model_full = build_s2a_model(cfg.model.s2a_model.s2a_full, device)

    safetensors.torch.load_model(t2s_model, str(resolve_path(args.t2s_ckpt)))
    safetensors.torch.load_model(codec_encoder, str(resolve_path(args.codec_encoder_ckpt)))
    safetensors.torch.load_model(codec_decoder, str(resolve_path(args.codec_decoder_ckpt)))
    safetensors.torch.load_model(s2a_model_1layer, str(resolve_path(args.s2a_1layer_ckpt)))
    safetensors.torch.load_model(s2a_model_full, str(resolve_path(args.s2a_full_ckpt)))

    manifest_path = resolve_path(args.manifest)
    records = read_jsonl(manifest_path)
    if args.utt_id:
        wanted = set(args.utt_id)
        records = [record for record in records if record.get("utt_id") in wanted]
    if args.limit > 0:
        records = records[: args.limit]

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "inference_meta.jsonl"

    with open(meta_path, "w", encoding="utf-8", newline="\n") as meta_f:
        for index, manifest_record in enumerate(records, start=1):
            item, sample_path = load_cached_item(manifest_path, manifest_record)
            utt_id = str(item.get("utt_id", manifest_record.get("utt_id", index)))
            audio, filled_semantic, target_mask = infer_one(
                item,
                t2s_model,
                codec_encoder,
                codec_decoder,
                s2a_model_1layer,
                s2a_model_full,
                device,
                n_timesteps_t2s=args.n_timesteps_t2s,
                cfg_t2s=args.cfg_t2s,
                rescale_cfg_t2s=args.rescale_cfg_t2s,
                n_timesteps_s2a=args.n_timesteps_s2a,
                cfg_s2a=args.cfg_s2a,
                rescale_cfg_s2a=args.rescale_cfg_s2a,
                path_base_dirs=(sample_path.parent, manifest_path.parent),
            )

            wav_path = out_dir / f"{utt_id}_fill_mask.wav"
            sf.write(wav_path, audio, 24000)
            semantic_path = None
            if args.save_semantic:
                semantic_path = out_dir / f"{utt_id}_filled_semantic.pt"
                torch.save(
                    {
                        "utt_id": utt_id,
                        "filled_semantic": filled_semantic,
                        "target_mask": target_mask,
                    },
                    semantic_path,
                )

            meta = {
                "utt_id": utt_id,
                "wav_path": str(wav_path),
                "semantic_path": str(semantic_path) if semantic_path else None,
                "mask_frame_count": int(target_mask.sum().item()),
                "semantic_len": int(filled_semantic.numel()),
                "source_wav": item.get("record", {}).get("wav"),
            }
            meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            print(
                f"[{index}/{len(records)}] wrote {wav_path} "
                f"mask_frame_count={meta['mask_frame_count']}",
                flush=True,
            )

    print(f"metadata={meta_path}", flush=True)


if __name__ == "__main__":
    main()



