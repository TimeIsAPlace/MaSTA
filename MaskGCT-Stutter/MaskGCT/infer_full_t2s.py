import argparse
import json
from pathlib import Path

import safetensors.torch
import soundfile as sf
import torch

from codec.amphion_codec.codec import CodecDecoder
from maskgct_s2a import MaskGCT_S2A
from maskgct_t2s import MaskGCT_T2S
from utils.util import load_config


MASKGCT_ROOT = Path(__file__).resolve().parent
DEFAULT_CFG = MASKGCT_ROOT / "config" / "maskgct.json"
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


def load_cached_item(manifest_dir, manifest_record):
    sample_path = Path(manifest_record["sample_path"])
    if not sample_path.is_absolute():
        sample_path = manifest_dir / sample_path
    return torch.load(sample_path, map_location="cpu")


def build_t2s_model(cfg, device):
    model = MaskGCT_T2S(cfg=cfg)
    model.eval().to(device)
    return model


def build_s2a_model(cfg, device):
    model = MaskGCT_S2A(cfg=cfg)
    model.eval().to(device)
    return model


def build_acoustic_decoder(cfg, device):
    decoder = CodecDecoder(cfg=cfg.decoder)
    decoder.eval().to(device)
    return decoder


@torch.no_grad()
def semantic_to_audio(
    semantic_code,
    s2a_model_1layer,
    s2a_model_full,
    codec_decoder,
    n_timesteps_s2a,
    cfg_s2a,
    rescale_cfg_s2a,
):
    num_quantizer = s2a_model_full.num_quantizer
    prompt = torch.empty(
        semantic_code.size(0),
        0,
        num_quantizer,
        dtype=torch.long,
        device=semantic_code.device,
    )

    cond = s2a_model_1layer.cond_emb(semantic_code)
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

    vq_emb = codec_decoder.vq2emb(predict_full.permute(2, 0, 1), n_quantizers=num_quantizer)
    audio = codec_decoder(vq_emb)
    return audio[0][0].detach().cpu().numpy()


@torch.no_grad()
def infer_one(
    item,
    t2s_model,
    s2a_model_1layer,
    s2a_model_full,
    codec_decoder,
    device,
    n_timesteps_t2s,
    cfg_t2s,
    rescale_cfg_t2s,
    n_timesteps_s2a,
    cfg_s2a,
    rescale_cfg_s2a,
):
    phone_ids = item["phone_ids"].long().unsqueeze(0).to(device)
    target_len = int(item["semantic_len"])
    prompt_semantic = torch.empty((1, 0), dtype=torch.long, device=device)

    generated_semantic = t2s_model.reverse_diffusion(
        prompt=prompt_semantic,
        target_len=target_len,
        phone_id=phone_ids,
        prompt_mask=torch.empty((1, 0), dtype=torch.bool, device=device),
        n_timesteps=n_timesteps_t2s,
        cfg=cfg_t2s,
        rescale_cfg=rescale_cfg_t2s,
    )

    audio = semantic_to_audio(
        generated_semantic,
        s2a_model_1layer,
        s2a_model_full,
        codec_decoder,
        n_timesteps_s2a=n_timesteps_s2a,
        cfg_s2a=cfg_s2a,
        rescale_cfg_s2a=rescale_cfg_s2a,
    )
    return audio, generated_semantic.detach().cpu().squeeze(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Full-sentence MaskGCT T2S inference on cached training data.")
    parser.add_argument("--manifest", required=True, help="Cached manifest.jsonl from dataset.py prepare-cache.")
    parser.add_argument("--t2s-ckpt", required=True, help="Full-generation T2S checkpoint.")
    parser.add_argument("--out-dir", default="wav/full_t2s_train")
    parser.add_argument("--config", default=str(DEFAULT_CFG))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--utt-id", nargs="*", default=[])
    parser.add_argument("--n-timesteps-t2s", type=int, default=25)
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
    parser.add_argument("--codec-decoder-ckpt", default=str(DEFAULT_CODEC_DECODER_CKPT))
    parser.add_argument("--s2a-1layer-ckpt", default=str(DEFAULT_S2A_1LAYER_CKPT))
    parser.add_argument("--s2a-full-ckpt", default=str(DEFAULT_S2A_FULL_CKPT))
    parser.add_argument("--save-semantic", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)

    t2s_model = build_t2s_model(cfg.model.t2s_model, device)
    codec_decoder = build_acoustic_decoder(cfg.model.acoustic_codec, device)
    s2a_model_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, device)
    s2a_model_full = build_s2a_model(cfg.model.s2a_model.s2a_full, device)

    safetensors.torch.load_model(t2s_model, str(resolve_path(args.t2s_ckpt)))
    safetensors.torch.load_model(codec_decoder, str(resolve_path(args.codec_decoder_ckpt)))
    safetensors.torch.load_model(s2a_model_1layer, str(resolve_path(args.s2a_1layer_ckpt)))
    safetensors.torch.load_model(s2a_model_full, str(resolve_path(args.s2a_full_ckpt)))

    manifest_path = resolve_path(args.manifest)
    manifest_dir = manifest_path.parent
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
            item = load_cached_item(manifest_dir, manifest_record)
            utt_id = str(item.get("utt_id", manifest_record.get("utt_id", index)))
            audio, generated_semantic = infer_one(
                item,
                t2s_model,
                s2a_model_1layer,
                s2a_model_full,
                codec_decoder,
                device,
                n_timesteps_t2s=args.n_timesteps_t2s,
                cfg_t2s=args.cfg_t2s,
                rescale_cfg_t2s=args.rescale_cfg_t2s,
                n_timesteps_s2a=args.n_timesteps_s2a,
                cfg_s2a=args.cfg_s2a,
                rescale_cfg_s2a=args.rescale_cfg_s2a,
            )

            wav_path = out_dir / f"{utt_id}_full_t2s.wav"
            sf.write(wav_path, audio, 24000)
            semantic_path = None
            if args.save_semantic:
                semantic_path = out_dir / f"{utt_id}_generated_semantic.pt"
                torch.save(
                    {
                        "utt_id": utt_id,
                        "generated_semantic": generated_semantic,
                        "target_semantic_len": int(item["semantic_len"]),
                    },
                    semantic_path,
                )

            meta = {
                "utt_id": utt_id,
                "wav_path": str(wav_path),
                "semantic_path": str(semantic_path) if semantic_path else None,
                "semantic_len": int(generated_semantic.numel()),
                "text": item.get("text"),
                "source_wav": item.get("record", {}).get("wav"),
            }
            meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(records)}] wrote {wav_path}", flush=True)

    print(f"metadata={meta_path}", flush=True)


if __name__ == "__main__":
    main()
