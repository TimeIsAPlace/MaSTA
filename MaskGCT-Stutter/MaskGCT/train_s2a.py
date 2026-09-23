#!/usr/bin/env python3
"""Distributed MaskGCT S2A training from cached semantic/acoustic tokens."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from random import randint

import safetensors.torch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import MaskGCTAudioTokenCollate, MaskGCTCachedAudioTokenDataset
from maskgct_utils import build_s2a_model
from utils.util import load_config


ROOT = Path(__file__).resolve().parent
NUM_ACOUSTIC_LAYERS = 12


def get_value(config, dotted_key, default=None):
    value = config
    for key in dotted_key.split("."):
        if isinstance(value, dict):
            if key not in value:
                return default
            value = value[key]
        elif hasattr(value, key):
            value = getattr(value, key)
        else:
            return default
    return value


class S2ACollate:
    """Adapt the current audio-token cache batch to the S2A input contract."""

    def __init__(self, stage: str):
        self.stage = stage
        self.base_collate = MaskGCTAudioTokenCollate()

    def __call__(self, items):
        batch = self.base_collate(items)
        layers = 1 if self.stage == "1layer" else NUM_ACOUSTIC_LAYERS
        if batch["acoustic_tokens"].ndim != 3:
            raise ValueError(
                "acoustic_tokens must be [batch, time, num_quantizers], "
                f"got {tuple(batch['acoustic_tokens'].shape)}"
            )
        if batch["acoustic_tokens"].size(2) != NUM_ACOUSTIC_LAYERS:
            raise ValueError(
                f"S2A requires {NUM_ACOUSTIC_LAYERS} cached acoustic layers, "
                f"got {batch['acoustic_tokens'].size(2)}"
            )
        if torch.any(batch["num_quantizers"] != NUM_ACOUSTIC_LAYERS):
            raise ValueError(
                f"Every cached sample must contain {NUM_ACOUSTIC_LAYERS} acoustic quantizers"
            )

        shared_lens = torch.minimum(batch["semantic_lens"], batch["acoustic_lens"])
        if torch.any(shared_lens < 2):
            bad_indices = torch.nonzero(shared_lens < 2, as_tuple=False).flatten().tolist()
            bad_utt_ids = [batch["utt_id"][index] for index in bad_indices]
            raise ValueError(f"S2A sequences are too short: {bad_utt_ids}")
        max_length = int(shared_lens.max().item())
        shared_mask = (
            torch.arange(max_length).unsqueeze(0) < shared_lens.unsqueeze(1)
        )
        return {
            "utt_id": batch["utt_id"],
            "semantic_tokens": batch["semantic_tokens"][:, :max_length],
            "semantic_lens": shared_lens,
            "semantic_mask": shared_mask,
            "acoustic_tokens": batch["acoustic_tokens"][:, :max_length, :layers],
            "acoustic_lens": shared_lens,
            "acoustic_mask": shared_mask,
            "num_quantizers": torch.full(
                (len(items),), layers, dtype=torch.long
            ),
        }


def move_batch(batch, device):
    for key in ("semantic_tokens", "acoustic_tokens", "acoustic_mask"):
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch


def compute_loss(model, batch, criterion):
    logits, mask_layer, final_mask, x0, _, _ = model(
        x0=batch["acoustic_tokens"],
        x_mask=batch["acoustic_mask"],
        cond_code=batch["semantic_tokens"],
    )
    layer = int(mask_layer.item())
    target = x0[:, :, layer]
    token_loss = criterion(logits.transpose(1, 2), target)
    final_mask = final_mask.squeeze(-1).float()
    loss_sum = (token_loss * final_mask).sum()
    mask_count = final_mask.sum()
    return loss_sum / (mask_count + 1e-6), mask_count, layer


@torch.no_grad()
def evaluate(hps, rank, model, loader):
    fp16 = bool(get_value(hps, "train.fp16_run", False))
    device = torch.device(f"cuda:{rank}")
    criterion = nn.CrossEntropyLoss(reduction="none")
    was_training = model.training
    model.eval()
    total_loss = torch.zeros((), device=device)
    total_count = torch.zeros((), device=device)
    for batch in loader:
        batch = move_batch(batch, device)
        with autocast(enabled=fp16):
            logits, mask_layer, final_mask, x0, _, _ = model(
                x0=batch["acoustic_tokens"],
                x_mask=batch["acoustic_mask"],
                cond_code=batch["semantic_tokens"],
            )
            target = x0[:, :, int(mask_layer.item())]
            losses = criterion(logits.transpose(1, 2), target)
            final_mask = final_mask.squeeze(-1).float()
            total_loss += (losses * final_mask).sum()
            total_count += final_mask.sum()
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
    if was_training:
        model.train()
    return float((total_loss / (total_count + 1e-6)).cpu()), float(total_count.cpu())


def clean_state_dict(model):
    return {k.replace("module.", ""): v for k, v in model.state_dict().items()}


def save_model(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(clean_state_dict(model), str(path))


def load_compatible_weights(model, path):
    """Load tensors whose names and shapes match the selected S2A stage."""
    source = safetensors.torch.load_file(str(path), device="cpu")
    target = model.state_dict()
    compatible = {k: v for k, v in source.items() if k in target and target[k].shape == v.shape}
    result = model.load_state_dict(compatible, strict=False)
    print(
        f"initialized {len(compatible)}/{len(target)} tensors from {path}; "
        f"missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}",
        flush=True,
    )


def train_worker(rank, world_size, args, hps):
    dist.init_process_group("nccl", init_method="env://", world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    seed = int(get_value(hps, "train.seed", 1234))
    torch.manual_seed(seed + rank)
    device = torch.device(f"cuda:{rank}")
    cfg = load_config(str(args.maskgct_config))
    model_cfg = cfg.model.s2a_model.s2a_1layer if args.stage == "1layer" else cfg.model.s2a_model.s2a_full
    if args.stage == "full":
        model_cfg.num_quantizer = NUM_ACOUSTIC_LAYERS
        model_cfg.predict_layer_1 = False
    model = build_s2a_model(model_cfg, device)
    if args.resume_checkpoint:
        safetensors.torch.load_model(model, str(args.resume_checkpoint))
    elif args.init_checkpoint:
        load_compatible_weights(model, args.init_checkpoint)

    train_set = MaskGCTCachedAudioTokenDataset(args.train_manifest)
    train_sampler = DistributedSampler(train_set, world_size, rank, shuffle=True)
    batch_size = int(get_value(hps, "train.batch_size", 4))
    workers = int(get_value(hps, "train.num_workers", 4))
    loader_args = dict(
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        collate_fn=S2ACollate(args.stage),
        persistent_workers=workers > 0,
    )
    train_loader = DataLoader(train_set, sampler=train_sampler, **loader_args)
    valid_loader = None
    if args.valid_manifest:
        valid_set = MaskGCTCachedAudioTokenDataset(args.valid_manifest)
        valid_sampler = DistributedSampler(valid_set, world_size, rank, shuffle=False)
        valid_loader = DataLoader(valid_set, sampler=valid_sampler, **loader_args)

    learning_rate = float(get_value(hps, "train.learning_rate", 1e-5))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate,
        betas=tuple(get_value(hps, "train.betas", [0.8, 0.99])),
        eps=float(get_value(hps, "train.eps", 1e-9)),
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(get_value(hps, "train.lr_decay", 0.999875))
    )
    # Only the randomly selected quantizer head participates in each step.
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model.train()

    fp16 = bool(get_value(hps, "train.fp16_run", False))
    accumulation = int(get_value(hps, "train.grad_accumulation_steps", 1))
    log_interval = int(get_value(hps, "train.log_interval", 50))
    save_interval = int(get_value(hps, "train.save_interval", 2000))
    epochs = int(get_value(hps, "train.epochs", 30000))
    output_dir = Path(args.output_dir)
    criterion = nn.CrossEntropyLoss(reduction="none")
    scaler = GradScaler(enabled=fp16)
    global_step = int(args.start_step)

    if rank == 0:
        print(
            f"stage={args.stage} "
            f"acoustic_layers={1 if args.stage == '1layer' else NUM_ACOUSTIC_LAYERS}"
        )
        print(f"train_manifest={args.train_manifest} records={len(train_set)}")
        if args.valid_manifest:
            print(f"valid_manifest={args.valid_manifest}")

    for epoch in range(int(args.start_epoch), epochs + 1):
        train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            with autocast(enabled=fp16):
                loss, mask_count, layer = compute_loss(model, batch, criterion)
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            update = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader)
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if global_step % log_interval == 0 and rank == 0:
                print(
                    f"epoch={epoch} batch={batch_index + 1}/{len(train_loader)} "
                    f"step={global_step} layer={layer} loss={float(loss):.4f} "
                    f"mask_count={float(mask_count):.0f}", flush=True,
                )
            if global_step % save_interval == 0:
                if valid_loader is not None:
                    valid_loss, valid_count = evaluate(hps, rank, model, valid_loader)
                    if rank == 0:
                        print(f"valid_loss={valid_loss:.4f} valid_mask_count={valid_count:.0f}")
                if rank == 0:
                    name = f"S2A_{args.stage}_{epoch}_{global_step}.safetensors"
                    save_model(model, output_dir / name)
                    print(f"saved: {output_dir / name}", flush=True)
            global_step += 1
        scheduler.step()
    if rank == 0:
        save_model(model, output_dir / f"S2A_{args.stage}_final.safetensors")
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="Train MaskGCT semantic-to-acoustic model")
    parser.add_argument("--stage", required=True, choices=("1layer", "full"))
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--valid-manifest")
    parser.add_argument("--config", default=str(ROOT / "config" / "train.json"))
    parser.add_argument("--maskgct-config", type=Path, default=ROOT / "config" / "maskgct.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "ckpt_s2a")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--start-step", type=int, default=0)
    return parser.parse_args()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("S2A training requires CUDA")
    args = parse_args()
    if args.init_checkpoint and args.resume_checkpoint:
        raise ValueError("use only one of --init-checkpoint and --resume-checkpoint")
    hps = load_config(args.config)
    world_size = torch.cuda.device_count()
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(randint(11451, 41919)))
    mp.spawn(train_worker, nprocs=world_size, args=(world_size, args, hps))


if __name__ == "__main__":
    main()
