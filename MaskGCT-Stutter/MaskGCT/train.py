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

from dataset import MaskGCTCachedAudioTokenDataset, MaskGCTAudioTokenCollate
from maskgct_utils import build_t2s_model
from utils.util import load_config


CFG_PATH = "./config/maskgct.json"
T2S_MODEL_CKPT = "./MaskGCT_model/t2s_model/model.safetensors"


def get_config_value(config, dotted_key, default=None):
    value = config
    for key in dotted_key.split("."):
        if isinstance(value, dict):
            if key not in value:
                return default
            value = value[key]
        else:
            if not hasattr(value, key):
                return default
            value = getattr(value, key)
    return value


def load_t2s_checkpoint(t2s_model, hps, save_model_path):
    keep_training_epochs = int(get_config_value(hps, "train.keep_training_epochs", 0))
    keep_training_steps = int(get_config_value(hps, "train.keep_training_steps", 0))

    if keep_training_epochs > 0:
        ckpt_path = Path(save_model_path) / f"T2S_{keep_training_epochs}_{keep_training_steps}.safetensors"
        safetensors.torch.load_model(t2s_model, str(ckpt_path))
        return keep_training_epochs, keep_training_steps

    if keep_training_epochs == -1:
        checkpoint = get_config_value(hps, "train.init_checkpoint", T2S_MODEL_CKPT)
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Initial checkpoint not found: {checkpoint}")
        safetensors.torch.load_model(t2s_model, str(checkpoint))
        return 0, keep_training_steps

    return keep_training_epochs, keep_training_steps


def save_t2s_checkpoint(t2s_model, save_path):
    state_dict = t2s_model.state_dict()
    clean_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    safetensors.torch.save_file(clean_state_dict, str(save_path))


def move_batch_to_device(batch, device):
    tensor_keys = [
        "phone_ids",
        "phone_mask",
        "semantic_tokens",
        "semantic_mask",
    ]
    for key in tensor_keys:
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch


def compute_t2s_loss(t2s_model, batch, criterion):
    logits, final_mask, x0, prompt_len, mask_prob = t2s_model(
        x0=batch["semantic_tokens"],
        x_mask=batch["semantic_mask"],
        phone_id=batch["phone_ids"],
        phone_mask=batch["phone_mask"],
    )
    feature_loss = criterion(logits.transpose(1, 2), x0)
    final_mask = final_mask.squeeze(-1).float()
    masked_loss = feature_loss * final_mask
    loss = masked_loss.sum() / (final_mask.sum() + 1e-6)
    return loss, final_mask


@torch.no_grad()
def evaluate_t2s(hps, rank, t2s_model, valid_loader):
    fp16_run = bool(get_config_value(hps, "train.fp16_run", False))
    device = torch.device(f"cuda:{rank}")
    criterion = nn.CrossEntropyLoss(reduction="none")
    was_training = t2s_model.training
    t2s_model.eval()

    total_loss = torch.zeros((), device=device)
    total_mask = torch.zeros((), device=device)
    for batch in valid_loader:
        batch = move_batch_to_device(batch, device)
        with autocast(enabled=fp16_run):
            logits, final_mask, x0, prompt_len, mask_prob = t2s_model(
                x0=batch["semantic_tokens"],
                x_mask=batch["semantic_mask"],
                phone_id=batch["phone_ids"],
                phone_mask=batch["phone_mask"],
            )
            feature_loss = criterion(logits.transpose(1, 2), x0)
            final_mask = final_mask.squeeze(-1).float()
            total_loss += (feature_loss * final_mask).sum()
            total_mask += final_mask.sum()

    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_mask, op=dist.ReduceOp.SUM)
    valid_loss = total_loss / (total_mask + 1e-6)

    if was_training:
        t2s_model.train()
    return float(valid_loss.detach().cpu()), float(total_mask.detach().cpu())


def train_epoch(
    hps,
    rank,
    epoch,
    t2s_model,
    optimizer,
    train_loader,
    sampler,
    global_step,
    valid_loader=None,
):
    fp16_run = bool(get_config_value(hps, "train.fp16_run", False))
    accumulation_steps = int(get_config_value(hps, "train.grad_accumulation_steps", 1))
    print_interval = int(get_config_value(hps, "train.log_interval", 50))
    save_interval = int(get_config_value(hps, "train.save_interval", 2000))
    save_model_path = Path(get_config_value(hps, "train.save_model_path", "ckpt"))

    device = torch.device(f"cuda:{rank}")
    criterion = nn.CrossEntropyLoss(reduction="none")
    scaler = GradScaler(enabled=fp16_run)

    sampler.set_epoch(epoch)
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(train_loader):
        batch = move_batch_to_device(batch, device)

        with autocast(enabled=fp16_run):
            final_loss, final_mask = compute_t2s_loss(t2s_model, batch, criterion)

        acc_loss = final_loss / accumulation_steps
        scaler.scale(acc_loss).backward()

        if (global_step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if global_step % print_interval == 0 and rank == 0:
            mask_count = float(final_mask.sum().detach().cpu())
            print(
                f"Epoch: {epoch} "
                f"[{100.0 * batch_idx / max(len(train_loader), 1):.0f}%] "
                f"Step: {global_step} "
                f"feature_loss:{float(final_loss.detach().cpu()):.4f} "
                f"mask_count:{mask_count:.0f}",
                flush=True,
            )

        if global_step % save_interval == 0:
            if valid_loader is not None:
                valid_loss, valid_mask_count = evaluate_t2s(
                    hps, rank, t2s_model, valid_loader
                )
                if rank == 0:
                    print(
                        f"Epoch: {epoch} Step: {global_step} "
                        f"valid_loss:{valid_loss:.4f} "
                        f"valid_random_mask_count:{valid_mask_count:.0f}",
                        flush=True,
                    )
            if rank == 0:
                save_model_path.mkdir(parents=True, exist_ok=True)
                ckpt_path = save_model_path / f"T2S_{epoch}_{global_step}.safetensors"
                print(f"saving: {ckpt_path} ...", flush=True)
                save_t2s_checkpoint(t2s_model, ckpt_path)
                print("checkpoint saved.", flush=True)
        
        global_step += 1

    return global_step


def train_t2s(rank, n_gpus, hps, train_manifest, valid_manifest):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=n_gpus,
        rank=rank,
    )
    torch.manual_seed(int(get_config_value(hps, "train.seed", 1234)))
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    device = f"cuda:{rank}"
    batch_size = int(get_config_value(hps, "train.batch_size", 16))
    epochs = int(get_config_value(hps, "train.epochs", 30000))
    learning_rate = float(get_config_value(hps, "train.learning_rate", 1e-5))
    betas = tuple(get_config_value(hps, "train.betas", [0.8, 0.99]))
    eps = float(get_config_value(hps, "train.eps", 1e-9))
    lr_decay = float(get_config_value(hps, "train.lr_decay", 0.999875))
    num_workers = int(get_config_value(hps, "train.num_workers", 4))
    save_model_path = Path(get_config_value(hps, "train.save_model_path", "ckpt"))

    cfg = load_config(CFG_PATH)
    t2s_model = build_t2s_model(cfg.model.t2s_model, device)
    epoch_begin, global_step = load_t2s_checkpoint(t2s_model, hps, save_model_path)

    train_dataset = MaskGCTCachedAudioTokenDataset(train_manifest)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
        drop_last=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=MaskGCTAudioTokenCollate(),
        persistent_workers=num_workers > 0,
    )
    valid_loader = None
    if valid_manifest:
        valid_dataset = MaskGCTCachedAudioTokenDataset(valid_manifest)
        valid_sampler = DistributedSampler(
            valid_dataset,
            num_replicas=n_gpus,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            sampler=valid_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=MaskGCTAudioTokenCollate(),
            persistent_workers=num_workers > 0,
        )

    optimizer = torch.optim.AdamW(
        t2s_model.parameters(),
        learning_rate,
        betas=betas,
        eps=eps,
    )
    for param_group in optimizer.param_groups:
        param_group.setdefault("initial_lr", learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=lr_decay,
        last_epoch=epoch_begin - 1,
    )

    t2s_model = DDP(t2s_model, device_ids=[rank], find_unused_parameters=False)
    t2s_model.train()

    if rank == 0:
        print(f"train_manifest={train_manifest}", flush=True)
        if valid_manifest:
            print(f"valid_manifest={valid_manifest}", flush=True)
        print(f"dataset_size={len(train_dataset)} epoch_begin={epoch_begin} global_step={global_step}", flush=True)

    for epoch in range(epoch_begin, epochs + 1):
        global_step = train_epoch(
            hps,
            rank,
            epoch,
            t2s_model,
            optimizer,
            train_loader,
            train_sampler,
            global_step,
            valid_loader=valid_loader,
        )
        scheduler.step()

    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="Train MaskGCT full-generation T2S with cached data.")
    parser.add_argument("--config", default=os.path.join("config", "train.json"))
    parser.add_argument(
        "--train-manifest",
        default=None,
        help="Path to cached manifest.jsonl. If omitted, uses data.train_manifest from config.",
    )
    parser.add_argument(
        "--valid-manifest",
        default=None,
        help="Path to cached dev manifest.jsonl. If omitted, uses data.valid_manifest from config.",
    )
    return parser.parse_args()


def main():
    assert torch.cuda.is_available(), "CPU training is not allowed."
    args = parse_args()
    hps = load_config(args.config)
    train_manifest = args.train_manifest or get_config_value(hps, "data.train_manifest")
    valid_manifest = args.valid_manifest or get_config_value(hps, "data.valid_manifest")
    if not train_manifest and Path("../data/token_cache/train/manifest.jsonl").exists():
        train_manifest = "../data/token_cache/train/manifest.jsonl"
    if not valid_manifest and Path("../data/token_cache/dev/manifest.jsonl").exists():
        valid_manifest = "../data/token_cache/dev/manifest.jsonl"
    if not train_manifest:
        raise ValueError("Please set --train-manifest or data.train_manifest in the train config.")

    n_gpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", str(randint(11451, 41919)))

    mp.spawn(
        train_t2s,
        nprocs=n_gpus,
        args=(n_gpus, hps, train_manifest, valid_manifest),
    )


if __name__ == "__main__":
    main()
