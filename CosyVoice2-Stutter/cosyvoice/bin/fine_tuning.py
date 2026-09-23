# Research snapshot: distributed/adapted by CosyVoice2-Stutter; see NOTICE for provenance.
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Stutter-control fine-tuning entrypoint.
# CLI arguments intentionally match cosyvoice/bin/train.py.

from __future__ import print_function
import argparse
import datetime
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from copy import deepcopy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import deepspeed

from hyperpyyaml import load_hyperpyyaml

from torch.distributed.elastic.multiprocessing.errors import record

from cosyvoice.utils.losses import DPOLoss
from cosyvoice.utils.executor import Executor
from cosyvoice.utils.train_utils import (
    init_distributed,
    init_dataset_and_dataloader,
    init_optimizer_and_scheduler,
    init_summarywriter, save_model,
    wrap_cuda_model, check_modify_and_save_config)


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--train_engine',
                        default='torch_ddp',
                        choices=['torch_ddp', 'deepspeed'],
                        help='Engine for paralleled training')
    parser.add_argument('--model', required=True, help='model which will be trained')
    parser.add_argument('--ref_model', required=False, help='ref model used in dpo')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--qwen_pretrain_path', required=False, help='qwen pretrain path')
    parser.add_argument('--onnx_path', required=False, help='onnx path, which is required for online feature extraction')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision training')
    parser.add_argument('--dpo',
                        action='store_true',
                        default=False,
                        help='Use Direct Preference Optimization')
    parser.add_argument('--deepspeed.save_states',
                        dest='save_states',
                        default='model_only',
                        choices=['model_only', 'model+optimizer'],
                        help='save model/optimizer states')
    parser.add_argument('--timeout',
                        default=60,
                        type=int,
                        help='timeout (in seconds) of cosyvoice_join.')
    parser.add_argument('--lora_target_modules',
                        default='',
                        type=str,
                        help='Comma/space separated Qwen linear module names to LoRA tune, e.g. q_proj,k_proj,v_proj. Empty disables LoRA.')
    parser.add_argument('--lora_rank',
                        default=8,
                        type=int,
                        help='LoRA rank.')
    parser.add_argument('--lora_alpha',
                        default=16.0,
                        type=float,
                        help='LoRA alpha scaling.')
    parser.add_argument('--lora_dropout',
                        default=0.05,
                        type=float,
                        help='LoRA dropout.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


class LoRALinear(nn.Module):
    def __init__(self, linear, rank=8, alpha=16.0, dropout=0.05):
        super().__init__()
        if rank <= 0:
            raise ValueError('lora rank must be positive')
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.weight = nn.Parameter(linear.weight.detach().clone())
        if linear.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone())
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, dtype=self.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, dtype=self.weight.dtype))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x):
        result = F.linear(x, self.weight, self.bias)
        lora_result = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
        return result + lora_result

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        merged_weight = self.weight + torch.matmul(self.lora_B, self.lora_A) * self.scaling
        destination[prefix + 'weight'] = merged_weight if keep_vars else merged_weight.detach()
        if self.bias is not None:
            destination[prefix + 'bias'] = self.bias if keep_vars else self.bias.detach()


def _parse_lora_targets(target_modules):
    targets = [i.strip() for i in target_modules.replace(',', ' ').split() if i.strip()]
    return sorted(set(targets))


def _get_submodule(root, module_name):
    module = root
    for part in module_name.split('.'):
        module = getattr(module, part)
    return module


def apply_lora(model, target_modules, rank, alpha, dropout):
    targets = _parse_lora_targets(target_modules)
    if not targets:
        return []
    replaced = []
    for name, module in list(model.llm.model.named_modules()):
        if not name:
            continue
        short_name = name.rsplit('.', 1)[-1]
        if short_name not in targets or not isinstance(module, nn.Linear):
            continue
        parent_name, child_name = name.rsplit('.', 1) if '.' in name else ('', name)
        parent = _get_submodule(model.llm.model, parent_name) if parent_name else model.llm.model
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        replaced.append('llm.model.{}'.format(name))
    if not replaced:
        raise ValueError('no Qwen linear modules matched lora_target_modules={}'.format(targets))
    logging.info('LoRA target modules: {}'.format(targets))
    logging.info('LoRA replaced modules: {}'.format(replaced))
    return replaced


def apply_stutter_finetune_freeze(model, configs):
    """Freeze Qwen body and train text embeddings plus small heads.

    The Qwen transformer blocks stay frozen, but the whole input embedding
    matrix remains trainable. This lets added control tokens change how nearby
    normal text tokens are represented during fine-tuning.
    """
    for _, param in model.named_parameters():
        param.requires_grad = False

    trainable_modules = ['speech_embedding', 'llm_embedding', 'llm_decoder']
    for module_name in trainable_modules:
        module = getattr(model, module_name, None)
        if module is None:
            raise ValueError('model has no module {}'.format(module_name))
        for param in module.parameters():
            param.requires_grad = True

    embed_tokens = model.llm.model.model.embed_tokens
    embed_tokens.weight.requires_grad = True

    for name, param in model.named_parameters():
        if '.lora_A' in name or '.lora_B' in name:
            param.requires_grad = True

    trainable, total = 0, 0
    trainable_names = []
    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
            trainable_names.append(name)
    logging.info('trainable parameters: {} / {} ({:.4f}%)'.format(trainable, total, 100.0 * trainable / max(total, 1)))
    logging.info('trainable parameter tensors: {}'.format(trainable_names))


@record
def main():
    args = get_args()
    os.environ['onnx_path'] = args.onnx_path
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    if args.model != 'llm':
        raise ValueError('train_stutter.py only supports --model llm')
    if args.dpo is True:
        raise ValueError('train_stutter.py does not support --dpo')
    gan = False

    override_dict = {k: None for k in ['llm', 'flow', 'hift', 'hifigan'] if k != args.model}
    if args.qwen_pretrain_path is not None:
        override_dict['qwen_pretrain_path'] = args.qwen_pretrain_path
    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)
    configs['train_conf'].update(vars(args))

    # Init env for ddp
    init_distributed(args)

    # Get dataset & dataloader
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
        init_dataset_and_dataloader(args, configs, gan, args.dpo)

    # Do some sanity checks and save config to args.model_dir
    configs = check_modify_and_save_config(args, configs)

    # Tensorboard summary
    writer = init_summarywriter(args)

    # load checkpoint
    model = configs[args.model]
    apply_lora(model,
               args.lora_target_modules,
               rank=args.lora_rank,
               alpha=args.lora_alpha,
               dropout=args.lora_dropout)
    start_step, start_epoch = 0, -1
    if args.checkpoint is not None:
        if os.path.exists(args.checkpoint):
            state_dict = torch.load(args.checkpoint, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            if 'step' in state_dict:
                start_step = state_dict['step']
            if 'epoch' in state_dict:
                start_epoch = state_dict['epoch']
        else:
            logging.warning('checkpoint {} do not exsist!'.format(args.checkpoint))

    apply_stutter_finetune_freeze(model, configs)

    # Dispatch model from cpu to gpu
    model = wrap_cuda_model(args, model)

    # Get optimizer & scheduler
    model, optimizer, scheduler, optimizer_d, scheduler_d = init_optimizer_and_scheduler(args, configs, model, gan)
    scheduler.set_step(start_step)
    if scheduler_d is not None:
        scheduler_d.set_step(start_step)

    # Save init checkpoints
    info_dict = deepcopy(configs['train_conf'])
    info_dict['step'] = start_step
    info_dict['epoch'] = start_epoch
    save_model(model, 'init', info_dict)

    ref_model, dpo_loss = None, None
    if args.dpo is True:
        # Kept unreachable to preserve structural parity with train.py.
        ref_model = deepcopy(configs[args.model])
        state_dict = torch.load(args.ref_model, map_location='cpu')
        ref_model.load_state_dict(state_dict, strict=False)
        dpo_loss = DPOLoss(beta=0.01, label_smoothing=0.0, ipo=False)
        ref_model = wrap_cuda_model(args, ref_model)

    # Get executor
    executor = Executor(gan=gan, ref_model=ref_model, dpo_loss=dpo_loss)
    executor.step = start_step

    # Init scaler, used for pytorch amp mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    print('start step {} start epoch {}'.format(start_step, start_epoch))

    # Start training loop
    for epoch in range(start_epoch + 1, info_dict['max_epoch']):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        dist.barrier()
        group_join = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))
        executor.train_one_epoc(model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, scaler, group_join, ref_model=ref_model)
        dist.destroy_process_group(group_join)


if __name__ == '__main__':
    main()
