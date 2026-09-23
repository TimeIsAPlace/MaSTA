# 训练与模型准备

本项目沿用已有 CosyVoice2 训练实现，重点微调 LLM。完整声学生成仍依赖 Flow
Matching 与 HiFT。不要在缺少训练/数据验证时宣称口吃标签已学会或音色完全保留。

## 两种微调过程

**Fine-grained 指令微调**对应原来的 finetune 流程。其监督文本携带具体位置的
重复、延长、阻塞、音素重复或填充标签，例如 `我<fill>嗯</fill>想去公园。`。
它学习事件类型和发生位置；数据准备不生成独立的 normal/stutter instruct 字段。

**Instruct 微调**对应原来的 instruct_finetune 流程。其输入包括独立的
`<|normal|><|endofprompt|>` 或 `<|stutter|><|endofprompt|>` 模式指令，
训练数据混合正常与口吃语音。口吃样本的目标文本保留事件标签，
因此模式选择与细粒度事件控制是两个层次，不是两个脚本名称不同但功能相同。

推荐工作流为：fine-grained 微调 → 将结果整理为完整模型目录 →
将该目录通过 `--pretrained_model_dir` 传给 instruct 微调。
两个脚本独立运行，未自动选择或接续上一个过程的 checkpoint。
它们内部的 stage 0–5 是各自的数据准备/训练步骤，不等同于这两个微调过程。

## 入口和阶段

| 入口 | 用途 |
| --- | --- |
| `train_fine_grained.sh` | Fine-grained 指令微调：文本内细粒度事件指令 |
| `train_instruct.sh` | Instruct 微调：独立的 normal / stutter 模式指令 |

均位于 `recipes/cosyvoice2/`，支持 `--help`。

| stage | 操作 |
| --- | --- |
| 0 | 准备分段/清单及 train/dev |
| 1 | 提取 speaker embedding |
| 2 | 提取语音 token |
| 3 | 生成 Parquet 和 data.list |
| 4 | 检查 tokenizer 长度与 Qwen embedding 容量 |
| 5 | torchrun 启动 LLM 训练 |

例如完成数据准备后，可用 `--stage 5 --stop_stage 5` 启动训练。
脚本仍会检查输入目录与预训练模型，请保留路径参数一致。

## 学习率与 batch

修改 `configs/cosyvoice2.yaml`：

- `train_conf.optim_conf.lr`：当前配方默认 2e-6。
- `batch.max_frames_in_batch`：当前 2000，动态按帧组 batch，非固定句数。
- `train_conf.accum_grad`：当前 2。
- `train_conf.max_epoch`：当前 200，按数据规模与验证集表现调整。

这些是移植来的配置值，未经此次发布调优。多卡用
`CUDA_VISIBLE_DEVICES=0,1`，并为并行任务指定不同的 `--rdzv_port`。

## Tokenizer 和 checkpoint

本仓库在 CosyVoice2Tokenizer 中注册了事件与模式 token。Stage 4 仅验证词表大小，
并不证明 token ID 的语义与已有 checkpoint 一致。使用来自同一训练流程的
tokenizer、Qwen 配置和权重。若报词表超过 embedding 容量，必须同步扩展 Qwen
模型和训练初始化 checkpoint，不能只跳过检查或删除事件 token。

本发布未提供任意 checkpoint 的自动扩词表/转换工具。
已有可工作的完整模型目录可直接用于推理。

## 训练结果如何用于合成

`exp/.../epoch_*_whole.pt` 是训练保存的 LLM 权重及训练元数据，合成入口需要完整目录。
在新的模型目录保留匹配的 `cosyvoice2.yaml`、`CosyVoice-BlankEN/`、Flow、HiFT 和 ONNX 文件，
再将选定的 LLM state dict 去掉 `epoch`、`step` 元数据后作为 `llm.pt`。
不要把优化器分片或任意字典直接重命名成 llm.pt。

如果使用多个检查点平均，可使用现有工具（需要验证损失 YAML）：

```bash
python cosyvoice/bin/average_model.py \
  --src_path exp/instruct_cosyvoice2/llm/torch_ddp \
  --dst_model CosyVoice2-0.5B-instruct/llm.pt \
  --val_best --num 5
```

目标目录及其他推理资源需事先准备好。此操作会写入目标 llm.pt，
应使用新的模型目录，并选择正确的 --num，确保有足够的检查点。

## 评估限制

当前配方按序号间隔划分验证集，可能包含同一说话人的训练和验证记录。
若做论文或泛化评估，需要自行划分说话人隔离的测试集。
应分别评估内容一致性、细粒度事件实现情况、模式指令遵循程度和说话人相似度。
本仓库没有附带这些指标的实验结果。
