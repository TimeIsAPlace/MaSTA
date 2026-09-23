# CosyVoice2-Stutter

基于 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) 的中文可控口吃语音研究工具箱。

包含两个不同的微调过程：**Fine-grained 指令微调**学习文本中具体位置的口吃事件控制；**Instruct 微调**学习 `<|normal|>` 和 `<|stutter|>` 句级模式控制。两者可组合用于保留原说话人身份的批量合成。此仓库是独立研究衍生版本，不是 FunAudioLLM 官方发行版。

[English](README.en.md) · [数据格式](docs/data-format.md) · [训练](docs/training.md) · [合成](docs/synthesis.md) · [常见问题](docs/troubleshooting.md)

## 功能

- Fine-grained 指令微调：用文本内事件标签控制阻塞、重复、延长和填充的位置与类型。
- Instruct 微调：用独立的模式指令选择正常语音或口吃语音，口吃文本仍可保留细粒度标签。
- 使用完整微调模型目录批量合成。
- AISHELL 风格标注转换；默认处理前 50,000 条有效记录。
- 每条使用同说话人的另一条录音作 prompt；没有可用参考时明确报错。
- 输出 WAV、文本、说话人映射与合成清单，已有输出可跳过。

本仓库仅发布代码和人工编写的格式示例。**不包含训练数据、真实转录、预训练或微调权重，也不提供已验证的性能指标。** 同说话人参考用于尽量保持音色，不能保证生成波形或音色完全一致。

## 安装

主要运行目标为 Linux、Python 3.10、NVIDIA CUDA GPU。Windows 可进行格式预检；训练和完整推理建议使用 Linux。以下命令均在克隆后的仓库根目录执行。

```bash
conda create -n cosyvoice2-stutter python=3.10 -y
conda activate cosyvoice2-stutter
# 系统需已有 ffmpeg；Debian/Ubuntu 可执行：
sudo apt-get install ffmpeg

python -m pip install "setuptools==80.9.0" wheel
python -m pip install "torch==2.3.1" "torchaudio==2.3.1" \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD:$PWD/third_party/Matcha-TTS:${PYTHONPATH:-}"
python scripts/check_environment.py
```

Matcha-TTS 所需源码已随仓库提供，无需 `git submodule update`，也无需另行安装它的完整训练环境。仓库保留了上游 requirements 的主要固定版本，并显式声明 Whisper 等依赖；这不是完整的传递依赖 lockfile。安装和 GPU 推理尚需在目标服务器验证。

## 准备模型

从 [CosyVoice2-0.5B 官方模型页](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) 获取模型，并遵守模型自身的许可。基础模型可用于初始化训练；它不等同于已学会本项目口吃标签的微调模型。

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='CosyVoice2-0.5B')"
```

合成时传入一个已准备好的完整模型目录：

```text
CosyVoice2-0.5B-instruct/
├── cosyvoice2.yaml
├── llm.pt
├── flow.pt
├── hift.pt
├── campplus.onnx
├── speech_tokenizer_v2.onnx
└── CosyVoice-BlankEN/
    └── ...模型配置、分词器及权重文件
```

额外模型文件可原样保留。目录内的词表、配置和权重必须相互匹配。单独的 `epoch_*.pt` 不等于完整推理目录，参见[训练文档](docs/training.md)。

## 快速开始

### 过程一：Fine-grained 指令微调

`train_fine_grained.sh` 对带事件标注的口吃语音进行微调，让模型学习文本内的
`[block]`、`[phone_rep]`、`<rep>…</rep>`、`<prolong>…</prolong>` 和
`<fill>…</fill>` 等细粒度指令。例如 `今<rep>今天</rep>天气很好。`
中的标签指定重复片段；此过程不单独写入 normal/stutter 模式指令。

```bash
CUDA_VISIBLE_DEVICES=0 bash recipes/cosyvoice2/train_fine_grained.sh \
  --annotation_dir data/Annotation --audio_dir data/Audio \
  --pretrained_model_dir CosyVoice2-0.5B
```

### 过程二：Instruct 微调

`train_instruct.sh` 在正常/口吃混合数据上学习独立的句级指令：
`<|normal|><|endofprompt|>` 或 `<|stutter|><|endofprompt|>`。
口吃样本的目标文本仍保留 fine-grained 事件标签，两层控制可以同时使用。

先按[数据格式](docs/data-format.md)准备口吃标注、对应长录音和正常语音清单：

```bash
CUDA_VISIBLE_DEVICES=0 bash recipes/cosyvoice2/train_instruct.sh \
  --annotation_dir data/Annotation \
  --audio_dir data/Audio \
  --aishell3_dir data/AISHELL-3 \
  --aishell3_content data/AISHELL-3/train/content.txt \
  --pretrained_model_dir CosyVoice2-0.5B-fine-grained
```

若按两过程顺序训练，请先将过程一的结果准备成完整的 `CosyVoice2-0.5B-fine-grained/`
目录，再传给过程二。脚本不会自动串联 checkpoint；也可显式选择另一个词表兼容的初始化目录。

学习率、动态 batch 帧数和训练 epoch 在
`configs/cosyvoice2.yaml` 配置。更多阶段说明见[训练文档](docs/training.md)。

### 合成 AISHELL 风格文本

将私有的 `aishell_stutter.txt` 放在仓库根目录，将原音频放在 `data/AISHELL-1/`。每行格式为 `音频相对路径<TAB>带标注文本`。

```bash
CUDA_VISIBLE_DEVICES=0 bash recipes/cosyvoice2/synthesize_aishell.sh \
  CosyVoice2-0.5B-instruct data data/aishell_stutter_synth_50k
```

第二个参数是 **包含 AISHELL-1 的目录**，即 `data`，不是 `data/AISHELL-1`。
默认处理 50,000 条；使用 Python 入口可调整数量、启用 FP16 或仅预检：

```bash
python recipes/cosyvoice2/synthesize_aishell.py \
  --model_dir CosyVoice2-0.5B-instruct \
  --input_file aishell_stutter.txt --audio_root data \
  --output_dir data/aishell_preview --num_lines 100 --dry_run
```

预检不需要 PyTorch 或模型权重，但需要真实存在的输入音频。若所选记录中某位说话人只有一个独立录音，扩大选择范围，或使用 `--spk2prompt` 指定该说话人的另一条录音。

## 仓库结构

```text
cosyvoice/                         模型与训练核心（保留上游命名空间）
recipes/cosyvoice2/                两种微调与两类批量合成入口
configs/                          CosyVoice2 模型与训练配置
recipes/sample_data/               人工编写的数据格式示例
third_party/Matcha-TTS/            Flow Matching 所需源码及许可证
tools/                            声纹、语音 token、Parquet 工具
scripts/                          环境诊断与发布检查
tests/                            无 GPU 的数据与参考音频选择测试
docs/                             数据、训练、推理及发布文档
```

## 开发与验证

```bash
python -m unittest discover -s tests -v
python scripts/check_release.py
```

GitHub Actions 运行 Python 编译、数据逻辑测试、shell 语法及发布内容检查，不执行 GPU 训练或评测。适配代码保留部分上游 CosyVoice1/3 类以满足内部导入，公开使用流程以 CosyVoice2 为限。

## 许可与来源

本仓库代码采用 [Apache-2.0](LICENSE)，保留各文件原有版权声明。Matcha-TTS 及其子组件遵循各自的 [MIT 许可证](third_party/Matcha-TTS/LICENSE)。来源、修改说明和快照限制见 [NOTICE](NOTICE) 与 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

数据集、音频、模型权重的许可与代码许可分别适用。使用已获授权的说话人录音；发布语音样例时标明合成来源。本项目用于语音研究，不用于诊断或评定说话人的健康状况。

参见 [贡献指南](CONTRIBUTING.md) 和 [GitHub 发布说明](docs/publishing.md)。
