# 合成与参考音频

本节的两个合成入口都调用 `inference_instruct2`，使用句级模式指令和文本内事件标签。
请使用完成 instruct 微调且词表匹配的完整 checkpoint；这些入口不是单独为
fine-grained-only checkpoint 设计的推理接口。

## AISHELL 风格批量合成

输入、模型和数据格式参见 README。shell 入口接受三个位置参数：

```bash
bash recipes/cosyvoice2/synthesize_aishell.sh \
  MODEL_DIR AUDIO_ROOT OUTPUT_DIR
```

默认模型 `CosyVoice2-0.5B`、音频根 `data`、输出
`data/aishell_stutter_synth_50k`。默认模型名称不代表它已微调；
使用口吃标签时应显式指定匹配的完整微调模型目录。

Python 入口提供更多控制：

```bash
CUDA_VISIBLE_DEVICES=0 python recipes/cosyvoice2/synthesize_aishell.py \
  --model_dir CosyVoice2-0.5B-instruct \
  --input_file aishell_stutter.txt --audio_root data \
  --output_dir data/aishell_stutter_small --num_lines 100 --fp16
```

直接运行 Python 默认 FP32；shell 默认 FP16。FP16 需要 CUDA。
开始大批量任务前，先用一个新输出目录合成小批并人工听检。

## prompt_wav 如何选择

每位说话人在所选记录中按输入顺序寻找第一条与目标文件不同的录音。
若第一条本身是目标，改用该说话人的另一条录音。会排除相同路径、
符号链接/硬链接指向的同一文件；没有不同录音则报错。

这不是按时长或音质自动选优。可以用 `--spk2prompt FILE` 指定经人工检查的参考：

```text
S_DEMO AISHELL-1/wav/train/S_DEMO/reference.wav
```

相对路径基于 audio_root。映射是一项由用户提供的说话人声明，脚本不做声纹验证；
请确认它确实属于该说话人且不是合成目标。不要为所有说话人共用一个参考文件。

## 输出与恢复

- `wavs/aishell_stutter_000001.wav`：按输入顺序命名的合成语音。
- `conversion_preview.jsonl`：前 100 条的文本、源音频、prompt 和输出路径。
- `manifest.jsonl`：全部记录的对应关系，成功完成后生成。
- `wav.scp / text / utt2spk / spk2utt / instruct`：完成后的 Kaldi 风格元数据。

WAV 先写临时文件，再替换为最终文件。相同任务重跑会跳过已有最终 WAV。
**更改输入顺序、模型或指令时请使用新的 output_dir**：现有跳过逻辑不验证参数指纹，
可能复用旧输出。强制重新合成使用 `--overwrite`。旧版或外部程序生成的损坏 WAV
也应删除后重跑或强制覆盖。

断点期间已有 WAV 可保留；最终 manifest 只有完整跑完才写入。
--dry_run 检查文本格式和文件路径，不验证音频解码、模型权重或声纹身份。

## 重合成已有 instruct 训练集

```bash
CUDA_VISIBLE_DEVICES=0 bash recipes/cosyvoice2/synthesize_dataset.sh \
  --model_dir CosyVoice2-0.5B-instruct \
  --data_dir data/instruct_cosyvoice2/train \
  --output_dir data/instruct_cosyvoice2_resynth/train
```

该入口读取 wav.scp、text、utt2spk、instruct，只合成包含 stutter token 的记录。
参考候选来自同一输入目录的同说话人记录，按时长接近 5 秒排序。
发布版同样严格拒绝使用目标自身作参考；缺少同说话人的另一条录音时请补充
--spk2prompt 映射。外部映射中的相对路径基于映射文件所在目录，
这与 AISHELL 入口的 audio_root 规则不同。
