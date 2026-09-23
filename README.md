# MaSTA · 可控口吃语音合成

MaSTA 汇集两个中文口吃语音合成研究实现：基于 CosyVoice2 的指令微调，以及基于 MaskGCT 的语义 token 区域填充。项目提供训练、数据准备、合成入口和音频对照演示。

[English](README.en.md) · [在线 Demo](https://timeisaplace.github.io/MaSTA/) · [本地预览说明](demo_page/README.md) · [发布指南](docs/PUBLISHING.md)

## 两个模型项目

| 项目 | 方法与输入 | 使用说明 |
| --- | --- | --- |
| CosyVoice2-Stutter | Fine-grained 微调学习文本内事件标签；Instruct 微调学习 normal/stutter 句级模式。使用完整微调模型和参考语音合成。 | [中文](CosyVoice2-Stutter/README.md) / [English](CosyVoice2-Stutter/README.en.md) |
| MaskGCT-Stutter | 三阶段 T2S 微调：标签条件适配、正常语音随机区域填充、口吃事件区域填充。推理依赖语义 token 缓存和 mask。 | [中文](MaskGCT-Stutter/README_zh.md) / [English](MaskGCT-Stutter/README.md) |

这两个实现分别配置环境，不能共享一套依赖安装：CosyVoice2 使用 PyTorch 2.3.1 / Transformers 4.51.3，MaskGCT 使用 PyTorch 2.0.1 / Transformers 4.41.1。训练和完整推理面向 Linux、Python 3.10 与 NVIDIA CUDA GPU。按各自 README 安装并在相应子目录执行模型命令。

代码不附带训练语料、模型权重或微调 checkpoint。仅下载官方基础模型不会获得本项目的口吃控制能力；请按模型文档准备兼容的微调结果。CPU 检查不代表 GPU 训练、生成质量或论文结果已经复现。

## 音频 Demo

[在线收听 Demo](https://timeisaplace.github.io/MaSTA/)：AS70、AISHELL-1 共 20 组对照，每组包含参考语音、CosyVoice2 和 MaskGCT 合成语音。AISHELL-1 的参考是原始流利语音。可直接在浏览器中收听，无需安装模型。

如需本地预览，在仓库根目录执行以下命令，然后打开 <http://localhost:8000/demo_page/>：

```bash
python -m http.server 8000 --bind 127.0.0.1
```

## 目录

```text
CosyVoice2-Stutter/    CosyVoice2 模型、微调及合成流程
MaskGCT-Stutter/       MaskGCT 三阶段训练及区域填充流程
demo_page/            静态演示页、元数据和音频样本
scripts/              整仓库检查入口
docs/                 发布说明和验证记录
.github/              CI、Pages、Issue/PR 模板与依赖更新配置
```

## 检查与贡献

无需下载模型即可运行：

```bash
python scripts/check_repository.py
```

该入口检查源码与本地链接、示例格式、Demo 音频和元数据，并执行两个模型的轻量测试；可用时还会运行 Node.js 语法检查和 Bash 语法检查。CI 在 Linux/Windows、Python 3.10/3.11 运行。结果与限制见[验证记录](docs/VALIDATION.md)。

欢迎通过 Issue 提供最小复现，或按[贡献指南](CONTRIBUTING.md)提交修改。安全问题见 [SECURITY.md](SECURITY.md)。

## 许可、引用与使用范围

各部分保留独立许可，详见 [LICENSE.md](LICENSE.md) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。演示音频的来源与待确认授权事项见 [音频说明](demo_page/AUDIO_SOURCES.md)；代码许可证不替音频、数据或权重授予许可。

本项目用于语音研究；生成内容应注明合成来源，使用获得授权的参考录音，不用于冒充说话人或临床诊断。引用项目请见 [CITATION.cff](CITATION.cff)。目前未填写尚未提供的论文、作者名单或 DOI。
