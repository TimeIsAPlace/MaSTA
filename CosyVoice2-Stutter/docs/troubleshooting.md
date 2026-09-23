# 常见问题

首先确认运行的是安装依赖的同一环境：

```bash
which python
python -m pip --version
python scripts/check_environment.py
```

| 错误 | 检查/处理 |
| --- | --- |
| No module named cosyvoice.cli | 从本仓库入口运行，确保 PYTHONPATH 包含仓库根目录，避免同名包遮蔽 |
| No module named matcha | PYTHONPATH 还需包含 third_party/Matcha-TTS |
| cannot import cached_download | 按 requirements 安装 diffusers==0.29.0；旧版与新版 Hub 不兼容 |
| No module named pkg_resources | 安装 setuptools==80.9.0，并检查当前 Python 环境 |
| AISHELL-1/AISHELL-1 重复 | audio_root 应为 AISHELL-1 的父目录 |
| speaker ... only one selected record | 增加所选记录，或指定同说话人的另一条参考音频 |
| tokenizer larger than embedding | 模型配置/词表/权重不匹配，见训练文档 |
| CUDA out of memory | 训练降低 max_frames_in_batch；推理可尝试 --fp16 和较短参考音频 |

仓库已有兼容版本要求，不要单独安装 Matcha-TTS 的原始全量 requirements，
也不要在工作环境中无约束升级所有包。环境检查会输出实际模块路径和版本；
检查通过仍不能代替完整模型加载和音频生成测试。
