# 发布前验证记录

验证对象：独立的 CosyVoice2-Stutter 源码目录（2026-09-23 整理）。

| 项目 | 结果 |
| --- | --- |
| Python 源码 AST 语法检查 | 通过 |
| shell 内嵌 Python 语法及本地脚本/配置引用 | 通过 |
| 4 个 shell 入口 bash -n | Git Bash 检查通过 |
| 标签转换、同说话人参考排除与 dry-run | 11 项 unittest 通过 |
| JSON / JSONL 与 LF 行尾检查 | 通过 |
| 大文件、权重、音频、常见凭据模式与个人绝对路径扫描 | 未发现命中；不是完整安全审计 |
| GitHub Actions | 已配置 Python 3.10 / 3.11 工作流，尚未在 GitHub 执行 |
| 全量依赖安装 / GPU 训练 / GPU 合成 | 本次未执行 |

本机是 Windows；数据测试不加载模型，不解码音频。完整模型、真实语料及 GPU
依赖需在使用者的服务器准备。发布目录通过检查不等于已经验证任意 checkpoint
的兼容性、音色保持或事件实现质量。

2026-09-23 整仓库整理后重新检查：发布检查与 11 项数据测试通过，
所有保留的 shell 脚本均通过 Git Bash `bash -n`。原先展开成路径文本的
示例链接已恢复为普通文件；上游示例中的个人路径已移除。未执行模型入口。

WSL 的本地磁盘镜像缺失，因此 shell 语法检查使用 Git Bash。
数据测试在本机默认 Python 3.7.5 下也曾通过，但不把它作为模型运行支持版本。

可在自己的 Linux 环境再次执行：

```bash
python -m unittest discover -s tests -v
python scripts/check_release.py
python scripts/check_environment.py --model_dir /path/to/complete-checkpoint
```

最后一个命令会实际加载用户指定的完整模型；不自动生成音频。
