# 上传到 GitHub

当前目录是独立发布目录，不依赖原研究工程的上级路径。
推荐仓库名称 `CosyVoice2-Stutter`，描述：

> Controllable Chinese stutter speech synthesis with CosyVoice2: fine-grained instruction tuning, instruct tuning and speaker-preserving synthesis.

## 上传前验证

```bash
python scripts/check_release.py
python -m unittest discover -s tests -v
```

检查仓库中未包含权重、真实音频、标注和凭据。默认 .gitignore 排除了常见产物，
但不阻止 git add -f。模型和数据如需单独发布，应另行确认对应权利和许可。

## Git 推送（推荐）

在 GitHub 新建空仓库，不预先生成 README 或许可证，然后在此目录执行：

```bash
git init -b main
git add .
git diff --cached --stat
git commit -m "Initial release: CosyVoice2 stutter research toolkit"
git remote add origin https://github.com/YOUR_ACCOUNT/CosyVoice2-Stutter.git
git push -u origin main
```

将 YOUR_ACCOUNT 替换为你的 GitHub 账号。发布目录没有预置账号、远程地址或访问令牌。
发布准备过程没有替你创建 GitHub 仓库、提交或推送。

也可用 GitHub 网页上传目录内容，但文件数量较多时建议使用 Git。
若使用交付的 ZIP，应先解压再提交源码，而不是把整个 ZIP 当成唯一仓库文件。
确认 .github、.gitignore、.gitattributes 等隐藏文件也上传。

## 发布页说明

建议首个版本先标为研究代码发布，并说明：

- 模型、真实数据和音频不在代码仓库中。
- 当前没有公开的性能指标或音频质量保证。
- CPU 检查/CI 与 GPU 端到端验证的范围不同。
- 该工程来源于本地研究快照，无法提供精确上游 commit。

如果以后公开微调模型，另加 model card，记录训练集来源、许可、数据切分、
基座模型、tokenizer 修改、训练参数、评测结果和使用限制。
