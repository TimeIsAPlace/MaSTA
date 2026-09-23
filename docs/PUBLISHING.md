# 发布到 GitHub

整个项目根目录作为一个 GitHub 仓库，两个模型保留自己的环境和许可证。
根 `.github/workflows/` 执行整仓库检查；子目录中的 workflow 仅供单独发布子项目时使用。

## 首次上传

1. 阅读 `demo_page/AUDIO_SOURCES.md`，补充当前音频的授权和实验来源；未获准公开的音频应在公开仓库前移除，并重新生成 Demo 元数据。
2. 执行 `python scripts/check_repository.py`。GPU 验证另行进行，不能用 CPU 检查代替。
3. 在 GitHub 新建空仓库，不自动生成 README 或许可证。
4. 在本项目根目录执行下列命令，替换远程地址中的账户和仓库名称：

```bash
git init -b main
git add .
git diff --cached --stat
git status --short
git commit -m "Prepare MaSTA research release"
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

本地已初始化 `main` 分支，尚未提交或设置远程；无需再次初始化。
现有 Git 仓库不需要重复添加 origin。上传源码目录内容，而非仅上传 ZIP。
提交前检查暂存清单：仅 Demo 白名单目录可含 WAV；模型、缓存、私人语料和凭据应被忽略。

## Demo 在线链接

在仓库 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。
进入 **Actions → Deploy demo → Run workflow**，选择默认分支。
部署成功后工作流显示实际网址，通常为 `https://账户.github.io/仓库名/`。
将该网址添加到仓库 About 的 Website，并可替换根 README 中的 Demo 导航为在线链接。
在部署成功前，README 的相对链接通向可用的本地预览说明，不伪造在线地址。

工作流手动触发，只上传 `demo_page/`，不会发布模型目录。配置依据
[GitHub Pages 官方文档](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)。

## 仓库设置与版本发布

- 填写 Description、Topics、Website 和真实维护者联系方式。
- 开启私人漏洞报告，验证 Security 页的报告入口。
- CI 首次通过后，为默认分支配置规则，要求 PR 与 Repository checks 通过。
- 将真实作者、仓库 URL、论文/DOI 和正式版本补入 `CITATION.cff`，不要编造尚未公开的信息。
- 正式发版时维护 `CHANGELOG.md`，创建版本 tag 和 Release，说明权重是否提供及 GPU 验证范围。
- 若发布权重，另附 model card，记录数据、授权、基座 revision、训练预算、评测与使用限制。

这些设置需在新建远程仓库后完成。本地整理不会创建远程仓库、推送或部署。
