# Listening demo

[在线收听](https://timeisaplace.github.io/MaSTA/) · [返回项目首页](../README.md) · [页面源码](index.html) · [音频来源](AUDIO_SOURCES.md)

静态页面展示 20 组、60 条音频，比较 AS70 / AISHELL-1 参考录音与
CosyVoice2 / MaskGCT 合成结果。AISHELL-1 参考列为流利语音，AS70 参考列为口吃语音。

从仓库根目录运行 `python -m http.server 8000 --bind 127.0.0.1`，
打开 <http://localhost:8000/demo_page/>。浏览页面不需要安装模型或 Python 音频包。

音频或文本变更后，安装 `python -m pip install -r demo_page/requirements.txt`，
运行 `python demo_page/build_demo.py` 重新生成 `data.js`，再运行
`python scripts/check_repository.py`。`data.js` 随源码提交，部署时直接使用。

在线页面已发布至 <https://timeisaplace.github.io/MaSTA/>。
修改页面或音频并推送后，在 [Actions](https://github.com/TimeIsAPlace/MaSTA/actions/workflows/pages.yml)
中手动运行 `Deploy demo` 更新网站，操作见[发布说明](../docs/PUBLISHING.md)。
Pages 只发布这个目录；仅修改根目录 README 无需重新部署。
