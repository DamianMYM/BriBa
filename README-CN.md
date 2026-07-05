# BriBa

**[English](README.md) | 中文**

BriBa 是一个本地优先的英语精听学习工作台。它可以把一段英语视频处理成可反复学习的材料：音频、英文字幕、可选中文翻译和可编辑学习笔记。

## 项目特色

- 按视频绝对起止时间截取学习片段。
- 使用 Whisper 兼容模型生成英文字幕。
- 支持人工逐句校对字幕。
- 使用本地 Ollama 生成逐句中文翻译，可显示或隐藏。
- 学习页支持音频播放、单句重听、字幕延后微调和学习笔记。
- 所有媒体处理和 AI 调用都在本机完成。

## 使用声明

BriBa 仅用于个人英语学习、精听练习、课堂演示和非商业研究。

本项目不提供、不托管、不分发任何影视资源、字幕资源或受版权保护的媒体内容。用户应只处理自己拥有合法使用权的素材，并自行承担导入、处理和导出内容的版权责任。

## 环境要求

- Python 3.10+
- FFmpeg，通常会通过 `imageio-ffmpeg` 自动提供
- Ollama，用于本地 AI 助手、学习笔记和中文翻译
- 如果使用 Whisper `large-v3`，建议使用 GPU

## 安装

```powershell
git clone https://github.com/DamianMYM/BriBa.git
cd BriBa

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 配置 Ollama

安装 Ollama：

```text
https://ollama.com/
```

检查本地服务：

```powershell
ollama list
```

需要时手动启动：

```powershell
ollama serve
```

BriBa 使用 Ollama 默认本地接口：

```text
http://127.0.0.1:11434
```

默认模型填写为：

```text
qwen3.5:27b
```

可以换成 `ollama list` 中已有的任意模型。

## 启动

```powershell
python app_gradio.py
```

打开：

```text
http://127.0.0.1:7860/
```

页面说明：

- `/`：学习首页
- `/new/`：新建材料
- `/review/`：校对字幕
- `/library/`：学习库

## 基本流程

1. 打开 `/new/`。
2. 上传视频或输入支持的下载链接。
3. 输入片段开始和结束时间。
4. 选择 Whisper 模型。
5. 生成学习库。
6. 回到 `/` 开始学习。
7. 需要中文辅助时，生成并显示中文翻译。
8. 字幕不准时，到 `/review/` 校对。

## 许可

作者保留软件著作权。

源码仅供学习、审阅和个人非商业使用。未经作者书面许可，不得修改、再发布、二次授权、商用，或发布衍生版本。

详见 [LICENSE](LICENSE)。
