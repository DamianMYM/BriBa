# BriBa

**English | [中文](README-CN.md)**

BriBa is a local-first English listening study workbench. It helps learners turn an English video clip into a reusable study library with audio, English subtitles, optional Chinese translations, and editable study notes.

## Highlights

- Clip a video by absolute start/end time.
- Generate English subtitles with Whisper-compatible ASR.
- Review and correct subtitle lines manually.
- Generate optional Chinese line-by-line translations with local Ollama.
- Study with audio, sentence replay, subtitle delay adjustment, and editable notes.
- Keep all media processing and AI calls on your own machine.

## Disclaimer

BriBa is for personal English learning, listening practice, classroom demonstration, and non-commercial research only.

This project does not provide, host, sell, or distribute any videos, subtitles, TV shows, movies, or copyrighted media. Users are responsible for ensuring they have the legal right to process any media they import.

## Requirements

- Python 3.10+
- FFmpeg, usually provided automatically through `imageio-ffmpeg`
- Ollama for local AI features
- A GPU is recommended for Whisper `large-v3`

## Install

```powershell
git clone https://github.com/DamianMYM/BriBa.git
cd BriBa

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Ollama

Install Ollama from:

```text
https://ollama.com/
```

Check the local service:

```powershell
ollama list
```

Start it manually if needed:

```powershell
ollama serve
```

BriBa uses Ollama's default local API:

```text
http://127.0.0.1:11434
```

The default model field is:

```text
qwen3.5:27b
```

You can replace it with any model shown by `ollama list`.

## Run

```powershell
python app_gradio.py
```

Open:

```text
http://127.0.0.1:7860/
```

Pages:

- `/` - study home
- `/new/` - create material
- `/review/` - correct subtitles
- `/library/` - manage local libraries

## Basic Workflow

1. Open `/new/`.
2. Upload a video or paste a supported download URL.
3. Enter the start and end time.
4. Choose a Whisper model.
5. Create the study library.
6. Study on `/`.
7. Optionally generate Chinese translations.
8. Correct subtitles in `/review/` if needed.

## License

Copyright is retained by the author.

The source code is visible for learning, review, and personal non-commercial use. Modification, redistribution, sublicensing, commercial use, or publishing derivative versions requires prior written permission from the author.

See [LICENSE](LICENSE).
