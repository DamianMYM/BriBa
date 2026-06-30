# Subtitle Fusion Workbench

This project now includes both a CLI pipeline and a simple drag-and-drop Gradio UI.

Current capabilities:

- Probe a video with `ffprobe`
- Detect embedded subtitle streams
- Extract a selected subtitle stream with `ffmpeg`
- Transcribe speech with Whisper-compatible ASR
- OCR the bottom subtitle band with Apple Vision
- Fuse ASR and OCR into a reviewable transcript prototype
- Export ASR, OCR, and fused results as `.srt`, `.txt`, and `.json`

## Environment check

You mentioned the `llava` conda environment. I checked the environment at `/Users/damianma/anaconda/anaconda3/envs/llava` and found:

- `gradio`: installed
- `tkinter`: installed
- `ffmpeg`: not installed
- `ffprobe`: not installed
- `whisper`: not installed
- `faster-whisper`: not installed

## Files

- `subtitle_pipeline.py`: CLI and reusable processing logic
- `app_gradio.py`: local drag-and-drop UI
- `vision_ocr.swift`: Apple Vision OCR helper used by the v2 prototype

## Requirements

- Python 3.11+
- `ffmpeg`
- `ffprobe`
- One ASR backend:
  - `faster-whisper`
  - or `openai-whisper`
- macOS with Apple Vision available through `swift`

## CLI usage

Auto mode:

```bash
python3 subtitle_pipeline.py /path/to/video.mp4 --output-dir ./out
```

Probe only:

```bash
python3 subtitle_pipeline.py /path/to/video.mp4 --probe-only
```

Force Whisper ASR:

```bash
PATH=/usr/local/bin:$PATH /Users/damianma/anaconda/anaconda3/envs/llava/bin/python subtitle_pipeline.py /path/to/video.mp4 --strategy asr --model base --language en
```

Run the v2 fusion prototype:

```bash
PATH=/usr/local/bin:$PATH /Users/damianma/anaconda/anaconda3/envs/llava/bin/python subtitle_pipeline.py /path/to/video.mp4 --output-dir ./out --strategy fusion --model base --language en --ocr-sample-fps 1.0
```

For your current machine, this exact command shape is the most reliable:

```bash
PATH=/usr/local/bin:$PATH /Users/damianma/anaconda/anaconda3/envs/llava/bin/python /Users/damianma/Documents/LanguageMaterials/subtitle_pipeline.py /Users/damianma/Downloads/SH_S1E1.mp4 --output-dir /Users/damianma/Documents/LanguageMaterials/output --strategy asr --model base --language en
```

The pipeline now sets `KMP_DUPLICATE_LIB_OK=TRUE` automatically before loading Whisper backends, which avoids the OpenMP duplicate-runtime crash we hit during setup.

## UI usage

Run the local UI:

```bash
PATH=/usr/local/bin:$PATH /Users/damianma/anaconda/anaconda3/envs/llava/bin/python app_gradio.py
```

Then open the local Gradio URL in your browser and drag a video file into the upload area.

## Recommended install path for `llava`

If you want this to run inside your usual `llava` environment, install the missing dependencies there:

```bash
/Users/damianma/anaconda/anaconda3/envs/llava/bin/pip install faster-whisper
```

For media tools, install `ffmpeg` and `ffprobe` through your package manager or conda-forge.

## Notes

- `auto` mode prefers embedded subtitle extraction when subtitle streams exist.
- If no subtitle stream is present, `auto` falls back to ASR.
- `ocr` now means bottom-region subtitle OCR using Apple Vision.
- `fusion` writes:
  - `*.asr.*`
  - `*.ocr.*`
  - `*.fused.*`
  - `*.review.json`
