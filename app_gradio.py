#!/usr/bin/env python3

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import gradio as gr

from subtitle_pipeline import PipelineError, process_video


APP_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = APP_DIR / "output"


def run_pipeline(
    uploaded_file,
    strategy: str,
    model_name: str,
    language: str,
    ocr_sample_fps: float,
) -> tuple[str, list[str]]:
    if not uploaded_file:
        return "请先拖入或选择一个视频文件。", []

    # Gradio 不同版本对 gr.File 的返回类型不一致：
    # - 可能是 str 文件路径
    # - 可能是 dict（例如 {"path": "..."}）
    # - 可能是带有 .path 属性的数据类对象
    file_path = None
    if isinstance(uploaded_file, str):
        file_path = uploaded_file
    elif isinstance(uploaded_file, dict):
        file_path = uploaded_file.get("path") or uploaded_file.get("name")
    else:
        file_path = getattr(uploaded_file, "path", None) or getattr(uploaded_file, "name", None)

    if not file_path:
        return "上传文件解析失败：未能获取文件路径。", []

    source_path = Path(str(file_path)).expanduser().resolve()
    session_dir = Path(tempfile.mkdtemp(prefix="subtitle-ui-", dir=str(DEFAULT_OUTPUT_DIR)))
    copied_path = session_dir / source_path.name
    shutil.copy2(source_path, copied_path)

    try:
        result = process_video(
            video_path=copied_path,
            output_dir=session_dir,
            strategy=strategy,
            model_name=model_name,
            language=language,
            ocr_sample_fps=ocr_sample_fps,
        )
    except PipelineError as exc:
        return f"处理失败：{exc}", []

    lines = [f"处理策略：{result.strategy_used}", result.summary]
    if result.probe_result is not None:
        lines.append("")
        lines.append("媒体信息：")
        lines.append(result.probe_result.path.name)
        lines.append(f"检测到字幕流数量：{len(result.probe_result.subtitle_streams)}")

    file_paths = [str(path) for path in result.created_files]
    if file_paths:
        lines.append("")
        lines.append("生成文件：")
        lines.extend(file_paths)

    return "\n".join(lines), file_paths


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Subtitle Extractor") as demo:
        gr.Markdown(
            """
            # Subtitle Extractor
            拖入一个视频文件。第二版原型支持 ASR、底部字幕 OCR，以及 ASR/OCR 融合校对。
            """
        )
        with gr.Row():
            file_input = gr.File(
                label="拖拽上传视频",
                file_count="single",
                type="file",
            )
            with gr.Column():
                strategy = gr.Dropdown(
                    choices=["auto", "embedded", "asr", "ocr", "fusion"],
                    value="fusion",
                    label="提取策略",
                )
                model_name = gr.Dropdown(
                    choices=["tiny", "base", "small", "medium", "large-v3"],
                    value="base",
                    label="Whisper 模型",
                )
                language = gr.Textbox(value="en", label="语言代码")
                ocr_sample_fps = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.5,
                    label="OCR 抽帧频率（fps）",
                )
                run_button = gr.Button("开始处理", variant="primary")

        status = gr.Textbox(label="运行结果", lines=12)
        downloads = gr.Files(label="下载生成文件")

        run_button.click(
            fn=run_pipeline,
            inputs=[file_input, strategy, model_name, language, ocr_sample_fps],
            outputs=[status, downloads],
        )

    return demo


if __name__ == "__main__":
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app = build_ui()
    app.launch()
