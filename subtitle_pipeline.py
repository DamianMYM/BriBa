#!/usr/bin/env python3

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


@dataclass
class SubtitleStream:
    index: int
    codec_name: str
    codec_type: str
    language: str | None
    title: str | None


@dataclass
class ProbeResult:
    path: Path
    subtitle_streams: list[SubtitleStream]
    raw_streams: list[dict[str, Any]]


@dataclass
class ProcessResult:
    strategy_used: str
    summary: str
    created_files: list[Path]
    probe_result: ProbeResult | None = None


@dataclass
class OCRSegment:
    start: float
    end: float
    text: str
    confidence: float


@dataclass
class OCRFrameText:
    timestamp: float
    text: str
    confidence: float


class PipelineError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe a video and extract subtitles when available."
    )
    parser.add_argument("video", help="Path to a local video file")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where extracted files will be written",
    )
    parser.add_argument(
        "--strategy",
        choices=["auto", "embedded", "ocr", "asr", "fusion"],
        default="auto",
        help="How to obtain subtitles from the video",
    )
    parser.add_argument(
        "--stream-index",
        type=int,
        help="Specific subtitle stream index to extract",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Only inspect the media file and print the detected streams",
    )
    parser.add_argument(
        "--format",
        default="srt",
        choices=["srt", "ass", "vtt"],
        help="Output subtitle format for embedded extraction",
    )
    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model name for ASR, for example tiny, base, small, medium, large-v3",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Expected speech language for ASR, for example en or zh",
    )
    parser.add_argument(
        "--ocr-sample-fps",
        type=float,
        default=1.0,
        help="Frames per second to sample for bottom-region subtitle OCR",
    )
    parser.add_argument(
        "--subtitle-region-ratio",
        type=float,
        default=0.28,
        help="Bottom portion of the frame to crop for OCR, expressed as 0-1 ratio of frame height",
    )
    return parser.parse_args()


def require_binary(name: str) -> str:
    binary = shutil.which(name)
    if not binary:
        raise PipelineError(
            f"Missing required binary: {name}. Install it first, then rerun."
        )
    return binary


def optional_binary(name: str) -> str | None:
    return shutil.which(name)


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise PipelineError(message) from exc


def run_command_with_env(
    command: list[str],
    extra_env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(extra_env)
    try:
        return subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise PipelineError(message) from exc


def probe_video(video_path: Path) -> ProbeResult:
    ffprobe = require_binary("ffprobe")
    result = run_command(
        [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            str(video_path),
        ]
    )
    payload = json.loads(result.stdout)
    raw_streams = payload.get("streams", [])
    subtitle_streams: list[SubtitleStream] = []

    for stream in raw_streams:
        if stream.get("codec_type") != "subtitle":
            continue
        tags = stream.get("tags", {})
        subtitle_streams.append(
            SubtitleStream(
                index=stream["index"],
                codec_name=stream.get("codec_name", "unknown"),
                codec_type=stream["codec_type"],
                language=tags.get("language"),
                title=tags.get("title"),
            )
        )

    return ProbeResult(
        path=video_path,
        subtitle_streams=subtitle_streams,
        raw_streams=raw_streams,
    )


def format_probe(result: ProbeResult) -> str:
    lines = [f"Video: {result.path}", f"Total streams: {len(result.raw_streams)}"]
    if not result.subtitle_streams:
        lines.append("Subtitle streams: none detected")
        return "\n".join(lines)

    lines.append("Subtitle streams:")
    for stream in result.subtitle_streams:
        label = stream.language or "unknown"
        title = f", title={stream.title}" if stream.title else ""
        lines.append(
            f"  - index={stream.index}, codec={stream.codec_name}, language={label}{title}"
        )
    return "\n".join(lines)


def choose_embedded_stream(
    result: ProbeResult, requested_index: int | None
) -> SubtitleStream:
    if not result.subtitle_streams:
        raise PipelineError(
            "No embedded subtitle stream found. Try OCR for hardcoded subtitles or ASR for speech recognition."
        )

    if requested_index is None:
        return result.subtitle_streams[0]

    for stream in result.subtitle_streams:
        if stream.index == requested_index:
            return stream

    available = ", ".join(str(stream.index) for stream in result.subtitle_streams)
    raise PipelineError(
        f"Requested stream index {requested_index} not found. Available subtitle indexes: {available}"
    )


def output_path_for(video_path: Path, output_dir: Path, suffix: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{video_path.stem}{suffix}"


def extract_embedded_subtitles(
    video_path: Path,
    output_dir: Path,
    stream: SubtitleStream,
    output_format: str,
) -> Path:
    ffmpeg = require_binary("ffmpeg")
    output_path = output_path_for(video_path, output_dir, f".embedded.{output_format}")
    run_command(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-map",
            f"0:{stream.index}",
            str(output_path),
        ]
    )
    return output_path


def write_text(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def timestamp_to_srt(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    ms = total_ms % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def segments_to_srt(segments: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for idx, segment in enumerate(segments, start=1):
        start = timestamp_to_srt(float(segment["start"]))
        end = timestamp_to_srt(float(segment["end"]))
        text = str(segment["text"]).strip()
        blocks.append(f"{idx}\n{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + "\n"


def segments_to_txt(segments: list[dict[str, Any]]) -> str:
    return "\n".join(str(segment["text"]).strip() for segment in segments) + "\n"


def normalized_line(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    compact = re.sub(r"[^\w\s']", "", compact.lower())
    return compact


def extract_audio(video_path: Path, output_dir: Path) -> Path:
    ffmpeg = require_binary("ffmpeg")
    audio_path = output_path_for(video_path, output_dir, ".audio.wav")
    run_command(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(audio_path),
        ]
    )
    return audio_path


def extract_subtitle_region_frames(
    video_path: Path,
    output_dir: Path,
    sample_fps: float,
    subtitle_region_ratio: float,
) -> Path:
    ffmpeg = require_binary("ffmpeg")
    frames_dir = output_dir / f"{video_path.stem}.ocr_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_pattern = frames_dir / "frame_%06d.png"
    crop_height_expr = f"{subtitle_region_ratio}*ih"
    crop_top_expr = f"ih-{subtitle_region_ratio}*ih"
    run_command(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps={sample_fps},crop=iw:{crop_height_expr}:0:{crop_top_expr}",
            str(frame_pattern),
        ]
    )
    return frames_dir


def run_vision_ocr(frames_dir: Path) -> list[dict[str, Any]]:
    swift = require_binary("swift")
    script_path = Path(__file__).resolve().parent / "vision_ocr.swift"
    module_cache_dir = frames_dir.parent / ".swift_module_cache"
    module_cache_dir.mkdir(parents=True, exist_ok=True)
    result = run_command_with_env(
        [swift, str(script_path), str(frames_dir)],
        {
            "CLANG_MODULE_CACHE_PATH": str(module_cache_dir),
            "SWIFT_MODULECACHE_PATH": str(module_cache_dir),
        },
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise PipelineError("Vision OCR returned invalid JSON output.") from exc


def can_run_tesseract() -> bool:
    binary = optional_binary("tesseract")
    if not binary:
        return False
    try:
        run_command_with_env(
            [binary, "--version"],
            {"DYLD_LIBRARY_PATH": "/usr/local/Cellar/libtiff/4.3.0/lib"},
        )
        return True
    except PipelineError:
        return False


def preprocess_ocr_frames(frames_dir: Path) -> Path:
    enhanced_dir = frames_dir.parent / f"{frames_dir.name}_enhanced"
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(frames_dir.glob("*.png")):
        img = Image.open(image_path).convert("L")
        img = ImageOps.autocontrast(img)
        img = img.resize((img.width * 4, img.height * 4))
        img = img.point(lambda pixel: 255 if pixel > 170 else 0)
        img.save(enhanced_dir / image_path.name)
    return enhanced_dir


def run_tesseract_ocr(frames_dir: Path) -> list[dict[str, Any]]:
    tesseract = require_binary("tesseract")
    enhanced_dir = preprocess_ocr_frames(frames_dir)
    payload: list[dict[str, Any]] = []
    for image_path in sorted(enhanced_dir.glob("*.png")):
        result = run_command_with_env(
            [
                tesseract,
                str(image_path),
                "stdout",
                "-l",
                "eng",
                "--psm",
                "6",
            ],
            {"DYLD_LIBRARY_PATH": "/usr/local/Cellar/libtiff/4.3.0/lib"},
        )
        lines = []
        for raw_line in result.stdout.splitlines():
            text = raw_line.strip()
            if not text:
                continue
            if not re.search(r"[A-Za-z]", text):
                continue
            lines.append({"text": text, "confidence": 1.0})
        payload.append({"file": image_path.name, "lines": lines})
    return payload


def parse_frame_timestamp(filename: str, sample_fps: float) -> float:
    match = re.search(r"(\d+)", filename)
    if not match:
        return 0.0
    index = int(match.group(1))
    return max(0.0, (index - 1) / sample_fps)


def collect_ocr_frame_texts(
    ocr_frames: list[dict[str, Any]],
    sample_fps: float,
) -> list[OCRFrameText]:
    texts: list[OCRFrameText] = []
    for frame in ocr_frames:
        lines = frame.get("lines", [])
        if not lines:
            continue
        candidates: list[tuple[str, float]] = []
        for line in lines:
            text = str(line.get("text", "")).strip()
            if not text:
                continue
            candidates.append((text, float(line.get("confidence", 0.0))))
        if not candidates:
            continue
        merged, confidence = max(
            candidates,
            key=lambda item: (
                sum(char.isalpha() for char in item[0]),
                len(item[0]),
            ),
        )
        if sum(char.isalpha() for char in merged) < 3 or len(merged.strip()) < 5:
            continue
        texts.append(
            OCRFrameText(
                timestamp=parse_frame_timestamp(str(frame.get("file", "")), sample_fps),
                text=merged,
                confidence=confidence,
            )
        )
    return texts


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalized_line(a), normalized_line(b)).ratio()


def aggregate_ocr_segments(
    frame_texts: list[OCRFrameText],
    sample_fps: float,
    min_similarity: float = 0.72,
) -> list[OCRSegment]:
    if not frame_texts:
        return []

    segments: list[OCRSegment] = []
    current_texts = [frame_texts[0].text]
    current_confidences = [frame_texts[0].confidence]
    start = frame_texts[0].timestamp
    end = frame_texts[0].timestamp + (1.0 / sample_fps)
    canonical = frame_texts[0].text

    for frame in frame_texts[1:]:
        is_similar = similarity(canonical, frame.text) >= min_similarity
        is_contiguous = frame.timestamp <= end + (1.5 / sample_fps)
        if is_similar and is_contiguous:
            current_texts.append(frame.text)
            current_confidences.append(frame.confidence)
            end = frame.timestamp + (1.0 / sample_fps)
            canonical = max(current_texts, key=lambda text: len(normalized_line(text)))
            continue

        segments.append(
            OCRSegment(
                start=start,
                end=end,
                text=max(current_texts, key=lambda text: len(normalized_line(text))),
                confidence=sum(current_confidences) / len(current_confidences),
            )
        )
        start = frame.timestamp
        end = frame.timestamp + (1.0 / sample_fps)
        current_texts = [frame.text]
        current_confidences = [frame.confidence]
        canonical = frame.text

    segments.append(
        OCRSegment(
            start=start,
            end=end,
            text=max(current_texts, key=lambda text: len(normalized_line(text))),
            confidence=sum(current_confidences) / len(current_confidences),
        )
    )
    return segments


def ocr_segments_to_dicts(segments: list[OCRSegment]) -> list[dict[str, Any]]:
    return [
        {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "confidence": segment.confidence,
        }
        for segment in segments
    ]


def write_segment_outputs(
    video_path: Path,
    output_dir: Path,
    prefix: str,
    segments: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> list[Path]:
    srt_path = output_path_for(video_path, output_dir, f".{prefix}.srt")
    txt_path = output_path_for(video_path, output_dir, f".{prefix}.txt")
    json_path = output_path_for(video_path, output_dir, f".{prefix}.json")
    write_text(srt_path, segments_to_srt(segments))
    write_text(txt_path, segments_to_txt(segments))
    payload = {"segments": segments}
    if metadata:
        payload.update(metadata)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return [srt_path, txt_path, json_path]


def load_asr_backend() -> tuple[str, Any]:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    try:
        from faster_whisper import WhisperModel

        return "faster-whisper", WhisperModel
    except ImportError:
        pass

    try:
        import whisper

        return "openai-whisper", whisper
    except ImportError as exc:
        raise PipelineError(
            "No ASR backend found. Install `faster-whisper` or `openai-whisper` in your active Python environment."
        ) from exc


def transcribe_audio(
    audio_path: Path,
    model_name: str,
    language: str,
) -> tuple[str, list[dict[str, Any]]]:
    backend_name, backend = load_asr_backend()

    if backend_name == "faster-whisper":
        model = backend(model_name)
        segments_iter, _info = model.transcribe(str(audio_path), language=language)
        segments = [
            {"start": segment.start, "end": segment.end, "text": segment.text}
            for segment in segments_iter
        ]
        return backend_name, segments

    model = backend.load_model(model_name)
    result = model.transcribe(str(audio_path), language=language)
    segments = [
        {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
        }
        for segment in result["segments"]
    ]
    return backend_name, segments


def transcribe_asr(
    video_path: Path,
    output_dir: Path,
    model_name: str,
    language: str,
) -> tuple[list[Path], list[dict[str, Any]]]:
    audio_path = extract_audio(video_path, output_dir)
    backend_name, segments = transcribe_audio(audio_path, model_name, language)
    paths = [audio_path]
    paths.extend(
        write_segment_outputs(
            video_path,
            output_dir,
            "asr",
            segments,
            metadata={
                "backend": backend_name,
                "model": model_name,
                "language": language,
            },
        )
    )
    return paths, segments


def extract_ocr_subtitles(
    video_path: Path,
    output_dir: Path,
    sample_fps: float,
    subtitle_region_ratio: float,
) -> tuple[list[Path], list[dict[str, Any]]]:
    frames_dir = extract_subtitle_region_frames(
        video_path, output_dir, sample_fps, subtitle_region_ratio
    )
    if can_run_tesseract():
        ocr_frames = run_tesseract_ocr(frames_dir)
        ocr_backend = "tesseract"
    else:
        ocr_frames = run_vision_ocr(frames_dir)
        ocr_backend = "vision"
    frame_texts = collect_ocr_frame_texts(ocr_frames, sample_fps)
    segments = ocr_segments_to_dicts(aggregate_ocr_segments(frame_texts, sample_fps))
    paths = [frames_dir]
    paths.extend(
        write_segment_outputs(
            video_path,
            output_dir,
            "ocr",
            segments,
            metadata={
                "ocr_backend": ocr_backend,
                "sample_fps": sample_fps,
                "subtitle_region_ratio": subtitle_region_ratio,
                "frame_count": len(ocr_frames),
            },
        )
    )
    return paths, segments


def overlap_duration(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def fuse_segments(
    asr_segments: list[dict[str, Any]],
    ocr_segments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fused: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    used_ocr_indexes: set[int] = set()

    for asr in asr_segments:
        asr_text = str(asr["text"]).strip()
        best_index = None
        best_overlap = 0.0
        for idx, ocr in enumerate(ocr_segments):
            if idx in used_ocr_indexes:
                continue
            overlap = overlap_duration(
                float(asr["start"]),
                float(asr["end"]),
                float(ocr["start"]),
                float(ocr["end"]),
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = idx

        if best_index is None or best_overlap <= 0:
            covered_by_used_ocr = False
            asr_duration = max(0.001, float(asr["end"]) - float(asr["start"]))
            for idx in used_ocr_indexes:
                ocr = ocr_segments[idx]
                overlap = overlap_duration(
                    float(asr["start"]),
                    float(asr["end"]),
                    float(ocr["start"]),
                    float(ocr["end"]),
                )
                if overlap / asr_duration >= 0.5:
                    covered_by_used_ocr = True
                    break
            if covered_by_used_ocr:
                continue
            fused.append(
                {
                    "start": asr["start"],
                    "end": asr["end"],
                    "text": asr_text,
                    "source": "asr",
                }
            )
            continue

        ocr = ocr_segments[best_index]
        used_ocr_indexes.add(best_index)
        ocr_text = str(ocr["text"]).strip()
        text_similarity = similarity(asr_text, ocr_text)
        if ocr_text and text_similarity < 0.85:
            review.append(
                {
                    "start": asr["start"],
                    "end": asr["end"],
                    "asr_text": asr_text,
                    "ocr_text": ocr_text,
                    "similarity": round(text_similarity, 3),
                }
            )

        final_text = ocr_text if ocr_text else asr_text
        source = (
            "ocr_override"
            if ocr_text and normalized_line(ocr_text) != normalized_line(asr_text)
            else "asr"
        )
        fused.append(
            {
                "start": min(float(asr["start"]), float(ocr["start"])),
                "end": max(float(asr["end"]), float(ocr["end"])),
                "text": final_text,
                "source": source,
            }
        )

    for idx, ocr in enumerate(ocr_segments):
        if idx in used_ocr_indexes:
            continue
        fused.append(
            {
                "start": ocr["start"],
                "end": ocr["end"],
                "text": ocr["text"],
                "source": "ocr_only",
            }
        )
        review.append(
            {
                "start": ocr["start"],
                "end": ocr["end"],
                "asr_text": "",
                "ocr_text": ocr["text"],
                "similarity": 0.0,
            }
        )

    fused.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    return fused, review


def process_video(
    video_path: Path,
    output_dir: Path,
    strategy: str = "auto",
    stream_index: int | None = None,
    output_format: str = "srt",
    model_name: str = "base",
    language: str = "en",
    probe_only: bool = False,
    ocr_sample_fps: float = 1.0,
    subtitle_region_ratio: float = 0.28,
) -> ProcessResult:
    if not video_path.exists():
        raise PipelineError(f"Video file does not exist: {video_path}")

    probe_result: ProbeResult | None = None
    ffprobe = optional_binary("ffprobe")
    if ffprobe:
        probe_result = probe_video(video_path)
    elif strategy in {"auto", "embedded"} or probe_only:
        raise PipelineError(
            "Missing required binary: ffprobe. Install it to inspect streams or auto-select extraction strategy."
        )

    if probe_result is not None and probe_only:
        return ProcessResult(
            strategy_used="probe",
            summary=format_probe(probe_result),
            created_files=[],
            probe_result=probe_result,
        )

    if strategy == "auto":
        strategy = "embedded" if probe_result and probe_result.subtitle_streams else "asr"

    if strategy == "embedded":
        if probe_result is None:
            raise PipelineError("Embedded extraction requires ffprobe to inspect subtitle streams.")
        stream = choose_embedded_stream(probe_result, stream_index)
        output_path = extract_embedded_subtitles(
            video_path, output_dir, stream, output_format
        )
        return ProcessResult(
            strategy_used="embedded",
            summary=f"Extracted embedded subtitles from stream {stream.index}.",
            created_files=[output_path],
            probe_result=probe_result,
        )

    if strategy == "ocr":
        created_files, segments = extract_ocr_subtitles(
            video_path, output_dir, ocr_sample_fps, subtitle_region_ratio
        )
        return ProcessResult(
            strategy_used="ocr",
            summary=f"Extracted {len(segments)} subtitle segments from the bottom subtitle region.",
            created_files=created_files,
            probe_result=probe_result,
        )

    if strategy == "asr":
        created_files, _segments = transcribe_asr(video_path, output_dir, model_name, language)
        return ProcessResult(
            strategy_used="asr",
            summary="Generated subtitles with Whisper-compatible ASR.",
            created_files=created_files,
            probe_result=probe_result,
        )

    if strategy == "fusion":
        asr_files, asr_segments = transcribe_asr(video_path, output_dir, model_name, language)
        ocr_files, ocr_segments = extract_ocr_subtitles(
            video_path, output_dir, ocr_sample_fps, subtitle_region_ratio
        )
        fused_segments, review = fuse_segments(asr_segments, ocr_segments)
        fused_files = write_segment_outputs(
            video_path,
            output_dir,
            "fused",
            fused_segments,
            metadata={
                "strategy": "fusion",
                "review_count": len(review),
                "ocr_sample_fps": ocr_sample_fps,
                "subtitle_region_ratio": subtitle_region_ratio,
            },
        )
        review_path = output_path_for(video_path, output_dir, ".review.json")
        review_path.write_text(
            json.dumps({"conflicts": review}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return ProcessResult(
            strategy_used="fusion",
            summary=(
                f"Generated ASR, OCR, and fused subtitles. "
                f"Fused segments: {len(fused_segments)}. Review conflicts: {len(review)}."
            ),
            created_files=asr_files + ocr_files + fused_files + [review_path],
            probe_result=probe_result,
        )

    raise PipelineError(f"Unsupported strategy: {strategy}")


def main() -> int:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    try:
        result = process_video(
            video_path=video_path,
            output_dir=output_dir,
            strategy=args.strategy,
            stream_index=args.stream_index,
            output_format=args.format,
            model_name=args.model,
            language=args.language,
            probe_only=args.probe_only,
            ocr_sample_fps=args.ocr_sample_fps,
            subtitle_region_ratio=args.subtitle_region_ratio,
        )
        if result.probe_result is not None:
            print(format_probe(result.probe_result))
        print(result.summary)
        for path in result.created_files:
            print(f"Created: {path}")
        return 0
    except PipelineError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
