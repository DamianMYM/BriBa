#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import os
import re
import wave
from pathlib import Path
from typing import Any

import numpy as np
import requests

from subtitle_pipeline import segments_to_srt, segments_to_txt, timestamp_to_srt


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_for_compare(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip().lower())
    return re.sub(r"[^a-z0-9']+", " ", text).strip()


def remove_repeated_phrases(text: str) -> str:
    words = re.findall(r"\S+", text.strip())
    if len(words) < 4:
        return text.strip()

    changed = True
    while changed:
        changed = False
        for size in range(1, min(10, len(words) // 2) + 1):
            idx = 0
            compact: list[str] = []
            while idx < len(words):
                current = [normalize_for_compare(word) for word in words[idx : idx + size]]
                next_part = [normalize_for_compare(word) for word in words[idx + size : idx + size * 2]]
                if current and current == next_part:
                    compact.extend(words[idx : idx + size])
                    idx += size * 2
                    changed = True
                    while idx + size <= len(words):
                        more = [normalize_for_compare(word) for word in words[idx : idx + size]]
                        if more != current:
                            break
                        idx += size
                    continue
                compact.append(words[idx])
                idx += 1
            words = compact
    return " ".join(words).strip()


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = remove_repeated_phrases(text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([,.!?;:])(?=[^\s])", r"\1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_text(text: str, max_words: int = 18) -> list[str]:
    text = clean_text(text)
    if not text:
        return []

    sentences = [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]
    chunks: list[str] = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) <= max_words:
            chunks.append(sentence)
            continue
        for start in range(0, len(words), max_words):
            chunks.append(" ".join(words[start : start + max_words]))
    return chunks or [text]


def split_segment(segment: dict[str, Any], max_words: int = 18) -> list[dict[str, Any]]:
    text = clean_text(str(segment.get("text", "")))
    chunks = split_text(text, max_words=max_words)
    if not chunks:
        return []
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", start + 0.5))
    duration = max(0.2, end - start)
    weights = [max(1, len(chunk)) for chunk in chunks]
    total = sum(weights)

    output: list[dict[str, Any]] = []
    cursor = start
    for idx, chunk in enumerate(chunks):
        if idx == len(chunks) - 1:
            chunk_end = end
        else:
            chunk_end = cursor + duration * (weights[idx] / total)
        output.append(
            {
                "start": round(cursor, 3),
                "end": round(max(cursor + 0.2, chunk_end), 3),
                "text": chunk,
                "speaker": segment.get("speaker", ""),
                "raw_text": segment.get("raw_text", segment.get("text", "")),
                "status": "auto_cleaned" if chunk != segment.get("text", "") else "draft",
                "note": segment.get("note", ""),
            }
        )
        cursor = chunk_end
    return output


def dedupe_adjacent_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for segment in segments:
        text = clean_text(str(segment.get("text", "")))
        if not text:
            continue
        segment = {**segment, "text": text}
        if output:
            prev = output[-1]
            prev_norm = normalize_for_compare(str(prev.get("text", "")))
            cur_norm = normalize_for_compare(text)
            if prev_norm and cur_norm and prev_norm == cur_norm:
                prev["end"] = max(float(prev["end"]), float(segment["end"]))
                prev["status"] = "auto_merged_duplicate"
                continue
        output.append(segment)
    return output


def auto_cleanup_segments(
    segments: list[dict[str, Any]],
    max_words: int = 18,
) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for segment in segments:
        cleaned.extend(split_segment(segment, max_words=max_words))
    cleaned = dedupe_adjacent_segments(cleaned)
    for idx, segment in enumerate(cleaned, start=1):
        segment["idx"] = idx
    return cleaned


def read_segments_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        segments = payload.get("segments", [])
    else:
        segments = payload
    return [dict(segment) for segment in segments]


def write_corrected_outputs(
    base_path: Path,
    segments: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> list[Path]:
    json_segments = []
    display_segments = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        speaker = str(segment.get("speaker", "")).strip()
        base_segment = {
            "start": float(segment.get("start", 0.0)),
            "end": float(segment.get("end", 0.0)),
            "text": text,
            "speaker": speaker,
            "status": segment.get("status", ""),
            "note": segment.get("note", ""),
        }
        json_segments.append(base_segment)
        display_text = f"{speaker}: {text}" if speaker else text
        display_segments.append(
            {
                "start": base_segment["start"],
                "end": base_segment["end"],
                "text": display_text,
                "speaker": speaker,
                "status": base_segment["status"],
                "note": base_segment["note"],
            }
        )

    json_path = base_path.with_suffix(".corrected.json")
    srt_path = base_path.with_suffix(".corrected.srt")
    txt_path = base_path.with_suffix(".corrected.txt")
    payload = {"segments": json_segments}
    if metadata:
        payload.update(metadata)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    srt_path.write_text(segments_to_srt(display_segments), encoding="utf-8")
    txt_path.write_text(segments_to_txt(display_segments), encoding="utf-8")
    return [json_path, srt_path, txt_path]


def segments_to_rows(segments: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for idx, segment in enumerate(segments, start=1):
        rows.append(
            [
                int(segment.get("idx", idx)),
                round(float(segment.get("start", 0.0)), 3),
                round(float(segment.get("end", 0.0)), 3),
                str(segment.get("speaker", "")),
                str(segment.get("text", "")),
                str(segment.get("status", "draft")),
                str(segment.get("note", "")),
            ]
        )
    return rows


def rows_to_segments(rows: Any) -> list[dict[str, Any]]:
    if hasattr(rows, "to_dict"):
        records = rows.to_dict("records")
        normalized_rows = [
            [
                record.get("idx", record.get("序号", idx + 1)),
                record.get("start", record.get("开始", 0.0)),
                record.get("end", record.get("结束", 0.0)),
                record.get("speaker", record.get("说话人", "")),
                record.get("text", record.get("文本", "")),
                record.get("status", record.get("状态", "draft")),
                record.get("note", record.get("备注", "")),
            ]
            for idx, record in enumerate(records)
        ]
    else:
        normalized_rows = rows or []

    segments: list[dict[str, Any]] = []
    for idx, row in enumerate(normalized_rows, start=1):
        if isinstance(row, dict):
            row = list(row.values())
        if len(row) < 5:
            continue
        text = str(row[4]).strip()
        if not text:
            continue
        segments.append(
            {
                "idx": int(float(row[0] or idx)),
                "start": float(row[1] or 0.0),
                "end": float(row[2] or 0.0),
                "speaker": str(row[3] or "").strip(),
                "text": text,
                "status": str(row[5] if len(row) > 5 else "reviewed"),
                "note": str(row[6] if len(row) > 6 else ""),
            }
        )
    return segments


def wav_samples(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
        width = wav.getsampwidth()
    if width != 2:
        raise ValueError("Only 16-bit PCM WAV is supported for speaker clustering.")
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sample_rate


def segment_audio_features(samples: np.ndarray, sample_rate: int, start: float, end: float) -> np.ndarray:
    start_idx = max(0, int(start * sample_rate))
    end_idx = min(len(samples), int(end * sample_rate))
    chunk = samples[start_idx:end_idx]
    if len(chunk) < sample_rate // 5:
        return np.zeros(8, dtype=np.float32)

    chunk = chunk - np.mean(chunk)
    energy = float(np.sqrt(np.mean(chunk**2)))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(chunk)))))
    spectrum = np.abs(np.fft.rfft(chunk[: min(len(chunk), sample_rate * 4)]))
    freqs = np.fft.rfftfreq(len(chunk[: min(len(chunk), sample_rate * 4)]), 1 / sample_rate)
    total = float(np.sum(spectrum) + 1e-8)
    centroid = float(np.sum(freqs * spectrum) / total)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / total))
    cumulative = np.cumsum(spectrum)
    rolloff = float(freqs[min(len(freqs) - 1, int(np.searchsorted(cumulative, cumulative[-1] * 0.85)))])
    return np.array(
        [
            energy,
            zcr,
            centroid / sample_rate,
            bandwidth / sample_rate,
            rolloff / sample_rate,
            float(np.percentile(chunk, 10)),
            float(np.percentile(chunk, 50)),
            float(np.percentile(chunk, 90)),
        ],
        dtype=np.float32,
    )


def cluster_speakers(
    wav_path: Path,
    segments: list[dict[str, Any]],
    speaker_count: int = 2,
) -> list[dict[str, Any]]:
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    samples, sample_rate = wav_samples(wav_path)
    features = np.vstack(
        [
            segment_audio_features(samples, sample_rate, float(segment["start"]), float(segment["end"]))
            for segment in segments
        ]
    )
    if len(segments) < speaker_count:
        speaker_count = max(1, len(segments))
    scaled = StandardScaler().fit_transform(features)
    labels = KMeans(n_clusters=speaker_count, random_state=42, n_init=10).fit_predict(scaled)
    output = []
    for segment, label in zip(segments, labels):
        output.append({**segment, "speaker": f"Speaker {int(label) + 1}", "status": "speaker_clustered"})
    return output


def extract_json_array(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError("AI response must be a JSON array.")
    return [dict(item) for item in payload]


def ollama_generate(model: str, prompt: str, timeout: int = 240) -> str:
    response = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.15,
                "top_p": 0.8,
            },
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return str(response.json().get("response", "")).strip()


def build_ai_optimize_prompt(
    chunk: list[dict[str, Any]],
    character_hints: str,
    context: str,
) -> str:
    payload = [
        {
            "idx": int(segment.get("idx", offset + 1)),
            "start": float(segment.get("start", 0.0)),
            "end": float(segment.get("end", 0.0)),
            "speaker": str(segment.get("speaker", "")),
            "text": str(segment.get("text", "")),
            "note": str(segment.get("note", "")),
        }
        for offset, segment in enumerate(chunk)
    ]
    return f"""
You are BriBa's transcript intelligence engine for English drama learning.
Your job is to improve ASR transcript rows so a human can review them faster.

Rules:
- Return ONLY a JSON array. No markdown. No explanation outside JSON.
- Keep the same number of rows and the same idx values.
- Do not change start/end timestamps.
- Improve punctuation, casing, spacing, and obvious ASR mistakes using context.
- Remove repeated phrases caused by ASR hallucination.
- Preserve uncertainty: if unsure, keep the original wording and add a short note.
- Speaker field may be improved only when context strongly supports it.
- If speaker is unknown, keep the existing Speaker label or use an empty string.
- status must be one of: ai_cleaned, needs_review, likely_ad_or_non_dialogue.
- note should be concise, in Chinese, explaining what changed or why review is needed.
- Do not translate the English dialogue into Chinese.

Known characters or user hints:
{character_hints or "(none)"}

Project context:
{context or "(none)"}

Rows:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return JSON array items shaped exactly like:
[
  {{"idx": 1, "speaker": "Speaker 1", "text": "Corrected English line.", "status": "ai_cleaned", "note": "修正标点"}},
  ...
]
""".strip()


def merge_ai_rows(
    original: list[dict[str, Any]],
    ai_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_idx = {int(item.get("idx")): item for item in ai_rows if str(item.get("idx", "")).strip()}
    output: list[dict[str, Any]] = []
    for segment in original:
        idx = int(segment.get("idx", len(output) + 1))
        ai = by_idx.get(idx)
        if not ai:
            output.append({**segment, "status": "needs_review", "note": "AI 未返回该行"})
            continue
        text = clean_text(str(ai.get("text", segment.get("text", ""))))
        status = str(ai.get("status", "ai_cleaned")).strip() or "ai_cleaned"
        if status not in {"ai_cleaned", "needs_review", "likely_ad_or_non_dialogue"}:
            status = "needs_review"
        output.append(
            {
                **segment,
                "speaker": str(ai.get("speaker", segment.get("speaker", ""))).strip(),
                "text": text or str(segment.get("text", "")),
                "status": status,
                "note": str(ai.get("note", segment.get("note", ""))).strip(),
            }
        )
    return output


def ai_optimize_segments(
    segments: list[dict[str, Any]],
    model: str = "qwen3.5:27b",
    character_hints: str = "",
    context: str = "",
    chunk_size: int = 16,
    max_rows: int = 80,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not segments:
        return [], ["没有可优化的字幕段。"]
    chunk_size = max(4, min(30, int(chunk_size or 16)))
    max_rows = int(max_rows or 0)
    selected = segments if max_rows <= 0 else segments[:max_rows]
    remaining = segments[len(selected) :]

    optimized: list[dict[str, Any]] = []
    messages: list[str] = []
    for start in range(0, len(selected), chunk_size):
        chunk = selected[start : start + chunk_size]
        prompt = build_ai_optimize_prompt(chunk, character_hints, context)
        try:
            response = ollama_generate(model, prompt)
            ai_rows = extract_json_array(response)
            optimized.extend(merge_ai_rows(chunk, ai_rows))
            messages.append(f"AI 优化 {start + 1}-{start + len(chunk)} 行完成。")
        except Exception as exc:
            fallback = [{**segment, "status": "needs_review", "note": f"AI 优化失败：{exc}"} for segment in chunk]
            optimized.extend(fallback)
            messages.append(f"AI 优化 {start + 1}-{start + len(chunk)} 行失败：{exc}")
    optimized.extend(remaining)
    for idx, segment in enumerate(optimized, start=1):
        segment["idx"] = idx
    return optimized, messages
