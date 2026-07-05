#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
ASSET_DIR = APP_DIR / "assets"
EXPORT_DIR = APP_DIR / "exports"
PROJECTS_PATH = DATA_DIR / "projects.json"


MOBILE_INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
  <title>BriBa Pack</title>
  <style>
    :root { color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f6f8fb; color: #102033; }
    header { position: sticky; top: 0; z-index: 2; display: flex; align-items: center; gap: 12px; padding: 14px 16px; background: rgba(255,255,255,.94); border-bottom: 1px solid #e5e7eb; backdrop-filter: blur(10px); }
    header img { width: 42px; height: 42px; }
    h1 { font-size: 24px; line-height: 1; margin: 0; }
    main { padding: 14px; max-width: 820px; margin: 0 auto; }
    button, .card { border: 1px solid #dbe3ee; background: #fff; border-radius: 8px; }
    button { width: 100%; padding: 12px; text-align: left; font: inherit; }
    button:active { transform: scale(.99); }
    .card { padding: 14px; margin: 0 0 12px; }
    .muted { color: #64748b; font-size: 13px; }
    .project-list, .line-list { display: grid; gap: 10px; }
    .back { color: #2563eb; cursor: pointer; margin: 2px 0 12px; }
    .hidden { display: none; }
    audio { width: 100%; margin: 8px 0 12px; }
    pre { white-space: pre-wrap; word-break: break-word; background: #fff; border: 1px solid #dbe3ee; border-radius: 8px; padding: 12px; line-height: 1.5; }
    .current { border-left: 4px solid #10b981; background: #ecfdf5; padding: 12px; border-radius: 6px; line-height: 1.55; margin: 8px 0 12px; }
    .tools { display:flex; gap:8px; margin-bottom:12px; }
    .tools button { text-align:center; padding:9px 6px; }
    .line.active { border-color:#10b981; background:#f0fdf4; }
    .time { color:#64748b; font-size:12px; margin-bottom:4px; }
  </style>
</head>
<body>
  <header>
    <img src="assets/briba-icon.svg" alt="BriBa" />
    <div>
      <h1>BriBa</h1>
      <div class="muted" id="pack-meta">Learning pack</div>
    </div>
  </header>
  <main>
    <section id="list-view">
      <div class="card">
        <strong>开始学习</strong>
        <div class="muted">选择一个学习库，进入精听播放器和逐句字幕。</div>
      </div>
      <div class="project-list" id="project-list"></div>
    </section>
    <section id="project-view" class="hidden">
      <div class="back" id="back-button">← 学习库</div>
      <div class="card">
        <h2 id="project-title"></h2>
        <div class="muted" id="project-meta"></div>
        <audio id="project-audio" controls playsinline></audio>
        <div class="current" id="current-line">点击字幕开始精听。</div>
        <div class="tools">
          <button id="prev-button">上一句</button>
          <button id="replay-button">重听</button>
          <button id="next-button">下一句</button>
        </div>
      </div>
      <h3>学习笔记</h3>
      <pre id="project-notes"></pre>
      <h3>字幕</h3>
      <div class="line-list" id="line-list"></div>
    </section>
  </main>
  <script>
    const state = { manifest: null, items: [], active: -1 };
    const $ = (id) => document.getElementById(id);

    async function textOrEmpty(path) {
      if (!path) return "";
      try {
        const res = await fetch(path);
        return res.ok ? await res.text() : "";
      } catch {
        return "";
      }
    }

    function parseTime(value) {
      const match = String(value || "").trim().match(/(\d+):(\d+):(\d+)[,.](\d+)/);
      if (!match) return 0;
      return Number(match[1]) * 3600 + Number(match[2]) * 60 + Number(match[3]) + Number(match[4].slice(0, 3).padEnd(3, "0")) / 1000;
    }

    function parseSrt(text) {
      return String(text || "").replace(/\r/g, "").split(/\n\s*\n/).map((block) => {
        const lines = block.split("\n").filter(Boolean);
        const timeLine = lines.find((line) => line.includes("-->"));
        if (!timeLine) return null;
        const [startRaw, endRaw] = timeLine.split("-->").map((item) => item.trim());
        return { start: parseTime(startRaw), end: parseTime(endRaw), text: lines.slice(lines.indexOf(timeLine) + 1).join(" ") };
      }).filter(Boolean);
    }

    function formatTime(value) {
      const total = Math.max(0, Math.floor(Number(value) || 0));
      const m = Math.floor(total / 60);
      const s = total % 60;
      return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
    }

    function showList() {
      $("list-view").classList.remove("hidden");
      $("project-view").classList.add("hidden");
      $("project-audio").pause();
    }

    function setActive(index) {
      if (index < 0 || index >= state.items.length || index === state.active) return;
      state.active = index;
      document.querySelectorAll(".line").forEach((node, idx) => node.classList.toggle("active", idx === index));
      $("current-line").textContent = state.items[index].text || "";
      const node = document.querySelector(`.line[data-index="${index}"]`);
      if (node) node.scrollIntoView({ block: "nearest" });
    }

    function jumpTo(index, play) {
      const item = state.items[index];
      if (!item) return;
      $("project-audio").currentTime = Math.max(0, Number(item.start) || 0);
      setActive(index);
      if (play) $("project-audio").play().catch(() => {});
    }

    function renderLines(items) {
      const list = $("line-list");
      list.innerHTML = "";
      items.forEach((item, index) => {
        const button = document.createElement("button");
        button.className = "line";
        button.dataset.index = String(index);
        button.innerHTML = `<div class="time">${formatTime(item.start)} - ${formatTime(item.end)}</div><div>${item.text || ""}</div>`;
        button.addEventListener("click", () => jumpTo(index, true));
        list.appendChild(button);
      });
    }

    async function showProject(project) {
      $("list-view").classList.add("hidden");
      $("project-view").classList.remove("hidden");
      $("project-title").textContent = project.title || project.source_name || "Untitled";
      $("project-meta").textContent = `${project.created_at || ""} · ${project.start_time || "开头"} → ${project.end_time || "结尾"}`;
      $("project-audio").src = project.audio || "";
      $("project-notes").textContent = await textOrEmpty(project.notes);
      state.items = parseSrt(await textOrEmpty(project.subtitles));
      state.active = -1;
      $("current-line").textContent = "点击字幕开始精听。";
      renderLines(state.items);
      window.scrollTo({ top: 0, behavior: "smooth" });
    }

    async function boot() {
      const res = await fetch("manifest.json");
      state.manifest = await res.json();
      $("pack-meta").textContent = `${state.manifest.title || "Learning pack"} · ${state.manifest.projects.length} 个学习库`;
      const list = $("project-list");
      list.innerHTML = "";
      state.manifest.projects.forEach((project) => {
        const button = document.createElement("button");
        button.innerHTML = `<strong>${project.title || project.source_name}</strong><div class="muted">${project.created_at || ""} · ${project.start_time || "开头"} → ${project.end_time || "结尾"}</div>`;
        button.addEventListener("click", () => showProject(project));
        list.appendChild(button);
      });
    }

    $("back-button").addEventListener("click", showList);
    $("prev-button").addEventListener("click", () => jumpTo(Math.max(0, state.active - 1), true));
    $("replay-button").addEventListener("click", () => jumpTo(state.active >= 0 ? state.active : 0, true));
    $("next-button").addEventListener("click", () => jumpTo(Math.min(state.items.length - 1, state.active + 1), true));
    $("project-audio").addEventListener("timeupdate", () => {
      const t = $("project-audio").currentTime || 0;
      const index = state.items.findIndex((item) => t >= item.start && t <= item.end);
      if (index >= 0) setActive(index);
    });
    boot().catch((error) => {
      $("project-list").innerHTML = `<div class="card">无法读取 BriBa Pack：${error.message}</div>`;
    });
  </script>
</body>
</html>
"""


def load_projects() -> list[dict[str, Any]]:
    if not PROJECTS_PATH.exists():
        return []
    try:
        payload = json.loads(PROJECTS_PATH.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def first_existing(files: list[str], suffix: str) -> Path | None:
    for file_path in files:
        path = Path(file_path)
        if file_path.endswith(suffix) and path.exists():
            return path
    return None


def preferred_existing(files: list[str], preferred_suffix: str, fallback_suffix: str) -> Path | None:
    return first_existing(files, preferred_suffix) or first_existing(files, fallback_suffix)


def copy_if_exists(src: Path | None, dst: Path, root: Path) -> str | None:
    if not src or not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.relative_to(root).as_posix()


def selected_projects(project_ids: list[str] | None = None) -> list[dict[str, Any]]:
    projects = load_projects()
    if not project_ids:
        return projects
    wanted = set(project_ids)
    return [project for project in projects if project.get("id") in wanted]


def project_has_exportable_files(project: dict[str, Any]) -> bool:
    files = [str(path) for path in project.get("files", [])]
    return bool(
        first_existing(files, ".listening.mp3")
        and (
            preferred_existing(files, ".corrected.srt", ".asr.srt")
            or preferred_existing(files, ".corrected.txt", ".asr.txt")
        )
    )


def export_mobile_pack(
    out_path: Path | None = None,
    project_ids: list[str] | None = None,
    title: str = "BriBa Learning Pack",
) -> Path:
    projects = [project for project in selected_projects(project_ids) if project_has_exportable_files(project)]
    if not projects:
        raise RuntimeError("No learning projects found. Create a project before exporting a mobile pack.")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = EXPORT_DIR / f"briba-pack-{stamp}.zip"
    out_path = out_path.resolve()

    build_dir = EXPORT_DIR / ".briba-pack-build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    (build_dir / "assets").mkdir(parents=True, exist_ok=True)
    icon_src = ASSET_DIR / "briba-icon.svg"
    if icon_src.exists():
        shutil.copy2(icon_src, build_dir / "assets" / "briba-icon.svg")

    manifest_projects: list[dict[str, Any]] = []
    for project in projects:
        project_id = str(project.get("id") or f"project-{len(manifest_projects) + 1}")
        files = [str(path) for path in project.get("files", [])]
        project_dir = build_dir / "projects" / project_id

        audio = copy_if_exists(first_existing(files, ".listening.mp3"), project_dir / "audio.mp3", build_dir)
        subtitles = copy_if_exists(preferred_existing(files, ".corrected.srt", ".asr.srt"), project_dir / "subtitles.srt", build_dir)
        transcript = copy_if_exists(preferred_existing(files, ".corrected.txt", ".asr.txt"), project_dir / "transcript.txt", build_dir)
        notes = copy_if_exists(first_existing(files, ".learning.md"), project_dir / "notes.md", build_dir)
        translations = copy_if_exists(first_existing(files, ".translations.zh.json"), project_dir / "translations.zh.json", build_dir)
        data = copy_if_exists(preferred_existing(files, ".corrected.json", ".asr.json"), project_dir / "asr.json", build_dir)

        manifest_projects.append(
            {
                "id": project_id,
                "title": project.get("title") or project.get("source_name") or project_id,
                "source_name": project.get("source_name", ""),
                "created_at": project.get("created_at", ""),
                "start_time": project.get("start_time", ""),
                "end_time": project.get("end_time", ""),
                "language": project.get("language", "en"),
                "audio": audio,
                "subtitles": subtitles,
                "transcript": transcript,
                "notes": notes,
                "translations": translations,
                "asr_json": data,
            }
        )

    manifest = {
        "format": "briba-pack",
        "version": 1,
        "title": title,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "projects": manifest_projects,
    }
    (build_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (build_dir / "index.html").write_text(MOBILE_INDEX_HTML, encoding="utf-8")

    if out_path.exists():
        out_path.unlink()
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in build_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(build_dir).as_posix())

    shutil.rmtree(build_dir)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a BriBa mobile learning pack.")
    parser.add_argument("--out", help="Output zip path. Defaults to exports/briba-pack-YYYYMMDD-HHMMSS.zip")
    parser.add_argument("--title", default="BriBa Learning Pack")
    parser.add_argument("--project-id", action="append", help="Export only the given project id. Can be repeated.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_path = export_mobile_pack(
        out_path=Path(args.out) if args.out else None,
        project_ids=args.project_id,
        title=args.title,
    )
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
