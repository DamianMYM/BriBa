#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from correction_tools import (
    ai_optimize_segments,
    auto_cleanup_segments,
    read_segments_json,
    rows_to_segments,
    segments_to_rows,
    write_corrected_outputs,
)
from mobile_pack import export_mobile_pack
from subtitle_pipeline import PipelineError, download_video, process_learning_clip


APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "output"
DATA_DIR = APP_DIR / "data"
ASSET_DIR = APP_DIR / "assets"
PROJECTS_PATH = DATA_DIR / "projects.json"
APP_LOG_PATH = OUTPUT_DIR / "briba-app.log"

REVIEW_COLUMNS = ["idx", "start", "end", "speaker", "text", "status", "note"]


BRIBA_ICON_SVG = """
<svg width="96" height="96" viewBox="0 0 96 96" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="BriBa mascot">
  <defs>
    <linearGradient id="body" x1="18" y1="15" x2="78" y2="84" gradientUnits="userSpaceOnUse">
      <stop stop-color="#36d6c3"/><stop offset="0.55" stop-color="#25a6d9"/><stop offset="1" stop-color="#4f64f4"/>
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="5" stdDeviation="5" flood-color="#102033" flood-opacity="0.18"/>
    </filter>
  </defs>
  <path d="M23 37c0-15 11-25 26-25 16 0 28 11 28 27v20c0 16-11 27-28 27-16 0-27-10-27-26V37z" fill="url(#body)" filter="url(#shadow)"/>
  <path d="M19 43c-7 1-12-3-13-9-1-7 4-12 11-11 5 1 9 5 10 10" fill="#36d6c3"/>
  <path d="M77 43c7 1 12-3 13-9 1-7-4-12-11-11-5 1-9 5-10 10" fill="#4f64f4"/>
  <path d="M28 50c0-10 8-17 20-17s20 7 20 17-8 17-20 17-20-7-20-17z" fill="#f8fbff"/>
  <circle cx="39" cy="48" r="5" fill="#102033"/><circle cx="58" cy="47" r="4" fill="#102033"/>
  <path d="M41 58c5 4 11 4 16 0" fill="none" stroke="#102033" stroke-width="3" stroke-linecap="round"/>
  <rect x="31" y="68" width="34" height="11" rx="5.5" fill="#102033" opacity="0.9"/>
</svg>
""".strip()


LEGACY_ROOT_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>BriBa</title>
  <style>
    :root { color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f6f8fb; color: #102033; }
    header { display:flex; align-items:center; justify-content:space-between; gap:16px; padding:14px 20px; background:#fff; border-bottom:1px solid #dbe3ee; position:sticky; top:0; z-index:3; }
    .brand { display:flex; align-items:center; gap:14px; }
    .logo { width:58px; height:58px; }
    h1 { margin:0; font-size:30px; line-height:1; letter-spacing:0; }
    .tagline { margin-top:5px; color:#64748b; font-size:14px; }
    nav { display:flex; gap:8px; flex-wrap:wrap; }
    nav a { color:#102033; background:#fff; border:1px solid #cbd5e1; border-radius:7px; padding:8px 11px; text-decoration:none; font-size:14px; }
    nav a:hover { border-color:#2563eb; color:#1d4ed8; }
    main { max-width: 1180px; margin:0 auto; padding:18px; }
    .topbar { display:grid; grid-template-columns:minmax(280px,1fr) auto; gap:10px; align-items:end; margin-bottom:14px; }
    label { display:block; font-size:13px; color:#64748b; margin-bottom:5px; }
    select, textarea, input { width:100%; box-sizing:border-box; border:1px solid #cbd5e1; border-radius:7px; padding:10px; font:inherit; background:#fff; }
    button { border:1px solid #cbd5e1; border-radius:7px; background:#fff; padding:10px 12px; font:inherit; cursor:pointer; }
    button.primary { background:#2563eb; color:#fff; border-color:#2563eb; }
    button:hover { filter: brightness(.98); }
    .study-wrap { display:grid; grid-template-columns:minmax(280px, 0.9fr) minmax(360px, 1.4fr); gap:16px; align-items:start; }
    .panel { background:#fff; border:1px solid #dbe3ee; border-radius:8px; padding:14px; }
    .study-side { position:sticky; top:96px; }
    .study-title { font-size:22px; font-weight:700; margin-bottom:6px; }
    .study-meta { color:#64748b; font-size:13px; margin-bottom:12px; line-height:1.5; }
    audio { width:100%; margin:8px 0 12px; }
    .current-line { border-left:4px solid #10b981; background:#ecfdf5; padding:12px; border-radius:6px; line-height:1.55; min-height:52px; }
    .study-tools { display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }
    .assistant { margin-top:14px; }
    .assistant h2 { font-size:18px; margin:0 0 10px; }
    .assistant-answer { white-space:pre-wrap; line-height:1.6; color:#102033; background:#f8fafc; border:1px solid #e2e8f0; border-radius:7px; padding:10px; min-height:48px; }
    .line-list { display:grid; gap:8px; max-height:76vh; overflow:auto; padding-right:4px; }
    .line-item { width:100%; text-align:left; border:1px solid #dbe3ee; background:#fff; border-radius:8px; padding:10px 12px; cursor:pointer; }
    .line-item:hover { border-color:#38bdf8; }
    .line-item.active { border-color:#10b981; background:#f0fdf4; }
    .line-time { color:#64748b; font-size:12px; margin-bottom:4px; }
    .line-speaker { color:#0f766e; font-size:12px; font-weight:700; margin-right:6px; }
    .empty { background:#fff; border:1px dashed #cbd5e1; border-radius:8px; padding:20px; color:#64748b; }
    @media (max-width: 860px) {
      header { align-items:flex-start; flex-direction:column; }
      .topbar, .study-wrap { grid-template-columns:1fr; }
      .study-side { position:static; }
      .line-list { max-height:none; }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="logo">__ICON__</div>
      <div>
        <h1>BriBa</h1>
        <div class="tagline">首页只用于学习。材料处理在独立页面。</div>
      </div>
    </div>
    <nav aria-label="工具页">
      <a href="/new/">新建材料</a>
      <a href="/review/">校对材料</a>
      <a href="/library/">学习库</a>
      <a href="/mobile/">手机包</a>
    </nav>
  </header>
  <main>
    <section class="topbar">
      <div>
        <label for="project-select">选择学习库</label>
        <select id="project-select"></select>
      </div>
      <button class="primary" id="start-button">开始学习</button>
    </section>
    <section id="study-root" class="empty">正在读取学习库...</section>
  </main>
  <script>
    const state = {
      projects: [],
      project: null,
      items: [],
      active: -1,
      subtitleDelay: 0.4,
      showCurrentLine: true,
      showTranslation: false,
      transcriptCollapsed: false
    };
    const $ = (id) => document.getElementById(id);

    function escapeHtml(value) {
      return String(value || "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
    }
    function formatTime(value) {
      const total = Math.max(0, Math.floor(Number(value) || 0));
      const m = Math.floor(total / 60);
      const s = total % 60;
      return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
    }
    async function api(url, options) {
      const res = await fetch(url, options);
      if (!res.ok) throw new Error(await res.text());
      return await res.json();
    }
    async function loadProjects() {
      const data = await api("/api/projects");
      state.projects = data.projects || [];
      const select = $("project-select");
      select.innerHTML = "";
      state.projects.forEach((project) => {
        const option = document.createElement("option");
        option.value = project.id;
        option.textContent = `${project.created_at || ""} | ${project.title || project.source_name || project.id}`;
        select.appendChild(option);
      });
      if (!state.projects.length) {
        $("study-root").innerHTML = '<div class="empty">还没有学习库。请先进入“新建材料”生成一个学习库。</div>';
        return;
      }
      await loadStudy(select.value);
    }
    function renderStudy(data) {
      state.project = data.project;
      state.items = data.items || [];
      state.active = -1;
      $("study-root").innerHTML = `
        <div class="study-wrap">
          <section class="panel study-side">
            <div class="study-title">${escapeHtml(data.project.title || "Untitled")}</div>
            <div class="study-meta">${escapeHtml(data.project.source_name || "")}<br>${escapeHtml(data.project.created_at || "")}</div>
            <audio controls preload="metadata" src="${data.audio_url || ""}"></audio>
            <div class="current-line">点击右侧任意一句开始精听。</div>
            <div class="study-tools">
              <button data-action="prev">上一句</button>
              <button data-action="replay">重听本句</button>
              <button data-action="next">下一句</button>
            </div>
            <section class="assistant">
              <h2>AI 学习助手</h2>
              <textarea id="assistant-question" rows="3" placeholder="例如：这段对话里有哪些值得背的表达？第 3 句语法怎么理解？"></textarea>
              <div class="study-tools">
                <input id="assistant-model" value="qwen3.5:27b" aria-label="Ollama 模型" />
                <button class="primary" id="assistant-button">提问</button>
              </div>
              <div class="assistant-answer" id="assistant-answer">选择一句或输入问题，让 BriBa 帮你分析。</div>
            </section>
          </section>
          <section class="line-list"></section>
        </div>`;
      renderLines();
      const audio = document.querySelector("audio");
      audio.addEventListener("timeupdate", () => {
        const t = audio.currentTime || 0;
        const index = state.items.findIndex((item) => t >= Number(item.start) && t <= Number(item.end));
        if (index >= 0) setActive(index);
      });
      document.querySelector('[data-action="prev"]').addEventListener("click", () => jumpTo(Math.max(0, state.active - 1), true));
      document.querySelector('[data-action="next"]').addEventListener("click", () => jumpTo(Math.min(state.items.length - 1, state.active + 1), true));
      document.querySelector('[data-action="replay"]').addEventListener("click", () => jumpTo(state.active >= 0 ? state.active : 0, true));
      $("assistant-button").addEventListener("click", askAssistant);
    }
    function renderLines() {
      const list = document.querySelector(".line-list");
      list.innerHTML = "";
      state.items.forEach((item, index) => {
        const button = document.createElement("button");
        button.className = "line-item";
        button.dataset.index = String(index);
        const speaker = item.speaker ? `<span class="line-speaker">${escapeHtml(item.speaker)}</span>` : "";
        button.innerHTML = `<div class="line-time">${formatTime(item.start)} - ${formatTime(item.end)}</div><div>${speaker}${escapeHtml(item.text)}</div>`;
        button.addEventListener("click", () => jumpTo(index, true));
        list.appendChild(button);
      });
    }
    function setActive(index) {
      if (index < 0 || index >= state.items.length || index === state.active) return;
      state.active = index;
      document.querySelectorAll(".line-item").forEach((node, idx) => node.classList.toggle("active", idx === index));
      document.querySelector(".current-line").textContent = state.items[index].text || "";
      const node = document.querySelector(`.line-item[data-index="${index}"]`);
      if (node) node.scrollIntoView({ block: "nearest" });
    }
    function jumpTo(index, play) {
      const item = state.items[index];
      const audio = document.querySelector("audio");
      if (!item || !audio) return;
      audio.currentTime = Math.max(0, Number(item.start) || 0);
      setActive(index);
      if (play) audio.play().catch(() => {});
    }
    async function loadStudy(projectId) {
      if (!projectId) return;
      $("study-root").innerHTML = '<div class="empty">正在打开学习库...</div>';
      renderStudy(await api(`/api/study/${encodeURIComponent(projectId)}`));
    }
    async function askAssistant() {
      const answer = $("assistant-answer");
      answer.textContent = "BriBa 正在思考...";
      try {
        const data = await api("/api/assistant", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            project_id: $("project-select").value,
            question: $("assistant-question").value,
            model: $("assistant-model").value
          })
        });
        answer.textContent = data.answer || "没有返回内容。";
      } catch (error) {
        answer.textContent = `AI 学习助手暂时不可用：${error.message}`;
      }
    }
    $("project-select").addEventListener("change", (event) => loadStudy(event.target.value));
    $("start-button").addEventListener("click", () => loadStudy($("project-select").value));
    loadProjects().catch((error) => {
      $("study-root").innerHTML = `<div class="empty">读取失败：${escapeHtml(error.message)}</div>`;
    });
  </script>
</body>
</html>
"""


ROOT_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>BriBa</title>
  <style>
    :root {
      color-scheme: light;
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif;
      --ink: #1d1d1f;
      --muted: #6e6e73;
      --line: #d2d2d7;
      --soft-line: #e8e8ed;
      --surface: rgba(255, 255, 255, 0.82);
      --surface-solid: #ffffff;
      --page: #f5f5f7;
      --accent: #0071e3;
      --accent-soft: #e8f2ff;
      --green: #0f8a5f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--page);
      color: var(--ink);
      -webkit-font-smoothing: antialiased;
    }
    a { color: inherit; text-decoration: none; }
    button, select, textarea, input {
      font: inherit;
      color: inherit;
    }
    .app-shell {
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    .site-header {
      position: sticky;
      top: 0;
      z-index: 10;
      border-bottom: 1px solid rgba(210, 210, 215, 0.72);
      background: rgba(251, 251, 253, 0.84);
      backdrop-filter: saturate(180%) blur(20px);
    }
    .header-inner {
      max-width: 1240px;
      height: 58px;
      margin: 0 auto;
      padding: 0 22px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }
    .brand {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      min-width: 0;
    }
    .logo {
      width: 34px;
      height: 34px;
      flex: 0 0 34px;
    }
    .logo svg {
      width: 100%;
      height: 100%;
      display: block;
    }
    .brand-title {
      font-size: 17px;
      font-weight: 700;
      line-height: 1;
      white-space: nowrap;
    }
    .brand-subtitle {
      margin-top: 3px;
      font-size: 12px;
      color: var(--muted);
      white-space: nowrap;
    }
    .tool-nav {
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .tool-nav a {
      min-height: 32px;
      display: inline-flex;
      align-items: center;
      border-radius: 8px;
      padding: 0 10px;
      color: #424245;
      font-size: 13px;
    }
    .tool-nav a:hover {
      background: #ededf2;
      color: var(--ink);
    }
    main {
      width: 100%;
      max-width: 1240px;
      margin: 0 auto;
      padding: 28px 22px 34px;
    }
    .intro {
      display: grid;
      grid-template-columns: minmax(260px, 1fr) minmax(340px, 560px);
      gap: 22px;
      align-items: end;
      margin-bottom: 20px;
    }
    .intro h1 {
      margin: 0;
      font-size: clamp(28px, 4.2vw, 50px);
      line-height: 1.05;
      font-weight: 720;
      letter-spacing: 0;
    }
    .intro p {
      max-width: 620px;
      margin: 12px 0 0;
      font-size: 16px;
      line-height: 1.55;
      color: var(--muted);
    }
    .library-bar {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 128px;
      gap: 10px;
      padding: 12px;
      border: 1px solid var(--soft-line);
      border-radius: 8px;
      background: var(--surface);
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.06);
    }
    label {
      display: block;
      margin-bottom: 6px;
      font-size: 12px;
      color: var(--muted);
    }
    select, textarea, input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface-solid);
      outline: none;
    }
    select, input {
      height: 42px;
      padding: 0 12px;
    }
    textarea {
      min-height: 88px;
      padding: 11px 12px;
      resize: vertical;
      line-height: 1.5;
    }
    select:focus, textarea:focus, input:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(0, 113, 227, 0.14);
    }
    button {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      min-height: 40px;
      padding: 0 13px;
      cursor: pointer;
    }
    button:hover { background: #f7f7fa; }
    button.primary {
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
      font-weight: 650;
    }
    button.primary:hover { background: #0077ed; }
    .library-bar .primary {
      align-self: end;
      height: 42px;
    }
    .empty {
      border: 1px dashed #c7c7cc;
      border-radius: 8px;
      padding: 24px;
      color: var(--muted);
      background: rgba(255,255,255,0.72);
    }
    .study-layout {
      display: grid;
      grid-template-columns: minmax(320px, 0.92fr) minmax(380px, 1.08fr);
      gap: 18px;
      align-items: start;
    }
    .panel {
      border: 1px solid var(--soft-line);
      border-radius: 8px;
      background: var(--surface-solid);
      box-shadow: 0 18px 45px rgba(0, 0, 0, 0.055);
    }
    .player-panel {
      position: sticky;
      top: 78px;
      overflow: hidden;
    }
    .now-card {
      padding: 22px;
      background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%);
      border-bottom: 1px solid var(--soft-line);
    }
    .eyebrow {
      margin-bottom: 8px;
      font-size: 12px;
      color: var(--muted);
    }
    .study-title {
      margin: 0;
      font-size: 26px;
      line-height: 1.18;
      font-weight: 720;
    }
    .study-meta {
      margin-top: 9px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
      overflow-wrap: anywhere;
    }
    .audio-wrap {
      padding: 16px 22px 0;
    }
    audio {
      width: 100%;
      display: block;
    }
    .learning-options {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 12px 22px 0;
      color: var(--muted);
      font-size: 12px;
    }
    .option-group {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
    }
    .chip-button {
      min-height: 32px;
      border-radius: 8px;
      padding: 0 10px;
      color: #424245;
      background: #f7f7fa;
      border-color: #e1e1e6;
      font-size: 12px;
    }
    .chip-button.active {
      background: #1d1d1f;
      color: #fff;
      border-color: #1d1d1f;
    }
    .sync-value {
      min-width: 60px;
      text-align: center;
      font-variant-numeric: tabular-nums;
    }
    .current-line {
      margin: 18px 22px 0;
      min-height: 112px;
      border-radius: 8px;
      background: #fbfbfd;
      border: 1px solid var(--soft-line);
      padding: 18px;
      display: flex;
      align-items: center;
      font-size: 22px;
      line-height: 1.42;
      font-weight: 650;
    }
    .current-line.is-empty {
      color: var(--muted);
      font-size: 16px;
      font-weight: 500;
    }
    .current-line.is-hidden {
      display: none;
    }
    .study-controls {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      padding: 14px 22px 22px;
    }
    .icon-button {
      height: 44px;
      padding: 0;
      font-size: 22px;
      line-height: 1;
    }
    .assistant {
      border-top: 1px solid var(--soft-line);
      padding: 20px 22px 22px;
    }
    .assistant-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 12px;
    }
    .assistant h2 {
      margin: 0;
      font-size: 18px;
      line-height: 1.2;
    }
    .assistant-head span {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .assistant-actions {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 92px;
      gap: 8px;
      margin-top: 8px;
    }
    .assistant-answer {
      margin-top: 12px;
      min-height: 72px;
      border: 1px solid var(--soft-line);
      border-radius: 8px;
      background: #fbfbfd;
      padding: 12px;
      color: #303033;
      line-height: 1.58;
      white-space: pre-wrap;
    }
    .transcript-panel {
      overflow: hidden;
    }
    .transcript-head {
      height: 58px;
      padding: 0 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--soft-line);
    }
    .transcript-head h2 {
      margin: 0;
      font-size: 17px;
    }
    .transcript-title {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .transcript-actions {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .count-pill {
      min-width: 48px;
      height: 26px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: 8px;
      background: #f2f2f7;
      color: var(--muted);
      font-size: 12px;
    }
    .line-list {
      max-height: calc(100vh - 178px);
      overflow: auto;
      padding: 10px;
      display: grid;
      gap: 8px;
    }
    .transcript-panel.is-collapsed .line-list {
      display: none;
    }
    .line-item {
      width: 100%;
      min-height: 66px;
      text-align: left;
      border: 1px solid transparent;
      border-radius: 8px;
      background: #fbfbfd;
      padding: 10px 12px;
      cursor: pointer;
    }
    .line-item:hover {
      background: #f5f9ff;
      border-color: #c7def8;
    }
    .line-item.active {
      background: var(--accent-soft);
      border-color: #89bfff;
    }
    .line-time {
      margin-bottom: 5px;
      color: var(--muted);
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }
    .line-speaker {
      display: inline-flex;
      margin-right: 7px;
      color: var(--green);
      font-size: 12px;
      font-weight: 700;
    }
    .line-text {
      line-height: 1.48;
      font-size: 15px;
    }
    .translation-text {
      display: none;
      margin-top: 7px;
      color: #08705b;
      line-height: 1.48;
      font-size: 14px;
    }
    .show-translation .translation-text {
      display: block;
    }
    @media (max-width: 940px) {
      .intro,
      .study-layout {
        grid-template-columns: 1fr;
      }
      .player-panel {
        position: static;
      }
      .line-list {
        max-height: none;
      }
    }
    @media (max-width: 620px) {
      main {
        padding: 20px 14px 28px;
      }
      .intro {
        gap: 14px;
      }
      .header-inner {
        height: auto;
        min-height: 58px;
        align-items: flex-start;
        flex-direction: column;
        padding-top: 12px;
        padding-bottom: 12px;
      }
      .tool-nav {
        width: 100%;
        justify-content: flex-start;
      }
      .library-bar {
        grid-template-columns: 1fr;
      }
      .brand-subtitle {
        white-space: normal;
      }
      .current-line {
        min-height: 92px;
        font-size: 19px;
      }
      .assistant-actions {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <header class="site-header">
      <div class="header-inner">
        <a class="brand" href="/" aria-label="BriBa 首页">
          <span class="logo">__ICON__</span>
          <span>
            <span class="brand-title">BriBa</span>
            <span class="brand-subtitle">精听学习空间</span>
          </span>
        </a>
        <nav class="tool-nav" aria-label="材料工具">
          <a href="/new/">新建材料</a>
          <a href="/review/">校对材料</a>
          <a href="/library/">学习库</a>
          <a href="/mobile/">手机包</a>
        </nav>
      </div>
    </header>

    <main>
      <section class="intro" aria-label="开始学习">
        <div>
          <h1>回到这一句，慢慢听懂。</h1>
          <p>首页只保留学习体验。处理视频、校对字幕和导出手机包都在独立页面完成。</p>
        </div>
        <div class="library-bar">
          <div>
            <label for="project-select">选择学习库</label>
            <select id="project-select"></select>
          </div>
          <button class="primary" id="start-button">开始学习</button>
        </div>
      </section>

      <section id="study-root" class="empty">正在读取学习库...</section>
    </main>
  </div>

  <script>
    const state = {
      projects: [],
      project: null,
      items: [],
      active: -1,
      subtitleDelay: 0.4,
      showCurrentLine: true,
      showTranslation: false,
      transcriptCollapsed: false
    };
    const $ = (id) => document.getElementById(id);

    function escapeHtml(value) {
      return String(value || "").replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }[ch]));
    }
    function formatTime(value) {
      const total = Math.max(0, Math.floor(Number(value) || 0));
      const m = Math.floor(total / 60);
      const s = total % 60;
      return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
    }
    async function api(url, options) {
      const res = await fetch(url, options);
      if (!res.ok) throw new Error(await res.text());
      return await res.json();
    }
    async function loadProjects() {
      const data = await api("/api/projects");
      state.projects = data.projects || [];
      const select = $("project-select");
      select.innerHTML = "";
      state.projects.forEach((project) => {
        const option = document.createElement("option");
        option.value = project.id;
        option.textContent = `${project.created_at || ""} · ${project.title || project.source_name || project.id}`;
        select.appendChild(option);
      });
      if (!state.projects.length) {
        $("study-root").innerHTML = '<div class="empty">还没有学习库。请先进入“新建材料”生成一个学习库。</div>';
        return;
      }
      await loadStudy(select.value);
    }
    function renderStudy(data) {
      state.project = data.project;
      state.items = data.items || [];
      state.active = -1;
      $("study-root").innerHTML = `
        <div class="study-layout">
          <section class="panel player-panel">
            <div class="now-card">
              <div class="eyebrow">正在学习</div>
              <h2 class="study-title">${escapeHtml(data.project.title || "Untitled")}</h2>
              <div class="study-meta">${escapeHtml(data.project.source_name || "")}<br>${escapeHtml(data.project.created_at || "")}</div>
            </div>
            <div class="audio-wrap">
              <audio controls preload="metadata" src="${data.audio_url || ""}"></audio>
            </div>
            <div class="learning-options">
              <div class="option-group">
                <button class="chip-button active" data-action="toggle-line" title="显示或隐藏当前句字幕">显示单句</button>
                <button class="chip-button" data-action="toggle-translation" title="显示或隐藏中文翻译">显示中文</button>
                <button class="chip-button" data-action="generate-translation" title="使用本地 Ollama 生成中文翻译">生成中文翻译</button>
              </div>
              <div class="option-group" aria-label="字幕同步微调">
                <span>字幕延后</span>
                <button class="chip-button" data-action="sync-down" title="让字幕更早出现">-</button>
                <span class="sync-value">0.4s</span>
                <button class="chip-button" data-action="sync-up" title="让字幕更晚出现">+</button>
              </div>
            </div>
            <div class="current-line is-empty">点击右侧任意一句开始精听。</div>
            <div class="study-controls">
              <button class="icon-button" data-action="prev" title="上一句" aria-label="上一句">‹</button>
              <button class="icon-button" data-action="replay" title="重听本句" aria-label="重听本句">↺</button>
              <button class="icon-button" data-action="next" title="下一句" aria-label="下一句">›</button>
            </div>
            <section class="assistant">
              <div class="assistant-head">
                <h2>AI 学习助手</h2>
                <span>本地 Ollama</span>
              </div>
              <textarea id="assistant-question" rows="3" placeholder="例如：这一句有哪些地道表达？语法结构怎么理解？"></textarea>
              <div class="assistant-actions">
                <input id="assistant-model" value="qwen3.5:27b" aria-label="Ollama 模型" />
                <button class="primary" id="assistant-button">提问</button>
              </div>
              <div class="assistant-answer" id="assistant-answer">选择一句或输入问题，让 BriBa 帮你分析。</div>
            </section>
          </section>

          <section class="panel transcript-panel">
            <div class="transcript-head">
              <div class="transcript-title">
                <h2>字幕精听</h2>
                <span class="count-pill">${state.items.length} 句</span>
              </div>
              <div class="transcript-actions">
                <button class="chip-button" data-action="toggle-transcript">折叠字幕</button>
              </div>
            </div>
            <div class="line-list"></div>
          </section>
        </div>`;
      renderLines();
      const audio = document.querySelector("audio");
      audio.addEventListener("timeupdate", () => {
        const t = Math.max(0, (audio.currentTime || 0) - state.subtitleDelay);
        const index = state.items.findIndex((item) => t >= Number(item.start) && t <= Number(item.end));
        if (index >= 0) setActive(index, false);
      });
      document.querySelector('[data-action="prev"]').addEventListener("click", () => jumpTo(Math.max(0, state.active - 1), true));
      document.querySelector('[data-action="next"]').addEventListener("click", () => jumpTo(Math.min(state.items.length - 1, state.active + 1), true));
      document.querySelector('[data-action="replay"]').addEventListener("click", () => jumpTo(state.active >= 0 ? state.active : 0, true));
      document.querySelector('[data-action="toggle-line"]').addEventListener("click", toggleCurrentLine);
      document.querySelector('[data-action="toggle-translation"]').addEventListener("click", toggleTranslation);
      document.querySelector('[data-action="generate-translation"]').addEventListener("click", generateTranslation);
      document.querySelector('[data-action="toggle-transcript"]').addEventListener("click", toggleTranscript);
      document.querySelector('[data-action="sync-down"]').addEventListener("click", () => adjustDelay(-0.2));
      document.querySelector('[data-action="sync-up"]').addEventListener("click", () => adjustDelay(0.2));
      updateLearningOptions();
      $("assistant-button").addEventListener("click", askAssistant);
    }
    function renderLines() {
      const list = document.querySelector(".line-list");
      list.innerHTML = "";
      state.items.forEach((item, index) => {
        const button = document.createElement("button");
        button.className = "line-item";
        button.dataset.index = String(index);
        const speaker = item.speaker ? `<span class="line-speaker">${escapeHtml(item.speaker)}</span>` : "";
        button.innerHTML = `
          <div class="line-time">${formatTime(item.start)} - ${formatTime(item.end)}</div>
          <div class="line-text">${speaker}${escapeHtml(item.text)}</div>
          <div class="translation-text">${escapeHtml(item.translation || "")}</div>`;
        button.addEventListener("click", () => jumpTo(index, true));
        list.appendChild(button);
      });
    }
    function setActive(index, shouldScroll) {
      if (index < 0 || index >= state.items.length || index === state.active) return;
      state.active = index;
      document.querySelectorAll(".line-item").forEach((node, idx) => node.classList.toggle("active", idx === index));
      const current = document.querySelector(".current-line");
      current.classList.remove("is-empty");
      current.innerHTML = `
        <div>
          <div>${escapeHtml(state.items[index].text || "")}</div>
          <div class="translation-text">${escapeHtml(state.items[index].translation || "")}</div>
        </div>`;
      const node = document.querySelector(`.line-item[data-index="${index}"]`);
      if (shouldScroll && node) node.scrollIntoView({ block: "nearest" });
    }
    function jumpTo(index, play) {
      const item = state.items[index];
      const audio = document.querySelector("audio");
      if (!item || !audio) return;
      audio.currentTime = Math.max(0, Number(item.start) || 0);
      setActive(index, true);
      if (play) audio.play().catch(() => {});
    }
    function updateLearningOptions() {
      const line = document.querySelector(".current-line");
      const lineButton = document.querySelector('[data-action="toggle-line"]');
      if (line) line.classList.toggle("is-hidden", !state.showCurrentLine);
      if (lineButton) {
        lineButton.classList.toggle("active", state.showCurrentLine);
        lineButton.textContent = state.showCurrentLine ? "隐藏单句" : "显示单句";
      }
      document.querySelector(".study-layout")?.classList.toggle("show-translation", state.showTranslation);
      const translationButton = document.querySelector('[data-action="toggle-translation"]');
      if (translationButton) {
        translationButton.classList.toggle("active", state.showTranslation);
        translationButton.textContent = state.showTranslation ? "隐藏中文" : "显示中文";
      }
      const transcript = document.querySelector(".transcript-panel");
      const transcriptButton = document.querySelector('[data-action="toggle-transcript"]');
      if (transcript) transcript.classList.toggle("is-collapsed", state.transcriptCollapsed);
      if (transcriptButton) transcriptButton.textContent = state.transcriptCollapsed ? "展开字幕" : "折叠字幕";
      const syncValue = document.querySelector(".sync-value");
      if (syncValue) syncValue.textContent = `${state.subtitleDelay.toFixed(1)}s`;
    }
    function toggleCurrentLine() {
      state.showCurrentLine = !state.showCurrentLine;
      updateLearningOptions();
    }
    function toggleTranslation() {
      state.showTranslation = !state.showTranslation;
      updateLearningOptions();
    }
    async function generateTranslation() {
      const button = document.querySelector('[data-action="generate-translation"]');
      const previous = button ? button.textContent : "";
      if (button) button.textContent = "翻译中...";
      try {
        const data = await api("/api/translate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            project_id: $("project-select").value,
            model: $("assistant-model")?.value || "qwen3.5:27b",
            limit: 24
          })
        });
        state.items = data.items || state.items;
        state.showTranslation = true;
        renderLines();
        if (state.active >= 0) {
          const active = state.active;
          state.active = -1;
          setActive(active, false);
        }
        updateLearningOptions();
        if (button) {
          button.textContent = Number(data.remaining || 0) > 0 ? `继续翻译（剩 ${data.remaining}）` : "已生成中文";
        }
        if (data.errors && data.errors.length) {
          alert(`已保存部分翻译，但这一批超时或失败：${data.errors[0]}`);
        }
      } catch (error) {
        alert(`中文翻译失败：${error.message}`);
      } finally {
        if (button && button.textContent === "翻译中...") button.textContent = previous || "生成中文翻译";
      }
    }
    function toggleTranscript() {
      state.transcriptCollapsed = !state.transcriptCollapsed;
      updateLearningOptions();
    }
    function adjustDelay(delta) {
      state.subtitleDelay = Math.max(-1.5, Math.min(2.5, Number((state.subtitleDelay + delta).toFixed(1))));
      updateLearningOptions();
    }
    window.bribaStudy = {
      toggleCurrentLine,
      toggleTranscript,
      adjustDelay,
      getState: () => ({
        active: state.active,
        subtitleDelay: state.subtitleDelay,
        showCurrentLine: state.showCurrentLine,
        transcriptCollapsed: state.transcriptCollapsed
      })
    };
    async function loadStudy(projectId) {
      if (!projectId) return;
      $("study-root").innerHTML = '<div class="empty">正在打开学习库...</div>';
      renderStudy(await api(`/api/study/${encodeURIComponent(projectId)}`));
    }
    async function askAssistant() {
      const answer = $("assistant-answer");
      answer.textContent = "BriBa 正在思考...";
      try {
        const data = await api("/api/assistant", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            project_id: $("project-select").value,
            question: $("assistant-question").value,
            model: $("assistant-model").value
          })
        });
        answer.textContent = data.answer || "没有返回内容。";
      } catch (error) {
        answer.textContent = `AI 学习助手暂时不可用：${error.message}`;
      }
    }
    $("project-select").addEventListener("change", (event) => loadStudy(event.target.value));
    $("start-button").addEventListener("click", () => loadStudy($("project-select").value));
    loadProjects().catch((error) => {
      $("study-root").innerHTML = `<div class="empty">读取失败：${escapeHtml(error.message)}</div>`;
    });
  </script>
</body>
</html>
"""


class AssistantRequest(BaseModel):
    project_id: str
    question: str
    model: str = "qwen3.5:27b"


class TranslationRequest(BaseModel):
    project_id: str
    model: str = "qwen3.5:27b"
    limit: int = 24


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    icon_path = ASSET_DIR / "briba-icon.svg"
    if not icon_path.exists():
        icon_path.write_text(BRIBA_ICON_SVG, encoding="utf-8")
    if not PROJECTS_PATH.exists():
        PROJECTS_PATH.write_text("[]\n", encoding="utf-8")


def as_text(value: Any) -> str:
    return "" if value is None else str(value)


def log_app_error(message: str) -> None:
    ensure_dirs()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    previous = APP_LOG_PATH.read_text(encoding="utf-8") if APP_LOG_PATH.exists() else ""
    APP_LOG_PATH.write_text(previous + f"[{stamp}] {message}\n", encoding="utf-8")


def load_projects() -> list[dict[str, Any]]:
    ensure_dirs()
    try:
        payload = json.loads(PROJECTS_PATH.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        payload = []
    return payload if isinstance(payload, list) else []


def save_projects(projects: list[dict[str, Any]]) -> None:
    ensure_dirs()
    PROJECTS_PATH.write_text(json.dumps(projects, ensure_ascii=False, indent=2), encoding="utf-8")


def project_label(project: dict[str, Any]) -> str:
    title = project.get("title") or project.get("source_name") or project.get("id")
    return f"{project.get('created_at', '')} | {title}"


def project_is_available(project: dict[str, Any]) -> bool:
    return any(Path(path).exists() for path in project.get("files", []))


def available_projects() -> list[dict[str, Any]]:
    return [project for project in load_projects() if project_is_available(project)]


def project_choices() -> list[str]:
    recover_orphan_projects()
    return [
        project_label(project)
        for project in sorted(available_projects(), key=lambda item: item.get("created_at", ""), reverse=True)
    ]


def find_project(value: str) -> dict[str, Any] | None:
    for project in load_projects():
        if project.get("id") == value or project_label(project) == value:
            return project
    return None


def uploaded_file_path(uploaded_file) -> Path | None:
    if not uploaded_file:
        return None
    if isinstance(uploaded_file, str):
        return Path(uploaded_file)
    if isinstance(uploaded_file, dict):
        value = uploaded_file.get("path") or uploaded_file.get("name")
        return Path(value) if value else None
    value = getattr(uploaded_file, "path", None) or getattr(uploaded_file, "name", None)
    return Path(value) if value else None


def prepare_source(uploaded_file, download_url: str, session_dir: Path) -> Path:
    url = (download_url or "").strip()
    if url:
        return download_video(url, session_dir)
    source_path = uploaded_file_path(uploaded_file)
    if not source_path:
        raise PipelineError("请上传一个视频文件，或者输入一个视频下载链接。")
    copied_path = session_dir / source_path.name
    shutil.copy2(source_path.expanduser().resolve(), copied_path)
    return copied_path


def project_files(project: dict[str, Any] | None) -> list[Path]:
    if not project:
        return []
    return [Path(path) for path in project.get("files", []) if Path(path).exists()]


def first_file(project: dict[str, Any] | None, suffix: str) -> Path | None:
    for path in project_files(project):
        if path.name.endswith(suffix):
            return path
    return None


def preferred_file(project: dict[str, Any] | None, preferred_suffix: str, fallback_suffix: str) -> Path | None:
    files = project_files(project)
    return next((path for path in files if path.name.endswith(preferred_suffix)), None) or first_file(project, fallback_suffix)


def preferred_segments_path(project: dict[str, Any] | None) -> Path | None:
    return preferred_file(project, ".corrected.json", ".asr.json")


def translation_path(project: dict[str, Any] | None) -> Path | None:
    existing = first_file(project, ".translations.zh.json")
    if existing:
        return existing
    if project and project.get("project_dir"):
        return Path(project["project_dir"]) / "translations.zh.json"
    return None


def load_translations(project: dict[str, Any] | None) -> list[str]:
    path = translation_path(project)
    if not path or not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    output: list[str] = []
    for item in payload:
        if isinstance(item, dict):
            output.append(str(item.get("zh") or item.get("translation") or "").strip())
        else:
            output.append(str(item).strip())
    return output


def read_text(path: Path | None, max_chars: int = 12000) -> str:
    if not path or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[:max_chars]


def output_url(path: Path | None) -> str:
    if not path:
        return ""
    try:
        rel = path.resolve().relative_to(OUTPUT_DIR.resolve()).as_posix()
        return f"/output/{rel}"
    except ValueError:
        return ""


def srt_time_to_seconds(value: str) -> float:
    match = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)", value.strip())
    if not match:
        return 0.0
    hours, minutes, seconds, millis = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis[:3].ljust(3, "0")) / 1000


def parse_srt(text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for block in re.split(r"\n\s*\n", text.replace("\r", "\n")):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        time_line = next((line for line in lines if "-->" in line), "")
        if not time_line:
            continue
        start_raw, end_raw = [part.strip() for part in time_line.split("-->", 1)]
        text_start = lines.index(time_line) + 1
        items.append(
            {
                "start": srt_time_to_seconds(start_raw),
                "end": srt_time_to_seconds(end_raw),
                "text": " ".join(lines[text_start:]).strip(),
                "speaker": "",
            }
        )
    return items


def learning_items(project: dict[str, Any]) -> list[dict[str, Any]]:
    translations = load_translations(project)
    segment_path = preferred_segments_path(project)
    if segment_path:
        try:
            segments = read_segments_json(segment_path)
            return [
                {
                    "start": float(segment.get("start", 0.0)),
                    "end": float(segment.get("end", 0.0)),
                    "text": str(segment.get("text", "")).strip(),
                    "speaker": str(segment.get("speaker", "")).strip(),
                    "translation": str(segment.get("translation") or segment.get("zh") or (translations[index] if index < len(translations) else "")).strip(),
                }
                for index, segment in enumerate(segments)
                if str(segment.get("text", "")).strip()
            ]
        except Exception:
            pass
    srt_path = preferred_file(project, ".corrected.srt", ".asr.srt")
    items = parse_srt(read_text(srt_path, 200000))
    for index, item in enumerate(items):
        item["translation"] = translations[index] if index < len(translations) else ""
    return items


def project_learning_context(project: dict[str, Any] | None, max_chars: int = 7000) -> str:
    if not project:
        return ""
    transcript_path = preferred_file(project, ".corrected.txt", ".asr.txt")
    notes_path = first_file(project, ".learning.md")
    parts = []
    notes = read_text(notes_path, 2500)
    transcript = read_text(transcript_path, max_chars)
    if notes:
        parts.append(f"学习笔记：\n{notes}")
    if transcript:
        parts.append(f"字幕文本：\n{transcript}")
    if not parts:
        items = learning_items(project)
        parts.append("\n".join(item["text"] for item in items[:120]))
    return "\n\n".join(parts)[:max_chars]


def recover_orphan_projects() -> int:
    ensure_dirs()
    projects = load_projects()
    known_dirs = {str(Path(project.get("project_dir", "")).resolve()) for project in projects if project.get("project_dir")}
    recovered = 0
    for project_dir in sorted(OUTPUT_DIR.glob("briba-*")):
        if not project_dir.is_dir() or str(project_dir.resolve()) in known_dirs:
            continue
        asr_json = next(project_dir.glob("*.asr.json"), None)
        audio = next(project_dir.glob("*.listening.mp3"), None)
        if not asr_json or not audio:
            continue
        try:
            payload = json.loads(asr_json.read_text(encoding="utf-8-sig"))
        except Exception:
            payload = {}
        files = [str(path) for path in sorted(project_dir.iterdir()) if path.is_file()]
        source_video = next((path for path in project_dir.glob("*.mp4") if not path.name.endswith(".clip.mp4")), None)
        source_name = source_video.name if source_video else asr_json.name.replace(".asr.json", "")
        created_at = datetime.fromtimestamp(asr_json.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        projects.append(
            {
                "id": uuid.uuid4().hex[:12],
                "title": f"已恢复：{Path(source_name).stem}",
                "source_name": source_name,
                "created_at": created_at,
                "project_dir": str(project_dir),
                "start_time": "",
                "end_time": "",
                "whisper_model": payload.get("model", "large-v3"),
                "language": payload.get("language", "en"),
                "ollama_model": "qwen3.5:27b",
                "use_ollama": False,
                "files": files,
                "summary": "Recovered from generated files after the UI event did not save a project record.",
                "recovered": True,
            }
        )
        known_dirs.add(str(project_dir.resolve()))
        recovered += 1
    if recovered:
        save_projects(projects)
    return recovered


def save_project_record(
    title: str,
    source_path: Path,
    session_dir: Path,
    start_time: str,
    end_time: str,
    whisper_model: str,
    language: str,
    ollama_model: str,
    use_ollama: bool,
    files: list[str],
    summary: str,
) -> dict[str, Any]:
    projects = load_projects()
    project = {
        "id": uuid.uuid4().hex[:12],
        "title": as_text(title).strip() or source_path.stem,
        "source_name": source_path.name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "project_dir": str(session_dir),
        "start_time": as_text(start_time).strip(),
        "end_time": as_text(end_time).strip(),
        "whisper_model": whisper_model,
        "language": as_text(language).strip() or "en",
        "ollama_model": as_text(ollama_model).strip() or "qwen3.5:27b",
        "use_ollama": use_ollama,
        "files": files,
        "summary": summary,
    }
    projects.append(project)
    save_projects(projects)
    return project


def add_project_files(project_id: str, paths: list[Path], status: str | None = "corrected") -> None:
    projects = load_projects()
    for project in projects:
        if project.get("id") == project_id:
            current = list(project.get("files", []))
            for path in paths:
                value = str(path)
                if value not in current:
                    current.append(value)
            project["files"] = current
            project["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if status:
                project["correction_status"] = status
    save_projects(projects)


def corrected_base_path(project: dict[str, Any]) -> Path:
    asr_json = first_file(project, ".asr.json")
    if asr_json:
        return asr_json.with_suffix("")
    return Path(project["project_dir"]) / Path(str(project.get("source_name", "briba"))).stem


def dataframe_from_segments(segments: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(segments_to_rows(segments), columns=REVIEW_COLUMNS)


def create_learning_project(
    title: str,
    uploaded_file,
    download_url: str,
    start_time: str,
    end_time: str,
    whisper_model: str,
    language: str,
    use_ollama: bool,
    ollama_model: str,
) -> tuple[str, list[str]]:
    session_dir = Path(tempfile.mkdtemp(prefix="briba-", dir=str(OUTPUT_DIR)))
    try:
        video_path = prepare_source(uploaded_file, download_url, session_dir)
        result = process_learning_clip(
            video_path=video_path,
            output_dir=session_dir,
            model_name=as_text(whisper_model).strip() or "large-v3",
            language=as_text(language).strip() or "en",
            start=as_text(start_time).strip() or None,
            end=as_text(end_time).strip() or None,
            ollama_model=as_text(ollama_model).strip() or "qwen3.5:27b",
            skip_ollama=not use_ollama,
        )
        files = [str(path) for path in result.created_files if path.exists()]
        project = save_project_record(
            as_text(title),
            video_path,
            session_dir,
            as_text(start_time),
            as_text(end_time),
            as_text(whisper_model).strip() or "large-v3",
            as_text(language).strip() or "en",
            as_text(ollama_model).strip() or "qwen3.5:27b",
            use_ollama,
            files,
            result.summary,
        )
        return f"创建成功：{project.get('title')}。回到首页即可开始学习。", files
    except Exception as exc:
        log_app_error(f"create_learning_project failed in {session_dir}: {exc}")
        return f"处理失败：{exc}", []


def prepare_manual_review(label: str) -> tuple[pd.DataFrame, str]:
    project = find_project(label)
    path = preferred_segments_path(project)
    if not project or not path:
        return pd.DataFrame(columns=REVIEW_COLUMNS), "没有找到可校对的字幕数据。"
    segments = read_segments_json(path)
    cleaned = auto_cleanup_segments(segments, max_words=16)
    return dataframe_from_segments(cleaned), f"已进入人工校对：{len(cleaned)} 行。主要改 speaker 和 text，确认后保存。"


def prepare_ai_review(
    label: str,
    model: str,
    character_hints: str,
    context: str,
    max_rows: int,
) -> tuple[pd.DataFrame, str]:
    project = find_project(label)
    path = preferred_segments_path(project)
    if not project or not path:
        return pd.DataFrame(columns=REVIEW_COLUMNS), "没有找到可校对的字幕数据。"
    segments = auto_cleanup_segments(read_segments_json(path), max_words=16)
    optimized, messages = ai_optimize_segments(
        segments,
        model=model.strip() or "qwen3.5:27b",
        character_hints=character_hints,
        context=context,
        chunk_size=12,
        max_rows=int(max_rows),
    )
    return dataframe_from_segments(optimized), "AI 智能校对完成。请检查表格，确认后保存。\n" + "\n".join(messages[-8:])


def save_review(label: str, rows) -> tuple[str, list[str]]:
    project = find_project(label)
    if not project:
        return "请先选择学习库。", []
    segments = rows_to_segments(rows)
    if not segments:
        return "没有可保存的校对内容。", []
    paths = write_corrected_outputs(
        corrected_base_path(project),
        segments,
        {
            "source_project_id": project.get("id"),
            "source_name": project.get("source_name"),
            "corrected_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    add_project_files(str(project.get("id")), paths)
    return f"已保存校对结果：{', '.join(path.name for path in paths)}", [str(path) for path in paths]


def refresh_dropdown() -> gr.Dropdown:
    choices = project_choices()
    return gr.update(choices=choices, value=choices[0] if choices else None)


def library_summary(label: str) -> tuple[str, list[str]]:
    project = find_project(label)
    if not project:
        return "请选择一个学习库。", []
    files = project_files(project)
    items = learning_items(project)
    audio = first_file(project, ".listening.mp3")
    body = f"""
## {project.get("title", "Untitled")}

来源：`{project.get("source_name", "")}`  
时间：`{project.get("start_time") or "视频开头"}` 到 `{project.get("end_time") or "视频结尾"}`  
字幕：{len(items)} 句  
音频：{"已生成" if audio else "未生成"}  
校对：`{project.get("correction_status", "未校对")}`
""".strip()
    return body, [str(path) for path in files]


def export_pack(title: str) -> tuple[str, str | None]:
    try:
        pack_path = export_mobile_pack(title=title.strip() or "BriBa Learning Pack")
    except Exception as exc:
        return f"导出失败：{exc}", None
    return f"已导出手机学习包：{pack_path}", str(pack_path)


def extract_json_list(text: str) -> list[Any]:
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
    return payload


def save_project_translations(project: dict[str, Any], translations: list[str]) -> Path:
    output = [{"index": index, "zh": value} for index, value in enumerate(translations)]
    path = translation_path(project)
    if not path:
        raise HTTPException(status_code=500, detail="Cannot decide translation output path")
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    add_project_files(str(project.get("id", "")), [path], status=None)
    return path


def translate_project_lines(project_id: str, model: str, limit: int = 24) -> dict[str, Any]:
    project = find_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    items = learning_items(project)
    if not items:
        raise HTTPException(status_code=400, detail="No subtitle items to translate")

    model_name = as_text(model).strip() or "qwen3.5:27b"
    translations = [str(item.get("translation", "")).strip() for item in items]
    missing_indexes = [index for index, value in enumerate(translations) if not value][: max(1, min(60, int(limit or 24)))]
    errors: list[str] = []

    for start in range(0, len(missing_indexes), 6):
        chunk_indexes = missing_indexes[start : start + 6]
        rows = [{"index": index, "text": items[index].get("text", "")} for index in chunk_indexes]
        prompt = f"""
你是 BriBa 的字幕翻译助手。
请把英文字幕逐句翻译成自然、简洁的中文，供中文母语者学习英语时参考。

要求：
- 只返回 JSON 数组，不要 Markdown，不要解释。
- 保持 index 不变。
- 不要改写英文原句。
- 中文翻译要贴近语境，但不要添加字幕里没有的信息。
- 如果一句话是人名、称呼或残句，也给出最自然的中文理解。

字幕：
{json.dumps(rows, ensure_ascii=False, indent=2)}

返回格式：
[
  {{"index": 0, "zh": "中文翻译"}}
]
""".strip()
        try:
            response = requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "top_p": 0.8, "num_predict": 900},
                },
                timeout=90,
            )
            response.raise_for_status()
            ai_rows = extract_json_list(str(response.json().get("response", "")))
            for row in ai_rows:
                if not isinstance(row, dict):
                    continue
                index = int(row.get("index", -1))
                if 0 <= index < len(translations):
                    translations[index] = str(row.get("zh") or row.get("translation") or "").strip()
        except Exception as exc:
            errors.append(f"{chunk_indexes[0] + 1}-{chunk_indexes[-1] + 1}: {exc}")
            break

    path = save_project_translations(project, translations)

    updated_items = learning_items(find_project(project_id) or project)
    remaining = sum(1 for value in translations if not value)
    translated = len(translations) - remaining
    return {
        "items": updated_items,
        "translation_file": str(path),
        "translated": translated,
        "remaining": remaining,
        "errors": errors,
    }


def assistant_answer(project_id: str, question: str, model: str) -> str:
    project = find_project(project_id)
    question = as_text(question).strip()
    if not project:
        return "请先选择一个学习库。"
    if not question:
        return "你可以问：这段对话里有哪些值得精听的表达？或者某一句语法为什么这样说？"
    context = project_learning_context(project)
    if not context:
        return "这个学习库还没有可供分析的字幕文本。"
    prompt = f"""
你是 BriBa 的英语学习助手，面向中文母语学习者。
请基于当前学习库的字幕和笔记回答用户问题。

要求：
- 用中文解释，必要时保留英文原句。
- 不要编造字幕里没有的剧情或台词。
- 如果问题适合精听训练，给出 2-4 个可执行练习。
- 回答要短而清楚。

当前学习库：{project.get("title") or project.get("source_name")}

学习材料：
{context}

用户问题：
{question}
""".strip()
    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={"model": as_text(model).strip() or "qwen3.5:27b", "prompt": prompt, "stream": False},
            timeout=180,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return f"AI 学习助手暂时不可用：{exc}\n请确认 Ollama 正在运行，并且模型已经安装。"
    return str(response.json().get("response", "")).strip() or "AI 没有返回内容。"


def new_material_ui() -> gr.Blocks:
    with gr.Blocks(title="BriBa - 新建材料") as demo:
        gr.Markdown("# 新建材料\n生成学习库后，回到 [学习首页](/) 开始学习。")
        with gr.Row():
            file_input = gr.File(label="上传本地视频", file_count="single", type="filepath")
            with gr.Column():
                title = gr.Textbox(label="学习库名称", placeholder="例如 English Drama Clip 01")
                download_url = gr.Textbox(label="视频下载链接，可留空")
                with gr.Row():
                    start_time = gr.Textbox(value="1分15s", label="开始时间")
                    end_time = gr.Textbox(value="10分55s", label="结束时间")
                with gr.Row():
                    whisper_model = gr.Dropdown(["large-v3", "medium", "small", "base"], value="large-v3", label="转录模型")
                    language = gr.Textbox(value="en", label="语言")
                with gr.Accordion("学习笔记，可选", open=False):
                    use_ollama = gr.Checkbox(value=False, label="创建时同时生成学习笔记")
                    ollama_model = gr.Textbox(value="qwen3.5:27b", label="Ollama 模型")
                create_button = gr.Button("生成学习库", variant="primary")
        status = gr.Textbox(label="状态", lines=3)
        files = gr.Files(label="生成文件")
        create_button.click(
            fn=create_learning_project,
            inputs=[title, file_input, download_url, start_time, end_time, whisper_model, language, use_ollama, ollama_model],
            outputs=[status, files],
        )
    return demo


def review_ui() -> gr.Blocks:
    choices = project_choices()
    initial = choices[0] if choices else None
    with gr.Blocks(title="BriBa - 校对材料") as demo:
        gr.Markdown("# 校对材料\n保存后，学习首页会自动使用校对后的字幕。")
        with gr.Row():
            selector = gr.Dropdown(choices=choices, value=initial, label="选择学习库", interactive=True)
            refresh = gr.Button("刷新")
        with gr.Row():
            manual_button = gr.Button("人工校对", variant="secondary")
            ai_button = gr.Button("AI 智能校对", variant="primary")
        with gr.Accordion("AI 智能校对设置", open=False):
            ai_model = gr.Textbox(value="qwen3.5:27b", label="AI 模型")
            ai_max_rows = gr.Slider(0, 300, value=80, step=10, label="AI 最多处理行数，0 表示全部")
            character_hints = gr.Textbox(label="角色提示，可选", placeholder="Character A, Character B, Teacher, Student")
            ai_context = gr.Textbox(label="场景提示，可选", lines=2, placeholder="British detective drama dialogue. Remove ads and ASR repetition.")
        status = gr.Textbox(label="状态", lines=4)
        table = gr.Dataframe(
            headers=REVIEW_COLUMNS,
            datatype=["number", "number", "number", "str", "str", "str", "str"],
            row_count=(8, "dynamic"),
            column_count=(7, "fixed"),
            interactive=True,
            label="校对表格：重点修改 speaker 和 text",
        )
        save_button = gr.Button("保存校对结果", variant="primary")
        files = gr.Files(label="校对输出文件")
        refresh.click(fn=refresh_dropdown, outputs=[selector])
        manual_button.click(fn=prepare_manual_review, inputs=[selector], outputs=[table, status])
        ai_button.click(fn=prepare_ai_review, inputs=[selector, ai_model, character_hints, ai_context, ai_max_rows], outputs=[table, status])
        save_button.click(fn=save_review, inputs=[selector, table], outputs=[status, files])
    return demo


def library_ui() -> gr.Blocks:
    choices = project_choices()
    initial = choices[0] if choices else None
    with gr.Blocks(title="BriBa - 学习库") as demo:
        gr.Markdown("# 学习库管理\n这里只负责查看材料和文件；真正学习请回到 [学习首页](/)。")
        with gr.Row():
            selector = gr.Dropdown(choices=choices, value=initial, label="历史学习库", interactive=True)
            refresh = gr.Button("刷新")
            open_button = gr.Button("查看信息", variant="secondary")
        result = gr.Markdown(label="学习库信息")
        files = gr.Files(label="文件")
        refresh.click(fn=refresh_dropdown, outputs=[selector])
        open_button.click(fn=library_summary, inputs=[selector], outputs=[result, files])
        selector.change(fn=library_summary, inputs=[selector], outputs=[result, files])
    return demo


def mobile_ui() -> gr.Blocks:
    with gr.Blocks(title="BriBa - 手机包") as demo:
        gr.Markdown("# 手机包\n校对完成后再导出手机包。手机端会使用同一套“选择学习库 -> 开始学习”的逻辑。")
        pack_title = gr.Textbox(value="BriBa Learning Pack", label="学习包标题")
        export_button = gr.Button("导出 BriBa Pack", variant="primary")
        export_status = gr.Textbox(label="导出结果", lines=3)
        export_file = gr.File(label="下载手机学习包")
        export_button.click(fn=export_pack, inputs=[pack_title], outputs=[export_status, export_file])
    return demo


def create_app() -> FastAPI:
    ensure_dirs()
    app = FastAPI(title="BriBa")
    app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
    app.mount("/assets", StaticFiles(directory=str(ASSET_DIR)), name="assets")

    @app.get("/", response_class=HTMLResponse)
    def root() -> str:
        return ROOT_HTML.replace("__ICON__", BRIBA_ICON_SVG)

    @app.get("/api/projects")
    def api_projects() -> dict[str, Any]:
        projects = []
        for project in sorted(available_projects(), key=lambda item: item.get("created_at", ""), reverse=True):
            projects.append(
                {
                    "id": project.get("id"),
                    "title": project.get("title") or project.get("source_name"),
                    "source_name": project.get("source_name", ""),
                    "created_at": project.get("created_at", ""),
                }
            )
        return {"projects": projects}

    @app.get("/api/study/{project_id}")
    def api_study(project_id: str) -> dict[str, Any]:
        project = find_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return {
            "project": {
                "id": project.get("id"),
                "title": project.get("title") or project.get("source_name"),
                "source_name": project.get("source_name", ""),
                "created_at": project.get("created_at", ""),
                "start_time": project.get("start_time", ""),
                "end_time": project.get("end_time", ""),
            },
            "audio_url": output_url(first_file(project, ".listening.mp3")),
            "items": learning_items(project),
        }

    @app.post("/api/assistant")
    def api_assistant(payload: AssistantRequest) -> dict[str, str]:
        return {"answer": assistant_answer(payload.project_id, payload.question, payload.model)}

    @app.post("/api/translate")
    def api_translate(payload: TranslationRequest) -> dict[str, Any]:
        return translate_project_lines(payload.project_id, payload.model, payload.limit)

    gr.mount_gradio_app(app, new_material_ui(), path="/new")
    gr.mount_gradio_app(app, review_ui(), path="/review")
    gr.mount_gradio_app(app, library_ui(), path="/library")
    gr.mount_gradio_app(app, mobile_ui(), path="/mobile")
    return app


if __name__ == "__main__":
    uvicorn.run(create_app(), host="127.0.0.1", port=7860)
