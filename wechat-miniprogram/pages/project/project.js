const app = getApp()
const fs = wx.getFileSystemManager()

let audio = null

function readText(relativePath) {
  if (!relativePath) return ""
  try {
    return fs.readFileSync(`${app.globalData.packRoot}/${relativePath}`, "utf8")
  } catch (error) {
    return ""
  }
}

function readJson(relativePath) {
  const text = readText(relativePath)
  if (!text) return null
  try {
    return JSON.parse(text)
  } catch (error) {
    return null
  }
}

function readTranslations(relativePath) {
  const payload = readJson(relativePath)
  if (!Array.isArray(payload)) return []
  return payload.map((item) => {
    if (typeof item === "string") return item
    if (item && typeof item === "object") return item.zh || item.translation || ""
    return ""
  })
}

function parseSrtTime(value) {
  const match = value.trim().match(/(\d+):(\d+):(\d+),(\d+)/)
  if (!match) return 0
  const [, hours, minutes, seconds, ms] = match
  return Number(hours) * 3600 + Number(minutes) * 60 + Number(seconds) + Number(ms.slice(0, 3).padEnd(3, "0")) / 1000
}

function parseSrt(text) {
  return text
    .replace(/\r/g, "")
    .split(/\n\s*\n/)
    .map((block) => {
      const lines = block.split("\n").filter(Boolean)
      const timeLine = lines.find((line) => line.includes("-->"))
      if (!timeLine) return null
      const [startRaw, endRaw] = timeLine.split("-->").map((item) => item.trim())
      const textLines = lines.slice(lines.indexOf(timeLine) + 1)
      return {
        start: parseSrtTime(startRaw),
        end: parseSrtTime(endRaw),
        startLabel: startRaw.replace(",", "."),
        endLabel: endRaw.replace(",", "."),
        text: textLines.join(" ")
      }
    })
    .filter(Boolean)
}

Page({
  data: {
    project: {},
    notes: "",
    notesKey: "",
    subtitleItems: [],
    currentSubtitle: {},
    currentSubtitleIndex: -1,
    playing: false,
    showCurrentLine: true,
    showTranslation: false,
    transcriptCollapsed: false,
    notesCollapsed: true,
    subtitleDelay: 0.4,
    subtitleDelayLabel: "0.4s"
  },

  onLoad(query) {
    const index = Number(query.index || 0)
    const manifest = app.globalData.manifest
    if (!manifest || !manifest.projects || !manifest.projects[index]) {
      wx.showModal({ title: "未找到学习库", content: "请先导入 BriBa Pack。", showCancel: false })
      return
    }

    const project = manifest.projects[index]
    wx.setNavigationBarTitle({ title: project.title || "BriBa" })
    const subtitles = readText(project.subtitles)
    const subtitleItems = parseSrt(subtitles)
    const translations = readTranslations(project.translations)
    subtitleItems.forEach((item, itemIndex) => {
      item.translation = translations[itemIndex] || ""
    })
    const notesPath = project.notes && project.notes !== project.transcript ? project.notes : ""
    const notesKey = `bribaNotes:${project.id || project.title || index}`
    const savedNotes = wx.getStorageSync(notesKey)
    const initialNotes = typeof savedNotes === "string" ? savedNotes : readText(notesPath)
    this.setData({
      project,
      notes: initialNotes,
      notesKey,
      subtitleItems
    })

    audio = wx.createInnerAudioContext()
    audio.src = `${app.globalData.packRoot}/${project.audio}`
    audio.onTimeUpdate(() => {
      const currentTime = Math.max(0, (audio.currentTime || 0) - this.data.subtitleDelay)
      const activeIndex = subtitleItems.findIndex((item) => currentTime >= item.start && currentTime <= item.end)
      if (activeIndex >= 0 && activeIndex !== this.data.currentSubtitleIndex) {
        this.setActiveLine(activeIndex)
      }
    })
    audio.onEnded(() => this.setData({ playing: false }))
  },

  onUnload() {
    if (audio) {
      audio.destroy()
      audio = null
    }
  },

  setActiveLine(index) {
    const item = this.data.subtitleItems[index]
    if (!item) return
    this.setData({
      currentSubtitleIndex: index,
      currentSubtitle: item
    })
  },

  jumpTo(index, shouldPlay = true) {
    const item = this.data.subtitleItems[index]
    if (!audio || !item) return
    audio.seek(Math.max(0, item.start))
    this.setActiveLine(index)
    if (shouldPlay) {
      audio.play()
      this.setData({ playing: true })
    }
  },

  jumpToLine(event) {
    this.jumpTo(Number(event.currentTarget.dataset.index || 0), true)
  },

  prevLine() {
    this.jumpTo(Math.max(0, this.data.currentSubtitleIndex - 1), true)
  },

  replayLine() {
    this.jumpTo(this.data.currentSubtitleIndex >= 0 ? this.data.currentSubtitleIndex : 0, true)
  },

  nextLine() {
    this.jumpTo(Math.min(this.data.subtitleItems.length - 1, this.data.currentSubtitleIndex + 1), true)
  },

  toggleAudio() {
    if (!audio) return
    if (this.data.playing) {
      audio.pause()
      this.setData({ playing: false })
    } else {
      if (this.data.currentSubtitleIndex < 0 && this.data.subtitleItems.length) {
        this.setActiveLine(0)
      }
      audio.play()
      this.setData({ playing: true })
    }
  },

  toggleCurrentLine() {
    this.setData({ showCurrentLine: !this.data.showCurrentLine })
  },

  toggleTranslation() {
    this.setData({ showTranslation: !this.data.showTranslation })
  },

  toggleTranscript() {
    this.setData({ transcriptCollapsed: !this.data.transcriptCollapsed })
  },

  toggleNotes() {
    this.setData({ notesCollapsed: !this.data.notesCollapsed })
  },

  onNotesInput(event) {
    const value = event.detail.value || ""
    this.setData({ notes: value })
    if (this.data.notesKey) {
      wx.setStorageSync(this.data.notesKey, value)
    }
  },

  updateDelay(delta) {
    const next = Math.max(-1.5, Math.min(2.5, Number((this.data.subtitleDelay + delta).toFixed(1))))
    this.setData({
      subtitleDelay: next,
      subtitleDelayLabel: `${next.toFixed(1)}s`
    })
  },

  delayDown() {
    this.updateDelay(-0.2)
  },

  delayUp() {
    this.updateDelay(0.2)
  }
})
