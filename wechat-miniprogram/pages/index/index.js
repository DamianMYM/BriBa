const app = getApp()
const fs = wx.getFileSystemManager()

function readJson(path) {
  const text = fs.readFileSync(path, "utf8")
  return JSON.parse(text)
}

Page({
  data: {
    projects: []
  },

  onShow() {
    if (app.globalData.manifest) {
      this.setData({ projects: app.globalData.manifest.projects || [] })
    }
  },

  choosePack() {
    wx.chooseMessageFile({
      count: 1,
      type: "file",
      extension: ["zip", "briba"],
      success: (res) => {
        const file = res.tempFiles[0]
        const packRoot = `${wx.env.USER_DATA_PATH}/briba-pack`
        try {
          try {
            fs.rmdirSync(packRoot, true)
          } catch (error) {}
          fs.mkdirSync(packRoot, true)
          fs.unzip({
            zipFilePath: file.path,
            targetPath: packRoot,
            success: () => {
              const manifest = readJson(`${packRoot}/manifest.json`)
              app.globalData.packRoot = packRoot
              app.globalData.manifest = manifest
              wx.setStorageSync("bribaPackRoot", packRoot)
              wx.setStorageSync("bribaManifest", manifest)
              this.setData({ projects: manifest.projects || [] })
              wx.showToast({ title: "导入成功", icon: "success" })
            },
            fail: (error) => {
              wx.showModal({ title: "解压失败", content: error.errMsg, showCancel: false })
            }
          })
        } catch (error) {
          wx.showModal({ title: "导入失败", content: error.message, showCancel: false })
        }
      }
    })
  },

  continueLast() {
    if (!this.data.projects.length) {
      this.choosePack()
      return
    }
    wx.navigateTo({ url: "/pages/project/project?index=0" })
  },

  openProject(event) {
    const index = event.currentTarget.dataset.index
    wx.navigateTo({ url: `/pages/project/project?index=${index}` })
  },

  clearPack() {
    wx.removeStorageSync("bribaPackRoot")
    wx.removeStorageSync("bribaManifest")
    app.globalData.packRoot = ""
    app.globalData.manifest = null
    this.setData({ projects: [] })
    wx.showToast({ title: "已清空", icon: "success" })
  }
})
