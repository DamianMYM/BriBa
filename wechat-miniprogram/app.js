App({
  onLaunch() {
    const packRoot = wx.getStorageSync("bribaPackRoot")
    const manifest = wx.getStorageSync("bribaManifest")
    if (packRoot && manifest) {
      this.globalData.packRoot = packRoot
      this.globalData.manifest = manifest
    }
  },

  globalData: {
    packRoot: "",
    manifest: null
  }
})
