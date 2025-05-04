// app.js
const modelService = require('./utils/modelService');

App({
  onLaunch() {
    // 初始化日志
    const logs = wx.getStorageSync('logs') || []
    logs.unshift(Date.now())
    wx.setStorageSync('logs', logs)

    // 初始化模型
    this.initializeModel();
  },

  initializeModel() {
    // 显示加载指示器
    let loadingShown = false;
    try {
      wx.showLoading({
        title: '连接后端服务...',
      });
      loadingShown = true;
    } catch (error) {
      console.error('显示加载指示器失败:', error);
    }

    // 检查后端API连接
    modelService.initModel()
      .then(result => {
        console.log('后端API连接成功:', result);
        this.globalData.modelReady = true;
        this.globalData.modelInfo = result;
        
        // 如果有等待API连接的回调，调用它们
        if (this.modelReadyCallback) {
          this.modelReadyCallback(result);
        }
      })
      .catch(error => {
        console.error('后端API连接失败:', error);
        this.globalData.modelReady = false;
        
        wx.showToast({
          title: '后端服务连接失败',
          icon: 'none'
        });
      })
      .finally(() => {
        // 确保只有在showLoading成功时才调用hideLoading
        if (loadingShown) {
          try {
            wx.hideLoading();
          } catch (error) {
            console.error('隐藏加载指示器失败:', error);
          }
        }
      });
  },

  globalData: {
    userInfo: null,
    modelReady: false,
    modelInfo: null
  }
})
