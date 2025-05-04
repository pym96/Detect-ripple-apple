// realtime.js
const modelService = require('../../utils/modelService');

// 类别颜色映射
const CLASS_COLORS = {
  0: '#4CAF50', // 100%成熟度 - 绿色
  1: '#4CAF50', // 50%成熟度 - 绿色（原为黄色，现统一为健康绿色）
  2: '#4CAF50', // 75%成熟度 - 绿色
  3: '#F44336'  // 腐烂苹果 - 红色
};

Page({
  data: {
    cameraContext: null,
    cameraPosition: 'back',
    isProcessing: false,
    isDetecting: false,
    detectionResult: null,
    detectionFrequency: 500, // 检测频率（毫秒）
    imageInfo: {
      width: 0,
      height: 0
    },
    // 界面状态
    overlayVisible: true,
    boxesInfo: [],
    fpsCounter: 0,
    fps: 0,
    lastFpsUpdate: 0,
    processingTimeAvg: 0,
    processingTimeAvgText: '0', // 预计算的处理时间文本
    processingTimes: [],
    detectionMode: 'continuous', // 'continuous' 或 'onDemand'
    noDetection: false, // 新增：表示是否没有检测到对象
    noDetectionCount: 0, // 新增：连续无检测的次数计数
    showNoDetectionMessage: false // 新增：是否显示无检测提示
  },

  onLoad() {
    console.log('实时检测页面加载');
    
    // 延迟初始化相机上下文，避免可能的渲染层错误
    setTimeout(() => {
      this.setData({
        cameraContext: wx.createCameraContext()
      });
      
      // 检查模型是否准备就绪
      const app = getApp();
      if (!app.globalData.modelReady) {
        wx.showLoading({
          title: '模型加载中...',
        });
        
        app.modelReadyCallback = () => {
          wx.hideLoading();
          console.log('模型准备就绪');
        };
      }
    }, 300);
    
    // 设置计算FPS的定时器
    this.fpsInterval = setInterval(() => {
      const now = Date.now();
      const elapsed = now - this.data.lastFpsUpdate;
      if (elapsed >= 1000) { // 每秒更新一次
        this.setData({
          fps: Math.round(this.data.fpsCounter * 1000 / elapsed),
          fpsCounter: 0,
          lastFpsUpdate: now
        });
      }
    }, 1000);
  },
  
  onUnload() {
    // 停止检测
    this.stopDetection();
    
    // 清除定时器
    if (this.fpsInterval) {
      clearInterval(this.fpsInterval);
    }
    if (this.detectionInterval) {
      clearInterval(this.detectionInterval);
    }
  },
  
  // 开始实时检测
  startDetection() {
    if (this.data.isDetecting) return;
    
    this.setData({
      isDetecting: true,
      boxesInfo: [],
      noDetection: false,
      noDetectionCount: 0,
      showNoDetectionMessage: false
    });
    
    // 连续检测模式
    if (this.data.detectionMode === 'continuous') {
      this.detectionInterval = setInterval(() => {
        if (!this.data.isProcessing) {
          this.detectFrame();
        }
      }, this.data.detectionFrequency);
    } else {
      // 单次检测（用于触发按钮模式）
      this.detectFrame();
    }
  },
  
  // 停止实时检测
  stopDetection() {
    if (this.detectionInterval) {
      clearInterval(this.detectionInterval);
    }
    
    this.setData({
      isDetecting: false,
      boxesInfo: [],
      noDetection: false,
      showNoDetectionMessage: false
    });
  },
  
  // 捕获和检测单帧
  detectFrame() {
    if (this.data.isProcessing || !this.data.cameraContext) return;
    
    this.setData({ isProcessing: true });
    
    // 捕获当前帧
    const context = this.data.cameraContext;
    context.takePhoto({
      quality: 'normal', // 使用较低质量以提高性能
      success: (res) => {
        this.processFrame(res.tempImagePath);
      },
      fail: (error) => {
        console.error('捕获帧失败:', error);
        this.setData({ isProcessing: false });
        
        // 如果是单次检测，需要重置检测状态
        if (this.data.detectionMode === 'onDemand') {
          this.setData({ isDetecting: false });
        }
      }
    });
  },
  
  // 生成检测框信息
  generateBoxesInfo(predictions, imageWidth, imageHeight) {
    const boxesInfo = [];
    
    // 检查predictions是否为数组
    if (Array.isArray(predictions) && predictions.length > 0) {
      // 处理数组形式的预测结果
      predictions.forEach((prediction, index) => {
        // 为检测框生成随机ID，用于动画和跟踪
        const boxId = 'box_' + Date.now() + '_' + index;
        const centerX = prediction.bbox.x * imageWidth;
        const centerY = prediction.bbox.y * imageHeight;
        const boxWidth = prediction.bbox.w * imageWidth;
        const boxHeight = prediction.bbox.h * imageHeight;
        
        // 计算左上角坐标
        const left = Math.max(0, centerX - boxWidth / 2);
        const top = Math.max(0, centerY - boxHeight / 2);
        
        // 确保边界框不超出图像范围
        const adjustedWidth = Math.min(boxWidth, imageWidth - left);
        const adjustedHeight = Math.min(boxHeight, imageHeight - top);
        
        // 预计算置信度百分比
        const confidencePercent = Math.round(prediction.confidence * 100);
        
        boxesInfo.push({
          id: boxId,
          label: prediction.displayLabel,
          confidence: prediction.confidence,
          confidencePercent: confidencePercent, // 预计算置信度百分比
          color: CLASS_COLORS[prediction.class] || '#2196F3',
          style: `left:${left}px; top:${top}px; width:${adjustedWidth}px; height:${adjustedHeight}px`
        });
      });
    } else if (predictions) {
      // 处理单个预测结果对象
      const prediction = predictions;
      const boxId = 'box_' + Date.now();
      const centerX = prediction.bbox.x * imageWidth;
      const centerY = prediction.bbox.y * imageHeight;
      const boxWidth = prediction.bbox.w * imageWidth;
      const boxHeight = prediction.bbox.h * imageHeight;
      
      // 计算左上角坐标
      const left = Math.max(0, centerX - boxWidth / 2);
      const top = Math.max(0, centerY - boxHeight / 2);
      
      // 确保边界框不超出图像范围
      const adjustedWidth = Math.min(boxWidth, imageWidth - left);
      const adjustedHeight = Math.min(boxHeight, imageHeight - top);
      
      // 预计算置信度百分比
      const confidencePercent = Math.round(prediction.confidence * 100);
      
      boxesInfo.push({
        id: boxId,
        label: prediction.displayLabel || prediction.friendlyLabel,
        confidence: prediction.confidence,
        confidencePercent: confidencePercent, // 预计算置信度百分比
        color: CLASS_COLORS[prediction.class] || '#2196F3',
        style: `left:${left}px; top:${top}px; width:${adjustedWidth}px; height:${adjustedHeight}px`
      });
    }
    
    return boxesInfo;
  },
  
  // 处理捕获的帧
  processFrame(imagePath) {
    const startTime = Date.now();
    
    // 首先获取图像信息
    wx.getImageInfo({
      src: imagePath,
      success: (imgInfo) => {
        this.setData({
          imageInfo: {
            width: imgInfo.width,
            height: imgInfo.height
          }
        });
        
        // 使用模型服务处理图像
        modelService.processImage(imagePath)
          .then(result => {
            // 增加FPS计数
            const endTime = Date.now();
            const processingTime = endTime - startTime;
            
            // 更新处理时间统计
            const processingTimes = [...this.data.processingTimes, processingTime];
            if (processingTimes.length > 10) {
              processingTimes.shift(); // 保持最近10次的记录
            }
            const processingTimeAvg = processingTimes.reduce((a, b) => a + b, 0) / processingTimes.length;
            const processingTimeAvgText = Math.round(processingTimeAvg).toString();
            
            // 生成界面显示的检测框信息
            let boxesInfo = [];
            let noDetection = false;
            let noDetectionCount = this.data.noDetectionCount;
            let showNoDetectionMessage = this.data.showNoDetectionMessage;
            
            if (!result) {
              // 没有检测到苹果
              noDetection = true;
              noDetectionCount += 1;
              
              // 如果连续5次以上没有检测到，显示提示消息
              if (noDetectionCount >= 5) {
                showNoDetectionMessage = true;
              }
            } else {
              // 有检测结果，重置无检测计数
              noDetection = false;
              noDetectionCount = 0;
              showNoDetectionMessage = false;
              
              // 处理检测结果
              boxesInfo = this.generateBoxesInfo(result.predictions, imgInfo.width, imgInfo.height);
            }
            
            // 更新状态
            this.setData({
              detectionResult: result,
              boxesInfo: boxesInfo,
              isProcessing: false,
              fpsCounter: this.data.fpsCounter + 1,
              processingTimes: processingTimes,
              processingTimeAvg: processingTimeAvg,
              processingTimeAvgText: processingTimeAvgText,
              noDetection: noDetection,
              noDetectionCount: noDetectionCount,
              showNoDetectionMessage: showNoDetectionMessage
            });
          })
          .catch(error => {
            console.error('处理帧失败:', error);
            
            // 增强错误处理
            let shouldStopDetection = false;
            
            // 检查是否是网络错误
            if (error.errMsg && error.errMsg.includes('request:fail')) {
              if (error.errMsg.includes('ERR_EMPTY_RESPONSE')) {
                console.log('检测到空响应错误，可能是服务器处理超时');
                // 在实时检测模式下，我们不想频繁弹出对话框，只在控制台记录错误
                // 但如果连续出现多次，可能需要暂停检测
                this.emptyResponseCount = (this.emptyResponseCount || 0) + 1;
                
                if (this.emptyResponseCount >= 3) {
                  shouldStopDetection = true;
                  wx.showModal({
                    title: '检测暂停',
                    content: '服务器多次返回空响应，可能正在处理大量数据。是否继续检测？',
                    success: (res) => {
                      if (res.confirm) {
                        console.log('用户选择继续');
                        this.emptyResponseCount = 0;
                        this.startDetection(); // 重新开始检测
                      }
                    }
                  });
                }
              } else if (error.errMsg.includes('timeout')) {
                console.log('请求超时');
                // 超时错误处理类似
                this.timeoutCount = (this.timeoutCount || 0) + 1;
                
                if (this.timeoutCount >= 3) {
                  shouldStopDetection = true;
                  wx.showModal({
                    title: '检测暂停',
                    content: '服务器多次请求超时。是否继续检测？',
                    success: (res) => {
                      if (res.confirm) {
                        console.log('用户选择继续');
                        this.timeoutCount = 0;
                        this.startDetection(); // 重新开始检测
                      }
                    }
                  });
                }
              }
            }
            
            this.setData({
              isProcessing: false,
              boxesInfo: []
            });
            
            // 如果需要停止检测
            if (shouldStopDetection) {
              this.stopDetection();
            }
          });
      },
      fail: (err) => {
        console.error('获取图像信息失败:', err);
        this.setData({ isProcessing: false });
        
        // 如果是单次检测，需要重置检测状态
        if (this.data.detectionMode === 'onDemand') {
          this.setData({ isDetecting: false });
        }
      }
    });
  },
  
  // 切换相机前后摄像头
  switchCamera() {
    const position = this.data.cameraPosition === 'back' ? 'front' : 'back';
    this.setData({
      cameraPosition: position
    });
  },
  
  // 开始/停止检测
  toggleDetection() {
    if (this.data.isDetecting) {
      this.stopDetection();
    } else {
      this.startDetection();
    }
  },
  
  // 切换检测模式 - 连续或单次
  switchDetectionMode() {
    const newMode = this.data.detectionMode === 'continuous' ? 'onDemand' : 'continuous';
    
    // 如果当前正在检测，先停止
    if (this.data.isDetecting) {
      this.stopDetection();
    }
    
    this.setData({
      detectionMode: newMode
    });
  },
  
  // 切换信息显示
  toggleOverlay() {
    this.setData({
      overlayVisible: !this.data.overlayVisible
    });
  },
  
  goBack() {
    wx.navigateBack();
  }
});