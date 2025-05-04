// camera.js
const modelService = require('../../utils/modelService');

// 类别颜色映射 - 更新为与inference.py一致
const CLASS_COLORS = {
  0: '#00FF00', // 绿色 - 100% 成熟度
  1: '#00A5FF', // 橙色 - 50% 成熟度
  2: '#00FF7F', // 浅绿色 - 75% 成熟度
  3: '#FF0000'  // 红色 - 腐烂
};

Page({
  data: {
    cameraContext: null,
    cameraPosition: 'back',
    imageSrc: '',
    resultImageSrc: '', // 带边界框的结果图像路径
    showResult: false,
    modelResult: null,
    formattedResults: [], // 预处理后的结果数据
    isProcessing: false,
    imageSize: { width: 0, height: 0 },
    boxPositions: [], // 用于存储计算后的边界框位置
    imageInfo: null, // 存储原始图片信息
    noDetection: false  // 新增：表示是否没有检测到对象
  },

  onLoad() {
    this.setData({
      cameraContext: wx.createCameraContext()
    });
    
    // 检查模型是否准备就绪
    const app = getApp();
    if (app.globalData.modelReady) {
      console.log('模型在相机页面已准备就绪');
    } else {
      console.log('模型在相机页面尚未准备就绪');
      // 设置模型准备就绪后要执行的回调
      app.modelReadyCallback = () => {
        console.log('模型现在在相机页面准备就绪');
      };
    }
  },

  switchCamera() {
    const position = this.data.cameraPosition === 'back' ? 'front' : 'back';
    this.setData({
      cameraPosition: position
    });
  },

  takePhoto() {
    if (this.data.isProcessing) return;
    
    this.setData({ isProcessing: true });
    
    const context = this.data.cameraContext;
    context.takePhoto({
      quality: 'high',
      success: (res) => {
        this.setData({
          imageSrc: res.tempImagePath,
          resultImageSrc: '', // 清空结果图像
          showResult: true,
          isProcessing: false,
          modelResult: null, // 重置之前的结果
          formattedResults: [], // 重置格式化后的结果
          noDetection: false // 重置无检测标志
        });
        
        // 获取原始图片信息
        this.getImageInfo(res.tempImagePath);
      },
      fail: (error) => {
        console.error('Camera error:', error);
        this.setData({ isProcessing: false });
        wx.showToast({
          title: '拍照失败',
          icon: 'none'
        });
      }
    });
  },
  
  // 获取图片信息，确保边界框在原始图片上的正确位置
  getImageInfo(imagePath) {
    wx.getImageInfo({
      src: imagePath,
      success: (res) => {
        console.log('原始图片信息:', res);
        this.setData({
          imageInfo: {
            width: res.width,
            height: res.height,
            orientation: res.orientation,
            path: res.path
          }
        });
      },
      fail: (err) => {
        console.error('获取图片信息失败:', err);
      }
    });
  },

  onImageLoad(e) {
    // 获取显示图像尺寸，用于边界框定位
    const { width, height } = e.detail;
    this.setData({
      imageSize: { width, height }
    });
  },

  // 在原图上绘制边界框
  drawBoundingBoxOnImage() {
    const modelResult = this.data.modelResult;
    const imageInfo = this.data.imageInfo;
    
    if (!modelResult || !modelResult.predictions || !imageInfo) {
      console.error('没有检测结果或图像信息');
      return Promise.reject('缺少必要数据');
    }
    
    return new Promise((resolve, reject) => {
      // 创建一个用于绘制的Canvas
      const canvasId = 'resultCanvas';
      const ctx = wx.createCanvasContext(canvasId);
      
      // 获取预测结果，可能是对象或数组
      const prediction = Array.isArray(modelResult.predictions) 
        ? (modelResult.predictions.length > 0 ? modelResult.predictions[0] : null)
        : modelResult.predictions;
        
      if (!prediction) {
        console.error('没有有效的预测结果');
        return reject('无有效预测');
      }
      
      const { width, height } = imageInfo;
      
      // 确保Canvas尺寸与原图一致
      ctx.canvas = { width, height };
      
      // 首先在Canvas上绘制原始图像
      ctx.drawImage(this.data.imageSrc, 0, 0, width, height);
      
      // 设置绘制边界框的样式
      const boxColor = this.getBoxColor(prediction.class);
      
      // 获取边界框坐标（已在modelService中转换为相对于原始图像的归一化坐标）
      const bbox = prediction.bbox;
      
      // 将归一化坐标转换为像素坐标
      // bbox包含中心点坐标和宽高，需要转换为左上角坐标
      const centerX = bbox.x * width;
      const centerY = bbox.y * height;
      const boxWidth = bbox.w * width;
      const boxHeight = bbox.h * height;
      
      // 计算左上角坐标
      const left = Math.max(0, centerX - boxWidth / 2);
      const top = Math.max(0, centerY - boxHeight / 2);
      
      console.log('绘制边界框:', {
        中心点坐标: `(${centerX.toFixed(2)}, ${centerY.toFixed(2)})`,
        宽高: `${boxWidth.toFixed(2)} x ${boxHeight.toFixed(2)}`,
        左上角: `(${left.toFixed(2)}, ${top.toFixed(2)})`
      });
      
      // 参考 inference.py 的 plot_one_box 函数改进边界框绘制
      // 计算适合图片大小的线宽
      const tl = Math.round(0.002 * (height + width) / 2) + 1;
      
      // 绘制边界框
      ctx.setStrokeStyle(boxColor);
      ctx.setLineWidth(tl);
      ctx.setLineCap('round');
      ctx.strokeRect(left, top, boxWidth, boxHeight);
      
      // 绘制标签背景和文字 (类似 inference.py)
      const confidencePercent = Math.round(prediction.confidence * 100);
      const text = `${prediction.friendlyLabel} ${confidencePercent}%`;
      const fontSize = Math.max(tl - 1, 1) * 3;  // 字体大小根据线宽自适应
      ctx.setFontSize(fontSize);
      
      // 估算文本宽度
      const textWidth = text.length * fontSize * 0.6;
      
      // 标签位置在边界框左上角
      const labelX = left;
      const labelY = top > fontSize ? top : top + fontSize;
      
      // 绘制标签背景 - 填充矩形
      ctx.setFillStyle(boxColor);
      ctx.fillRect(labelX, labelY - fontSize - 3, textWidth, fontSize + 3);
      
      // 绘制白色文字
      ctx.setFillStyle('#FFFFFF');
      ctx.fillText(text, labelX, labelY - 2);
      
      // 完成绘制并保存为图片
      ctx.draw(false, () => {
        // 增加延时确保绘制完成
        setTimeout(() => {
          wx.canvasToTempFilePath({
            canvasId: canvasId,
            width: width,
            height: height,
            destWidth: width, // 确保输出图像尺寸与原图一致
            destHeight: height,
            fileType: 'jpg',
            quality: 0.9, // 使用较高质量
            success: (res) => {
              console.log('生成带边界框的图片成功', res.tempFilePath);
              this.setData({
                resultImageSrc: res.tempFilePath
              });
              resolve(res.tempFilePath);
            },
            fail: (err) => {
              console.error('生成带边界框的图片失败', err);
              reject(err);
            }
          });
        }, 300); // 适当增加延时，确保绘制完成
      });
    });
  },

  // 预处理和格式化结果数据
  formatResults(results) {
    if (!results) return [];
    
    return [{
      bbox: results.bbox,
      class: results.class,
      confidence: results.confidence,
      displayLabel: `${results.friendlyLabel} (${Math.round(results.confidence * 100)}%)`,
      confidenceText: `${Math.round(results.confidence * 100)}%`,
      original: results
    }];
  },

  getBoxColor(classId) {
    return CLASS_COLORS[classId] || '#2196F3'; // 默认为蓝色
  },

  processImage() {
    if (this.data.isProcessing || !this.data.imageSrc) return;
    
    this.setData({
      isProcessing: true,
      modelResult: null,
      formattedResults: [],
      noDetection: false
    });
    
    wx.showLoading({
      title: '处理中...',
    });
    
    // 调用模型服务处理图像
    modelService.processImage(this.data.imageSrc)
      .then(result => {
        wx.hideLoading();
        
        if (!result) {
          // 没有检测到苹果
          this.setData({
            isProcessing: false,
            noDetection: true
          });
          return;
        }
        
        // 更新UI显示
        this.setData({
          modelResult: result,
          formattedResults: this.formatResults(result),
          processingTimeText: `${result.processingTime}ms`,
          isProcessing: false,
          resultImageSrc: result.resultImageUrl || ''
        });
        
        // 尝试绘制边界框
        this.drawBoundingBoxOnImage()
          .then(() => {
            console.log('边界框绘制成功');
          })
          .catch(err => {
            console.error('边界框绘制失败:', err);
          });
      })
      .catch(error => {
        console.error('处理图像失败:', error);
        wx.hideLoading();
        
        // 增强错误处理和用户反馈
        let errorMessage = '处理失败';
        
        // 检查是否是网络错误
        if (error.errMsg && error.errMsg.includes('request:fail')) {
          if (error.errMsg.includes('ERR_EMPTY_RESPONSE')) {
            errorMessage = '服务器响应为空，可能正在处理大图像';
            console.log('检测到空响应错误，可能是服务器处理超时');
          } else if (error.errMsg.includes('timeout')) {
            errorMessage = '请求超时，请稍后重试';
          } else {
            errorMessage = '网络连接错误，请检查网络';
          }
        }
        
        wx.showModal({
          title: '处理失败',
          content: errorMessage + '，是否重试？',
          success: (res) => {
            if (res.confirm) {
              console.log('用户选择重试');
              // 短暂延迟后重试
              setTimeout(() => {
                this.processImage();
              }, 1000);
            } else {
              console.log('用户取消重试');
            }
          }
        });
        
        this.setData({ isProcessing: false });
      });
  },

  retakePhoto() {
    this.setData({
      showResult: false,
      modelResult: null,
      formattedResults: [],
      resultImageSrc: '',
      imageSrc: '',
      noDetection: false
    });
  },

  goBack() {
    wx.navigateBack();
  }
});