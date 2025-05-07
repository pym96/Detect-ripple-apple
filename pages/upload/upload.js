// upload.js
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
    imageSrc: '',
    resultImageSrc: '', // 带边界框的结果图像路径
    showImage: false,
    modelResult: null,
    formattedResults: [], // 预处理后的结果数据
    isProcessing: false,
    imageSize: { width: 0, height: 0 },
    imageInfo: null, // 存储原始图片信息
    noDetection: false,  // 新增：表示是否没有检测到对象
    healthLevel: 0,  // 健康度
    rotLevel: 0  // 腐烂度
  },

  onLoad() {
    // 检查模型是否准备就绪
    const app = getApp();
    if (app.globalData.modelReady) {
      console.log('模型在上传页面已准备就绪');
    } else {
      console.log('模型在上传页面尚未准备就绪');
      // 设置模型准备就绪后要执行的回调
      app.modelReadyCallback = () => {
        console.log('模型现在在上传页面准备就绪');
      };
    }
  },

  chooseImage() {
    if (this.data.isProcessing) return;
    
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['album'],
      success: (res) => {
        const tempFilePath = res.tempFiles[0].tempFilePath;
        this.setData({
          imageSrc: tempFilePath,
          resultImageSrc: '', // 清空结果图像
          showImage: true,
          modelResult: null, // 重置之前的结果
          formattedResults: [], // 重置格式化后的结果
          noDetection: false,  // 重置无检测标志
          processingTimeText: '', // 重置处理时间
          healthLevel: 0,  // 重置健康度
          rotLevel: 0  // 重置腐烂度
        });
        
        // 获取原始图片信息
        this.getImageInfo(tempFilePath);
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
    // 获取显示图像尺寸
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
      
      // 确保预测结果中有友好标签
      if (!prediction.friendlyLabel) {
        prediction.friendlyLabel = modelService.MODEL_LABELS[prediction.class] || `类别${prediction.class}`;
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
      
      // 检查坐标是否合理，如果不合理则进行调整
      if (left > width || top > height || boxWidth > width * 2 || boxHeight > height * 2) {
        console.warn('检测到异常的边界框坐标，可能需要进一步调整归一化处理');
      }
      
      console.log('绘制边界框:', {
        中心点坐标: `(${centerX.toFixed(2)}, ${centerY.toFixed(2)})`,
        宽高: `${boxWidth.toFixed(2)} x ${boxHeight.toFixed(2)}`,
        左上角: `(${left.toFixed(2)}, ${top.toFixed(2)})`
      });
      
      // 如果有原始边界框数据，也打印出来以便调试
      if (prediction.originalBox) {
        console.log('原始边界框数据:', prediction.originalBox);
      }
      
      // 绘制边界框，确保线宽足够明显且不会超出图像范围
      const lineWidth = 5; // 增加线宽使边框更明显
      ctx.setStrokeStyle(boxColor);
      ctx.setLineWidth(lineWidth);
      ctx.strokeRect(left, top, boxWidth, boxHeight);
      
      // 绘制标签背景
      const confidencePercent = Math.round(prediction.confidence * 100);
      const text = `${prediction.friendlyLabel} ${confidencePercent}%`;
      const padding = 6;
      const fontSize = 16; // 增大字体使标签更清晰
      ctx.setFontSize(fontSize);
      // 由于无法直接测量文本宽度，使用字符数估算
      const textWidth = text.length * fontSize * 0.6;
      
      // 确保标签不会超出图像边界
      let labelX = left;
      let labelY = top - fontSize - padding;
      
      // 如果标签会超出顶部，则放在边界框内部顶部
      if (labelY < 0) {
        labelY = top + fontSize + padding;
      }
      
      // 绘制半透明背景，使标签更易读
      ctx.setGlobalAlpha(0.8);
      ctx.setFillStyle(boxColor);
      ctx.fillRect(labelX, labelY - fontSize - padding, textWidth + padding * 2, fontSize + padding * 2);
      ctx.setGlobalAlpha(1.0);
      
      // 绘制白色文字
      ctx.setFillStyle('#FFFFFF');
      ctx.fillText(text, labelX + padding, labelY - padding / 2);
      
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
    
    // 检查results是否为数组，如果是，则取第一个元素
    // 如果results是predictions数组，则处理第一个预测结果
    const prediction = Array.isArray(results) 
      ? (results.length > 0 ? results[0] : null)
      : results;
    
    if (!prediction) return [];
    
    // 获取友好标签，确保有值
    let displayLabel = '';
    let isRotten = false;
    
    // 检查class是否为数字类型
    if (typeof prediction.class === 'number') {
      // 检查是否为腐烂苹果（类别3）
      isRotten = prediction.class === 3;
      // 使用MODEL_LABELS获取友好标签
      const friendlyLabel = modelService.MODEL_LABELS[prediction.class];
      displayLabel = friendlyLabel ? `${friendlyLabel} ` : `类别${prediction.class}`;
    } else {
      // 如果class不是数字，可能是字符串类型的标签（如'rotten_apple'）
      // 将字符串标签转换为友好显示格式
      let friendlyLabel = '';
      
      isRotten = prediction.class === 'rotten_apple' || (prediction.class && String(prediction.class).includes('rotten'));
      if (isRotten) {
        friendlyLabel = '腐烂';
      } else {
        // 默认为健康
        friendlyLabel = '健康';
      }
      
      displayLabel = `${friendlyLabel}`;
    }
    
    console.log('格式化结果:', { class: prediction.class, isRotten, displayLabel });
    
    return [{
      bbox: prediction.bbox,
      class: prediction.class,
      confidence: prediction.confidence,
      displayLabel: displayLabel,
      isRotten: isRotten, // 添加是否腐烂的标志
      original: prediction
    }];
  },
  
  getBoxColor(classId) {
    return CLASS_COLORS[classId] || '#2196F3';// 默认为蓝色
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
        const formattedResults = this.formatResults(result.predictions);
        this.setData({
          modelResult: result,
          formattedResults: formattedResults,
          processingTimeText: `${result.processingTime}`,
          isProcessing: false,
          resultImageSrc: result.resultImageUrl || ''
          // 不在这里设置健康度和腐烂度，而是在下面根据检测结果设置
        });
        
        // 根据格式化后的结果设置健康度和腐烂度
        // 使用formattedResults中的isRotten标志，它已经在formatResults函数中正确处理
        const isRotten = formattedResults.length > 0 && formattedResults[0].isRotten;
        
        // 获取检测结果的类别，用于日志记录
        const prediction = Array.isArray(result.predictions) 
          ? (result.predictions.length > 0 ? result.predictions[0] : null)
          : result.predictions;
        
        console.log('检测结果类别:', prediction ? prediction.class : '无', '是否腐烂:', isRotten);
        
        if (prediction) {
          // 获取置信度并转换为百分比
          const confidencePercent = Math.round(prediction.confidence * 100);
          
          if (isRotten) {
            // 如果是腐烂苹果，腐烂度就是置信度，健康度是100减去腐烂度
            this.setData({
              rotLevel: confidencePercent,
              healthLevel: 100 - confidencePercent
            });
            console.log(`腐烂苹果: 腐烂度=${confidencePercent}%, 健康度=${100 - confidencePercent}%`);
          } else {
            // 如果是健康苹果，健康度就是置信度，腐烂度是100减去健康度
            this.setData({
              healthLevel: confidencePercent,
              rotLevel: 100 - confidencePercent
            });
            console.log(`健康苹果: 健康度=${confidencePercent}%, 腐烂度=${100 - confidencePercent}%`);
          }
        }
        
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

  resetImage() {
    this.setData({
      imageSrc: '',
      resultImageSrc: '',
      showImage: false,
      modelResult: null,
      formattedResults: [],
      noDetection: false,
      processingTimeText: '',
      isProcessing: false,
      healthLevel: 0,  // 重置健康度
      rotLevel: 0  // 重置腐烂度
    });
  },

  goBack() {
    wx.navigateBack();
  }
});