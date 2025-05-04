// modelService.js - 用于处理与后端API通信的工具

// 模型类别
const MODEL_CLASSES = {
  0: '100%成熟度',
  1: '50%成熟度',
  2: '75%成熟度',
  3: '腐烂苹果'
};

// 模型类别对应的友好标签
const MODEL_LABELS = {
  0: '健康',  // 100%成熟度 -> 健康
  1: '健康',  // 50%成熟度 -> 健康
  2: '健康',  // 75%成熟度 -> 健康
  3: '腐烂'   // 腐烂苹果 -> 腐烂
};

// 后端API配置
const API_CONFIG = {
  baseUrl: 'http://123.57.63.76:5000',  // 后端服务器地址 (使用内网IP)
  detectEndpoint: '/api/detect',     // 检测API端点
  healthEndpoint: '/api/health'      // 健康检查端点
};

// 分类颜色 - 与后端一致
const CLASS_COLORS = {
  0: '#00FF00', // 绿色 - 100% 成熟度
  1: '#00A5FF', // 橙色 - 50% 成熟度
  2: '#00FF7F', // 浅绿色 - 75% 成熟度
  3: '#FF0000'  // 红色 - 腐烂
};

/**
 * 检查后端API是否可用
 * @returns {Promise<boolean>} - API是否可用
 */
async function checkApiHealth() {
  try {
    console.log('开始健康检查请求...');
    const response = await new Promise((resolve, reject) => {
      const requestTask = wx.request({
        url: `${API_CONFIG.baseUrl}${API_CONFIG.healthEndpoint}`,
        method: 'GET',
        timeout: 10000, // 设置10秒超时
        success: (res) => {
          console.log('健康检查请求成功返回:', res.statusCode);
          resolve(res);
        },
        fail: (err) => {
          console.error('健康检查请求失败:', err);
          reject(err);
        }
      });
      
      // 设置请求完成的回调
      requestTask.onHeadersReceived((res) => {
        console.log('健康检查收到响应头:', res.header);
      });
    });
    
    const isHealthy = response.statusCode === 200 && response.data && response.data.status === 'ok';
    console.log('API健康状态:', isHealthy ? '正常' : '异常');
    return isHealthy;
  } catch (error) {
    console.error('API健康检查失败:', error);
    return false;
  }
}

/**
 * 将图像文件转换为Base64编码
 * @param {string} filePath - 图像文件路径
 * @returns {Promise<string>} - Base64编码的图像数据
 */
async function imageToBase64(filePath) {
  return new Promise((resolve, reject) => {
    wx.getFileSystemManager().readFile({
      filePath: filePath,
      encoding: 'base64',
      success: res => {
        resolve(res.data);
      },
      fail: err => {
        console.error('转换图像到Base64失败:', err);
        reject(err);
      }
    });
  });
}

/**
 * 使用后端API处理图像
 * @param {string} imagePath - 要处理的图像路径
 * @param {number} retryCount - 重试次数，默认为2
 * @returns {Promise} - 解析为模型预测结果的Promise
 */
async function processImage(imagePath, retryCount = 2) {
  try {
    console.log(`开始处理图像: ${imagePath}，剩余重试次数: ${retryCount}`);
    const startTime = Date.now();
    
    // 1. 检查API是否可用
    const isApiHealthy = await checkApiHealth();
    if (!isApiHealthy) {
      throw new Error('后端API不可用');
    }
    
    // 2. 将图像转换为Base64
    const base64Image = await imageToBase64(imagePath);
    console.log('图像转换为Base64完成');
    
    // 3. 调用后端API进行推理 - 添加超时和重试机制
    const response = await new Promise((resolve, reject) => {
      // 创建请求任务
      const requestTask = wx.request({
        url: `${API_CONFIG.baseUrl}${API_CONFIG.detectEndpoint}`,
        method: 'POST',
        header: {
          'content-type': 'application/x-www-form-urlencoded'
        },
        data: {
          image: `data:image/jpeg;base64,${base64Image}`
        },
        timeout: 30000, // 设置30秒超时
        success: (res) => {
          console.log('API请求成功返回:', res.statusCode);
          resolve(res);
        },
        fail: (err) => {
          console.error('API请求失败:', err);
          reject(err);
        },
        complete: () => {
          console.log('API请求完成');
        }
      });
      
      // 设置请求完成的回调
      requestTask.onHeadersReceived((res) => {
        console.log('收到响应头:', res.header);
      });
    }).catch(async (error) => {
      // 处理请求错误，如果还有重试次数，则重试
      console.error(`请求失败: ${JSON.stringify(error)}，剩余重试次数: ${retryCount}`);
      
      if (retryCount > 0) {
        console.log(`尝试重新请求，等待2秒...`);
        // 等待2秒后重试
        await new Promise(resolve => setTimeout(resolve, 2000));
        return processImage(imagePath, retryCount - 1);
      }
      
      // 重试次数用完，抛出错误
      throw error;
    });
    
    // 4. 处理API响应
    if (!response) {
      throw new Error('未收到API响应');
    }
    
    if (response.statusCode !== 200) {
      throw new Error(`API返回错误状态码: ${response.statusCode}`);
    }
    
    if (!response.data) {
      throw new Error('API返回空数据');
    }
    
    if (!response.data.success) {
      throw new Error(response.data.message || 'API操作未成功');
    }
    
    const apiResult = response.data;
    console.log(`推理完成, 检测到 ${apiResult.detection_count} 个目标`);
    
    // 5. 如果没有检测结果，返回null
    if (!apiResult.detections || apiResult.detections.length === 0) {
      console.log('没有检测到苹果');
      return null;
    }
    
    // 6. 格式化检测结果
    const processedDetections = apiResult.detections.map(det => {
      // 检查边界框坐标是否合理，如果不合理则进行归一化处理
      const box = det.box;
      let x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3];
      
      // 如果坐标值异常大，可能是绝对像素值而非归一化值，需要归一化处理
      // 假设正常图像尺寸不会超过10000像素
      const needsNormalization = x1 > 10 || y1 > 10 || x2 > 10 || y2 > 10;
      
      // 获取图像尺寸（如果无法获取，使用1作为默认值进行归一化）
      // 这里假设后端返回的坐标是基于原始图像尺寸的像素值
      const imgWidth = 1;
      const imgHeight = 1;
      
      // 如果需要归一化，将像素坐标转换为0-1范围的归一化坐标
      if (needsNormalization) {
        console.log('检测到异常大的坐标值，执行归一化处理:', box);
        
        // 计算边界框的宽度和高度
        const boxWidth = x2 - x1;
        const boxHeight = y2 - y1;
        
        // 估算图像尺寸 - 假设边界框不会超过整个图像的80%
        // 这是一个启发式方法，基于大多数检测场景中目标不会占据整个图像
        const estimatedImgWidth = boxWidth / 0.8;
        const estimatedImgHeight = boxHeight / 0.8;
        
        // 使用估算的图像尺寸进行归一化
        x1 = x1 / estimatedImgWidth;
        y1 = y1 / estimatedImgHeight;
        x2 = x2 / estimatedImgWidth;
        y2 = y2 / estimatedImgHeight;
        
        // 确保归一化后的值在0-1范围内
        x1 = Math.max(0, Math.min(1, x1));
        y1 = Math.max(0, Math.min(1, y1));
        x2 = Math.max(0, Math.min(1, x2));
        y2 = Math.max(0, Math.min(1, y2));
        
        console.log('归一化后的坐标:', [x1, y1, x2, y2]);
      }
      
      return {
        class: det.class_id,
        className: det.class_name,
        confidence: det.confidence,
        bbox: {
          x: (x1 + x2) / 2, // 中心点x（归一化坐标）
          y: (y1 + y2) / 2, // 中心点y（归一化坐标）
          w: x2 - x1,       // 宽度（归一化值）
          h: y2 - y1        // 高度（归一化值）
        },
        // 转换为页面显示所需的格式
        displayLabel: MODEL_LABELS[det.class_id] || `类别${det.class_id}`,
        confidenceText: (det.confidence * 100).toFixed(1) + '%',
        color: CLASS_COLORS[det.class_id] || '#FF0000',
        // 保存原始边界框数据以便调试
        originalBox: det.box
      };
    });
    
    // 7. 按置信度排序
    processedDetections.sort((a, b) => b.confidence - a.confidence);
    
    // 8. 输出所有检测结果的信息
    console.log("\n检测结果按类别:");
    console.log(apiResult.class_counts);
    
    // 9. 计算处理时间
    const processingTime = Date.now() - startTime;
    console.log(`总处理时间: ${processingTime}ms`);
    
    // 10. 返回结果对象
    return {
      predictions: processedDetections,
      processingTime: processingTime,
      resultImageUrl: `${API_CONFIG.baseUrl}${apiResult.result_image}`, // 带边界框的结果图像URL
      health_level: apiResult.health_level || 0, // 添加健康度
      rot_level: apiResult.rot_level || 0 // 添加腐烂度
    };
  } catch (error) {
    console.error('处理图像时出错:', error);
    throw error;
  }
}

/**
 * 初始化模型服务
 * @returns {Promise<object>} - 初始化结果
 */
async function initModel() {
  try {
    console.log('检查后端API是否可用...');
    const isApiHealthy = await checkApiHealth();
    
    if (!isApiHealthy) {
      throw new Error('后端API不可用，请确保后端服务已启动');
    }
    
    return {
      status: 'success',
      message: '后端API连接成功',
      timestamp: Date.now()
    };
  } catch (error) {
    console.error('API连接失败:', error);
    throw error;
  }
}

// 导出函数和常量
module.exports = {
  initModel,
  processImage,
  MODEL_CLASSES,
  MODEL_LABELS,
  CLASS_COLORS
};