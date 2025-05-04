// onnxUtils.js - ONNX模型工具

// 移除Node.js模块，微信小程序不支持
// const fs = require('fs');
// const path = require('path');
// const onnx = require('onnxruntime-node');
// const Jimp = require('jimp');

// NMS 阈值 - 过滤重叠边界框 (从inference.py采用相同值)
const NMS_THRESHOLD = 0.45;
// 置信度阈值 - 过滤低置信度检测 (从inference.py采用相同值)
const CONFIDENCE_THRESHOLD = 0.25;
// 模型输入尺寸
const MODEL_INPUT_SIZE = 640;
// 类别数量
const NUM_CLASSES = 4;

/**
 * 图像预处理
 * @param {string} imagePath - 图像路径
 * @returns {Promise<Object>} - 预处理结果
 */
function preprocessImage(imagePath) {
  return new Promise((resolve, reject) => {
    try {
      const startTime = new Date().getTime(); // 记录开始时间
      
      // 使用微信小程序API获取图像信息
      wx.getImageInfo({
        src: imagePath,
        success: (imgInfo) => {
          const originalWidth = imgInfo.width;
          const originalHeight = imgInfo.height;
          
          console.log(`原始图像尺寸: ${originalWidth}x${originalHeight}`);
          
          // 计算缩放比例和填充
          const scale = Math.min(
            MODEL_INPUT_SIZE / originalWidth,
            MODEL_INPUT_SIZE / originalHeight
          );
          
          // 缩放后的尺寸
          const newWidth = Math.floor(originalWidth * scale);
          const newHeight = Math.floor(originalHeight * scale);
          
          // 计算填充量
          const xPad = Math.floor((MODEL_INPUT_SIZE - newWidth) / 2);
          const yPad = Math.floor((MODEL_INPUT_SIZE - newHeight) / 2);
          
          console.log(`缩放比例: ${scale}, 填充: (${xPad}, ${yPad})`);
          
          // 创建Canvas用于图像处理
          const canvasId = 'preProcessCanvas';
          const ctx = wx.createCanvasContext(canvasId);
          
          // 清空canvas并填充灰色(128,128,128)
          ctx.clearRect(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
          ctx.fillStyle = 'rgb(128, 128, 128)';
          ctx.fillRect(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
          
          // 将图像绘制到画布中央，周围用灰色填充
          ctx.drawImage(imagePath, xPad, yPad, newWidth, newHeight);
          
          ctx.draw(false, () => {
            // 从Canvas获取图像数据
            wx.canvasGetImageData({
              canvasId: canvasId,
              x: 0,
              y: 0,
              width: MODEL_INPUT_SIZE,
              height: MODEL_INPUT_SIZE,
              success: (res) => {
                // 将RGBA数据转换为模型输入格式
                const imageData = res.data;
                const tensorData = normalizeImageData(imageData);
                
                resolve({
                  data: tensorData,
                  scale: scale,
                  offset: { x: xPad, y: yPad },
                  originalSize: { width: originalWidth, height: originalHeight },
                  startTime: startTime // 添加开始处理的时间戳
                });
              },
              fail: (err) => {
                console.error('获取Canvas图像数据失败:', err);
                reject(err);
              }
            });
          });
        },
        fail: (err) => {
          console.error('获取图像信息失败:', err);
          reject(err);
        }
      });
    } catch (error) {
      console.error('预处理图像时出错:', error);
      reject(error);
    }
  });
}

/**
 * 将图像数据从RGBA归一化为RGB格式的Float32Array
 * @param {Uint8ClampedArray} imageData - RGBA格式的图像数据
 * @returns {Float32Array} - 归一化的RGB数据
 */
function normalizeImageData(imageData) {
  const pixelCount = imageData.length / 4; // RGBA每像素4个值
  const normalizedData = new Float32Array(pixelCount * 3); // RGB每像素3个值
  
  // 单独存储每个通道 [R,R,...,G,G,...,B,B,...]
  for (let i = 0; i < pixelCount; i++) {
    // RGB归一化到[0,1]
    normalizedData[i] = imageData[i * 4] / 255.0;                 // R
    normalizedData[i + pixelCount] = imageData[i * 4 + 1] / 255.0;  // G
    normalizedData[i + pixelCount * 2] = imageData[i * 4 + 2] / 255.0;  // B
  }
  
  return normalizedData;
}

/**
 * 运行模型推理
 * @param {string} modelPath - 模型路径
 * @param {Float32Array} imageData - 预处理后的图像数据
 * @returns {Promise<Array>} - 模型输出
 */
function runInference(modelPath, imageData) {
  return new Promise((resolve, reject) => {
    try {
      // 小程序环境不能直接运行ONNX模型
      // 这里生成模拟数据用于开发测试
      console.log(`模拟推理: ${modelPath}`);
      console.log(`输入数据长度: ${imageData.length}`);
      
      setTimeout(() => {
        // 随机确定是否检测到苹果 (20%概率没有检测)
        const detectApple = Math.random() > 0.2;
        
        if (!detectApple) {
          console.log('模拟无检测结果');
          resolve([]);
          return;
        }
        
        // 生成模拟的检测结果 (4个类别，随机坐标)
        const numDetections = Math.floor(Math.random() * 3) + 1;
        const detections = [];
        
        for (let i = 0; i < numDetections; i++) {
          const classId = Math.floor(Math.random() * NUM_CLASSES);
          detections.push({
            bbox: {
              x: 0.3 + Math.random() * 0.4, // 中心x [0.3-0.7]
              y: 0.3 + Math.random() * 0.4, // 中心y [0.3-0.7]
              w: 0.2 + Math.random() * 0.2, // 宽度 [0.2-0.4]
              h: 0.2 + Math.random() * 0.2  // 高度 [0.2-0.4]
            },
            class: classId,
            confidence: 0.7 + Math.random() * 0.3 // 置信度 [0.7-1.0]
          });
        }
        
        resolve(detections);
      }, 500);
    } catch (error) {
      console.error('模型推理出错:', error);
      reject(error);
    }
  });
}

/**
 * 计算两个边界框的IoU
 * @param {Object} box1 - 第一个边界框
 * @param {Object} box2 - 第二个边界框
 * @returns {number} - IoU值
 */
function calculateIoU(box1, box2) {
  // 将中心点坐标转换为左上角和右下角坐标
  const box1X1 = box1.x - box1.w / 2;
  const box1Y1 = box1.y - box1.h / 2;
  const box1X2 = box1.x + box1.w / 2;
  const box1Y2 = box1.y + box1.h / 2;
  
  const box2X1 = box2.x - box2.w / 2;
  const box2Y1 = box2.y - box2.h / 2;
  const box2X2 = box2.x + box2.w / 2;
  const box2Y2 = box2.y + box2.h / 2;
  
  // 计算交集区域
  const interX1 = Math.max(box1X1, box2X1);
  const interY1 = Math.max(box1Y1, box2Y1);
  const interX2 = Math.min(box1X2, box2X2);
  const interY2 = Math.min(box1Y2, box2Y2);
  
  // 如果没有交集，则IoU为0
  if (interX2 < interX1 || interY2 < interY1) {
    return 0.0;
  }
  
  // 计算交集面积
  const interArea = (interX2 - interX1) * (interY2 - interY1);
  
  // 计算并集面积
  const box1Area = (box1X2 - box1X1) * (box1Y2 - box1Y1);
  const box2Area = (box2X2 - box2X1) * (box2Y2 - box2Y1);
  const unionArea = box1Area + box2Area - interArea;
  
  // 返回IoU
  return interArea / unionArea;
}

/**
 * 对检测结果应用NMS
 * @param {Array} detections - 检测结果数组
 * @param {number} iouThreshold - IoU阈值
 * @returns {Array} - 过滤后的检测结果
 */
function applyNMS(detections, iouThreshold) {
  if (detections.length === 0) {
    return [];
  }
  
  // 按照置信度排序
  detections.sort((a, b) => b.confidence - a.confidence);
  
  const selectedBoxes = [];
  const indexesToRemove = new Set();
  
  // 应用NMS
  for (let i = 0; i < detections.length; i++) {
    if (indexesToRemove.has(i)) continue;
    
    selectedBoxes.push(detections[i]);
    
    for (let j = i + 1; j < detections.length; j++) {
      if (indexesToRemove.has(j)) continue;
      
      // 如果两个边界框属于同一类别且IoU大于阈值，则移除较低置信度的框
      if (detections[i].class === detections[j].class) {
        const iou = calculateIoU(detections[i].bbox, detections[j].bbox);
        if (iou > iouThreshold) {
          indexesToRemove.add(j);
        }
      }
    }
  }
  
  return selectedBoxes;
}

/**
 * 后处理模型输出
 * @param {Array} detections - 模型输出数据
 * @param {Object} preprocessInfo - 预处理信息
 * @returns {Array} - 处理后的检测结果
 */
function postprocessOutput(detections, preprocessInfo) {
  try {
    // 输出日志
    console.log('开始处理模型输出');
    
    // 检查是否为空
    if (!detections || detections.length === 0) {
      console.warn('模型输出为空');
      return [];
    }
    
    console.log('检测结果数量:', detections.length);
    
    // 过滤掉置信度低于阈值的检测
    const filteredByConfidence = detections.filter(detection => 
      detection && detection.confidence && detection.confidence > CONFIDENCE_THRESHOLD
    );
    
    if (filteredByConfidence.length === 0) {
      console.warn('所有检测结果的置信度都低于阈值');
      return [];
    }
    
    console.log(`过滤后的检测结果数量 (置信度 > ${CONFIDENCE_THRESHOLD}):`, filteredByConfidence.length);
    
    // 为每个检测结果添加类别名称
    const processedDetections = filteredByConfidence.map(detection => {
      if (!detection || typeof detection.class === 'undefined') {
        console.warn('检测结果缺少类别信息:', detection);
        return null;
      }
      
      // 获取类别名称
      const className = getClassName(detection.class);
      
      // 确保边界框存在并格式正确
      const bbox = detection.bbox || {};
      const validBbox = {
        x: typeof bbox.x === 'number' ? bbox.x : 0.5,
        y: typeof bbox.y === 'number' ? bbox.y : 0.5,
        w: typeof bbox.w === 'number' ? bbox.w : 0.2,
        h: typeof bbox.h === 'number' ? bbox.h : 0.2
      };
      
      return {
        ...detection,
        className: className,
        bbox: validBbox
      };
    }).filter(Boolean); // 移除null项
    
    if (processedDetections.length === 0) {
      console.warn('处理后没有有效的检测结果');
      return [];
    }
    
    console.log('处理后的检测结果:', processedDetections);
    
    // 应用NMS过滤重叠边界框
    const filteredDetections = applyNMS(processedDetections, NMS_THRESHOLD);
    console.log(`NMS后保留了 ${filteredDetections.length} 个检测框, 原始框数: ${processedDetections.length}`);
    
    return filteredDetections;
  } catch (error) {
    console.error('处理模型输出时出错:', error);
    return [];
  }
}

/**
 * 根据类别ID获取类别名称
 * @param {number} classId - 类别ID
 * @returns {string} - 类别名称
 */
function getClassName(classId) {
  const classNames = {
    0: '100%成熟度',
    1: '50%成熟度',
    2: '75%成熟度',
    3: '腐烂苹果'
  };
  return classNames[classId] || `未知类别(${classId})`;
}

module.exports = {
  preprocessImage,
  runInference,
  postprocessOutput,
  calculateIoU,
  applyNMS
}; 