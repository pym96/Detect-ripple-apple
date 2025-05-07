#!/usr/bin/env python3
import os
import base64
import cv2
import numpy as np
import yaml
import time
import json
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path
import io
import traceback
from openvino.runtime import Core  # 使用OpenVINO Runtime

app = Flask(__name__)
CORS(app)  # 启用跨域请求支持

# 配置
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
RESULT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_optimized')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'openvino_model/best.xml')  # OpenVINO XML文件
MODEL_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'openvino_model/best.bin')  # OpenVINO BIN文件
DATA_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.yaml')
CONF_THRESHOLD = 0.25  # 置信度阈值

# 确保上传和结果目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传大小为16MB

# --- Helper Functions ---
def get_class_names(data_yaml):
    """从data.yaml文件加载类别名称"""
    try:
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
            if 'names' in data:
                if isinstance(data['names'], list):
                    return data['names']
                elif isinstance(data['names'], dict):
                    max_idx = max(int(k) for k in data['names'].keys())
                    result = [''] * (max_idx + 1)
                    for idx, name in data['names'].items():
                        result[int(idx)] = name
                    return result
    except Exception as e:
        print(f"警告：无法从{data_yaml}加载类别名称: {e}")
    return ['100-_ripeness', '50-_ripeness', '75-_ripeness', 'rotten_apple']

def get_class_colors(class_names):
    """为每个类别生成视觉上不同的颜色"""
    color_map = {
        '100-_ripeness': [0, 255, 0],    # Green
        '50-_ripeness': [0, 165, 255],   # Orange
        '75-_ripeness': [0, 255, 127],   # Light Green
        'rotten_apple': [0, 0, 255]      # Red
    }
    colors = []
    np.random.seed(42)
    for class_name in class_names:
        if class_name in color_map:
            colors.append(color_map[class_name])
        else:
            colors.append([int(c) for c in np.random.randint(0, 255, 3)])
    return colors

def plot_one_box(box, img, color, label=None, line_thickness=3):
    """在图像上绘制一个边界框"""
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA) # Filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def preprocess_image(img_path, input_shape):
    """预处理图像用于YOLOv8输入"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像文件: {img_path}")
    
    h, w = img.shape[:2]
    target_h, target_w = input_shape[2], input_shape[3]

    # 计算缩放比例和填充
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    dh, dw = (target_h - new_h) / 2, (target_w - new_w) / 2

    # 缩放和填充
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # 归一化 BGR -> RGB, HWC -> CHW, / 255.0
    blob = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    blob = blob.transpose(2, 0, 1) # HWC to CHW
    blob = np.expand_dims(blob, axis=0).astype(np.float32) / 255.0 # Add batch dimension and normalize
    return blob, img, (scale, dw, dh) # Return original image and scaling info

def postprocess_results(outputs, original_image, scaling_info, conf_thres, iou_thres=0.45):
    """后处理YOLOv8输出"""
    scale, dw, dh = scaling_info
    original_h, original_w = original_image.shape[:2]
    
    # ONNX model output shape might be [1, num_classes + 4, num_proposals]
    # We need to transpose it to [1, num_proposals, num_classes + 4]
    outputs = np.transpose(outputs, (0, 2, 1)) # [1, 84, 8400] -> [1, 8400, 84] for YOLOv8
    
    boxes = []
    scores = []
    class_ids = []

    for pred in outputs[0]: # Iterate through proposals
        # pred shape: [x_center, y_center, width, height, obj_conf, class1_conf, class2_conf, ...]
        box = pred[:4]
        obj_conf = pred[4] # This might not be present in all exported models
        class_confidences = pred[4:] # Assume obj_conf is NOT separate, common case

        if obj_conf < conf_thres: # Initial filter (optional depending on model export)
             continue
        
        class_id = np.argmax(class_confidences)
        score = class_confidences[class_id] # Use class confidence as score

        if score >= conf_thres:
             # Convert box from center xywh to xyxy
            cx, cy, w, h = box
            x1 = (cx - w / 2 - dw) / scale
            y1 = (cy - h / 2 - dh) / scale
            x2 = (cx + w / 2 - dw) / scale
            y2 = (cy + h / 2 - dh) / scale
            
            # Clip boxes to image dimensions
            x1 = max(0, min(x1, original_w))
            y1 = max(0, min(y1, original_h))
            x2 = max(0, min(x2, original_w))
            y2 = max(0, min(y2, original_h))

            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS) using OpenCV
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)

    detections = []
    if len(indices) > 0:
        # Flatten indices if it's nested
        if isinstance(indices, (list, tuple)) and len(indices) > 0 and isinstance(indices[0], (list, tuple, np.ndarray)):
             indices = indices.flatten()
             
        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            
            detections.append({
                'box': [float(b) for b in box],
                'confidence': float(score),
                'class_id': int(class_id),
                'class_name': "" # Will be filled later
            })
            
    return detections

# 检查文件扩展名是否允许
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- OpenVINO Runtime Session (全局初始化) ---
print("初始化OpenVINO Runtime...")
try:
    # 创建OpenVINO核心
    core = Core()
    
    # 读取模型
    model = core.read_model(MODEL_XML)
    
    # 编译模型 (默认为CPU)
    compiled_model = core.compile_model(model)
    
    # 获取输入和输出信息
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    
    # 获取输入形状
    input_shape = input_layer.shape
    input_name = input_layer.any_name
    output_name = output_layer.any_name
    
    print(f"OpenVINO模型加载成功: 输入形状={input_shape}")
    
except Exception as e:
    print(f"警告: OpenVINO模型加载失败: {e}")
    traceback.print_exc()
    compiled_model = None

# --- 使用OpenVINO模型进行推理 ---
def run_openvino_inference(img_path, conf_thres=CONF_THRESHOLD, iou_thres=0.45):
    """使用OpenVINO Runtime运行优化推理"""
    start_time = time.time()

    # 1. 加载类别名称
    class_names = get_class_names(DATA_YAML)
    colors = get_class_colors(class_names)
    
    # 2. 检查模型和图像
    if compiled_model is None:
        return None, [], "错误: OpenVINO模型未正确加载"
    
    if not os.path.isfile(img_path):
        return None, [], f"错误: 图像文件'{img_path}'未找到"

    # 3. 预处理图像
    try:
        input_tensor, original_img, scaling_info = preprocess_image(img_path, input_shape)
    except Exception as e:
        traceback.print_exc()
        return None, [], f"错误: 图像预处理失败: {e}"

    # 4. 运行推理
    try:
        # OpenVINO推理
        results = compiled_model([input_tensor])
        output_tensor = results[output_layer]
    except Exception as e:
        traceback.print_exc()
        return None, [], f"错误: OpenVINO推理失败: {e}"

    # 5. 后处理结果
    try:
        detections = postprocess_results(output_tensor, original_img, scaling_info, conf_thres, iou_thres)
        
        # 添加类别名称到检测结果
        for det in detections:
            class_id = det['class_id']
            if 0 <= class_id < len(class_names):
                det['class_name'] = class_names[class_id]
            else:
                det['class_name'] = f"未知类别 {class_id}"
    except Exception as e:
        traceback.print_exc()
        return None, [], f"错误: 结果后处理失败: {e}"

    # 6. 绘制检测结果并统计类别
    result_img = original_img.copy()
    class_counts = {}
    for det in detections:
        x1, y1, x2, y2 = map(int, det['box'])
        conf = det['confidence']
        class_id = det['class_id']
        class_name = det['class_name']

        # 获取此类别的颜色
        color = colors[class_id] if 0 <= class_id < len(colors) else [0, 0, 255]
        label = f"{class_name} {conf:.2f}"
        plot_one_box([x1, y1, x2, y2], result_img, color, label)

        # 统计类别
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    total_time = time.time() - start_time
    print(f"图像 {os.path.basename(img_path)} 处理完成，用时: {total_time:.4f}秒, 检测到 {len(detections)} 个目标")
    
    return result_img, detections, class_counts

# 处理上传的图像文件
@app.route('/api/detect', methods=['POST'])
def detect():
    # 检查是否有文件部分
    if 'file' not in request.files and 'image' not in request.form:
        return jsonify({
            'success': False,
            'message': '没有文件或图像数据'
        }), 400
    
    timestamp = int(time.time())
    img_path = None
    
    try:
        # 处理文件上传
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'message': '没有选择文件'
                }), 400
            
            if file and allowed_file(file.filename):
                filename = secure_filename(f"{timestamp}_{file.filename}")
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(img_path)
        
        # 处理Base64编码的图像
        elif 'image' in request.form:
            image_data = request.form['image']
            # 检查是否是Base64格式
            if ';base64,' in image_data:
                image_data = image_data.split(';base64,')[1]
            
            # 解码Base64数据
            image_bytes = base64.b64decode(image_data)
            
            # 将字节转换为numpy数组
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # 保存图像
            filename = f"{timestamp}_image.jpg"
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            cv2.imwrite(img_path, img)
        
        # 运行推理 - 这里使用优化后的OpenVINO推理
        result_img, detections, class_counts = run_openvino_inference(img_path)
        
        if result_img is None:
            return jsonify({
                'success': False,
                'message': detections  # 在这种情况下，detections包含错误消息
            }), 500
        
        # 保存结果图像
        result_filename = f"{timestamp}_result.jpg"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        cv2.imwrite(result_path, result_img)
        
        # 构建结果URL（在实际部署中，这应该是完整的URL）
        result_url = f"/api/results/{result_filename}"
        
        # 计算健康度和腐烂度 - 修改为根据检测结果的置信度计算
        health_level = 0.0
        rot_level = 0.0
        
        # 如果只检测到一个苹果，则直接根据置信度计算健康度和腐烂度
        if len(detections) == 1:
            det = detections[0]
            confidence_percent = det['confidence'] * 100
            
            if det['class_name'] == 'rotten_apple':
                # 腐烂苹果：腐烂度 = 置信度，健康度 = (100 - 置信度)
                rot_level = confidence_percent
                health_level = 100 - confidence_percent
            else:
                # 健康苹果：健康度 = 置信度，腐烂度 = (100 - 置信度)
                health_level = confidence_percent
                rot_level = 100 - confidence_percent
                
        # 多个检测结果或无检测结果的情况下，使用类别计数方法
        else:
            ripeness_count = 0
            # 计算健康苹果的总数（所有成熟度类别）
            if '100-_ripeness' in class_counts:
                ripeness_count += class_counts['100-_ripeness']
            if '75-_ripeness' in class_counts:
                ripeness_count += class_counts['75-_ripeness']
            if '50-_ripeness' in class_counts:
                ripeness_count += class_counts['50-_ripeness']
            
            # 计算腐烂苹果的数量
            rotten_count = class_counts.get('rotten_apple', 0)
            
            # 计算总苹果数
            total_apples = ripeness_count + rotten_count
            
            if total_apples > 0:
                # 健康度：健康苹果占总数的百分比
                health_level = (ripeness_count / total_apples) * 100.0
                # 腐烂度：腐烂苹果占总数的百分比
                rot_level = (rotten_count / total_apples) * 100.0
        
        # 返回JSON响应
        inference_time = time.time() - timestamp
        return jsonify({
            'success': True,
            'detections': detections,
            'class_counts': class_counts,
            'result_image': result_url,
            'detection_count': len(detections),
            'health_level': round(health_level, 2),  # 保留两位小数
            'rot_level': round(rot_level, 2),  # 保留两位小数
            'processing_time': round(inference_time * 1000, 2)  # 处理时间（毫秒）
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'处理请求时出错: {str(e)}'
        }), 500

# 提供结果图像
@app.route('/api/results/<filename>', methods=['GET'])
def get_result(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename))

# 健康检查端点
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'model': os.path.basename(MODEL_XML),
        'backend': 'OpenVINO Runtime',
        'timestamp': int(time.time())
    })

# 主函数
if __name__ == '__main__':
    # 打印服务器信息
    print(f"模型XML文件: {MODEL_XML}")
    print(f"模型BIN文件: {MODEL_BIN}")
    print(f"数据YAML: {DATA_YAML}")
    print(f"上传文件夹: {UPLOAD_FOLDER}")
    print(f"结果文件夹: {RESULT_FOLDER}")
    
    # 加载类别名称并打印
    class_names = get_class_names(DATA_YAML)
    print(f"使用类别名称: {class_names}")
    
    # 启动Flask服务器
    # 设置debug=False以减少不必要的日志并提高性能
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True) 