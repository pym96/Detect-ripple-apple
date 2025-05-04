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
from ultralytics import YOLO
import io

app = Flask(__name__)
CORS(app)  # 启用跨域请求支持

# 配置
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
RESULT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best.onnx')
DATA_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.yaml')
CONF_THRESHOLD = 0.6

# 确保上传和结果目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传大小为16MB

# 加载类别名称
def get_class_names(data_yaml):
    """从data.yaml文件加载类别名称"""
    try:
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
            if 'names' in data:
                # 如果names是列表
                if isinstance(data['names'], list):
                    return data['names']
                # 如果names是字典（以索引为键）
                elif isinstance(data['names'], dict):
                    # 按键排序以获得正确顺序
                    max_idx = max(int(k) for k in data['names'].keys())
                    result = [''] * (max_idx + 1)
                    for idx, name in data['names'].items():
                        result[int(idx)] = name
                    return result
    except Exception as e:
        print(f"警告：无法从{data_yaml}加载类别名称: {e}")
    
    # 如果data.yaml无法读取，使用默认类别名称
    return ['100-_ripeness', '50-_ripeness', '75-_ripeness', 'rotten_apple']

# 为每个类别生成颜色
def get_class_colors(class_names):
    """为每个类别生成视觉上不同的颜色"""
    # 为常见的苹果成熟度类别定义一致的颜色
    color_map = {
        '100-_ripeness': [0, 255, 0],    # 绿色表示完全成熟
        '50-_ripeness': [0, 165, 255],   # 橙色表示半成熟
        '75-_ripeness': [0, 255, 127],   # 浅绿色表示大部分成熟
        'rotten_apple': [0, 0, 255]      # 红色表示腐烂
    }
    
    colors = []
    np.random.seed(42)  # 为了可重现的随机颜色
    
    for class_name in class_names:
        if class_name in color_map:
            colors.append(color_map[class_name])
        else:
            # 为任何其他类别生成随机颜色
            colors.append([int(c) for c in np.random.randint(0, 255, 3)])
    
    return colors

# 在图像上绘制一个边界框
def plot_one_box(box, img, color, label=None, line_thickness=3):
    """在图像上绘制一个边界框"""
    # 在图像img上绘制一个边界框
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # 线/字体粗细
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    
    if label:
        tf = max(tl - 1, 1)  # 字体粗细
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # 填充
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

# 检查文件扩展名是否允许
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 使用ONNX模型进行推理
def run_inference(img_path, conf_thres=CONF_THRESHOLD):
    """使用ONNX模型运行推理"""
    # 加载类别名称
    class_names = get_class_names(DATA_YAML)
    colors = get_class_colors(class_names)
    
    # 检查图像文件是否存在
    if not os.path.isfile(img_path):
        return None, [], f"错误：图像文件'{img_path}'未找到"
    
    try:
        # 使用YOLO模型（它可以处理ONNX格式）
        model = YOLO(MODEL_PATH, task='detect')  # 明确指定任务类型为检测
        results = model.predict(
            source=img_path,
            conf=conf_thres,
            verbose=False
        )[0]
        
        # 获取检测信息
        img = cv2.imread(img_path)
        detections = []
        
        for i, det in enumerate(results.boxes.data):
            x1, y1, x2, y2, conf, cls = det
            class_id = int(cls)
            
            # 获取类别名称
            if class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                class_name = f"Class {class_id}"
            
            detections.append({
                'box': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(conf),
                'class_id': class_id,
                'class_name': class_name
            })
        
        # 在图像上绘制检测结果
        result_img = img.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det['box'])
            conf = det['confidence']
            class_id = det['class_id']
            class_name = det['class_name']
            
            # 获取此类别的颜色（安全处理超出范围的索引）
            if class_id < len(colors):
                color = colors[class_id]
            else:
                color = [0, 0, 255]  # 默认为红色
                
            label = f"{class_name} {conf:.2f}"
            plot_one_box([x1, y1, x2, y2], result_img, color, label)
        
        # 按类别统计检测结果
        class_counts = {}
        for det in detections:
            cls_name = det['class_name']
            if cls_name in class_counts:
                class_counts[cls_name] += 1
            else:
                class_counts[cls_name] = 1
        
        return result_img, detections, class_counts
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, [], f"推理过程中出错: {str(e)}"

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
        
        # 运行推理
        result_img, detections, class_counts = run_inference(img_path)
        
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
        
        # 计算健康度和腐烂度
        health_level = 0.0
        rot_level = 0.0
        
        # 健康度基于成熟度类别（100%、75%、50%成熟度）
        # 腐烂度基于腐烂苹果类别
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
        return jsonify({
            'success': True,
            'detections': detections,
            'class_counts': class_counts,
            'result_image': result_url,
            'detection_count': len(detections),
            'health_level': round(health_level, 2),  # 保留两位小数
            'rot_level': round(rot_level, 2)  # 保留两位小数
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
        'model': os.path.basename(MODEL_PATH),
        'timestamp': int(time.time())
    })

# 主函数
if __name__ == '__main__':
    # 打印服务器信息
    print(f"模型路径: {MODEL_PATH}")
    print(f"数据YAML: {DATA_YAML}")
    print(f"上传文件夹: {UPLOAD_FOLDER}")
    print(f"结果文件夹: {RESULT_FOLDER}")
    
    # 加载类别名称并打印
    class_names = get_class_names(DATA_YAML)
    print(f"使用类别名称: {class_names}")
    
    # 启动Flask服务器
    app.run(host='0.0.0.0', port=5000, debug=True)