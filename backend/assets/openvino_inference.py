#!/usr/bin/env python3
import os
import cv2
import numpy as np
import yaml
import time
from pathlib import Path
import traceback
import onnxruntime as ort  # 使用ONNX Runtime替代OpenVINO

# --- Helper Functions (Adapted from flask_inference.py) ---

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
    """Preprocess the image for YOLOv8 input."""
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
    """Postprocess YOLOv8 outputs."""
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

# --- Main ONNX Runtime Inference Function ---

def run_optimized_inference(img_path, model_path, data_yaml, conf_thres=0.6, iou_thres=0.45):
    """使用 ONNX Runtime 运行优化推理"""
    start_time = time.time()

    # 1. Load Class Names and Colors
    class_names = get_class_names(data_yaml)
    colors = get_class_colors(class_names)

    # 2. Initialize ONNX Runtime Session
    try:
        # 创建ONNX运行时优化的会话
        # 使用 MacOS 上的ExecutionProvider
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4  # 并行线程数
        
        # 创建ONNX会话
        session = ort.InferenceSession(
            model_path, 
            sess_options=sess_options,
            providers=['CPUExecutionProvider']  # MacOS上使用CPU执行
        )
        
        # 获取输入形状
        inputs = session.get_inputs()
        input_shape = inputs[0].shape  # [1, 3, 640, 640]
        input_name = inputs[0].name
        
        # 获取输出名称
        output_name = session.get_outputs()[0].name
        
        print(f"模型加载时间: {time.time() - start_time:.4f} 秒")

    except Exception as e:
        traceback.print_exc()
        return None, [], f"错误：无法加载或初始化ONNX模型 '{model_path}': {e}"

    # 3. Check Image File
    if not os.path.isfile(img_path):
        return None, [], f"错误：图像文件 '{img_path}' 未找到"

    # 4. Preprocess Image
    preprocess_start = time.time()
    try:
        input_tensor, original_img, scaling_info = preprocess_image(img_path, input_shape)
    except Exception as e:
        traceback.print_exc()
        return None, [], f"错误：图像预处理失败: {e}"
    print(f"图像预处理时间: {time.time() - preprocess_start:.4f} 秒")

    # 5. Run Inference
    inference_start = time.time()
    try:
        outputs = session.run([output_name], {input_name: input_tensor})
        output_tensor = outputs[0]  # 获取输出张量

    except Exception as e:
        traceback.print_exc()
        return None, [], f"错误：ONNX运行时推理失败: {e}"
    print(f"纯推理时间: {time.time() - inference_start:.4f} 秒")

    # 6. Postprocess Results
    postprocess_start = time.time()
    try:
        detections = postprocess_results(output_tensor, original_img, scaling_info, conf_thres, iou_thres)
        
        # Add class names to detections
        for det in detections:
            class_id = det['class_id']
            if 0 <= class_id < len(class_names):
                det['class_name'] = class_names[class_id]
            else:
                det['class_name'] = f"未知类别 {class_id}"
                
    except Exception as e:
        traceback.print_exc()
        return None, [], f"错误：结果后处理失败: {e}"
    print(f"后处理时间: {time.time() - postprocess_start:.4f} 秒")

    # 7. Draw Detections and Count Classes
    result_img = original_img.copy()
    class_counts = {}
    for det in detections:
        x1, y1, x2, y2 = map(int, det['box'])
        conf = det['confidence']
        class_id = det['class_id']
        class_name = det['class_name']

        # Get color safely
        color = colors[class_id] if 0 <= class_id < len(colors) else [0, 0, 255]
        label = f"{class_name} {conf:.2f}"
        plot_one_box([x1, y1, x2, y2], result_img, color, label)

        # Count classes
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    total_time = time.time() - start_time
    print(f"总处理时间 (ONNX Runtime): {total_time:.4f} 秒")
    
    return result_img, detections, class_counts


# --- Example Usage ---
if __name__ == '__main__':
    # Define paths relative to this script's location
    SCRIPT_DIR = Path(__file__).resolve().parent
    DEFAULT_MODEL_PATH = SCRIPT_DIR / 'best.onnx'
    DEFAULT_DATA_YAML = SCRIPT_DIR / 'data.yaml'
    # Create dummy image/folders for testing if they don't exist
    DEFAULT_IMAGE_DIR = SCRIPT_DIR / 'uploads'
    DEFAULT_RESULT_DIR = SCRIPT_DIR / 'results_optimized'
    DEFAULT_IMAGE_PATH = DEFAULT_IMAGE_DIR / 'test_apple.jpg'

    DEFAULT_IMAGE_DIR.mkdir(exist_ok=True)
    DEFAULT_RESULT_DIR.mkdir(exist_ok=True)

    print("--- ONNX Runtime 推理测试 ---")
    print(f"模型: {DEFAULT_MODEL_PATH}")
    print(f"数据配置: {DEFAULT_DATA_YAML}")
    print(f"测试图像: {DEFAULT_IMAGE_PATH}")

    if not DEFAULT_MODEL_PATH.is_file():
        print(f"错误：模型文件 '{DEFAULT_MODEL_PATH}' 不存在。请确保模型文件在此处。")
    elif not DEFAULT_DATA_YAML.is_file():
         print(f"警告：数据配置文件 '{DEFAULT_DATA_YAML}' 不存在。将使用默认类别名称。")
    elif not DEFAULT_IMAGE_PATH.is_file():
         print(f"错误：测试图像 '{DEFAULT_IMAGE_PATH}' 不存在。请提供有效的测试图像。")
    else:
        result_img, detections, class_counts = run_optimized_inference(
            str(DEFAULT_IMAGE_PATH),
            str(DEFAULT_MODEL_PATH),
            str(DEFAULT_DATA_YAML),
            conf_thres=0.2, # 降低置信度阈值
        )

        if result_img is not None:
            print(f"检测到 {len(detections)} 个目标:")
            for det in detections:
                print(f"  - 类别: {det['class_name']}, 置信度: {det['confidence']:.2f}, 边界框: {det['box']}")
            print(f"类别统计: {class_counts}")

            # Save the result image
            output_path = DEFAULT_RESULT_DIR / f"{DEFAULT_IMAGE_PATH.stem}_optimized_result.jpg"
            cv2.imwrite(str(output_path), result_img)
            print(f"结果图像已保存至: {output_path}")
        else:
            # Detections list contains the error message if result_img is None
            print(f"推理失败: {detections}")

    print("--- 测试完成 ---") 