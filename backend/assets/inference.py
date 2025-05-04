#!/usr/bin/env python3
import os
import argparse
import cv2
import numpy as np
import yaml
import torch
from ultralytics import YOLO
from pathlib import Path
import onnxruntime as ort
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with trained YOLOv8 model')
    parser.add_argument('--model', type=str, default='/Users/panyiming/WeChatProjects/miniprogram-1/backend/assets/best.onnx',
                        help='Model weights path (.pt or .onnx)')
    parser.add_argument('--img', type=str, required=True,
                        help='Image path (jpg or png)')
    parser.add_argument('--data', type=str, default='dataset/data.yaml',
                        help='Path to data.yaml file for class names')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='Confidence threshold for detection')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on (cpu, cuda, or blank for auto)')
    parser.add_argument('--save', action='store_true',
                        help='Save the output image with detections')
    parser.add_argument('--show', action='store_true', default=True,
                        help='Show the result with cv2.imshow')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output for debugging')
    return parser.parse_args()

def get_class_names(data_yaml):
    """Load class names from data.yaml file"""
    try:
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
            if 'names' in data:
                # If names is a list
                if isinstance(data['names'], list):
                    return data['names']
                # If names is a dict (with indices as keys)
                elif isinstance(data['names'], dict):
                    # Sort by key to get the correct order
                    max_idx = max(int(k) for k in data['names'].keys())
                    result = [''] * (max_idx + 1)
                    for idx, name in data['names'].items():
                        result[int(idx)] = name
                    return result
    except Exception as e:
        print(f"Warning: Could not load class names from {data_yaml}: {e}")
    
    # Default class names if data.yaml can't be read
    return ['100-_ripeness', '50-_ripeness', '75-_ripeness', 'rotten_apple']

def get_class_colors(class_names):
    """Generate visually distinct colors for each class"""
    # Define consistent colors for common apple ripeness categories
    color_map = {
        '100-_ripeness': [0, 255, 0],    # Green for fully ripe
        '50-_ripeness': [0, 165, 255],   # Orange for half ripe
        '75-_ripeness': [0, 255, 127],   # Light green for mostly ripe
        'rotten_apple': [0, 0, 255]      # Red for rotten
    }
    
    colors = []
    np.random.seed(42)  # for reproducible random colors
    
    for class_name in class_names:
        if class_name in color_map:
            colors.append(color_map[class_name])
        else:
            # Generate random color for any other class
            colors.append([int(c) for c in np.random.randint(0, 255, 3)])
    
    return colors

def plot_one_box(box, img, color, label=None, line_thickness=3):
    """Plot one bounding box on image"""
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def predict_with_pt(model_path, img_path, conf_thres, device='', class_names=None, verbose=False):
    """Run inference with PyTorch .pt model"""
    # Load model
    model = YOLO(model_path, task='detect')  # 明确指定任务类型为检测
    
    # Print model class information if verbose
    if verbose:
        print(f"Model loaded from: {model_path}")
        print(f"Model class names (from model): {model.names}")
        print(f"Class names (from data.yaml): {class_names}")
    
    # Run inference
    results = model.predict(
        source=img_path,
        conf=conf_thres,
        device=device,
        verbose=verbose
    )[0]
    
    # Print raw detection info if verbose
    if verbose:
        print("\nRaw detection info:")
        print(f"Boxes: {results.boxes}")
        if len(results.boxes) > 0:
            print(f"  - Shape: {results.boxes.xyxy.shape}")
            print(f"  - Classes detected: {results.boxes.cls.tolist()}")
            print(f"  - Confidences: {results.boxes.conf.tolist()}")
    
    # Get detection information
    img = cv2.imread(img_path)
    detections = []
    
    # Create class mapping: try to use model's internal names if available
    if hasattr(model, 'names') and model.names:
        if verbose:
            print("\nUsing model's internal class names")
        model_class_names = model.names
    else:
        if verbose:
            print("\nFalling back to data.yaml class names")
        model_class_names = {i: name for i, name in enumerate(class_names)}
    
    if verbose:
        print(f"Class mapping: {model_class_names}")
    
    for i, det in enumerate(results.boxes.data):
        x1, y1, x2, y2, conf, cls = det
        class_id = int(cls)
        
        # Try to get class name from model's internal names first
        if class_id in model_class_names:
            class_name = model_class_names[class_id]
        elif class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"Class {class_id}"
        
        if verbose:
            print(f"\nDetection #{i+1}:")
            print(f"  - Box: {[float(x1), float(y1), float(x2), float(y2)]}")
            print(f"  - Class ID: {class_id}")
            print(f"  - Class Name: {class_name}")
            print(f"  - Confidence: {float(conf)}")
        
        detections.append({
            'box': [float(x1), float(y1), float(x2), float(y2)],
            'confidence': float(conf),
            'class_id': class_id,
            'class_name': class_name
        })
    
    return img, detections, results

def main():
    args = parse_args()
    
    # Check if the image file exists
    if not os.path.isfile(args.img):
        print(f"Error: Image file '{args.img}' not found")
        return
    
    # Load class names from data.yaml
    class_names = get_class_names(args.data)
    if args.verbose:
        print(f"Data YAML path: {args.data}")
        print(f"Class names from data.yaml: {class_names}")
    else:
        print(f"Using class names: {class_names}")
    
    # Get colors for each class with ripeness-specific colors
    colors = get_class_colors(class_names)
    
    # Use .pt model path if provided, otherwise try .onnx
    model_path = args.model
    img_path = args.img
    
    # Run inference based on model type
    if model_path.endswith('.pt'):
        img, detections, results = predict_with_pt(
            model_path=model_path,
            img_path=img_path,
            conf_thres=args.conf_thres,
            device=args.device,
            class_names=class_names,
            verbose=args.verbose
        )
    elif model_path.endswith('.onnx'):
        print("Using ONNX model directly with ultralytics")
        # Use YOLO for ONNX model as well - it handles both formats
        model = YOLO(model_path, task='detect')  # 明确指定任务类型为检测
        results = model.predict(
            source=img_path,
            conf=args.conf_thres,
            device=args.device,
            verbose=args.verbose
        )[0]
        
        img = cv2.imread(img_path)
        detections = []
        
        for i, det in enumerate(results.boxes.data):
            x1, y1, x2, y2, conf, cls = det
            class_id = int(cls)
            
            # Get class name
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
    else:
        print(f"Error: Unsupported model format for '{model_path}'")
        print("Please use either a PyTorch (.pt) or ONNX (.onnx) model.")
        return
    
    # Print class counts
    class_counts = {}
    for det in detections:
        cls_name = det['class_name']
        if cls_name in class_counts:
            class_counts[cls_name] += 1
        else:
            class_counts[cls_name] = 1
    
    print("\nDetections by class:")
    for cls_name, count in class_counts.items():
        print(f"  {cls_name}: {count}")
    
    # Draw detections on the image
    result_img = img.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det['box'])
        conf = det['confidence']
        class_id = det['class_id']
        class_name = det['class_name']
        
        # Get color for this class (safely handle out of range indices)
        if class_id < len(colors):
            color = colors[class_id]
        else:
            color = [0, 0, 255]  # Default to red
            
        label = f"{class_name} {conf:.2f}"
        plot_one_box([x1, y1, x2, y2], result_img, color, label)
    
    # Display results
    print(f"\nDetected {len(detections)} objects:")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det['class_name']} (Confidence: {det['confidence']:.2f})")
    
    # Save result if requested
    if args.save:
        output_name = os.path.splitext(os.path.basename(args.img))[0] + "_detection.jpg"
        cv2.imwrite(output_name, result_img)
        print(f"\nDetection result saved as '{output_name}'")
    
    # Display the image
    if args.show:
        cv2.imshow("Detection Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()