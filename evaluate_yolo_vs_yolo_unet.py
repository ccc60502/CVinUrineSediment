# -*- coding: utf-8 -*-
"""
YOLO单独检测 vs YOLO+改进UNet两阶段方法对比评估脚本

评估指标：
- Precision（精确率）
- Recall（召回率）
- F1-Score
- mAP@0.5, mAP@0.75
- TP/FP/FN统计
- 过滤率（YOLO+UNet过滤掉的框比例）
- 假阳性减少率
- 假阴性增加率
- 推理时间（YOLO单独、YOLO+UNet总时间、UNet平均时间）
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import time
import json
from tqdm import tqdm

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("警告: 未安装pandas，将跳过CSV/Excel导出（可运行: pip install pandas）")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: 未安装matplotlib，将跳过图表生成（可运行: pip install matplotlib）")

# ================== 配置参数 ==================
# YOLO模型路径
YOLO_MODEL_PATH = r"E:\Data_Industry\Unet_wdir\2stage_wdir\best.pt"

# UNet模型路径（改进版，含注意力门控和转置卷积）
UNET_MODEL_PATH = r"E:\Data_Industry\膀胱癌细胞检测_文章\Unet_attention_文件\模型评估文件\focalT_epoch150\last_model_1.pth"

# 测试数据路径
TEST_IMAGE_DIR = r"E:\Data_Industry\dateByJ_1004560_20251201\Just_Cancer\image"
TEST_LABEL_DIR = r"E:\Data_Industry\dateByJ_1004560_20251201\Just_Cancer\labels"  # YOLO格式标签目录

# 输出结果目录
OUTPUT_DIR = r"E:\Data_Industry\膀胱癌细胞检测_文章\YOLO_vs_YOLO_UNet对比评估"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# YOLO参数
YOLO_CONF = 0.25
YOLO_CLASSES = [0]

# UNet参数
UNET_INPUT_SIZE = (1024, 1024)
UNET_NUM_CLASSES = 2
MASK_THRESHOLD = 0.6
FILTER_BY_UNET = True
UNET_FILTER_THRESHOLD = 0.01

# IoU阈值（用于mAP计算）
IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

# 预热次数
WARMUP_ITERATIONS = 5
# ==============================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载模型
print("加载模型...")
yolo_model = YOLO(YOLO_MODEL_PATH)

# 加载改进版UNet
print("加载改进版UNet模型（含注意力门控和转置卷积）...")
from model.unet_resnet_enhanced import Unet
unet_model = Unet(num_classes=UNET_NUM_CLASSES, enhanced=True)
state_dict = torch.load(UNET_MODEL_PATH, map_location=device)
if any(k.startswith('model.') for k in state_dict.keys()):
    unet_model.load_state_dict(state_dict, strict=False)
else:
    new_state_dict = {'model.' + k: v for k, v in state_dict.items()}
    unet_model.load_state_dict(new_state_dict, strict=False)
unet_model.eval()
unet_model.to(device)

from utils.utils import cvtColor, preprocess_input, resize_image
print("模型加载完成！")


def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def load_yolo_labels(label_path, img_width, img_height):
    """加载YOLO格式的标签文件"""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            boxes.append([x1, y1, x2, y2, class_id])
    
    return boxes


def yolo_only_detection(image_path):
    """仅使用YOLO进行检测，返回检测结果和推理时间"""
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    results = yolo_model.predict(
        source=image_path,
        conf=YOLO_CONF,
        classes=YOLO_CLASSES,
        save=False,
        verbose=False
    )[0]
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    inference_time = (time.time() - start_time) * 1000  # 毫秒
    
    detections = []
    for box in results.boxes:
        bbox = box.xyxy[0].cpu().numpy().astype(float)
        conf = box.conf[0].item()
        detections.append({
            'bbox': bbox.tolist(),
            'conf': conf,
            'class': int(box.cls[0].item())
        })
    
    return detections, inference_time


def yolo_unet_two_stage(image_path):
    """YOLO+UNet两阶段方法，返回检测结果和详细时间信息"""
    image_pil = Image.open(image_path).convert("RGB")
    orig_np = np.array(image_pil)
    h, w = orig_np.shape[:2]
    
    # YOLO检测阶段
    torch.cuda.synchronize() if device.type == 'cuda' else None
    yolo_start = time.time()
    
    results = yolo_model.predict(
        source=image_path,
        conf=YOLO_CONF,
        classes=YOLO_CLASSES,
        save=False,
        verbose=False
    )[0]
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    yolo_time = (time.time() - yolo_start) * 1000  # 毫秒
    
    detections = []
    unet_times = []
    filtered_count = 0
    
    # UNet分割和过滤阶段
    for idx, box in enumerate(results.boxes):
        bbox = box.xyxy[0].cpu().numpy().astype(int)
        conf = box.conf[0].item()
        
        # 裁剪和预处理
        x1, y1, x2, y2 = bbox
        w_box, h_box = x2 - x1, y2 - y1
        
        expand_w = int(w_box * 0.2)
        expand_h = int(h_box * 0.2)
        x1_exp = max(0, x1 - expand_w)
        y1_exp = max(0, y1 - expand_h)
        x2_exp = min(w, x2 + expand_w)
        y2_exp = min(h, y2 + expand_h)
        
        patch = image_pil.crop((x1_exp, y1_exp, x2_exp, y2_exp))
        patch = cvtColor(patch)
        patch_resized, nw, nh = resize_image(patch, (UNET_INPUT_SIZE[1], UNET_INPUT_SIZE[0]))
        
        pad_x = (UNET_INPUT_SIZE[1] - nw) // 2
        pad_y = (UNET_INPUT_SIZE[0] - nh) // 2
        
        # UNet推理
        patch_input = np.expand_dims(
            np.transpose(preprocess_input(np.array(patch_resized, np.float32)), (2, 0, 1)), 0
        )
        patch_tensor = torch.from_numpy(patch_input).type(torch.FloatTensor).to(device)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        unet_start = time.time()
        
        with torch.no_grad():
            output = unet_model(patch_tensor)[0]
            pr = F.softmax(output, dim=0).cpu().numpy()
            prob_map = pr[1]
            mask_1024 = (prob_map >= MASK_THRESHOLD).astype(np.uint8)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        unet_time = (time.time() - unet_start) * 1000  # 毫秒
        unet_times.append(unet_time)
        
        # 映射回原图
        mask_actual = mask_1024[pad_y:pad_y+nh, pad_x:pad_x+nw]
        crop_w, crop_h = x2_exp - x1_exp, y2_exp - y1_exp
        mask_resized = cv2.resize(mask_actual, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
        
        instance_mask = np.zeros((h, w), dtype=np.uint8)
        y_end = min(y1_exp + crop_h, h)
        x_end = min(x1_exp + crop_w, w)
        actual_h = y_end - y1_exp
        actual_w = x_end - x1_exp
        if actual_h < crop_h or actual_w < crop_w:
            mask_resized = mask_resized[:actual_h, :actual_w]
        instance_mask[y1_exp:y_end, x1_exp:x_end] = mask_resized
        
        # UNet过滤
        if FILTER_BY_UNET:
            x1_orig, y1_orig, x2_orig, y2_orig = bbox
            x1_orig = max(0, x1_orig)
            y1_orig = max(0, y1_orig)
            x2_orig = min(w, x2_orig)
            y2_orig = min(h, y2_orig)
            
            bbox_mask = instance_mask[y1_orig:y2_orig, x1_orig:x2_orig]
            bbox_area = (x2_orig - x1_orig) * (y2_orig - y1_orig)
            if bbox_area > 0:
                cancer_pixels = np.sum(bbox_mask == 1)
                cancer_ratio = cancer_pixels / bbox_area
                if cancer_ratio < UNET_FILTER_THRESHOLD:
                    filtered_count += 1
                    continue  # 过滤掉这个检测框
        
        detections.append({
            'bbox': bbox.tolist(),
            'conf': conf,
            'class': 0
        })
    
    total_time = yolo_time + sum(unet_times)
    avg_unet_time = np.mean(unet_times) if unet_times else 0
    
    return detections, {
        'total_time': total_time,
        'yolo_time': yolo_time,
        'unet_total_time': sum(unet_times),
        'unet_avg_time': avg_unet_time,
        'unet_count': len(unet_times),
        'filtered_count': filtered_count
    }


def evaluate_detection(pred_boxes, gt_boxes, iou_threshold=0.5):
    """评估检测结果，返回详细指标"""
    if len(gt_boxes) == 0:
        if len(pred_boxes) == 0:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}
        else:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': len(pred_boxes), 'fn': 0}
    
    if len(pred_boxes) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': len(gt_boxes)}
    
    matched_gt = set()
    tp = 0
    fp = 0
    
    pred_boxes_sorted = sorted(pred_boxes, key=lambda x: x['conf'], reverse=True)
    
    for pred_box in pred_boxes_sorted:
        pred_bbox = pred_box['bbox']
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            gt_bbox = gt_box[:4]
            iou = calculate_iou(pred_bbox, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
    
    fn = len(gt_boxes) - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def calculate_map(pred_boxes_list, gt_boxes_list, iou_thresholds):
    """计算mAP（mean Average Precision）- 简化版本"""
    aps = []
    
    for iou_thresh in iou_thresholds:
        # 收集所有预测和真实框
        all_preds = []
        all_gts = []
        
        for pred_boxes, gt_boxes in zip(pred_boxes_list, gt_boxes_list):
            for pb in pred_boxes:
                all_preds.append(pb)
            for gb in gt_boxes:
                all_gts.append(gb[:4])  # 只取坐标
        
        if len(all_gts) == 0:
            aps.append(0.0)
            continue
        
        # 按置信度排序
        all_preds_sorted = sorted(all_preds, key=lambda x: x['conf'], reverse=True)
        
        # 计算TP/FP
        matched_gts = set()
        tp_list = []
        fp_list = []
        
        for pred_box in all_preds_sorted:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(all_gts):
                if gt_idx in matched_gts:
                    continue
                iou = calculate_iou(pred_box['bbox'], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_thresh:
                tp_list.append(1)
                fp_list.append(0)
                matched_gts.add(best_gt_idx)
            else:
                tp_list.append(0)
                fp_list.append(1)
        
        # 计算累积TP和FP
        tp_cumsum = np.cumsum(tp_list)
        fp_cumsum = np.cumsum(fp_list)
        
        # 计算Precision和Recall
        recalls = tp_cumsum / len(all_gts) if len(all_gts) > 0 else np.zeros(len(tp_cumsum))
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum) if len(tp_cumsum) > 0 else np.zeros(len(tp_cumsum))
        
        # 计算AP（使用11点插值法）
        ap = 0
        for r in np.arange(0, 1.1, 0.1):
            p_max = max([p for p, rec in zip(precisions, recalls) if rec >= r], default=0.0)
            ap += p_max / 11
        aps.append(ap)
    
    return np.mean(aps) if len(aps) > 0 else 0.0


def main():
    """主评估函数"""
    # 获取测试图像列表
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(TEST_IMAGE_DIR).glob(f'*{ext}'))
        image_files.extend(Path(TEST_IMAGE_DIR).glob(f'*{ext.upper()}'))
    
    if len(image_files) == 0:
        print(f"错误：在 {TEST_IMAGE_DIR} 中未找到图像文件！")
        return
    
    print(f"找到 {len(image_files)} 张测试图像")
    
    # 预热
    print("预热模型...")
    if len(image_files) > 0:
        for _ in range(WARMUP_ITERATIONS):
            yolo_only_detection(str(image_files[0]))
            yolo_unet_two_stage(str(image_files[0]))
    print("预热完成！")
    
    # 存储结果
    per_sample_results = []
    
    # YOLO单独检测统计
    yolo_only_stats = {
        'precision': [], 'recall': [], 'f1': [],
        'tp': 0, 'fp': 0, 'fn': 0,
        'inference_time': []
    }
    
    # YOLO+UNet统计
    yolo_unet_stats = {
        'precision': [], 'recall': [], 'f1': [],
        'tp': 0, 'fp': 0, 'fn': 0,
        'total_time': [], 'yolo_time': [], 'unet_avg_time': [],
        'filtered_count': 0, 'original_count': 0
    }
    
    # 用于mAP计算
    yolo_only_preds = []
    yolo_unet_preds = []
    all_gt_boxes = []
    
    # 逐图像评估
    print("\n开始评估...")
    for image_path in tqdm(image_files, desc="评估进度"):
        image_name = image_path.stem
        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]
        
        # 加载ground truth
        label_path = os.path.join(TEST_LABEL_DIR, f"{image_name}.txt")
        gt_boxes = load_yolo_labels(label_path, w, h)
        
        # YOLO单独检测
        yolo_detections, yolo_time = yolo_only_detection(str(image_path))
        
        # YOLO+UNet两阶段
        yolo_unet_detections, time_info = yolo_unet_two_stage(str(image_path))
        
        # 评估（IoU=0.5）
        yolo_result = evaluate_detection(yolo_detections, gt_boxes, 0.5)
        yolo_unet_result = evaluate_detection(yolo_unet_detections, gt_boxes, 0.5)
        
        # 累积统计
        yolo_only_stats['precision'].append(yolo_result['precision'])
        yolo_only_stats['recall'].append(yolo_result['recall'])
        yolo_only_stats['f1'].append(yolo_result['f1'])
        yolo_only_stats['tp'] += yolo_result['tp']
        yolo_only_stats['fp'] += yolo_result['fp']
        yolo_only_stats['fn'] += yolo_result['fn']
        yolo_only_stats['inference_time'].append(yolo_time)
        
        yolo_unet_stats['precision'].append(yolo_unet_result['precision'])
        yolo_unet_stats['recall'].append(yolo_unet_result['recall'])
        yolo_unet_stats['f1'].append(yolo_unet_result['f1'])
        yolo_unet_stats['tp'] += yolo_unet_result['tp']
        yolo_unet_stats['fp'] += yolo_unet_result['fp']
        yolo_unet_stats['fn'] += yolo_unet_result['fn']
        yolo_unet_stats['total_time'].append(time_info['total_time'])
        yolo_unet_stats['yolo_time'].append(time_info['yolo_time'])
        yolo_unet_stats['unet_avg_time'].append(time_info['unet_avg_time'])
        yolo_unet_stats['filtered_count'] += time_info['filtered_count']
        yolo_unet_stats['original_count'] += len(yolo_detections)
        
        # 用于mAP计算
        yolo_only_preds.append(yolo_detections)
        yolo_unet_preds.append(yolo_unet_detections)
        all_gt_boxes.append(gt_boxes)
        
        # 存储每个样本结果
        per_sample_results.append({
            '图像名称': image_name,
            'GT目标数': len(gt_boxes),
            'YOLO_检测数': len(yolo_detections),
            'YOLO+UNet_检测数': len(yolo_unet_detections),
            '过滤数量': time_info['filtered_count'],
            '过滤率_%': (time_info['filtered_count'] / len(yolo_detections) * 100) if len(yolo_detections) > 0 else 0,
            'YOLO_Precision': yolo_result['precision'],
            'YOLO_Recall': yolo_result['recall'],
            'YOLO_F1': yolo_result['f1'],
            'YOLO_TP': yolo_result['tp'],
            'YOLO_FP': yolo_result['fp'],
            'YOLO_FN': yolo_result['fn'],
            'YOLO+UNet_Precision': yolo_unet_result['precision'],
            'YOLO+UNet_Recall': yolo_unet_result['recall'],
            'YOLO+UNet_F1': yolo_unet_result['f1'],
            'YOLO+UNet_TP': yolo_unet_result['tp'],
            'YOLO+UNet_FP': yolo_unet_result['fp'],
            'YOLO+UNet_FN': yolo_unet_result['fn'],
            'Precision提升': yolo_unet_result['precision'] - yolo_result['precision'],
            'Recall变化': yolo_unet_result['recall'] - yolo_result['recall'],
            'F1提升': yolo_unet_result['f1'] - yolo_result['f1'],
            'FP减少': yolo_result['fp'] - yolo_unet_result['fp'],
            'FN增加': yolo_unet_result['fn'] - yolo_result['fn'],
            'YOLO_推理时间_ms': yolo_time,
            'YOLO+UNet_总时间_ms': time_info['total_time'],
            'YOLO+UNet_YOLO时间_ms': time_info['yolo_time'],
            'YOLO+UNet_UNet平均时间_ms': time_info['unet_avg_time'],
            '时间增加_ms': time_info['total_time'] - yolo_time,
            '时间增加_%': ((time_info['total_time'] - yolo_time) / yolo_time * 100) if yolo_time > 0 else 0,
        })
    
    if len(per_sample_results) == 0:
        print("错误：没有成功评估的样本！")
        return
    
    # 计算mAP
    print("\n计算mAP...")
    yolo_only_map = calculate_map(yolo_only_preds, all_gt_boxes, IOU_THRESHOLDS)
    yolo_unet_map = calculate_map(yolo_unet_preds, all_gt_boxes, IOU_THRESHOLDS)
    
    # 计算平均指标
    print("\n" + "="*70)
    print("评估结果汇总")
    print("="*70)
    
    # YOLO单独检测
    yolo_avg_precision = np.mean(yolo_only_stats['precision']) * 100
    yolo_avg_recall = np.mean(yolo_only_stats['recall']) * 100
    yolo_avg_f1 = np.mean(yolo_only_stats['f1']) * 100
    yolo_avg_time = np.mean(yolo_only_stats['inference_time'])
    
    # YOLO+UNet
    unet_avg_precision = np.mean(yolo_unet_stats['precision']) * 100
    unet_avg_recall = np.mean(yolo_unet_stats['recall']) * 100
    unet_avg_f1 = np.mean(yolo_unet_stats['f1']) * 100
    unet_avg_total_time = np.mean(yolo_unet_stats['total_time'])
    unet_avg_yolo_time = np.mean(yolo_unet_stats['yolo_time'])
    unet_avg_unet_time = np.mean(yolo_unet_stats['unet_avg_time'])
    
    # 过滤统计
    total_filtered = yolo_unet_stats['filtered_count']
    total_original = yolo_unet_stats['original_count']
    filter_rate = (total_filtered / total_original * 100) if total_original > 0 else 0
    
    # FP减少率
    fp_reduction = ((yolo_only_stats['fp'] - yolo_unet_stats['fp']) / yolo_only_stats['fp'] * 100) if yolo_only_stats['fp'] > 0 else 0
    
    print(f"\n{'指标':<25} {'YOLO单独':<20} {'YOLO+UNet':<20}")
    print("-" * 70)
    print(f"{'Precision (%)':<25} {yolo_avg_precision:<20.4f} {unet_avg_precision:<20.4f}")
    print(f"{'Recall (%)':<25} {yolo_avg_recall:<20.4f} {unet_avg_recall:<20.4f}")
    print(f"{'F1-Score (%)':<25} {yolo_avg_f1:<20.4f} {unet_avg_f1:<20.4f}")
    print(f"{'mAP@0.5:0.95 (%)':<25} {yolo_only_map*100:<20.4f} {yolo_unet_map*100:<20.4f}")
    print(f"{'TP (总数)':<25} {yolo_only_stats['tp']:<20} {yolo_unet_stats['tp']:<20}")
    print(f"{'FP (总数)':<25} {yolo_only_stats['fp']:<20} {yolo_unet_stats['fp']:<20}")
    print(f"{'FN (总数)':<25} {yolo_only_stats['fn']:<20} {yolo_unet_stats['fn']:<20}")
    print(f"{'平均推理时间 (ms)':<25} {yolo_avg_time:<20.4f} {unet_avg_total_time:<20.4f}")
    print(f"{'YOLO时间 (ms)':<25} {yolo_avg_time:<20.4f} {unet_avg_yolo_time:<20.4f}")
    print(f"{'UNet平均时间 (ms/框)':<25} {'-':<20} {unet_avg_unet_time:<20.4f}")
    print(f"{'过滤框数量':<25} {'-':<20} {total_filtered:<20}")
    print(f"{'过滤率 (%)':<25} {'-':<20} {filter_rate:<20.4f}")
    print(f"{'FP减少率 (%)':<25} {'-':<20} {fp_reduction:<20.4f}")
    
    # 改进幅度
    precision_improve = unet_avg_precision - yolo_avg_precision
    recall_change = unet_avg_recall - yolo_avg_recall
    f1_improve = unet_avg_f1 - yolo_avg_f1
    time_increase = unet_avg_total_time - yolo_avg_time
    
    print("\n改进幅度：")
    print("-" * 70)
    print(f"Precision提升: {precision_improve:+.2f}% ({precision_improve/yolo_avg_precision*100:+.2f}%)")
    print(f"Recall变化: {recall_change:+.2f}% ({recall_change/yolo_avg_recall*100:+.2f}%)")
    print(f"F1-Score提升: {f1_improve:+.2f}% ({f1_improve/yolo_avg_f1*100:+.2f}%)")
    print(f"mAP提升: {(yolo_unet_map-yolo_only_map)*100:+.2f}%")
    print(f"推理时间增加: {time_increase:+.2f} ms ({time_increase/yolo_avg_time*100:+.2f}%)")
    
    # 保存汇总结果
    summary_results = {
        'yolo_only': {
            'precision_percent': float(yolo_avg_precision),
            'recall_percent': float(yolo_avg_recall),
            'f1_percent': float(yolo_avg_f1),
            'map_percent': float(yolo_only_map * 100),
            'tp': yolo_only_stats['tp'],
            'fp': yolo_only_stats['fp'],
            'fn': yolo_only_stats['fn'],
            'avg_inference_time_ms': float(yolo_avg_time)
        },
        'yolo_unet': {
            'precision_percent': float(unet_avg_precision),
            'recall_percent': float(unet_avg_recall),
            'f1_percent': float(unet_avg_f1),
            'map_percent': float(yolo_unet_map * 100),
            'tp': yolo_unet_stats['tp'],
            'fp': yolo_unet_stats['fp'],
            'fn': yolo_unet_stats['fn'],
            'avg_total_time_ms': float(unet_avg_total_time),
            'avg_yolo_time_ms': float(unet_avg_yolo_time),
            'avg_unet_time_ms': float(unet_avg_unet_time),
            'filtered_count': total_filtered,
            'filter_rate_percent': float(filter_rate),
            'fp_reduction_percent': float(fp_reduction)
        },
        'improvement': {
            'precision_improve_percent': float(precision_improve),
            'recall_change_percent': float(recall_change),
            'f1_improve_percent': float(f1_improve),
            'map_improve_percent': float((yolo_unet_map - yolo_only_map) * 100),
            'time_increase_ms': float(time_increase),
            'time_increase_percent': float(time_increase / yolo_avg_time * 100)
        },
        'total_samples': len(per_sample_results)
    }
    
    with open(os.path.join(OUTPUT_DIR, 'summary_results.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    
    # 生成详细表格
    print("\n" + "="*70)
    print("生成每个样本的详细结果表格...")
    
    if HAS_PANDAS:
        df = pd.DataFrame(per_sample_results)
        csv_path = os.path.join(OUTPUT_DIR, 'per_sample_results.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ CSV表格已保存: {csv_path}")
        
        try:
            excel_path = os.path.join(OUTPUT_DIR, 'per_sample_results.xlsx')
            df.to_excel(excel_path, index=False, engine='openpyxl')
            print(f"✓ Excel表格已保存: {excel_path}")
        except ImportError:
            print("⚠ 未安装openpyxl，跳过Excel导出")
        
        print("\n前10个样本的结果预览：")
        preview_cols = ['图像名称', 'YOLO_Precision', 'YOLO+UNet_Precision', 'Precision提升',
                       'YOLO_推理时间_ms', 'YOLO+UNet_总时间_ms', '时间增加_%']
        print(df[preview_cols].head(10).to_string(index=False))
    else:
        import csv
        csv_path = os.path.join(OUTPUT_DIR, 'per_sample_results.csv')
        if per_sample_results:
            fieldnames = per_sample_results[0].keys()
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(per_sample_results)
            print(f"✓ CSV表格已保存: {csv_path}")
    
    # 生成可视化图表
    if HAS_MATPLOTLIB:
        print("\n" + "="*70)
        print("生成可视化图表...")
        
        fig = plt.figure(figsize=(18, 12))
        
        # 子图1: Precision对比
        ax1 = plt.subplot(3, 3, 1)
        sample_names = [r['图像名称'] for r in per_sample_results]
        yolo_precisions = [r['YOLO_Precision']*100 for r in per_sample_results]
        unet_precisions = [r['YOLO+UNet_Precision']*100 for r in per_sample_results]
        x_pos = np.arange(len(sample_names))
        width = 0.35
        ax1.bar(x_pos - width/2, yolo_precisions, width, label='YOLO单独', alpha=0.8, color='#3498db')
        ax1.bar(x_pos + width/2, unet_precisions, width, label='YOLO+UNet', alpha=0.8, color='#e74c3c')
        ax1.set_xlabel('样本编号')
        ax1.set_ylabel('Precision (%)')
        ax1.set_title('每个样本的Precision对比', fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos[::max(1, len(x_pos)//10)])
        ax1.set_xticklabels([f"S{i+1}" for i in range(len(sample_names))][::max(1, len(x_pos)//10)], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 子图2: Recall对比
        ax2 = plt.subplot(3, 3, 2)
        yolo_recalls = [r['YOLO_Recall']*100 for r in per_sample_results]
        unet_recalls = [r['YOLO+UNet_Recall']*100 for r in per_sample_results]
        ax2.bar(x_pos - width/2, yolo_recalls, width, label='YOLO单独', alpha=0.8, color='#3498db')
        ax2.bar(x_pos + width/2, unet_recalls, width, label='YOLO+UNet', alpha=0.8, color='#e74c3c')
        ax2.set_xlabel('样本编号')
        ax2.set_ylabel('Recall (%)')
        ax2.set_title('每个样本的Recall对比', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos[::max(1, len(x_pos)//10)])
        ax2.set_xticklabels([f"S{i+1}" for i in range(len(sample_names))][::max(1, len(x_pos)//10)], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 子图3: F1对比
        ax3 = plt.subplot(3, 3, 3)
        yolo_f1s = [r['YOLO_F1']*100 for r in per_sample_results]
        unet_f1s = [r['YOLO+UNet_F1']*100 for r in per_sample_results]
        ax3.bar(x_pos - width/2, yolo_f1s, width, label='YOLO单独', alpha=0.8, color='#3498db')
        ax3.bar(x_pos + width/2, unet_f1s, width, label='YOLO+UNet', alpha=0.8, color='#e74c3c')
        ax3.set_xlabel('样本编号')
        ax3.set_ylabel('F1-Score (%)')
        ax3.set_title('每个样本的F1-Score对比', fontsize=12, fontweight='bold')
        ax3.set_xticks(x_pos[::max(1, len(x_pos)//10)])
        ax3.set_xticklabels([f"S{i+1}" for i in range(len(sample_names))][::max(1, len(x_pos)//10)], rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 子图4: 平均指标对比
        ax4 = plt.subplot(3, 3, 4)
        metrics = ['Precision', 'Recall', 'F1-Score']
        yolo_means = [yolo_avg_precision, yolo_avg_recall, yolo_avg_f1]
        unet_means = [unet_avg_precision, unet_avg_recall, unet_avg_f1]
        x_metrics = np.arange(len(metrics))
        ax4.bar(x_metrics - width/2, yolo_means, width, label='YOLO单独', alpha=0.8, color='#3498db')
        ax4.bar(x_metrics + width/2, unet_means, width, label='YOLO+UNet', alpha=0.8, color='#e74c3c')
        ax4.set_ylabel('百分比 (%)')
        ax4.set_title('平均指标对比', fontsize=12, fontweight='bold')
        ax4.set_xticks(x_metrics)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim([0, 100])
        
        # 子图5: TP/FP/FN对比
        ax5 = plt.subplot(3, 3, 5)
        categories = ['TP', 'FP', 'FN']
        yolo_counts = [yolo_only_stats['tp'], yolo_only_stats['fp'], yolo_only_stats['fn']]
        unet_counts = [yolo_unet_stats['tp'], yolo_unet_stats['fp'], yolo_unet_stats['fn']]
        x_cat = np.arange(len(categories))
        ax5.bar(x_cat - width/2, yolo_counts, width, label='YOLO单独', alpha=0.8, color='#3498db')
        ax5.bar(x_cat + width/2, unet_counts, width, label='YOLO+UNet', alpha=0.8, color='#e74c3c')
        ax5.set_ylabel('数量')
        ax5.set_title('TP/FP/FN总数对比', fontsize=12, fontweight='bold')
        ax5.set_xticks(x_cat)
        ax5.set_xticklabels(categories)
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 子图6: 推理时间对比
        ax6 = plt.subplot(3, 3, 6)
        time_categories = ['YOLO单独', 'YOLO+UNet\n总时间', 'YOLO+UNet\nYOLO时间', 'YOLO+UNet\nUNet平均']
        time_values = [yolo_avg_time, unet_avg_total_time, unet_avg_yolo_time, unet_avg_unet_time]
        colors_time = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
        ax6.bar(time_categories, time_values, alpha=0.8, color=colors_time, edgecolor='black')
        ax6.set_ylabel('时间 (ms)')
        ax6.set_title('推理时间对比', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        for i, (cat, val) in enumerate(zip(time_categories, time_values)):
            ax6.text(i, val + max(time_values) * 0.02, f'{val:.1f}ms', ha='center', va='bottom', fontsize=9)
        
        # 子图7: 改进幅度
        ax7 = plt.subplot(3, 3, 7)
        improvements = [precision_improve, recall_change, f1_improve]
        colors_improve = ['green' if x > 0 else 'red' for x in improvements]
        ax7.bar(metrics, improvements, alpha=0.8, color=colors_improve, edgecolor='black')
        ax7.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax7.set_ylabel('提升幅度 (%)')
        ax7.set_title('改进幅度', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')
        
        # 子图8: 过滤率分布
        ax8 = plt.subplot(3, 3, 8)
        filter_rates = [r['过滤率_%'] for r in per_sample_results if r['过滤率_%'] > 0]
        if filter_rates:
            ax8.hist(filter_rates, bins=20, alpha=0.7, color='#9b59b6', edgecolor='black')
            ax8.axvline(np.mean(filter_rates), color='red', linestyle='--', linewidth=2,
                       label=f'平均值: {np.mean(filter_rates):.2f}%')
            ax8.set_xlabel('过滤率 (%)')
            ax8.set_ylabel('样本数')
            ax8.set_title('过滤率分布', fontsize=12, fontweight='bold')
            ax8.legend()
            ax8.grid(True, alpha=0.3, axis='y')
        
        # 子图9: 时间增加分布
        ax9 = plt.subplot(3, 3, 9)
        time_increases = [r['时间增加_%'] for r in per_sample_results]
        ax9.hist(time_increases, bins=20, alpha=0.7, color='#f39c12', edgecolor='black')
        ax9.axvline(np.mean(time_increases), color='red', linestyle='--', linewidth=2,
                   label=f'平均值: {np.mean(time_increases):.2f}%')
        ax9.set_xlabel('时间增加 (%)')
        ax9.set_ylabel('样本数')
        ax9.set_title('时间增加分布', fontsize=12, fontweight='bold')
        ax9.legend()
        ax9.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, 'comparison_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ 对比图表已保存: {plot_path}")
        plt.close()
    else:
        print("\n⚠ 未安装matplotlib，跳过图表生成")
    
    print(f"\n所有结果已保存到: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()

