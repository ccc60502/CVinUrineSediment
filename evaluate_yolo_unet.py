# -*- coding: utf-8 -*-
"""
评估脚本：比较YOLO单独检测和YOLO+UNet两阶段方法
支持两种评估模式：
1. 检测评估：使用YOLO格式的ground truth标注（.txt文件）
2. 分割评估：使用分割mask作为ground truth（.png文件）
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict
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
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.rcParams['font.size'] = 12
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: 未安装matplotlib，将跳过图表生成（可运行: pip install matplotlib）")

# ================== 配置参数 ==================
# YOLO模型路径
YOLO_MODEL_PATH = r"E:\Data_Industry\Unet_wdir\2stage_wdir\best.pt"

# UNet模型路径
UNET_MODEL_PATH = r"E:\Data_Industry\split_images\edit_1004560_MergedBatch\VOC2012\Unet_model_file\attention_UNet\edit_1004560_MergedBatch_ByJ20251201.pth"

# 测试数据路径
TEST_IMAGE_DIR = r"E:\Data_Industry\dateByJ_1004560_20251201\Just_Cancer\image"
TEST_LABEL_DIR = r"E:\Data_Industry\dateByJ_1004560_20251201\Just_Cancer\labels"  # YOLO格式标签目录，如果为None则使用分割mask评估
TEST_MASK_DIR = None   # 分割mask目录，如果为None则使用YOLO格式标签评估

# 输出结果目录
OUTPUT_DIR = r"E:\Data_Industry\膀胱癌细胞检测_文章\YOLO_Unet比较评估"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# YOLO参数
YOLO_CONF = 0.25
YOLO_CLASSES = [0]
IOU_THRESHOLD = 0.5  # 用于计算mAP的IoU阈值

# UNet参数
UNET_INPUT_SIZE = (1024, 1024)
UNET_NUM_CLASSES = 2
MASK_THRESHOLD = 0.6
FILTER_BY_UNET = True
UNET_FILTER_THRESHOLD = 0.01

# 评估模式
EVAL_MODE = "detection"  # "detection", "segmentation", "both"
# ==============================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
print("加载模型...")
yolo_model = YOLO(YOLO_MODEL_PATH)
from model.unet_resnet import Unet
unet_model = Unet(num_classes=UNET_NUM_CLASSES)
unet_model.load_state_dict(torch.load(UNET_MODEL_PATH, map_location=device))
unet_model.eval()
unet_model.to(device)
from utils.utils import cvtColor, preprocess_input, resize_image

print("模型加载完成！")


def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 计算交集
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # 计算并集
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


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
            y_center = float(parts[2]) * img_width
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            
            # 转换为x1, y1, x2, y2格式
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            boxes.append([x1, y1, x2, y2, class_id])
    
    return boxes


def load_segmentation_mask(mask_path):
    """加载分割mask"""
    if not os.path.exists(mask_path):
        return None
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    # 将mask转换为二值（0或1）
    mask = (mask > 0).astype(np.uint8)
    return mask


def yolo_only_detection(image_path):
    """仅使用YOLO进行检测"""
    results = yolo_model.predict(
        source=image_path,
        conf=YOLO_CONF,
        classes=YOLO_CLASSES,
        save=False,
        verbose=False
    )[0]
    
    detections = []
    for box in results.boxes:
        bbox = box.xyxy[0].cpu().numpy().astype(float)
        conf = box.conf[0].item()
        detections.append({
            'bbox': bbox.tolist(),
            'conf': conf,
            'class': int(box.cls[0].item())
        })
    
    return detections


def yolo_unet_two_stage(image_path):
    """YOLO+UNet两阶段方法"""
    image_pil = Image.open(image_path).convert("RGB")
    orig_np = np.array(image_pil)
    h, w = orig_np.shape[:2]
    
    # YOLO检测
    results = yolo_model.predict(
        source=image_path,
        conf=YOLO_CONF,
        classes=YOLO_CLASSES,
        save=False,
        verbose=False
    )[0]
    
    detections = []
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    
    for idx, box in enumerate(results.boxes):
        bbox = box.xyxy[0].cpu().numpy().astype(int)
        conf = box.conf[0].item()
        
        # 裁剪和预处理（复用原有逻辑）
        x1, y1, x2, y2 = bbox
        w_box, h_box = x2 - x1, y2 - y1
        
        # 扩展边界
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
        
        record = {
            "orig_xy": (x1_exp, y1_exp),
            "crop_size": (x2_exp - x1_exp, y2_exp - y1_exp),
            "orig_bbox": bbox,
            "resized_size": (nw, nh),
            "pad_offset": (pad_x, pad_y)
        }
        
        # UNet推理
        patch_input = np.expand_dims(
            np.transpose(preprocess_input(np.array(patch_resized, np.float32)), (2, 0, 1)), 0
        )
        patch_tensor = torch.from_numpy(patch_input).type(torch.FloatTensor).to(device)
        
        with torch.no_grad():
            output = unet_model(patch_tensor)
            pr = F.softmax(output[0], dim=0).cpu().numpy()
            prob_map = pr[1]
            mask_1024 = (prob_map >= MASK_THRESHOLD).astype(np.uint8)
        
        # 映射回原图
        mask_actual = mask_1024[pad_y:pad_y+nh, pad_x:pad_x+nw]
        crop_w, crop_h = record["crop_size"]
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
                    continue  # 过滤掉这个检测框
        
        detections.append({
            'bbox': bbox.tolist(),
            'conf': conf,
            'class': 0,
            'mask': instance_mask
        })
        combined_mask = np.maximum(combined_mask, instance_mask)
    
    return detections, combined_mask


def evaluate_detection(pred_boxes, gt_boxes, iou_threshold=0.5):
    """评估检测结果"""
    if len(gt_boxes) == 0:
        if len(pred_boxes) == 0:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}
        else:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': len(pred_boxes), 'fn': 0}
    
    if len(pred_boxes) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': len(gt_boxes)}
    
    # 匹配预测框和真实框
    matched_gt = set()
    tp = 0
    fp = 0
    
    # 按置信度排序
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


def evaluate_segmentation(pred_mask, gt_mask):
    """评估分割结果"""
    if gt_mask is None:
        return None
    
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)
    
    # 确保尺寸一致
    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # 计算IoU
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union if union > 0 else 0.0
    
    # 计算Dice系数
    dice = 2 * intersection / (pred_mask.sum() + gt_mask.sum()) if (pred_mask.sum() + gt_mask.sum()) > 0 else 0.0
    
    # 计算像素准确率
    pixel_acc = (pred_mask == gt_mask).sum() / pred_mask.size
    
    return {
        'iou': iou,
        'dice': dice,
        'pixel_acc': pixel_acc
    }


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
    
    # 初始化统计变量（用于计算平均值）
    yolo_only_stats = {
        'precision': [],
        'recall': [],
        'f1': [],
        'tp': 0,
        'fp': 0,
        'fn': 0
    }
    
    yolo_unet_stats = {
        'precision': [],
        'recall': [],
        'f1': [],
        'tp': 0,
        'fp': 0,
        'fn': 0
    }
    
    segmentation_stats = {
        'iou': [],
        'dice': [],
        'pixel_acc': []
    }
    
    # 存储每个样本的详细结果
    per_sample_results = []
    
    # 逐图像评估
    for image_path in tqdm(image_files, desc="评估进度"):
        image_name = image_path.stem
        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]
        
        # 加载ground truth
        gt_boxes = []
        gt_mask = None
        
        if TEST_LABEL_DIR and os.path.exists(os.path.join(TEST_LABEL_DIR, f"{image_name}.txt")):
            gt_boxes = load_yolo_labels(os.path.join(TEST_LABEL_DIR, f"{image_name}.txt"), w, h)
        
        if TEST_MASK_DIR and os.path.exists(os.path.join(TEST_MASK_DIR, f"{image_name}.png")):
            gt_mask = load_segmentation_mask(os.path.join(TEST_MASK_DIR, f"{image_name}.png"))
        
        # YOLO单独检测
        yolo_detections = yolo_only_detection(str(image_path))
        
        # YOLO+UNet两阶段
        yolo_unet_detections, combined_mask = yolo_unet_two_stage(str(image_path))
        
        # 检测评估
        if len(gt_boxes) > 0 or EVAL_MODE in ["detection", "both"]:
            yolo_result = evaluate_detection(yolo_detections, gt_boxes, IOU_THRESHOLD)
            yolo_unet_result = evaluate_detection(yolo_unet_detections, gt_boxes, IOU_THRESHOLD)
            
            yolo_only_stats['precision'].append(yolo_result['precision'])
            yolo_only_stats['recall'].append(yolo_result['recall'])
            yolo_only_stats['f1'].append(yolo_result['f1'])
            yolo_only_stats['tp'] += yolo_result['tp']
            yolo_only_stats['fp'] += yolo_result['fp']
            yolo_only_stats['fn'] += yolo_result['fn']
            
            yolo_unet_stats['precision'].append(yolo_unet_result['precision'])
            yolo_unet_stats['recall'].append(yolo_unet_result['recall'])
            yolo_unet_stats['f1'].append(yolo_unet_result['f1'])
            yolo_unet_stats['tp'] += yolo_unet_result['tp']
            yolo_unet_stats['fp'] += yolo_unet_result['fp']
            yolo_unet_stats['fn'] += yolo_unet_result['fn']
            
            # 保存每个样本的详细结果
            per_sample_results.append({
                '图像名称': image_name,
                'YOLO_检测数': len(yolo_detections),
                'YOLO+UNet_检测数': len(yolo_unet_detections),
                'GT目标数': len(gt_boxes),
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
            })
        
        # 分割评估
        if gt_mask is not None and EVAL_MODE in ["segmentation", "both"]:
            seg_result = evaluate_segmentation(combined_mask, gt_mask)
            if seg_result:
                segmentation_stats['iou'].append(seg_result['iou'])
                segmentation_stats['dice'].append(seg_result['dice'])
                segmentation_stats['pixel_acc'].append(seg_result['pixel_acc'])
                
                # 添加到每个样本结果中
                if per_sample_results:
                    per_sample_results[-1].update({
                        'IoU': seg_result['iou'],
                        'Dice': seg_result['dice'],
                        'Pixel_Accuracy': seg_result['pixel_acc']
                    })
    
    # 计算平均指标
    print("\n" + "="*60)
    print("评估结果汇总")
    print("="*60)
    
    # 预先计算平均值（用于后续可视化）
    yolo_precision = np.mean(yolo_only_stats['precision']) if yolo_only_stats['precision'] else 0.0
    yolo_recall = np.mean(yolo_only_stats['recall']) if yolo_only_stats['recall'] else 0.0
    yolo_f1 = np.mean(yolo_only_stats['f1']) if yolo_only_stats['f1'] else 0.0
    
    unet_precision = np.mean(yolo_unet_stats['precision']) if yolo_unet_stats['precision'] else 0.0
    unet_recall = np.mean(yolo_unet_stats['recall']) if yolo_unet_stats['recall'] else 0.0
    unet_f1 = np.mean(yolo_unet_stats['f1']) if yolo_unet_stats['f1'] else 0.0
    
    if EVAL_MODE in ["detection", "both"]:
        print("\n【检测评估结果】")
        print("-" * 60)
        print(f"{'指标':<20} {'YOLO单独':<20} {'YOLO+UNet':<20}")
        print("-" * 60)
        
        print(f"{'Precision':<20} {yolo_precision:<20.4f} {unet_precision:<20.4f}")
        print(f"{'Recall':<20} {yolo_recall:<20.4f} {unet_recall:<20.4f}")
        print(f"{'F1-Score':<20} {yolo_f1:<20.4f} {unet_f1:<20.4f}")
        print(f"{'TP (总数)':<20} {yolo_only_stats['tp']:<20} {yolo_unet_stats['tp']:<20}")
        print(f"{'FP (总数)':<20} {yolo_only_stats['fp']:<20} {yolo_unet_stats['fp']:<20}")
        print(f"{'FN (总数)':<20} {yolo_only_stats['fn']:<20} {yolo_unet_stats['fn']:<20}")
        
        # 计算改进幅度
        precision_improve = (unet_precision - yolo_precision) / yolo_precision * 100 if yolo_precision > 0 else 0
        recall_improve = (unet_recall - yolo_recall) / yolo_recall * 100 if yolo_recall > 0 else 0
        f1_improve = (unet_f1 - yolo_f1) / yolo_f1 * 100 if yolo_f1 > 0 else 0
        
        print("\n改进幅度：")
        print(f"  Precision: {precision_improve:+.2f}%")
        print(f"  Recall: {recall_improve:+.2f}%")
        print(f"  F1-Score: {f1_improve:+.2f}%")
    
    if EVAL_MODE in ["segmentation", "both"] and segmentation_stats['iou']:
        print("\n【分割评估结果】")
        print("-" * 60)
        print(f"{'指标':<20} {'平均值':<20} {'标准差':<20}")
        print("-" * 60)
        
        mean_iou = np.mean(segmentation_stats['iou'])
        mean_dice = np.mean(segmentation_stats['dice'])
        mean_pixel_acc = np.mean(segmentation_stats['pixel_acc'])
        
        std_iou = np.std(segmentation_stats['iou'])
        std_dice = np.std(segmentation_stats['dice'])
        std_pixel_acc = np.std(segmentation_stats['pixel_acc'])
        
        print(f"{'IoU':<20} {mean_iou:<20.4f} {std_iou:<20.4f}")
        print(f"{'Dice系数':<20} {mean_dice:<20.4f} {std_dice:<20.4f}")
        print(f"{'像素准确率':<20} {mean_pixel_acc:<20.4f} {std_pixel_acc:<20.4f}")
    
    # 保存结果到JSON文件
    results = {
        'yolo_only': {
            'precision': float(np.mean(yolo_only_stats['precision'])) if yolo_only_stats['precision'] else 0.0,
            'recall': float(np.mean(yolo_only_stats['recall'])) if yolo_only_stats['recall'] else 0.0,
            'f1': float(np.mean(yolo_only_stats['f1'])) if yolo_only_stats['f1'] else 0.0,
            'tp': yolo_only_stats['tp'],
            'fp': yolo_only_stats['fp'],
            'fn': yolo_only_stats['fn']
        },
        'yolo_unet': {
            'precision': float(np.mean(yolo_unet_stats['precision'])) if yolo_unet_stats['precision'] else 0.0,
            'recall': float(np.mean(yolo_unet_stats['recall'])) if yolo_unet_stats['recall'] else 0.0,
            'f1': float(np.mean(yolo_unet_stats['f1'])) if yolo_unet_stats['f1'] else 0.0,
            'tp': yolo_unet_stats['tp'],
            'fp': yolo_unet_stats['fp'],
            'fn': yolo_unet_stats['fn']
        }
    }
    
    if segmentation_stats['iou']:
        results['segmentation'] = {
            'iou': float(np.mean(segmentation_stats['iou'])),
            'dice': float(np.mean(segmentation_stats['dice'])),
            'pixel_acc': float(np.mean(segmentation_stats['pixel_acc']))
        }
    
    # 保存汇总结果到JSON
    with open(os.path.join(OUTPUT_DIR, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # ==================== 生成每个样本的详细表格 ====================
    if per_sample_results:
        print("\n" + "="*60)
        print("生成每个样本的详细结果表格...")
        
        if HAS_PANDAS:
            # 创建DataFrame
            df = pd.DataFrame(per_sample_results)
            
            # 保存为CSV
            csv_path = os.path.join(OUTPUT_DIR, 'per_sample_results.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"✓ CSV表格已保存: {csv_path}")
            
            # 保存为Excel（如果安装了openpyxl）
            try:
                excel_path = os.path.join(OUTPUT_DIR, 'per_sample_results.xlsx')
                df.to_excel(excel_path, index=False, engine='openpyxl')
                print(f"✓ Excel表格已保存: {excel_path}")
            except ImportError:
                print("⚠ 未安装openpyxl，跳过Excel导出（可运行: pip install openpyxl）")
            
            # 打印表格预览（前10行）
            print("\n前10个样本的结果预览：")
            print(df.head(10).to_string(index=False))
        else:
            # 如果没有pandas，使用CSV模块手动写入
            import csv
            csv_path = os.path.join(OUTPUT_DIR, 'per_sample_results.csv')
            if per_sample_results:
                fieldnames = per_sample_results[0].keys()
                with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(per_sample_results)
                print(f"✓ CSV表格已保存: {csv_path}")
            
            # 打印前5个样本的结果
            print("\n前5个样本的结果预览：")
            for i, result in enumerate(per_sample_results[:5]):
                print(f"\n样本 {i+1}: {result['图像名称']}")
                print(f"  YOLO: P={result['YOLO_Precision']:.4f}, R={result['YOLO_Recall']:.4f}, F1={result['YOLO_F1']:.4f}")
                print(f"  YOLO+UNet: P={result['YOLO+UNet_Precision']:.4f}, R={result['YOLO+UNet_Recall']:.4f}, F1={result['YOLO+UNet_F1']:.4f}")
    
    # ==================== 生成可视化图表 ====================
    if HAS_MATPLOTLIB:
        print("\n" + "="*60)
        print("生成可视化图表...")
        
        if EVAL_MODE in ["detection", "both"] and per_sample_results:
            # 创建图表
            fig = plt.figure(figsize=(16, 12))
            
            # 子图1: Precision对比（每个样本）
            ax1 = plt.subplot(3, 2, 1)
            sample_names = [r['图像名称'] for r in per_sample_results]
            yolo_precisions = [r['YOLO_Precision'] for r in per_sample_results]
            unet_precisions = [r['YOLO+UNet_Precision'] for r in per_sample_results]
            x_pos = np.arange(len(sample_names))
            width = 0.35
            ax1.bar(x_pos - width/2, yolo_precisions, width, label='YOLO单独', alpha=0.8, color='#3498db')
            ax1.bar(x_pos + width/2, unet_precisions, width, label='YOLO+UNet', alpha=0.8, color='#e74c3c')
            ax1.set_xlabel('样本编号')
            ax1.set_ylabel('Precision')
            ax1.set_title('每个样本的Precision对比', fontsize=14, fontweight='bold')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels([f"S{i+1}" for i in range(len(sample_names))], rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([0, 1.1])
            
            # 子图2: Recall对比（每个样本）
            ax2 = plt.subplot(3, 2, 2)
            yolo_recalls = [r['YOLO_Recall'] for r in per_sample_results]
            unet_recalls = [r['YOLO+UNet_Recall'] for r in per_sample_results]
            ax2.bar(x_pos - width/2, yolo_recalls, width, label='YOLO单独', alpha=0.8, color='#3498db')
            ax2.bar(x_pos + width/2, unet_recalls, width, label='YOLO+UNet', alpha=0.8, color='#e74c3c')
            ax2.set_xlabel('样本编号')
            ax2.set_ylabel('Recall')
            ax2.set_title('每个样本的Recall对比', fontsize=14, fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([f"S{i+1}" for i in range(len(sample_names))], rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, 1.1])
            
            # 子图3: F1-Score对比（每个样本）
            ax3 = plt.subplot(3, 2, 3)
            yolo_f1s = [r['YOLO_F1'] for r in per_sample_results]
            unet_f1s = [r['YOLO+UNet_F1'] for r in per_sample_results]
            ax3.bar(x_pos - width/2, yolo_f1s, width, label='YOLO单独', alpha=0.8, color='#3498db')
            ax3.bar(x_pos + width/2, unet_f1s, width, label='YOLO+UNet', alpha=0.8, color='#e74c3c')
            ax3.set_xlabel('样本编号')
            ax3.set_ylabel('F1-Score')
            ax3.set_title('每个样本的F1-Score对比', fontsize=14, fontweight='bold')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels([f"S{i+1}" for i in range(len(sample_names))], rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim([0, 1.1])
            
            # 子图4: 平均指标对比（柱状图）
            ax4 = plt.subplot(3, 2, 4)
            metrics = ['Precision', 'Recall', 'F1-Score']
            yolo_means = [yolo_precision, yolo_recall, yolo_f1]
            unet_means = [unet_precision, unet_recall, unet_f1]
            x_metrics = np.arange(len(metrics))
            ax4.bar(x_metrics - width/2, yolo_means, width, label='YOLO单独', alpha=0.8, color='#3498db')
            ax4.bar(x_metrics + width/2, unet_means, width, label='YOLO+UNet', alpha=0.8, color='#e74c3c')
            ax4.set_xlabel('指标')
            ax4.set_ylabel('分数')
            ax4.set_title('平均指标对比', fontsize=14, fontweight='bold')
            ax4.set_xticks(x_metrics)
            ax4.set_xticklabels(metrics)
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.set_ylim([0, 1.1])
            # 添加数值标签
            for i, (yolo_val, unet_val) in enumerate(zip(yolo_means, unet_means)):
                ax4.text(i - width/2, yolo_val + 0.02, f'{yolo_val:.3f}', ha='center', va='bottom', fontsize=9)
                ax4.text(i + width/2, unet_val + 0.02, f'{unet_val:.3f}', ha='center', va='bottom', fontsize=9)
            
            # 子图5: TP/FP/FN对比（堆叠柱状图）
            ax5 = plt.subplot(3, 2, 5)
            categories = ['TP', 'FP', 'FN']
            yolo_counts = [yolo_only_stats['tp'], yolo_only_stats['fp'], yolo_only_stats['fn']]
            unet_counts = [yolo_unet_stats['tp'], yolo_unet_stats['fp'], yolo_unet_stats['fn']]
            x_cat = np.arange(len(categories))
            ax5.bar(x_cat - width/2, yolo_counts, width, label='YOLO单独', alpha=0.8, color='#3498db')
            ax5.bar(x_cat + width/2, unet_counts, width, label='YOLO+UNet', alpha=0.8, color='#e74c3c')
            ax5.set_xlabel('类别')
            ax5.set_ylabel('数量')
            ax5.set_title('TP/FP/FN总数对比', fontsize=14, fontweight='bold')
            ax5.set_xticks(x_cat)
            ax5.set_xticklabels(categories)
            ax5.legend()
            ax5.grid(True, alpha=0.3, axis='y')
            # 添加数值标签
            for i, (yolo_val, unet_val) in enumerate(zip(yolo_counts, unet_counts)):
                ax5.text(i - width/2, yolo_val + max(yolo_counts + unet_counts) * 0.01, 
                        f'{yolo_val}', ha='center', va='bottom', fontsize=9)
                ax5.text(i + width/2, unet_val + max(yolo_counts + unet_counts) * 0.01, 
                        f'{unet_val}', ha='center', va='bottom', fontsize=9)
            
            # 子图6: 改进幅度（折线图）
            ax6 = plt.subplot(3, 2, 6)
            precision_improve_list = [r['Precision提升'] for r in per_sample_results]
            recall_improve_list = [r['Recall变化'] for r in per_sample_results]
            f1_improve_list = [r['F1提升'] for r in per_sample_results]
            ax6.plot(x_pos, precision_improve_list, marker='o', label='Precision提升', linewidth=2, markersize=6)
            ax6.plot(x_pos, recall_improve_list, marker='s', label='Recall变化', linewidth=2, markersize=6)
            ax6.plot(x_pos, f1_improve_list, marker='^', label='F1提升', linewidth=2, markersize=6)
            ax6.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax6.set_xlabel('样本编号')
            ax6.set_ylabel('改进幅度')
            ax6.set_title('每个样本的改进幅度', fontsize=14, fontweight='bold')
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels([f"S{i+1}" for i in range(len(sample_names))], rotation=45)
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(OUTPUT_DIR, 'evaluation_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ 对比图表已保存: {plot_path}")
            plt.close()
            
            # 生成单独的箱线图（展示分布）
            fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
            data_to_plot = [
                [yolo_precisions, unet_precisions],
                [yolo_recalls, unet_recalls],
                [yolo_f1s, unet_f1s]
            ]
            titles = ['Precision分布', 'Recall分布', 'F1-Score分布']
            labels = ['YOLO单独', 'YOLO+UNet']
            
            for idx, (ax, data, title) in enumerate(zip(axes, data_to_plot, titles)):
                bp = ax.boxplot(data, labels=labels, patch_artist=True)
                bp['boxes'][0].set_facecolor('#3498db')
                bp['boxes'][0].set_alpha(0.7)
                bp['boxes'][1].set_facecolor('#e74c3c')
                bp['boxes'][1].set_alpha(0.7)
                ax.set_ylabel('分数')
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_ylim([0, 1.1])
            
            plt.tight_layout()
            boxplot_path = os.path.join(OUTPUT_DIR, 'evaluation_distribution.png')
            plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
            print(f"✓ 分布箱线图已保存: {boxplot_path}")
            plt.close()
        
            # 分割评估可视化
            if EVAL_MODE in ["segmentation", "both"] and segmentation_stats['iou']:
                fig3, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # IoU分布
                axes[0].hist(segmentation_stats['iou'], bins=20, alpha=0.7, color='#2ecc71', edgecolor='black')
                axes[0].axvline(np.mean(segmentation_stats['iou']), color='red', linestyle='--', 
                               linewidth=2, label=f'平均值: {np.mean(segmentation_stats["iou"]):.4f}')
                axes[0].set_xlabel('IoU')
                axes[0].set_ylabel('样本数')
                axes[0].set_title('IoU分布直方图', fontsize=12, fontweight='bold')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3, axis='y')
                
                # Dice分布
                axes[1].hist(segmentation_stats['dice'], bins=20, alpha=0.7, color='#9b59b6', edgecolor='black')
                axes[1].axvline(np.mean(segmentation_stats['dice']), color='red', linestyle='--', 
                               linewidth=2, label=f'平均值: {np.mean(segmentation_stats["dice"]):.4f}')
                axes[1].set_xlabel('Dice系数')
                axes[1].set_ylabel('样本数')
                axes[1].set_title('Dice系数分布直方图', fontsize=12, fontweight='bold')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3, axis='y')
                
                # 像素准确率分布
                axes[2].hist(segmentation_stats['pixel_acc'], bins=20, alpha=0.7, color='#f39c12', edgecolor='black')
                axes[2].axvline(np.mean(segmentation_stats['pixel_acc']), color='red', linestyle='--', 
                               linewidth=2, label=f'平均值: {np.mean(segmentation_stats["pixel_acc"]):.4f}')
                axes[2].set_xlabel('像素准确率')
                axes[2].set_ylabel('样本数')
                axes[2].set_title('像素准确率分布直方图', fontsize=12, fontweight='bold')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                seg_plot_path = os.path.join(OUTPUT_DIR, 'segmentation_distribution.png')
                plt.savefig(seg_plot_path, dpi=300, bbox_inches='tight')
                print(f"✓ 分割评估分布图已保存: {seg_plot_path}")
                plt.close()
    else:
        print("\n⚠ 未安装matplotlib，跳过图表生成")
    
    print(f"\n所有结果已保存到: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()

