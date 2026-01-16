# -*- coding: utf-8 -*-
"""
原始UNet模型评估脚本（无转置卷积和注意力门控）
计算指标：
- Mean IoU (%)
- Dice (%)
- FW IoU (%)
- 推理时间 (ms/图)

可选：支持与增强版UNet对比评估
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.rcParams['font.size'] = 12
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: 未安装matplotlib，将跳过图表生成（可运行: pip install matplotlib）")

# ================== 配置参数 ==================
# 原始UNet模型路径（无转置卷积和注意力门控）
ORIGINAL_UNET_MODEL_PATH = r"E:\Data_Industry\膀胱癌细胞检测_文章\Unet_resnet50文件\2\last_model_1.pth"

# 可选：增强版UNet模型路径（用于对比）
ENHANCED_UNET_MODEL_PATH = None  # 如果为None，则不进行对比评估
# ENHANCED_UNET_MODEL_PATH = r"E:\Data_Industry\split_images\edit_1004560_MergedBatch\VOC2012\Unet_model_file\attention_UNet\edit_1004560_MergedBatch_ByJ20251201.pth"

# 测试数据路径
TEST_IMAGE_DIR = r"E:\Data_Industry\split_images\edit_1004560_MergedBatch\VOC2012\JPEGImages"
TEST_MASK_DIR = r"E:\Data_Industry\split_images\edit_1004560_MergedBatch\VOC2012\SegmentationClass"  # Ground truth mask目录

# 输出结果目录
OUTPUT_DIR = r"E:\Data_Industry\膀胱癌细胞检测_文章\Unet_resnet50文件\2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# UNet参数
UNET_INPUT_SIZE = (1024, 1024)  # (height, width)
UNET_NUM_CLASSES = 2  # 背景 + 癌细胞

# 推理参数
MASK_THRESHOLD = 0.5  # 用于二值化的阈值（如果使用概率图）
USE_PROB_THRESHOLD = False  # True: 使用概率图阈值, False: 使用argmax

# 预热次数（用于稳定推理时间测量）
WARMUP_ITERATIONS = 5

# 是否进行对比评估
COMPARE_WITH_ENHANCED = ENHANCED_UNET_MODEL_PATH is not None
# ==============================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载原始UNet模型
print("加载原始UNet模型（无转置卷积和注意力门控）...")
from model.unet_resnet import Unet as Unet_Original
original_unet_model = Unet_Original(num_classes=UNET_NUM_CLASSES)
original_unet_model.load_state_dict(torch.load(ORIGINAL_UNET_MODEL_PATH, map_location=device))
original_unet_model.eval()
original_unet_model.to(device)
print("原始UNet模型加载完成！")

# 可选：加载增强版UNet模型（用于对比）
enhanced_unet_model = None
if COMPARE_WITH_ENHANCED:
    print("加载增强版UNet模型（用于对比）...")
    from model.unet_resnet_enhanced import Unet as Unet_Enhanced_Wrapper
    # 使用enhanced=True加载增强版（默认启用注意力门控和转置卷积）
    enhanced_unet_model = Unet_Enhanced_Wrapper(num_classes=UNET_NUM_CLASSES, enhanced=True)
    enhanced_unet_model.load_state_dict(torch.load(ENHANCED_UNET_MODEL_PATH, map_location=device))
    enhanced_unet_model.eval()
    enhanced_unet_model.to(device)
    print("增强版UNet模型加载完成！")

from utils.utils import cvtColor, preprocess_input, resize_image


def calculate_dice(pred_mask, gt_mask, num_classes):
    """
    计算Dice系数（每个类别分别计算，然后取平均）
    pred_mask: numpy array, shape (H, W), 值为类别索引
    gt_mask: numpy array, shape (H, W), 值为类别索引
    num_classes: 类别数
    """
    dices = []
    for i in range(num_classes):
        pred_binary = (pred_mask == i).astype(np.uint8)
        gt_binary = (gt_mask == i).astype(np.uint8)
        
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        pred_sum = pred_binary.sum()
        gt_sum = gt_binary.sum()
        
        if pred_sum + gt_sum == 0:
            # 如果预测和真实都没有该类，Dice为1（完美匹配）
            dices.append(1.0)
        else:
            dice = 2.0 * intersection / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
            dices.append(dice)
    
    return np.mean(dices)


def calculate_mean_iou(pred_mask, gt_mask, num_classes):
    """
    计算Mean IoU
    pred_mask: numpy array, shape (H, W), 值为类别索引
    gt_mask: numpy array, shape (H, W), 值为类别索引
    num_classes: 类别数
    """
    ious = []
    for i in range(num_classes):
        pred_binary = (pred_mask == i).astype(np.uint8)
        gt_binary = (gt_mask == i).astype(np.uint8)
        
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        
        if union > 0:
            iou = intersection / union
            # 只对真实标签中存在的类别计算IoU
            if gt_binary.sum() > 0:
                ious.append(iou)
    
    return np.mean(ious) if len(ious) > 0 else 0.0


def calculate_fw_iou(pred_mask, gt_mask, num_classes):
    """
    计算Frequency Weighted IoU
    pred_mask: numpy array, shape (H, W), 值为类别索引
    gt_mask: numpy array, shape (H, W), 值为类别索引
    num_classes: 类别数
    """
    ious = []
    frequencies = []
    
    for i in range(num_classes):
        pred_binary = (pred_mask == i).astype(np.uint8)
        gt_binary = (gt_mask == i).astype(np.uint8)
        
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        freq = gt_binary.sum()
        
        frequencies.append(freq)
        iou = intersection / union if union > 0 else 0.0
        ious.append(iou)
    
    total_freq = sum(frequencies)
    if total_freq == 0:
        return 0.0
    
    fw_iou = sum(f * iou for f, iou in zip(frequencies, ious)) / total_freq
    return fw_iou


def load_gt_mask(mask_path):
    """加载ground truth mask"""
    if not os.path.exists(mask_path):
        return None
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    
    # 将mask转换为类别索引（0=背景, 1=癌细胞）
    # 假设mask中0为背景，非0为前景（癌细胞）
    mask = (mask > 0).astype(np.uint8)
    return mask


def unet_inference(image_path, model, model_name="UNet"):
    """
    UNet推理
    返回: (pred_mask, inference_time_ms)
    """
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    original_h, original_w = np.array(image).shape[:2]
    
    # 预处理
    image = cvtColor(image)
    image_resized, nw, nh = resize_image(image, (UNET_INPUT_SIZE[1], UNET_INPUT_SIZE[0]))
    
    # 转换为tensor
    image_data = np.expand_dims(
        np.transpose(preprocess_input(np.array(image_resized, np.float32)), (2, 0, 1)), 0
    )
    image_tensor = torch.from_numpy(image_data).type(torch.FloatTensor).to(device)
    
    # 推理
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        output = model(image_tensor)[0]  # [C, H, W]
        
        if USE_PROB_THRESHOLD:
            # 使用概率图阈值
            pr = F.softmax(output, dim=0).cpu().numpy()
            prob_map = pr[1]  # 前景概率图
            pred_mask_resized = (prob_map >= MASK_THRESHOLD).astype(np.uint8)
        else:
            # 使用argmax
            pred_mask_resized = output.argmax(dim=0).cpu().numpy().astype(np.uint8)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
    
    # 提取实际图像区域（去除padding）
    pad_x = (UNET_INPUT_SIZE[1] - nw) // 2
    pad_y = (UNET_INPUT_SIZE[0] - nh) // 2
    pred_mask_actual = pred_mask_resized[pad_y:pad_y+nh, pad_x:pad_x+nw]
    
    # 恢复到原始尺寸
    pred_mask = cv2.resize(pred_mask_actual, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    
    return pred_mask, inference_time


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
    
    # 预热（稳定GPU）
    print("预热模型...")
    if len(image_files) > 0:
        for _ in range(WARMUP_ITERATIONS):
            unet_inference(str(image_files[0]), original_unet_model, "原始UNet")
            if COMPARE_WITH_ENHANCED:
                unet_inference(str(image_files[0]), enhanced_unet_model, "增强UNet")
    print("预热完成！")
    
    # 存储每个样本的结果
    per_sample_results = []
    
    # 原始UNet统计
    original_total_mean_iou = []
    original_total_dice = []
    original_total_fw_iou = []
    original_total_inference_time = []
    
    # 增强UNet统计（如果进行对比）
    enhanced_total_mean_iou = []
    enhanced_total_dice = []
    enhanced_total_fw_iou = []
    enhanced_total_inference_time = []
    
    # 逐图像评估
    print("\n开始评估...")
    for image_path in tqdm(image_files, desc="评估进度"):
        image_name = image_path.stem
        
        # 加载ground truth mask
        mask_path = os.path.join(TEST_MASK_DIR, f"{image_name}.png")
        if not os.path.exists(mask_path):
            # 尝试其他扩展名
            for ext in ['.jpg', '.png', '.bmp']:
                alt_path = os.path.join(TEST_MASK_DIR, f"{image_name}{ext}")
                if os.path.exists(alt_path):
                    mask_path = alt_path
                    break
        
        gt_mask = load_gt_mask(mask_path)
        if gt_mask is None:
            print(f"警告: 未找到 {image_name} 的ground truth mask，跳过")
            continue
        
        # 原始UNet推理
        original_pred_mask, original_inference_time_ms = unet_inference(
            str(image_path), original_unet_model, "原始UNet"
        )
        
        # 确保尺寸一致
        if original_pred_mask.shape != gt_mask.shape:
            gt_mask_resized = cv2.resize(gt_mask, (original_pred_mask.shape[1], original_pred_mask.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
        else:
            gt_mask_resized = gt_mask
        
        # 计算原始UNet指标
        original_mean_iou_value = calculate_mean_iou(original_pred_mask, gt_mask_resized, UNET_NUM_CLASSES)
        original_dice_value = calculate_dice(original_pred_mask, gt_mask_resized, UNET_NUM_CLASSES)
        original_fw_iou_value = calculate_fw_iou(original_pred_mask, gt_mask_resized, UNET_NUM_CLASSES)
        
        original_total_mean_iou.append(original_mean_iou_value)
        original_total_dice.append(original_dice_value)
        original_total_fw_iou.append(original_fw_iou_value)
        original_total_inference_time.append(original_inference_time_ms)
        
        # 增强UNet推理（如果进行对比）
        enhanced_mean_iou_value = None
        enhanced_dice_value = None
        enhanced_fw_iou_value = None
        enhanced_inference_time_ms = None
        
        if COMPARE_WITH_ENHANCED:
            enhanced_pred_mask, enhanced_inference_time_ms = unet_inference(
                str(image_path), enhanced_unet_model, "增强UNet"
            )
            
            # 计算增强UNet指标
            enhanced_mean_iou_value = calculate_mean_iou(enhanced_pred_mask, gt_mask_resized, UNET_NUM_CLASSES)
            enhanced_dice_value = calculate_dice(enhanced_pred_mask, gt_mask_resized, UNET_NUM_CLASSES)
            enhanced_fw_iou_value = calculate_fw_iou(enhanced_pred_mask, gt_mask_resized, UNET_NUM_CLASSES)
            
            enhanced_total_mean_iou.append(enhanced_mean_iou_value)
            enhanced_total_dice.append(enhanced_dice_value)
            enhanced_total_fw_iou.append(enhanced_fw_iou_value)
            enhanced_total_inference_time.append(enhanced_inference_time_ms)
        
        # 存储结果
        result = {
            '图像名称': image_name,
            '原始UNet_Mean_IoU': original_mean_iou_value,
            '原始UNet_Dice': original_dice_value,
            '原始UNet_FW_IoU': original_fw_iou_value,
            '原始UNet_推理时间_ms': original_inference_time_ms,
            '原始UNet_Mean_IoU_%': original_mean_iou_value * 100,
            '原始UNet_Dice_%': original_dice_value * 100,
            '原始UNet_FW_IoU_%': original_fw_iou_value * 100,
        }
        
        if COMPARE_WITH_ENHANCED:
            result.update({
                '增强UNet_Mean_IoU': enhanced_mean_iou_value,
                '增强UNet_Dice': enhanced_dice_value,
                '增强UNet_FW_IoU': enhanced_fw_iou_value,
                '增强UNet_推理时间_ms': enhanced_inference_time_ms,
                '增强UNet_Mean_IoU_%': enhanced_mean_iou_value * 100,
                '增强UNet_Dice_%': enhanced_dice_value * 100,
                '增强UNet_FW_IoU_%': enhanced_fw_iou_value * 100,
                'Mean_IoU提升': (enhanced_mean_iou_value - original_mean_iou_value) * 100,
                'Dice提升': (enhanced_dice_value - original_dice_value) * 100,
                'FW_IoU提升': (enhanced_fw_iou_value - original_fw_iou_value) * 100,
                '推理时间变化_ms': enhanced_inference_time_ms - original_inference_time_ms,
            })
        
        per_sample_results.append(result)
    
    if len(per_sample_results) == 0:
        print("错误：没有成功评估的样本！")
        return
    
    # 计算平均指标
    print("\n" + "="*60)
    print("评估结果汇总")
    print("="*60)
    
    # 原始UNet平均指标
    original_avg_mean_iou = np.mean(original_total_mean_iou) * 100
    original_avg_dice = np.mean(original_total_dice) * 100
    original_avg_fw_iou = np.mean(original_total_fw_iou) * 100
    original_avg_inference_time = np.mean(original_total_inference_time)
    
    original_std_mean_iou = np.std(original_total_mean_iou) * 100
    original_std_dice = np.std(original_total_dice) * 100
    original_std_fw_iou = np.std(original_total_fw_iou) * 100
    original_std_inference_time = np.std(original_total_inference_time)
    
    print(f"\n【原始UNet评估结果】")
    print(f"{'指标':<20} {'平均值':<20} {'标准差':<20}")
    print("-" * 60)
    print(f"{'Mean IoU (%)':<20} {original_avg_mean_iou:<20.4f} {original_std_mean_iou:<20.4f}")
    print(f"{'Dice (%)':<20} {original_avg_dice:<20.4f} {original_std_dice:<20.4f}")
    print(f"{'FW IoU (%)':<20} {original_avg_fw_iou:<20.4f} {original_std_fw_iou:<20.4f}")
    print(f"{'推理时间 (ms/图)':<20} {original_avg_inference_time:<20.4f} {original_std_inference_time:<20.4f}")
    
    # 增强UNet平均指标（如果进行对比）
    if COMPARE_WITH_ENHANCED:
        enhanced_avg_mean_iou = np.mean(enhanced_total_mean_iou) * 100
        enhanced_avg_dice = np.mean(enhanced_total_dice) * 100
        enhanced_avg_fw_iou = np.mean(enhanced_total_fw_iou) * 100
        enhanced_avg_inference_time = np.mean(enhanced_total_inference_time)
        
        enhanced_std_mean_iou = np.std(enhanced_total_mean_iou) * 100
        enhanced_std_dice = np.std(enhanced_total_dice) * 100
        enhanced_std_fw_iou = np.std(enhanced_total_fw_iou) * 100
        enhanced_std_inference_time = np.std(enhanced_total_inference_time)
        
        print(f"\n【增强UNet评估结果】")
        print(f"{'指标':<20} {'平均值':<20} {'标准差':<20}")
        print("-" * 60)
        print(f"{'Mean IoU (%)':<20} {enhanced_avg_mean_iou:<20.4f} {enhanced_std_mean_iou:<20.4f}")
        print(f"{'Dice (%)':<20} {enhanced_avg_dice:<20.4f} {enhanced_std_dice:<20.4f}")
        print(f"{'FW IoU (%)':<20} {enhanced_avg_fw_iou:<20.4f} {enhanced_std_fw_iou:<20.4f}")
        print(f"{'推理时间 (ms/图)':<20} {enhanced_avg_inference_time:<20.4f} {enhanced_std_inference_time:<20.4f}")
        
        # 改进幅度
        print(f"\n【改进幅度】")
        print("-" * 60)
        mean_iou_improve = enhanced_avg_mean_iou - original_avg_mean_iou
        dice_improve = enhanced_avg_dice - original_avg_dice
        fw_iou_improve = enhanced_avg_fw_iou - original_avg_fw_iou
        time_change = enhanced_avg_inference_time - original_avg_inference_time
        
        print(f"Mean IoU提升: {mean_iou_improve:+.2f}% ({mean_iou_improve/original_avg_mean_iou*100:+.2f}%)")
        print(f"Dice提升: {dice_improve:+.2f}% ({dice_improve/original_avg_dice*100:+.2f}%)")
        print(f"FW IoU提升: {fw_iou_improve:+.2f}% ({fw_iou_improve/original_avg_fw_iou*100:+.2f}%)")
        print(f"推理时间变化: {time_change:+.2f} ms ({time_change/original_avg_inference_time*100:+.2f}%)")
    
    # 保存汇总结果到JSON
    summary_results = {
        'original_unet': {
            'mean_iou_percent': float(original_avg_mean_iou),
            'dice_percent': float(original_avg_dice),
            'fw_iou_percent': float(original_avg_fw_iou),
            'inference_time_ms': float(original_avg_inference_time),
            'std_mean_iou_percent': float(original_std_mean_iou),
            'std_dice_percent': float(original_std_dice),
            'std_fw_iou_percent': float(original_std_fw_iou),
            'std_inference_time_ms': float(original_std_inference_time),
        },
        'total_samples': len(per_sample_results)
    }
    
    if COMPARE_WITH_ENHANCED:
        summary_results['enhanced_unet'] = {
            'mean_iou_percent': float(enhanced_avg_mean_iou),
            'dice_percent': float(enhanced_avg_dice),
            'fw_iou_percent': float(enhanced_avg_fw_iou),
            'inference_time_ms': float(enhanced_avg_inference_time),
            'std_mean_iou_percent': float(enhanced_std_mean_iou),
            'std_dice_percent': float(enhanced_std_dice),
            'std_fw_iou_percent': float(enhanced_std_fw_iou),
            'std_inference_time_ms': float(enhanced_std_inference_time),
        }
        summary_results['improvement'] = {
            'mean_iou_improve_percent': float(mean_iou_improve),
            'dice_improve_percent': float(dice_improve),
            'fw_iou_improve_percent': float(fw_iou_improve),
            'time_change_ms': float(time_change),
        }
    
    with open(os.path.join(OUTPUT_DIR, 'summary_results.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    
    # ==================== 生成每个样本的详细表格 ====================
    print("\n" + "="*60)
    print("生成每个样本的详细结果表格...")
    
    if HAS_PANDAS:
        df = pd.DataFrame(per_sample_results)
        
        # 保存为CSV
        csv_path = os.path.join(OUTPUT_DIR, 'per_sample_results.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ CSV表格已保存: {csv_path}")
        
        # 保存为Excel
        try:
            excel_path = os.path.join(OUTPUT_DIR, 'per_sample_results.xlsx')
            df.to_excel(excel_path, index=False, engine='openpyxl')
            print(f"✓ Excel表格已保存: {excel_path}")
        except ImportError:
            print("⚠ 未安装openpyxl，跳过Excel导出（可运行: pip install openpyxl）")
        
        # 打印表格预览
        print("\n前10个样本的结果预览：")
        if COMPARE_WITH_ENHANCED:
            cols = ['图像名称', '原始UNet_Mean_IoU_%', '增强UNet_Mean_IoU_%', 'Mean_IoU提升', 
                   '原始UNet_Dice_%', '增强UNet_Dice_%', 'Dice提升']
        else:
            cols = ['图像名称', '原始UNet_Mean_IoU_%', '原始UNet_Dice_%', '原始UNet_FW_IoU_%', '原始UNet_推理时间_ms']
        print(df[cols].head(10).to_string(index=False))
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
    
    # ==================== 生成可视化图表 ====================
    if HAS_MATPLOTLIB:
        print("\n" + "="*60)
        print("生成可视化图表...")
        
        if COMPARE_WITH_ENHANCED:
            # 对比评估图表
            fig = plt.figure(figsize=(18, 12))
            
            # 子图1-3: 原始UNet指标分布
            for idx, (metric_name, metric_data, metric_avg, metric_std, color) in enumerate([
                ('Mean IoU', original_total_mean_iou, original_avg_mean_iou, original_std_mean_iou, '#3498db'),
                ('Dice', original_total_dice, original_avg_dice, original_std_dice, '#2ecc71'),
                ('FW IoU', original_total_fw_iou, original_avg_fw_iou, original_std_fw_iou, '#9b59b6')
            ], 1):
                ax = plt.subplot(3, 3, idx)
                ax.hist([x * 100 for x in metric_data], bins=20, alpha=0.6, color=color, 
                       edgecolor='black', label='原始UNet')
                ax.hist([x * 100 for x in [enhanced_total_mean_iou, enhanced_total_dice, enhanced_total_fw_iou][idx-1]], 
                       bins=20, alpha=0.6, color='#e74c3c', edgecolor='black', label='增强UNet')
                ax.axvline(metric_avg, color=color, linestyle='--', linewidth=2, 
                           label=f'原始UNet平均: {metric_avg:.2f}%')
                ax.axvline([enhanced_avg_mean_iou, enhanced_avg_dice, enhanced_avg_fw_iou][idx-1], 
                          color='#e74c3c', linestyle='--', linewidth=2,
                          label=f'增强UNet平均: {[enhanced_avg_mean_iou, enhanced_avg_dice, enhanced_avg_fw_iou][idx-1]:.2f}%')
                ax.set_xlabel(f'{metric_name} (%)')
                ax.set_ylabel('样本数')
                ax.set_title(f'{metric_name}分布对比', fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
            
            # 子图4: 平均指标对比
            ax4 = plt.subplot(3, 3, 4)
            metrics = ['Mean IoU', 'Dice', 'FW IoU']
            original_values = [original_avg_mean_iou, original_avg_dice, original_avg_fw_iou]
            enhanced_values = [enhanced_avg_mean_iou, enhanced_avg_dice, enhanced_avg_fw_iou]
            x_pos = np.arange(len(metrics))
            width = 0.35
            ax4.bar(x_pos - width/2, original_values, width, label='原始UNet', alpha=0.8, color='#3498db')
            ax4.bar(x_pos + width/2, enhanced_values, width, label='增强UNet', alpha=0.8, color='#e74c3c')
            ax4.set_ylabel('百分比 (%)')
            ax4.set_title('平均指标对比', fontsize=12, fontweight='bold')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(metrics)
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.set_ylim([0, 100])
            
            # 子图5: 推理时间对比
            ax5 = plt.subplot(3, 3, 5)
            ax5.bar(['原始UNet', '增强UNet'], [original_avg_inference_time, enhanced_avg_inference_time], 
                   alpha=0.8, color=['#3498db', '#e74c3c'], edgecolor='black')
            ax5.set_ylabel('时间 (ms)')
            ax5.set_title('平均推理时间对比', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='y')
            
            # 子图6: 改进幅度
            ax6 = plt.subplot(3, 3, 6)
            improvements = [mean_iou_improve, dice_improve, fw_iou_improve]
            colors_improve = ['green' if x > 0 else 'red' for x in improvements]
            ax6.bar(metrics, improvements, alpha=0.8, color=colors_improve, edgecolor='black')
            ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax6.set_ylabel('提升幅度 (%)')
            ax6.set_title('改进幅度', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3, axis='y')
            
            # 子图7-9: 每个样本的指标趋势
            sample_indices = range(len(per_sample_results))
            for idx, (metric_name, original_col, enhanced_col, ax_idx) in enumerate([
                ('Mean IoU', '原始UNet_Mean_IoU_%', '增强UNet_Mean_IoU_%', 7),
                ('Dice', '原始UNet_Dice_%', '增强UNet_Dice_%', 8),
                ('FW IoU', '原始UNet_FW_IoU_%', '增强UNet_FW_IoU_%', 9)
            ]):
                ax = plt.subplot(3, 3, ax_idx)
                ax.plot(sample_indices, [r[original_col] for r in per_sample_results], 
                       marker='o', label='原始UNet', linewidth=1.5, markersize=3, alpha=0.7, color='#3498db')
                ax.plot(sample_indices, [r[enhanced_col] for r in per_sample_results], 
                       marker='s', label='增强UNet', linewidth=1.5, markersize=3, alpha=0.7, color='#e74c3c')
                ax.set_xlabel('样本编号')
                ax.set_ylabel(f'{metric_name} (%)')
                ax.set_title(f'每个样本的{metric_name}趋势', fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(OUTPUT_DIR, 'comparison_results.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ 对比评估图表已保存: {plot_path}")
            plt.close()
        else:
            # 仅原始UNet评估图表（与evaluate_unet.py类似）
            fig = plt.figure(figsize=(16, 10))
            
            # 子图1-4: 指标分布和推理时间
            for idx, (metric_name, metric_data, metric_avg, metric_std, color) in enumerate([
                ('Mean IoU', original_total_mean_iou, original_avg_mean_iou, original_std_mean_iou, '#3498db'),
                ('Dice', original_total_dice, original_avg_dice, original_std_dice, '#2ecc71'),
                ('FW IoU', original_total_fw_iou, original_avg_fw_iou, original_std_fw_iou, '#9b59b6'),
                ('推理时间', original_total_inference_time, original_avg_inference_time, original_std_inference_time, '#f39c12')
            ], 1):
                ax = plt.subplot(2, 3, idx)
                if idx == 4:
                    ax.hist(metric_data, bins=20, alpha=0.7, color=color, edgecolor='black')
                    ax.axvline(metric_avg, color='red', linestyle='--', linewidth=2, 
                              label=f'平均值: {metric_avg:.2f} ms')
                    ax.set_xlabel('推理时间 (ms)')
                else:
                    ax.hist([x * 100 for x in metric_data], bins=20, alpha=0.7, color=color, edgecolor='black')
                    ax.axvline(metric_avg, color='red', linestyle='--', linewidth=2, 
                              label=f'平均值: {metric_avg:.2f}%')
                    ax.set_xlabel(f'{metric_name} (%)')
                ax.set_ylabel('样本数')
                ax.set_title(f'{metric_name}分布', fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
            
            # 子图5: 指标对比箱线图
            ax5 = plt.subplot(2, 3, 5)
            data_to_plot = [
                [x * 100 for x in original_total_mean_iou],
                [x * 100 for x in original_total_dice],
                [x * 100 for x in original_total_fw_iou]
            ]
            bp = ax5.boxplot(data_to_plot, labels=['Mean IoU', 'Dice', 'FW IoU'], patch_artist=True)
            colors_box = ['#3498db', '#2ecc71', '#9b59b6']
            for patch, color in zip(bp['boxes'], colors_box):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax5.set_ylabel('百分比 (%)')
            ax5.set_title('指标分布对比', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='y')
            
            # 子图6: 每个样本的指标趋势
            ax6 = plt.subplot(2, 3, 6)
            sample_indices = range(len(per_sample_results))
            ax6.plot(sample_indices, [r['原始UNet_Mean_IoU_%'] for r in per_sample_results], 
                    marker='o', label='Mean IoU', linewidth=1.5, markersize=4, alpha=0.7)
            ax6.plot(sample_indices, [r['原始UNet_Dice_%'] for r in per_sample_results], 
                    marker='s', label='Dice', linewidth=1.5, markersize=4, alpha=0.7)
            ax6.plot(sample_indices, [r['原始UNet_FW_IoU_%'] for r in per_sample_results], 
                    marker='^', label='FW IoU', linewidth=1.5, markersize=4, alpha=0.7)
            ax6.set_xlabel('样本编号')
            ax6.set_ylabel('百分比 (%)')
            ax6.set_title('每个样本的指标趋势', fontsize=12, fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(OUTPUT_DIR, 'evaluation_results.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ 评估结果图表已保存: {plot_path}")
            plt.close()
    else:
        print("\n⚠ 未安装matplotlib，跳过图表生成")
    
    print(f"\n所有结果已保存到: {OUTPUT_DIR}")
    print("="*60)
    print("\n评估完成！")
    print(f"原始UNet平均指标:")
    print(f"  Mean IoU: {original_avg_mean_iou:.2f}%")
    print(f"  Dice: {original_avg_dice:.2f}%")
    print(f"  FW IoU: {original_avg_fw_iou:.2f}%")
    print(f"  推理时间: {original_avg_inference_time:.2f} ms/图")
    
    if COMPARE_WITH_ENHANCED:
        print(f"\n增强UNet平均指标:")
        print(f"  Mean IoU: {enhanced_avg_mean_iou:.2f}%")
        print(f"  Dice: {enhanced_avg_dice:.2f}%")
        print(f"  FW IoU: {enhanced_avg_fw_iou:.2f}%")
        print(f"  推理时间: {enhanced_avg_inference_time:.2f} ms/图")


if __name__ == "__main__":
    main()

