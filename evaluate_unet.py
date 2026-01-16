# -*- coding: utf-8 -*-
"""
UNet模型单独评估脚本
计算指标：
- Mean IoU (%)
- Dice (%)
- FW IoU (%)
- 推理时间 (ms/图)
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
# UNet模型路径
UNET_MODEL_PATH = r"E:\Data_Industry\膀胱癌细胞检测_文章\Unet_attention_文件\last_model_1.pth"

# 测试数据路径
TEST_IMAGE_DIR = r"E:\Data_Industry\split_images\edit_1004560_MergedBatch\VOC2012\JPEGImages"
TEST_MASK_DIR = r"E:\Data_Industry\split_images\edit_1004560_MergedBatch\VOC2012\SegmentationClass"  # Ground truth mask目录

# 输出结果目录
OUTPUT_DIR = r"E:\Data_Industry\膀胱癌细胞检测_文章\AT-UNet单独评估\1219重新运行"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# UNet参数
UNET_INPUT_SIZE = (1024, 1024)  # (height, width)
UNET_NUM_CLASSES = 2  # 背景 + 癌细胞

# 推理参数
MASK_THRESHOLD = 0.5  # 用于二值化的阈值（如果使用概率图）
USE_PROB_THRESHOLD = False  # True: 使用概率图阈值, False: 使用argmax

# 预热次数（用于稳定推理时间测量）
WARMUP_ITERATIONS = 5
# ==============================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载模型
print("加载增强版UNet模型（含注意力门控和转置卷积）...")
from model.unet_resnet_enhanced import Unet
# 使用增强版UNet（默认启用注意力门控和转置卷积）
unet_model = Unet(num_classes=UNET_NUM_CLASSES, enhanced=True)
# 加载权重
state_dict = torch.load(UNET_MODEL_PATH, map_location=device)
# 如果权重键名有 'model.' 前缀（增强版包装），需要处理
if any(k.startswith('model.') for k in state_dict.keys()):
    # 权重已经是正确的格式（有model.前缀）
    unet_model.load_state_dict(state_dict, strict=False)
else:
    # 如果权重没有model.前缀，可能是直接保存的增强版内部模型
    # 尝试添加model.前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith('model.'):
            new_state_dict['model.' + k] = v
        else:
            new_state_dict[k] = v
    unet_model.load_state_dict(new_state_dict, strict=False)
unet_model.eval()
unet_model.to(device)
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


def unet_inference(image_path, model):
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
            unet_inference(str(image_files[0]), unet_model)
    print("预热完成！")
    
    # 存储每个样本的结果
    per_sample_results = []
    
    # 总体统计
    total_mean_iou = []
    total_dice = []
    total_fw_iou = []
    total_inference_time = []
    
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
        
        # UNet推理
        pred_mask, inference_time_ms = unet_inference(str(image_path), unet_model)
        
        # 确保尺寸一致
        if pred_mask.shape != gt_mask.shape:
            gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # 计算指标
        mean_iou_value = calculate_mean_iou(pred_mask, gt_mask, UNET_NUM_CLASSES)
        dice_value = calculate_dice(pred_mask, gt_mask, UNET_NUM_CLASSES)
        fw_iou_value = calculate_fw_iou(pred_mask, gt_mask, UNET_NUM_CLASSES)
        
        # 存储结果
        per_sample_results.append({
            '图像名称': image_name,
            'Mean_IoU': mean_iou_value,
            'Dice': dice_value,
            'FW_IoU': fw_iou_value,
            '推理时间_ms': inference_time_ms,
            'Mean_IoU_%': mean_iou_value * 100,
            'Dice_%': dice_value * 100,
            'FW_IoU_%': fw_iou_value * 100,
        })
        
        total_mean_iou.append(mean_iou_value)
        total_dice.append(dice_value)
        total_fw_iou.append(fw_iou_value)
        total_inference_time.append(inference_time_ms)
    
    if len(per_sample_results) == 0:
        print("错误：没有成功评估的样本！")
        return
    
    # 计算平均指标
    print("\n" + "="*60)
    print("评估结果汇总")
    print("="*60)
    
    avg_mean_iou = np.mean(total_mean_iou) * 100
    avg_dice = np.mean(total_dice) * 100
    avg_fw_iou = np.mean(total_fw_iou) * 100
    avg_inference_time = np.mean(total_inference_time)
    
    std_mean_iou = np.std(total_mean_iou) * 100
    std_dice = np.std(total_dice) * 100
    std_fw_iou = np.std(total_fw_iou) * 100
    std_inference_time = np.std(total_inference_time)
    
    print(f"\n{'指标':<20} {'平均值':<20} {'标准差':<20}")
    print("-" * 60)
    print(f"{'Mean IoU (%)':<20} {avg_mean_iou:<20.4f} {std_mean_iou:<20.4f}")
    print(f"{'Dice (%)':<20} {avg_dice:<20.4f} {std_dice:<20.4f}")
    print(f"{'FW IoU (%)':<20} {avg_fw_iou:<20.4f} {std_fw_iou:<20.4f}")
    print(f"{'推理时间 (ms/图)':<20} {avg_inference_time:<20.4f} {std_inference_time:<20.4f}")
    
    # 保存汇总结果到JSON
    summary_results = {
        'mean_iou_percent': float(avg_mean_iou),
        'dice_percent': float(avg_dice),
        'fw_iou_percent': float(avg_fw_iou),
        'inference_time_ms': float(avg_inference_time),
        'std_mean_iou_percent': float(std_mean_iou),
        'std_dice_percent': float(std_dice),
        'std_fw_iou_percent': float(std_fw_iou),
        'std_inference_time_ms': float(std_inference_time),
        'total_samples': len(per_sample_results)
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
        
        # 打印表格预览（前10行）
        print("\n前10个样本的结果预览：")
        print(df[['图像名称', 'Mean_IoU_%', 'Dice_%', 'FW_IoU_%', '推理时间_ms']].head(10).to_string(index=False))
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
            print(f"  Mean IoU: {result['Mean_IoU_%']:.2f}%")
            print(f"  Dice: {result['Dice_%']:.2f}%")
            print(f"  FW IoU: {result['FW_IoU_%']:.2f}%")
            print(f"  推理时间: {result['推理时间_ms']:.2f} ms")
    
    # ==================== 生成可视化图表 ====================
    if HAS_MATPLOTLIB:
        print("\n" + "="*60)
        print("生成可视化图表...")
        
        # 创建综合对比图
        fig = plt.figure(figsize=(16, 10))
        
        # 子图1: Mean IoU分布
        ax1 = plt.subplot(2, 3, 1)
        ax1.hist([r['Mean_IoU_%'] for r in per_sample_results], bins=20, alpha=0.7, color='#3498db', edgecolor='black')
        ax1.axvline(avg_mean_iou, color='red', linestyle='--', linewidth=2, 
                   label=f'平均值: {avg_mean_iou:.2f}%')
        ax1.set_xlabel('Mean IoU (%)')
        ax1.set_ylabel('样本数')
        ax1.set_title('Mean IoU分布', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 子图2: Dice分布
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist([r['Dice_%'] for r in per_sample_results], bins=20, alpha=0.7, color='#2ecc71', edgecolor='black')
        ax2.axvline(avg_dice, color='red', linestyle='--', linewidth=2, 
                   label=f'平均值: {avg_dice:.2f}%')
        ax2.set_xlabel('Dice (%)')
        ax2.set_ylabel('样本数')
        ax2.set_title('Dice系数分布', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 子图3: FW IoU分布
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist([r['FW_IoU_%'] for r in per_sample_results], bins=20, alpha=0.7, color='#9b59b6', edgecolor='black')
        ax3.axvline(avg_fw_iou, color='red', linestyle='--', linewidth=2, 
                   label=f'平均值: {avg_fw_iou:.2f}%')
        ax3.set_xlabel('FW IoU (%)')
        ax3.set_ylabel('样本数')
        ax3.set_title('FW IoU分布', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 子图4: 推理时间分布
        ax4 = plt.subplot(2, 3, 4)
        ax4.hist([r['推理时间_ms'] for r in per_sample_results], bins=20, alpha=0.7, color='#f39c12', edgecolor='black')
        ax4.axvline(avg_inference_time, color='red', linestyle='--', linewidth=2, 
                   label=f'平均值: {avg_inference_time:.2f} ms')
        ax4.set_xlabel('推理时间 (ms)')
        ax4.set_ylabel('样本数')
        ax4.set_title('推理时间分布', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 子图5: 指标对比（箱线图）
        ax5 = plt.subplot(2, 3, 5)
        data_to_plot = [
            [r['Mean_IoU_%'] for r in per_sample_results],
            [r['Dice_%'] for r in per_sample_results],
            [r['FW_IoU_%'] for r in per_sample_results]
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
        ax6.plot(sample_indices, [r['Mean_IoU_%'] for r in per_sample_results], 
                marker='o', label='Mean IoU', linewidth=1.5, markersize=4, alpha=0.7)
        ax6.plot(sample_indices, [r['Dice_%'] for r in per_sample_results], 
                marker='s', label='Dice', linewidth=1.5, markersize=4, alpha=0.7)
        ax6.plot(sample_indices, [r['FW_IoU_%'] for r in per_sample_results], 
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
        
        # 生成汇总对比图
        fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 平均指标柱状图
        ax1 = axes[0]
        metrics = ['Mean IoU', 'Dice', 'FW IoU']
        values = [avg_mean_iou, avg_dice, avg_fw_iou]
        stds = [std_mean_iou, std_dice, std_fw_iou]
        colors_bar = ['#3498db', '#2ecc71', '#9b59b6']
        bars = ax1.bar(metrics, values, yerr=stds, capsize=5, alpha=0.8, color=colors_bar, edgecolor='black')
        ax1.set_ylabel('百分比 (%)')
        ax1.set_title('平均指标对比（含标准差）', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim([0, 100])
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 1,
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
        
        # 推理时间信息
        ax2 = axes[1]
        ax2.bar(['平均推理时间'], [avg_inference_time], alpha=0.8, color='#f39c12', edgecolor='black')
        ax2.errorbar(['平均推理时间'], [avg_inference_time], yerr=[std_inference_time], 
                    fmt='none', color='black', capsize=10, capthick=2)
        ax2.set_ylabel('时间 (ms)')
        ax2.set_title(f'推理时间统计\n平均值: {avg_inference_time:.2f} ms, 标准差: {std_inference_time:.2f} ms', 
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        # 添加数值标签
        ax2.text(0, avg_inference_time + std_inference_time + max(avg_inference_time * 0.05, 1),
                f'{avg_inference_time:.2f} ms', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        summary_plot_path = os.path.join(OUTPUT_DIR, 'summary_comparison.png')
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ 汇总对比图已保存: {summary_plot_path}")
        plt.close()
    else:
        print("\n⚠ 未安装matplotlib，跳过图表生成")
    
    print(f"\n所有结果已保存到: {OUTPUT_DIR}")
    print("="*60)
    print("\n评估完成！")
    print(f"平均指标:")
    print(f"  Mean IoU: {avg_mean_iou:.2f}%")
    print(f"  Dice: {avg_dice:.2f}%")
    print(f"  FW IoU: {avg_fw_iou:.2f}%")
    print(f"  推理时间: {avg_inference_time:.2f} ms/图")


if __name__ == "__main__":
    main()

