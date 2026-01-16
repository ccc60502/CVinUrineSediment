# -*- coding: utf-8 -*-
# filename: yolo_unet_two_stage_predict.py

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO
from pathlib import Path

# ================== 请修改这几行路径 ==================
# 1. YOLOv11 检测模型
YOLO_MODEL_PATH = r"E:\Data_Industry\Unet_wdir\2stage_wdir\best.pt"

# 2. 你的UNet模型权重（和predict.py里一样）
UNET_MODEL_PATH = r"E:\Data_Industry\膀胱癌细胞检测_文章\Unet_attention_文件\模型评估文件\last_model_1.pth"

# 2.5. UNet模型版本选择
USE_ENHANCED_UNET = True  # True: 使用增强版UNet（注意力门控+转置卷积）
                          # False: 使用原始版UNet（双线性插值上采样）

# 2.6. 增强版UNet模块开关（仅当USE_ENHANCED_UNET=True时有效）
USE_ATTENTION = True          # 是否使用注意力门控
USE_TRANSPOSE = True          # 是否使用转置卷积上采样
USE_NUCLEUS_CYTOPLASM = False  # 是否使用核质分离感知模块

# 3. 输入图片或文件夹
INPUT_PATH = r"E:\Data_Industry\Unet_wdir\test_wdir\复杂样本输入"   # 可以是单张图或文件夹

# 4. 输出文件夹
OUTPUT_FOLDER = r"E:\Data_Industry\Unet_wdir\test_wdir\yolo_Unet复杂样本输出"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
debug_dir = None
# YOLO 参数
YOLO_CONF = 0.39
YOLO_CLASSES = [0]   # 你要检测的类别（比如0是癌细胞）

# UNET 参数（和你原来的predict.py保持一致）
UNET_INPUT_SIZE = (1024, 1024)   # 你原来用的就是480
UNET_NUM_CLASSES = 1 + 1       # 背景 + 细胞（你原来num_classes=1）

# 后处理参数
MASK_THRESHOLD = 0.9

# 过滤参数（根据UNet结果过滤YOLO检测框）
FILTER_BY_UNET = True          # 是否启用UNet过滤功能
UNET_FILTER_THRESHOLD = 0.01   # UNet在YOLO框内的癌细胞像素比例阈值，低于此值则删除该框
                                # 建议范围：0.01-0.05，值越小越严格

# 重叠框合并参数
MERGE_OVERLAPPING_BOXES = True  # 是否合并重叠的检测框
MERGE_IOU_THRESHOLD = 0.05      # 合并重叠框的IoU阈值，高于此值则合并
                                # 建议范围：0.2-0.5，值越大越严格（只合并高度重叠的框）
# ====================================================


# 加载 YOLO 模型
yolo_model = YOLO(YOLO_MODEL_PATH)

# 加载UNet模型（根据参数选择原始版或增强版）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if USE_ENHANCED_UNET:
    print("加载增强版UNet模型...")
    from model.unet_resnet_enhanced import Unet
    # 使用增强版UNet（可配置各个模块）
    unet_model = Unet(
        num_classes=UNET_NUM_CLASSES, 
        enhanced=True,
        use_attention=USE_ATTENTION,
        use_transpose=USE_TRANSPOSE,
        use_nucleus_cytoplasm=USE_NUCLEUS_CYTOPLASM
    )
    # 加载权重（自动处理权重键名）
    state_dict = torch.load(UNET_MODEL_PATH, map_location=device)
    # 处理权重键名（增强版有model.前缀）
    if any(k.startswith('model.') for k in state_dict.keys()):
        # 权重已经有model.前缀，直接加载
        unet_model.load_state_dict(state_dict, strict=False)
    else:
        # 如果权重没有model.前缀，添加前缀
        new_state_dict = {'model.' + k: v for k, v in state_dict.items()}
        unet_model.load_state_dict(new_state_dict, strict=False)
    module_status = []
    if USE_ATTENTION:
        module_status.append("注意力门控")
    if USE_TRANSPOSE:
        module_status.append("转置卷积")
    if USE_NUCLEUS_CYTOPLASM:
        module_status.append("核质分离")
    print(f"增强版UNet模型加载完成！启用模块: {', '.join(module_status) if module_status else '无'}")
else:
    print("加载原始版UNet模型（双线性插值上采样）...")
    from model.unet_resnet import Unet
    # 使用原始版UNet
    unet_model = Unet(num_classes=UNET_NUM_CLASSES)
    # 加载权重
    unet_model.load_state_dict(torch.load(UNET_MODEL_PATH, map_location=device), strict=False)
    print("原始版UNet模型加载完成！")

unet_model.eval()
unet_model.to(device)

from utils.utils import cvtColor, preprocess_input, resize_image  # 直接用你原来的工具函数


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


def merge_overlapping_boxes(boxes, confidences, iou_threshold=0.3):
    """
    合并重叠的检测框
    返回合并后的框列表和对应的置信度列表
    """
    if len(boxes) == 0:
        return [], []
    
    # 转换为numpy数组便于处理
    boxes = np.array(boxes)
    confidences = np.array(confidences)
    
    # 按置信度排序（高置信度优先）
    sorted_indices = np.argsort(confidences)[::-1]
    boxes = boxes[sorted_indices]
    confidences = confidences[sorted_indices]
    
    merged_boxes = []
    merged_confs = []
    used = [False] * len(boxes)
    
    for i in range(len(boxes)):
        if used[i]:
            continue
        
        # 当前框
        current_box = boxes[i].copy()
        current_conf = confidences[i]
        group_indices = [i]
        
        # 查找所有与当前框重叠的框
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            
            iou = calculate_iou(current_box, boxes[j])
            if iou >= iou_threshold:
                group_indices.append(j)
                used[j] = True
        
        # 合并这一组框：计算最小外接矩形
        if len(group_indices) > 1:
            # 多个框，计算合并后的框
            group_boxes = boxes[group_indices]
            x1_min = np.min(group_boxes[:, 0])
            y1_min = np.min(group_boxes[:, 1])
            x2_max = np.max(group_boxes[:, 2])
            y2_max = np.max(group_boxes[:, 3])
            merged_box = np.array([x1_min, y1_min, x2_max, y2_max], dtype=int)
            # 合并后的置信度取平均值
            merged_conf = np.mean(confidences[group_indices])
        else:
            # 单个框，直接使用
            merged_box = current_box
            merged_conf = current_conf
        
        merged_boxes.append(merged_box)
        merged_confs.append(merged_conf)
        used[i] = True
    
    return merged_boxes, merged_confs


def crop_and_prepare_patch(image_pil, bbox, expand_ratio=0.6):
    x1, y1, x2, y2 = map(int, bbox)
    w, h = x2 - x1, y2 - y1

    # 扩展边界
    expand_w = int(w * expand_ratio)
    expand_h = int(h * expand_ratio)
    x1 = max(0, x1 - expand_w)
    y1 = max(0, y1 - expand_h)
    x2 = min(image_pil.width, x2 + expand_w)
    y2 = min(image_pil.height, y2 + expand_h)

    patch = image_pil.crop((x1, y1, x2, y2))

    # 必须和训练时完全一致的预处理
    patch = cvtColor(patch)
    patch_resized, nw, nh = resize_image(patch, (UNET_INPUT_SIZE[1], UNET_INPUT_SIZE[0]))
    
    # 计算padding偏移量（resize_image会在480x480画布上居中放置图像）
    pad_x = (UNET_INPUT_SIZE[1] - nw) // 2  # 左右padding
    pad_y = (UNET_INPUT_SIZE[0] - nh) // 2  # 上下padding
    
    record = {
        "orig_xy": (x1, y1),
        "crop_size": (x2 - x1, y2 - y1),  # (width, height) - PIL坐标系
        "orig_bbox": bbox,
        "resized_size": (nw, nh),      # 实际图像区域在480x480中的尺寸 (width, height)
        "pad_offset": (pad_x, pad_y)   # padding偏移量，用于mask映射
    }
    
    patch_input = np.expand_dims(np.transpose(preprocess_input(np.array(patch_resized, np.float32)), (2, 0, 1)), 0)
    patch_tensor = torch.from_numpy(patch_input).type(torch.FloatTensor).to(device)

    return patch_tensor, record, patch_resized


def unet_predict(patch_tensor, patch_idx, patch_for_debug):
    global debug_dir
    with torch.no_grad():
        output = unet_model(patch_tensor)
        pr = output[0]

        pr = F.softmax(pr, dim=0).cpu().numpy()

        prob_map = pr[1]
        mask_480 = (prob_map >= MASK_THRESHOLD).astype(np.uint8)

        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"cell_{patch_idx:03d}_crop.jpg"),
                    cv2.cvtColor(patch_for_debug, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(debug_dir, f"cell_{patch_idx:03d}_prob.jpg"),
                    (np.clip(prob_map, 0, 1) * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(debug_dir, f"cell_{patch_idx:03d}_mask_480.jpg"),
                    (mask_480 * 255).astype(np.uint8))

        return mask_480


def mask_back_to_original(mask_480, record, orig_shape):
    """
    把UNET的mask映射回原图坐标
    关键修复：正确处理resize时的padding，避免坐标偏移
    """
    H, W = orig_shape[:2]
    
    # 1. 从480x480的mask中提取实际图像区域（去除padding）
    pad_x, pad_y = record["pad_offset"]
    nw, nh = record["resized_size"]  # (width, height)
    
    # 提取实际图像区域的mask（去除左右上下的padding）
    # mask_480是numpy数组，形状是(height, width)，所以索引是[行, 列]
    mask_actual = mask_480[pad_y:pad_y+nh, pad_x:pad_x+nw]  # 形状: (nh, nw)
    
    # 2. 将实际图像区域的mask resize回crop区域的原始尺寸
    crop_w, crop_h = record["crop_size"]  # (width, height) - 注意顺序！
    # cv2.resize需要(width, height)，返回的numpy数组形状是(height, width)
    mask_resized = cv2.resize(mask_actual, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
    # mask_resized形状: (crop_h, crop_w)

    # 3. 放回原图（处理边界情况）
    x1, y1 = record["orig_xy"]
    full_mask = np.zeros((H, W), dtype=np.uint8)

    # 计算实际可用的区域（防止越界）
    # numpy数组索引是[行(y), 列(x)]，所以y对应高度，x对应宽度
    y_end = min(y1 + crop_h, H)  # crop_h是高度
    x_end = min(x1 + crop_w, W)  # crop_w是宽度
    actual_h = y_end - y1
    actual_w = x_end - x1
    
    # 如果实际区域小于resize后的mask，需要裁剪mask
    if actual_h < crop_h or actual_w < crop_w:
        mask_resized = mask_resized[:actual_h, :actual_w]
    
    # 安全地放置mask（numpy数组索引：[行(y), 列(x)]）
    full_mask[y1:y_end, x1:x_end] = mask_resized

    return full_mask  # 这个就是每个细胞独立的实例mask


def process_one_image(img_path):
    global debug_dir
    debug_dir = os.path.join(OUTPUT_FOLDER, "debug_unet_patches", Path(img_path).stem)
    os.makedirs(debug_dir, exist_ok=True)
    # 每次处理一张图就新建一个子文件夹，干净清晰

    image_pil = Image.open(img_path).convert("RGB")
    orig_np = np.array(image_pil)
    h, w = orig_np.shape[:2]

    results = yolo_model.predict(source=img_path, conf=YOLO_CONF,
                                 classes=YOLO_CLASSES, save=False, verbose=False)[0]

    # 提取所有检测框和置信度
    all_boxes = []
    all_confs = []
    for box in results.boxes:
        bbox = box.xyxy[0].cpu().numpy().astype(int)
        conf = box.conf[0].item()
        all_boxes.append(bbox)
        all_confs.append(conf)
    
    # 合并重叠的检测框
    if MERGE_OVERLAPPING_BOXES and len(all_boxes) > 1:
        merged_boxes, merged_confs = merge_overlapping_boxes(
            all_boxes, all_confs, iou_threshold=MERGE_IOU_THRESHOLD
        )
        print(f"  原始检测框: {len(all_boxes)} 个, 合并后: {len(merged_boxes)} 个")
    else:
        merged_boxes = all_boxes
        merged_confs = all_confs

    vis = orig_np.copy()
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    
    # 存储所有mask用于最终可视化（避免颜色叠加）
    all_masks = []
    all_bboxes = []
    all_confs_final = []
    all_labels = []
    
    # 存储有效的检测结果（通过UNet验证的）
    valid_detections = []

    for idx, (bbox, conf) in enumerate(zip(merged_boxes, merged_confs)):
        # 1 crop & 预处理（使用合并后的bbox）
        patch_tensor, record, patch_resized_pil_or_np = crop_and_prepare_patch(image_pil, bbox, expand_ratio=0.2)

        # 转成numpy供调试保存用（不管是PIL还是np都转成np）
        if isinstance(patch_resized_pil_or_np, Image.Image):
            patch_debug = np.array(patch_resized_pil_or_np)
        else:
            patch_debug = patch_resized_pil_or_np

        # 2 UNET推理（现在传三个参数，永不缺变量！）
        unet_mask_480 = unet_predict(patch_tensor, idx, patch_debug)

        # 3 映射回原图（修复了padding和坐标系统导致的偏移问题）
        instance_mask = mask_back_to_original(unet_mask_480, record, orig_np.shape)
        
        # 4 根据UNet结果过滤YOLO检测框
        cancer_ratio = None
        if FILTER_BY_UNET:
            # 提取原始YOLO框内的mask区域（不是扩展后的框）
            x1_orig, y1_orig, x2_orig, y2_orig = bbox
            # 确保坐标不越界
            x1_orig = max(0, x1_orig)
            y1_orig = max(0, y1_orig)
            x2_orig = min(w, x2_orig)
            y2_orig = min(h, y2_orig)
            
            # 提取YOLO框内的mask
            bbox_mask = instance_mask[y1_orig:y2_orig, x1_orig:x2_orig]
            
            # 计算YOLO框内癌细胞的像素比例
            bbox_area = (x2_orig - x1_orig) * (y2_orig - y1_orig)
            if bbox_area > 0:
                cancer_pixels = np.sum(bbox_mask == 1)
                cancer_ratio = cancer_pixels / bbox_area
                
                # 如果癌细胞比例低于阈值，跳过这个检测框
                if cancer_ratio < UNET_FILTER_THRESHOLD:
                    print(f"  过滤检测框 {idx}: YOLO置信度={conf:.3f}, UNet癌细胞比例={cancer_ratio:.4f} < {UNET_FILTER_THRESHOLD}")
                    continue  # 跳过这个检测框，不进行后续处理
        
        # 记录有效的检测结果（如果通过了过滤或未启用过滤）
        if FILTER_BY_UNET:
            valid_detections.append({
                'bbox': bbox,
                'conf': conf,
                'instance_mask': instance_mask,
                'cancer_ratio': cancer_ratio
            })
        
        # 调试：保存提取的实际mask区域（去除padding后的）和映射后的mask
        pad_x, pad_y = record["pad_offset"]
        nw, nh = record["resized_size"]
        mask_actual_debug = unet_mask_480[pad_y:pad_y+nh, pad_x:pad_x+nw]
        cv2.imwrite(os.path.join(debug_dir, f"cell_{idx:03d}_mask_actual.jpg"),
                    (mask_actual_debug * 255).astype(np.uint8))
        
        # 保存映射回crop尺寸的mask（用于对比验证）
        crop_w, crop_h = record["crop_size"]
        mask_resized_debug = cv2.resize(mask_actual_debug, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(debug_dir, f"cell_{idx:03d}_mask_resized_to_crop.jpg"),
                    (mask_resized_debug * 255).astype(np.uint8))
        
        # 在原图上绘制crop区域和mask，用于验证位置
        debug_vis = orig_np.copy()
        x1, y1 = record["orig_xy"]
        x2, y2 = x1 + crop_w, y1 + crop_h
        cv2.rectangle(debug_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框：crop区域
        mask_color_debug = np.zeros_like(debug_vis)
        mask_color_debug[instance_mask == 1] = [255, 0, 0]  # 红色：mask区域
        cv2.addWeighted(mask_color_debug, 0.5, debug_vis, 1.0, 0, debug_vis)
        cv2.imwrite(os.path.join(debug_dir, f"cell_{idx:03d}_mapped_to_original.jpg"),
                    cv2.cvtColor(debug_vis, cv2.COLOR_RGB2BGR))
        
        # 累积mask（用于最终合并，避免颜色叠加问题）
        combined_mask = np.maximum(combined_mask, instance_mask)
        
        # 存储mask和相关信息用于最终可视化（避免循环中多次叠加颜色）
        all_masks.append(instance_mask)
        all_bboxes.append(bbox)
        all_confs_final.append(conf)
        label_text = f"Cancer {conf:.2f}"
        if FILTER_BY_UNET and cancer_ratio is not None:
            label_text += f" R:{cancer_ratio:.2f}"
        all_labels.append(label_text)
    
    # 5 统一可视化（避免颜色叠加问题）
    # 先绘制所有mask（一次性绘制，避免叠加）
    if len(all_masks) > 0:
        # 创建统一的mask颜色层
        mask_color_layer = np.zeros_like(vis)
        for instance_mask in all_masks:
            # 只在mask区域绘制颜色，使用逻辑或避免叠加
            mask_color_layer[instance_mask == 1] = [220, 101, 107]  # 深红色（BGR）
        
        # 一次性叠加mask颜色层（透明度0.4）
        cv2.addWeighted(mask_color_layer, 0.4, vis, 1.0, 0, vis)
        
        # 绘制所有mask的边界轮廓（不透明）
        for instance_mask in all_masks:
            contours, _ = cv2.findContours(instance_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (220, 101, 107), thickness=1)
        
        # 绘制所有检测框和标签
        for bbox, label_text in zip(all_bboxes, all_labels):
            x1, y1, x2, y2 = bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (220, 63, 70), 2)
            cv2.putText(vis, label_text, (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (220, 63, 70), 1)

    # 保存最终结果
    filename = Path(img_path).name
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, filename), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    # 保存纯mask
    # cv2.imwrite(os.path.join(OUTPUT_FOLDER, Path(filename).stem + "_mask.png"),
    #             (combined_mask * 255).astype(np.uint8))

    original_count = len(results.boxes)
    merged_count = len(merged_boxes) if MERGE_OVERLAPPING_BOXES else original_count
    filtered_count = len(all_bboxes)  # 最终保留的检测框数量
    merge_info = f", 合并后 {merged_count} 个" if MERGE_OVERLAPPING_BOXES and merged_count != original_count else ""
    filter_info = f", 过滤后 {filtered_count} 个" if FILTER_BY_UNET and filtered_count != merged_count else ""
    print(f"完成：{filename}  →  YOLO检测到 {original_count} 个细胞{merge_info}{filter_info}")


def main():
    if os.path.isfile(INPUT_PATH):
        process_one_image(INPUT_PATH)
    elif os.path.isdir(INPUT_PATH):
        for p in Path(INPUT_PATH).rglob("*.png") or Path(INPUT_PATH).rglob("*.jpg"):
            process_one_image(str(p))
    else:
        print("路径错误！")


if __name__ == "__main__":
    main()
