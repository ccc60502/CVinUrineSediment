# -*- coding: utf-8 -*-
# filename: yolo_only_predict.py
# 功能：仅使用YOLO11模型进行目标检测，不进行UNet分割

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO

# ================== 请修改这几行路径 ==================
# 1. YOLOv11 检测模型
YOLO_MODEL_PATH = r"E:\Data_Industry\Unet_wdir\2stage_wdir\best.pt"

# 2. 输入图片或文件夹
INPUT_PATH = r"E:\Data_Industry\Unet_wdir\test_wdir\复杂样本输入"   # 可以是单张图或文件夹

# 3. 输出文件夹
OUTPUT_FOLDER = r"E:\Data_Industry\Unet_wdir\test_wdir\yolo复杂样本输出"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# YOLO 参数
YOLO_CONF = 0.39          # 置信度阈值
YOLO_CLASSES = [0]       # 要检测的类别（比如0是癌细胞）
# ====================================================

# 检测框和文字颜色 (RGB格式，与yolo_unet_two_stage_predict.py保持一致)
BOX_COLOR = (220, 63, 70)      # 框的颜色 (RGB格式)
TEXT_COLOR = (220, 63, 70)     # 文字的颜色 (RGB格式)
BOX_THICKNESS = 2              # 框的线宽
TEXT_FONT = cv2.FONT_HERSHEY_TRIPLEX
TEXT_SCALE = 0.5               # 文字大小
TEXT_THICKNESS = 1             # 文字线宽

# 加载 YOLO 模型
print(f"加载YOLO模型: {YOLO_MODEL_PATH}")
yolo_model = YOLO(YOLO_MODEL_PATH)
print("YOLO模型加载完成！")


def process_one_image(img_path):
    """
    处理单张图像，进行YOLO检测并可视化结果
    """
    # 读取图像（使用PIL读取RGB格式，与yolo_unet_two_stage_predict.py保持一致）
    image_pil = Image.open(img_path).convert("RGB")
    orig_np = np.array(image_pil)
    h, w = orig_np.shape[:2]
    
    print(f"\n处理图像: {Path(img_path).name} (尺寸: {w}x{h})")
    
    # YOLO检测
    results = yolo_model.predict(
        source=img_path, 
        conf=YOLO_CONF,
        classes=YOLO_CLASSES, 
        save=False, 
        verbose=False
    )[0]
    
    # 创建可视化图像（RGB格式）
    vis = orig_np.copy()
    
    # 统计检测结果
    detection_count = len(results.boxes)
    
    # 绘制检测框和标签
    for idx, box in enumerate(results.boxes):
        # 获取边界框坐标
        bbox = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = bbox
        
        # 获取置信度
        conf = box.conf[0].item()
        
        # 绘制检测框（RGB格式颜色）
        cv2.rectangle(vis, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        
        # 准备标签文本
        label_text = f"Cancer {conf:.2f}"
        
        # 绘制文字（RGB格式颜色，与yolo_unet_two_stage_predict.py保持一致）
        cv2.putText(
            vis, 
            label_text, 
            (x1, max(y1 - 10, 10)),
            TEXT_FONT, 
            TEXT_SCALE, 
            TEXT_COLOR, 
            TEXT_THICKNESS
        )
    
    # 保存结果（转换为BGR格式保存，与yolo_unet_two_stage_predict.py保持一致）
    filename = Path(img_path).name
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    
    print(f"完成：{filename}  →  检测到 {detection_count} 个目标")
    print(f"结果已保存至: {output_path}")


def main():
    """
    主函数：处理单张图像或整个文件夹
    """
    if os.path.isfile(INPUT_PATH):
        # 单张图像
        process_one_image(INPUT_PATH)
    elif os.path.isdir(INPUT_PATH):
        # 文件夹：处理所有图像
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(INPUT_PATH).rglob(f"*{ext}"))
            image_files.extend(Path(INPUT_PATH).rglob(f"*{ext.upper()}"))
        
        if len(image_files) == 0:
            print(f"错误：在文件夹 {INPUT_PATH} 中未找到图像文件")
            return
        
        print(f"找到 {len(image_files)} 张图像，开始处理...")
        for img_path in image_files:
            process_one_image(str(img_path))
        
        print(f"\n所有图像处理完成！结果保存在: {OUTPUT_FOLDER}")
    else:
        print(f"错误：路径不存在或无效: {INPUT_PATH}")


if __name__ == "__main__":
    main()

