# 训练与预测使用说明 - 增强版UNet（含核质分离感知模块）

## 📋 概述

本文档介绍如何使用增强版UNet进行训练和预测。增强版UNet包含三个核心创新模块：

1. **注意力门控（Attention Gate）**：在跳跃连接处增强癌细胞区域，抑制背景噪声
2. **转置卷积上采样（Transposed Convolution）**：可学习的上采样方式，替代固定双线性插值
3. **核质分离感知模块（Nucleus-Cytoplasm Separation Module）**：利用癌细胞核占比大的形态学先验，通过不同大小的卷积核分别提取细胞核和细胞质的特征

---

## 🚀 第一部分：训练

### 1.1 基本训练命令

#### 使用增强版UNet（全部模块启用，推荐）

```bash
python train.py \
    --data-path "E:\Data_Industry\split_images\edit_1004560_MergedBatch" \
    --num-classes 1 \
    --batch-size 2 \
    --epochs 30 \
    --use-enhanced True \
    --use-attention True \
    --use-transpose True \
    --use-nucleus-cytoplasm True \
    --device cuda
```

#### 使用原版UNet（对比实验）

```bash
python train.py \
    --data-path "E:\Data_Industry\split_images\edit_1004560_MergedBatch" \
    --num-classes 1 \
    --batch-size 2 \
    --epochs 30 \
    --use-enhanced False \
    --device cuda
```

### 1.2 模块配置说明

增强版UNet支持灵活配置各个模块，可以根据需求启用或禁用：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use-enhanced` | bool | True | 是否使用增强版UNet |
| `--use-attention` | bool | True | 是否使用注意力门控（仅增强版有效） |
| `--use-transpose` | bool | True | 是否使用转置卷积上采样（仅增强版有效） |
| `--use-nucleus-cytoplasm` | bool | True | 是否使用核质分离感知模块（仅增强版有效） |

### 1.3 不同配置的训练示例

#### 配置1：只使用注意力门控

```bash
python train.py \
    --data-path "你的数据路径" \
    --num-classes 1 \
    --batch-size 2 \
    --epochs 30 \
    --use-enhanced True \
    --use-attention True \
    --use-transpose False \
    --use-nucleus-cytoplasm False \
    --device cuda
```

#### 配置2：使用注意力门控 + 转置卷积（不含核质分离）

```bash
python train.py \
    --data-path "你的数据路径" \
    --num-classes 1 \
    --batch-size 2 \
    --epochs 30 \
    --use-enhanced True \
    --use-attention True \
    --use-transpose True \
    --use-nucleus-cytoplasm False \
    --device cuda
```

#### 配置3：使用注意力门控 + 核质分离（不含转置卷积）

```bash
python train.py \
    --data-path "你的数据路径" \
    --num-classes 1 \
    --batch-size 2 \
    --epochs 30 \
    --use-enhanced True \
    --use-attention True \
    --use-transpose False \
    --use-nucleus-cytoplasm True \
    --device cuda
```

#### 配置4：全部模块启用（推荐，最佳性能）

```bash
python train.py \
    --data-path "你的数据路径" \
    --num-classes 1 \
    --batch-size 2 \
    --epochs 30 \
    --use-enhanced True \
    --use-attention True \
    --use-transpose True \
    --use-nucleus-cytoplasm True \
    --device cuda
```

### 1.4 从预训练权重继续训练

#### 从原版权重迁移到增强版

```bash
python train.py \
    --data-path "你的数据路径" \
    --num-classes 1 \
    --batch-size 2 \
    --epochs 30 \
    --weights "weights/original_model.pth" \
    --use-enhanced True \
    --use-attention True \
    --use-transpose True \
    --use-nucleus-cytoplasm True \
    --device cuda
```

**注意**：从原版权重加载时，只有编码器（ResNet50）部分的权重会被加载，解码器部分和新增模块会重新初始化。

#### 从增强版权重继续训练

```bash
python train.py \
    --data-path "你的数据路径" \
    --num-classes 1 \
    --batch-size 2 \
    --epochs 30 \
    --weights "weights/enhanced_model.pth" \
    --use-enhanced True \
    --use-attention True \
    --use-transpose True \
    --use-nucleus-cytoplasm True \
    --device cuda
```

### 1.5 训练输出示例

训练时会显示模型配置信息：

```
✅ 使用增强版UNet (注意力门控=True, 转置卷积=True, 核质分离=True)
📦 权重加载完成: 成功加载 234 个参数
⚠️  无法加载 45 个参数（可能是架构变化导致的）
📊 模型参数统计: 总参数=45.23M, 可训练参数=45.23M
```

### 1.6 训练建议

#### 首次训练增强版（含核质分离模块）

- **推荐配置**：全部模块启用（`--use-attention True --use-transpose True --use-nucleus-cytoplasm True`）
- **学习率**：保持默认（1e-4）或稍微降低10-20%
- **训练轮次**：建议30-50个epoch
- **批大小**：根据显存调整，建议2-4

#### 显存优化

如果显存不足，可以关闭转置卷积（转置卷积会增加显存占用）：

```bash
python train.py \
    --use-enhanced True \
    --use-attention True \
    --use-transpose False \
    --use-nucleus-cytoplasm True \
    --batch-size 2
```

#### 速度优化

如果训练速度慢，可以只使用注意力门控和核质分离模块：

```bash
python train.py \
    --use-enhanced True \
    --use-attention True \
    --use-transpose False \
    --use-nucleus-cytoplasm True
```

---

## 🔮 第二部分：预测

### 2.1 预测脚本配置

预测脚本 `yolo_unet_two_stage_predict.py` 支持灵活配置UNet模型版本和各个模块。

#### 关键配置参数

在脚本开头修改以下参数：

```python
# UNet模型版本选择
USE_ENHANCED_UNET = True  # True: 使用增强版UNet
                          # False: 使用原始版UNet

# 增强版UNet模块开关（仅当USE_ENHANCED_UNET=True时有效）
USE_ATTENTION = True          # 是否使用注意力门控
USE_TRANSPOSE = True          # 是否使用转置卷积上采样
USE_NUCLEUS_CYTOPLASM = True  # 是否使用核质分离感知模块
```

### 2.2 不同配置的预测示例

#### 配置1：使用增强版UNet（全部模块启用）

```python
USE_ENHANCED_UNET = True
USE_ATTENTION = True
USE_TRANSPOSE = True
USE_NUCLEUS_CYTOPLASM = True
```

#### 配置2：使用增强版UNet（不含核质分离模块）

```python
USE_ENHANCED_UNET = True
USE_ATTENTION = True
USE_TRANSPOSE = True
USE_NUCLEUS_CYTOPLASM = False
```

#### 配置3：使用增强版UNet（只使用注意力门控和核质分离）

```python
USE_ENHANCED_UNET = True
USE_ATTENTION = True
USE_TRANSPOSE = False
USE_NUCLEUS_CYTOPLASM = True
```

#### 配置4：使用原始版UNet

```python
USE_ENHANCED_UNET = False
# 以下参数在原始版中无效
USE_ATTENTION = True
USE_TRANSPOSE = True
USE_NUCLEUS_CYTOPLASM = True
```

### 2.3 运行预测

#### 预测单张图片

```python
INPUT_PATH = r"path/to/your/image.jpg"
```

#### 预测整个文件夹

```python
INPUT_PATH = r"path/to/your/folder"
```

脚本会自动处理文件夹中的所有 `.png` 和 `.jpg` 图片。

### 2.4 预测输出示例

运行预测时会显示模型加载信息：

```
加载增强版UNet模型...
增强版UNet模型加载完成！启用模块: 注意力门控, 转置卷积, 核质分离
```

### 2.5 重要注意事项

#### ⚠️ 模型配置必须与训练时一致

**关键**：预测时使用的模块配置必须与训练时完全一致！

例如，如果训练时使用了：
```bash
--use-attention True --use-transpose True --use-nucleus-cytoplasm True
```

那么预测时也必须设置：
```python
USE_ATTENTION = True
USE_TRANSPOSE = True
USE_NUCLEUS_CYTOPLASM = True
```

**不匹配的配置会导致模型加载失败或预测结果异常！**

#### 权重文件路径

确保 `UNET_MODEL_PATH` 指向正确的权重文件：

```python
UNET_MODEL_PATH = r"E:\Data_Industry\膀胱癌细胞检测_文章\Unet_attention_文件\模型评估文件\last_model_1.pth"
```

---

## 📊 第三部分：核质分离感知模块详解

### 3.1 模块原理

核质分离感知模块（Nucleus-Cytoplasm Separation Module）是专门针对癌细胞"核大"形态学特征设计的创新模块。

#### 设计思想

1. **核分支**：使用5×5大卷积核，提取高对比度、致密区域（对应细胞核）
2. **质分支**：使用3×3小卷积核，提取低对比度、稀疏区域（对应细胞质）
3. **核质比估计**：根据特征判断核质比，生成权重图
4. **加权融合**：核区域权重高，质区域权重低

#### 为什么不需要额外标注？

- **架构引导学习**：大卷积核自然倾向于提取致密特征，小卷积核自然倾向于提取稀疏特征
- **端到端优化**：所有参数通过反向传播自动优化，核质分离是"副产品"，但有助于分割精度
- **形态学先验**：利用癌细胞"核大"的形态学特征，不需要额外的核质标注

### 3.2 模块优势

1. **针对性强**：专门针对癌细胞"核大"的形态学特征
2. **提升精度**：通过核质分离，提升边界分割精度
3. **无需额外数据**：不需要核质分离的标注，只需现有的分割mask
4. **可解释性**：核分支和质分支的输出可以可视化

### 3.3 使用建议

#### 何时启用核质分离模块？

- ✅ **推荐启用**：处理癌细胞分割任务时，核质分离模块能显著提升分割精度
- ✅ **推荐启用**：当癌细胞核占比大、边界模糊时，核质分离模块特别有效
- ⚠️ **可选**：如果显存充足，建议启用所有模块以获得最佳性能

#### 何时禁用核质分离模块？

- ⚠️ **可选禁用**：如果显存不足，可以关闭核质分离模块（但建议保留注意力门控）
- ⚠️ **可选禁用**：如果训练速度要求高，可以关闭核质分离模块

---

## 🎯 第四部分：完整工作流程示例

### 4.1 完整训练流程

#### 步骤1：准备数据

确保数据路径正确，包含 `train.txt` 和 `val.txt`：

```
your_data_path/
├── Images/
├── SegmentationClass/
├── train.txt
└── val.txt
```

#### 步骤2：开始训练

```bash
python train.py \
    --data-path "E:\Data_Industry\split_images\edit_1004560_MergedBatch" \
    --num-classes 1 \
    --batch-size 2 \
    --epochs 50 \
    --use-enhanced True \
    --use-attention True \
    --use-transpose True \
    --use-nucleus-cytoplasm True \
    --device cuda
```

#### 步骤3：查看训练结果

训练完成后，模型权重保存在：
- `exp_XXX/weights/best_model_1.pth`（最佳模型）
- `exp_XXX/weights/last_model_1.pth`（最后一轮模型）

训练曲线图保存在：
- `exp_XXX/weights/training_curves.png`

### 4.2 完整预测流程

#### 步骤1：配置预测脚本

编辑 `yolo_unet_two_stage_predict.py`：

```python
# 模型路径
YOLO_MODEL_PATH = r"E:\Data_Industry\Unet_wdir\2stage_wdir\best.pt"
UNET_MODEL_PATH = r"exp_XXX/weights/best_model_1.pth"

# 模型配置（必须与训练时一致！）
USE_ENHANCED_UNET = True
USE_ATTENTION = True
USE_TRANSPOSE = True
USE_NUCLEUS_CYTOPLASM = True

# 输入输出路径
INPUT_PATH = r"path/to/your/images"
OUTPUT_FOLDER = r"path/to/output"
```

#### 步骤2：运行预测

```bash
python yolo_unet_two_stage_predict.py
```

#### 步骤3：查看结果

预测结果保存在 `OUTPUT_FOLDER`，包含：
- 可视化结果图片（带检测框和分割mask）
- 调试文件夹 `debug_unet_patches/`（包含每个检测框的详细处理过程）

---

## ⚠️ 常见问题

### Q1: 训练时显示"无法加载某些参数"？

**A**: 这是正常的。从原版权重加载到增强版时，只有编码器权重会被加载，解码器和新增模块会重新初始化。

### Q2: 预测时模型加载失败？

**A**: 检查以下几点：
1. 模型配置是否与训练时一致（`USE_ATTENTION`, `USE_TRANSPOSE`, `USE_NUCLEUS_CYTOPLASM`）
2. 权重文件路径是否正确
3. 权重文件是否完整（没有损坏）

### Q3: 显存不足怎么办？

**A**: 可以关闭转置卷积模块：
```bash
--use-transpose False
```
或减小批大小：
```bash
--batch-size 1
```

### Q4: 核质分离模块会增加多少计算量？

**A**: 核质分离模块会增加约5-10%的计算量和显存占用，但能显著提升分割精度。

### Q5: 如何选择最佳配置？

**A**: 推荐配置（最佳性能）：
- `--use-attention True`
- `--use-transpose True`
- `--use-nucleus-cytoplasm True`

如果显存不足，可以关闭转置卷积；如果速度要求高，可以只使用注意力门控和核质分离。


**关键提醒**：预测时的模块配置必须与训练时完全一致！

