# Open Vocabulary Detection & Segmentation

基于 GroundingDINO + MobileSAM + CLIP 的开放词汇检测与分割工具包。

## 功能特性

- **开放词汇检测**：无需训练，检测任意类别
- **精细分割**：MobileSAM 生成高质量分割掩码
- **自由文本描述**：支持自然语言描述进行目标检测
- **即开即用**：简单的 API 接口，快速集成
- **GPU 加速**：支持 CUDA 加速，RTX 5060 可流畅运行

## 模型组合

| 模型 | 用途 | 特点 |
|------|------|------|
| GroundingDINO | 开放词汇检测 | 支持任意文本描述的目标检测 |
| MobileSAM | 图像分割 | 轻量级 SAM，速度快 |
| CLIP | 语义对齐 | 图像-文本特征对齐验证 |

## 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.7 (推荐)
- GPU 显存 >= 6GB

## 安装

### 1. 克隆项目

```bash
cd /home/weiyu/Open-Vocabulary/src
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载模型权重

```bash
# 创建权重目录
mkdir -p weights

# 下载 GroundingDINO 权重
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P weights/

# 下载配置文件
wget https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py -P weights/

# 下载 MobileSAM 权重
wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt -P weights/
```

## 快速开始

### 命令行使用

```bash
# 基础检测 - 检测指定类别
python demo.py --image your_image.jpg --classes person car dog

# 使用自由文本描述
python demo.py --image your_image.jpg --caption "red car on the street"

# 调整检测阈值
python demo.py --image your_image.jpg --classes person --box_threshold 0.3 --text_threshold 0.2

# 仅检测，不使用 SAM 分割
python demo.py --image your_image.jpg --classes person --no_sam

# 使用 CPU 运行
python demo.py --image your_image.jpg --classes person --device cpu
```

### Python API 使用

```python
from open_vocab import create_pipeline
import cv2

# 创建检测流水线
pipeline = create_pipeline(
    grounding_dino_config="./weights/GroundingDINO_SwinT_OGC.py",
    grounding_dino_checkpoint="./weights/groundingdino_swint_ogc.pth",
    mobile_sam_checkpoint="./weights/mobile_sam.pt",
    device="cuda",
    use_clip=True,
)

# 方式 1: 检测指定类别
results = pipeline.detect_and_segment(
    image="your_image.jpg",
    classes=["person", "car", "dog", "cat"],
    box_threshold=0.35,
    text_threshold=0.25,
    use_sam=True,
)

# 方式 2: 使用自由文本描述
results = pipeline.detect_with_caption(
    image="your_image.jpg",
    caption="red car on the street",
    box_threshold=0.3,
    text_threshold=0.2,
)

# 保存可视化结果
output = cv2.cvtColor(results['visualization'], cv2.COLOR_RGB2BGR)
cv2.imwrite("output.jpg", output)

# 获取检测结果
boxes = results['boxes']      # 检测框坐标 [N, 4]
scores = results['scores']    # 置信度分数 [N]
labels = results['labels']    # 类别标签 [N]
masks = results['masks']      # 分割掩码 [N, H, W]
```

## 项目结构

```
open_vocab/
├── __init__.py              # 包入口
├── core.py                  # 主流水线（OpenVocabularyPipeline）
├── models/
│   ├── grounding_dino.py   # GroundingDINO 模型封装
│   ├── mobile_sam.py       # MobileSAM 模型封装
│   └── clip_model.py       # CLIP 模型封装
├── utils/                   # 工具函数
└── configs/                 # 配置文件

demo.py                      # 命令行演示脚本
example_usage.py             # 使用示例代码
requirements.txt             # 依赖列表
```

## API 参考

### OpenVocabularyPipeline

主类，整合 GroundingDINO、MobileSAM 和 CLIP。

#### 初始化参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `grounding_dino_config` | str | GroundingDINO 配置文件路径 |
| `grounding_dino_checkpoint` | str | GroundingDINO 权重文件路径 |
| `mobile_sam_checkpoint` | str | MobileSAM 权重文件路径 |
| `device` | str | 运行设备，"cuda" 或 "cpu" |
| `use_clip` | bool | 是否使用 CLIP 进行语义验证 |

#### 方法

**detect_and_segment()**

使用指定类别列表进行检测和分割。

```python
results = pipeline.detect_and_segment(
    image="path/to/image.jpg",      # 图像路径或 numpy 数组
    classes=["person", "car"],      # 要检测的类别列表
    box_threshold=0.35,             # 检测框置信度阈值
    text_threshold=0.25,            # 文本匹配阈值
    use_sam=True,                   # 是否使用 SAM 分割
)
```

**detect_with_caption()**

使用自由文本描述进行检测和分割。

```python
results = pipeline.detect_with_caption(
    image="path/to/image.jpg",
    caption="red car on the street",  # 自然语言描述
    box_threshold=0.3,
    text_threshold=0.2,
)
```

#### 返回值

| 字段 | 类型 | 说明 |
|------|------|------|
| `boxes` | ndarray | 检测框坐标 [N, 4]，格式 (x1, y1, x2, y2) |
| `scores` | ndarray | 置信度分数 [N] |
| `labels` | list | 类别标签列表 [N] |
| `masks` | ndarray | 分割掩码 [N, H, W] (如果 use_sam=True) |
| `visualization` | ndarray | 可视化结果图像 |

## 使用示例

### 示例 1: 基础检测

```python
from open_vocab import create_pipeline

pipeline = create_pipeline(
    grounding_dino_config="./weights/GroundingDINO_SwinT_OGC.py",
    grounding_dino_checkpoint="./weights/groundingdino_swint_ogc.pth",
    mobile_sam_checkpoint="./weights/mobile_sam.pt",
    device="cuda",
)

results = pipeline.detect_and_segment(
    image="street.jpg",
    classes=["person", "car", "bicycle"],
)
```

### 示例 2: 自定义类别

```python
# 可以检测任意类别，无需训练
custom_classes = [
    "coffee cup",
    "laptop computer",
    "wooden chair",
    "potted plant",
]

results = pipeline.detect_and_segment(
    image="office.jpg",
    classes=custom_classes,
    box_threshold=0.3,
)
```

### 示例 3: 访问分割掩码

```python
results = pipeline.detect_and_segment(
    image="people.jpg",
    classes=["person"],
)

for i, (box, label, score, mask) in enumerate(
    zip(results['boxes'], results['labels'], results['scores'], results['masks'])
):
    print(f"目标 {i+1}: {label}, 置信度: {score:.3f}")
    print(f"  检测框: {box}")
    print(f"  掩码像素数: {mask.sum()}")
```

### 示例 4: 批量处理

```python
from pathlib import Path

image_dir = Path("./images")
output_dir = Path("./outputs")
output_dir.mkdir(exist_ok=True)

for image_path in image_dir.glob("*.jpg"):
    results = pipeline.detect_and_segment(
        image=str(image_path),
        classes=["person", "vehicle"],
    )
    
    output_path = output_dir / f"result_{image_path.name}"
    cv2.imwrite(str(output_path), results['visualization'])
```

## 参数调优

### 检测阈值

- **box_threshold**: 检测框置信度阈值（默认 0.35）
  - 值越高，检测结果越精确，但可能漏检
  - 值越低，召回率越高，但可能引入误检

- **text_threshold**: 文本匹配阈值（默认 0.25）
  - 控制文本描述与图像特征的匹配程度

### 推荐设置

| 场景 | box_threshold | text_threshold |
|------|---------------|----------------|
| 高精度要求 | 0.4 - 0.5 | 0.3 - 0.4 |
| 高召回要求 | 0.25 - 0.3 | 0.15 - 0.2 |
| 平衡模式 | 0.35 | 0.25 |

## 性能参考

在 RTX 5060 上的推理速度：

| 图像尺寸 | GroundingDINO | MobileSAM | 总时间 |
|----------|---------------|-----------|--------|
| 640x480 | ~100ms | ~50ms | ~150ms |
| 1280x720 | ~200ms | ~100ms | ~300ms |
| 1920x1080 | ~400ms | ~200ms | ~600ms |

## 常见问题

### Q: 模型加载失败？

确保模型权重文件路径正确，且文件已完整下载。

### Q: CUDA 内存不足？

1. 使用更小的输入图像尺寸
2. 在 CPU 上运行：`device="cpu"`
3. 关闭 SAM 分割：`use_sam=False`

### Q: 检测不到目标？

1. 降低 `box_threshold` 和 `text_threshold`
2. 检查类别名称是否准确
3. 尝试使用更具体的描述

### Q: 分割质量不好？

MobileSAM 对检测框质量敏感，确保 GroundingDINO 的检测框准确。

## 依赖库

- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- transformers >= 4.30.0
- groundingdino-py >= 0.4.0
- open_clip_torch >= 2.20.0
- opencv-python >= 4.8.0
- supervision >= 0.14.0

## 许可证

本项目仅供学习和研究使用。

## 相关链接

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [CLIP](https://github.com/openai/CLIP)
