# 快速开始指南

## 5 分钟快速体验

### 1. 安装依赖（2 分钟）

```bash
cd /home/weiyu/Open-Vocabulary/src

# 安装核心依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install open3d scipy

# 注意：transformers 版本必须是 4.27.1
pip install transformers==4.27.1
```

### 2. 下载模型（1 分钟）

```bash
mkdir -p weights

# GroundingDINO
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P weights/
wget https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py -P weights/

# MobileSAM
wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt -P weights/
```

### 3. 运行第一个示例（2 分钟）

#### 选项 A: 2D 检测

```bash
python demo.py --image your_image.jpg --classes person chair table
```

#### 选项 B: 自由文本描述

```bash
python demo.py --image your_image.jpg --caption "red car on the street"
```

## 15 分钟完整体验

### 步骤 1: 准备测试数据

下载测试数据集（以 TUM RGB-D 为例）：

```bash
# 创建一个简单的测试序列
mkdir -p test_data/{rgb,depth}

# 使用 OpenCV 生成测试图像
python << 'EOF'
import cv2
import numpy as np

# 生成 10 帧测试数据
for i in range(10):
    # RGB 图像
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.imwrite(f"test_data/rgb/{i:05d}.png", rgb)
    
    # 深度图（模拟）
    depth = np.ones((480, 640), dtype=np.uint16) * 1000  # 1 米
    cv2.imwrite(f"test_data/depth/{i:05d}.png", depth)

print("测试数据已生成")
EOF
```

### 步骤 2: 运行 3D SLAM 示例

```bash
python example_3d_slam.py
```

在文件中取消注释 `example_rgbd_sequence_processing()` 函数。

### 步骤 3: 查看结果

```bash
# 查看生成的场景图
cat scene_graph.json

# 可视化（需要 Open3D）
python -c "
from open_vocab import create_pipeline, create_object_slam_pipeline
# ... 加载和可视化代码
"
```

## 30 分钟深入理解

### 阅读顺序

1. **README.md** (5 分钟) - 了解项目功能
2. **IMPLEMENTATION_SUMMARY.md** (10 分钟) - 理解系统架构
3. **example_3d_slam.py** (15 分钟) - 学习使用方法

### 代码探索路径

```
1. open_vocab/__init__.py          # 入口，了解可用模块
2. open_vocab/core.py              # 2D 检测核心
3. open_vocab/slam_pipeline.py     # 3D SLAM 核心
4. open_vocab/models/slam_3d.py    # 3D 数据结构
5. open_vocab/models/multiview_fusion.py  # 多视图融合
```

## 常见问题快速解决

### ❌ 导入错误

```python
ModuleNotFoundError: No module named 'torch'
```

**解决**: 
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### ❌ 版本冲突

```
ValueError: not enough values to unpack
```

**解决**: 确保 transformers 版本正确
```bash
pip install transformers==4.27.1
```

### ❌ CUDA 内存不足

```
RuntimeError: CUDA out of memory
```

**解决**: 
- 减小输入图像尺寸
- 使用 CPU: `device="cpu"`
- 关闭 SAM: `use_sam=False`

### ❌ 模型加载失败

```
FileNotFoundError: weights/groundingdino_swint_ogc.pth
```

**解决**:
```bash
ls weights/  # 检查文件是否存在
# 重新下载缺失的文件
```

## 下一步

完成快速开始后，你可以：

1. ✅ 使用 2D 检测功能
2. ✅ 尝试 3D SLAM 基础功能
3. ⏳ 用自己的数据测试
4. ⏳ 修改参数优化效果
5. ⏳ 集成到现有项目

## 获取帮助

- 📖 查看 README.md 了解详细 API
- 💡 查看 example_*.py 学习用法
- 🔍 查看代码注释理解实现
- 📝 查看 IMPLEMENTATION_SUMMARY.md 了解架构

祝你使用愉快！🚀
