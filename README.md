# Open Vocabulary Detection & 3D Object SLAM

基于 GroundingDINO + MobileSAM + CLIP 的开放词汇检测与 3D 物体级 SLAM 系统。

## 功能特性

### 2D 检测
- **开放词汇检测**：无需训练，检测任意类别
- **精细分割**：MobileSAM 生成高质量分割掩码
- **自由文本描述**：支持自然语言描述进行目标检测
- **即开即用**：简单的 API 接口，快速集成
- **GPU 加速**：支持 CUDA 加速，RTX 5060 可流畅运行

### 3D SLAM（新增）✨
- **RGB-D 处理**：从深度图像生成 3D 点云
- **多视图融合**：融合多帧观测，构建一致的对象表示
- **对象地图**：维护 3D 对象实例及其属性
- **场景图构建**：自动推断对象间的空间和语义关系
- **类 ConceptGraphs**：实现类似 ConceptGraphs 的 3D 场景理解流程

## 模型组合

| 模型 | 用途 | 特点 |
|------|------|------|
| GroundingDINO | 开放词汇检测 | 支持任意文本描述的目标检测 |
| MobileSAM | 图像分割 | 轻量级 SAM，速度快 |
| CLIP | 语义对齐 | 图像 - 文本特征对齐验证 |
| RGB-D Fusion | 3D 重建 | 多视图几何融合 |

## 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.7 (推荐)
- GPU 显存 >= 6GB
- Open3D >= 0.15 (3D 可视化)

## 安装

### 1. 克隆项目

```bash
cd /home/weiyu/Open-Vocabulary/src
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
pip install open3d  # 3D 可视化
pip install scipy   # 数据关联算法
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

### 2D 检测（基础用法）

```bash
# 命令行使用
python demo.py --image your_image.jpg --classes person car dog

# 自由文本描述
python demo.py --image your_image.jpg --caption "red car on the street"
```

### 3D SLAM（新功能）

```python
from open_vocab import create_pipeline, create_object_slam_pipeline

# Step 1: 创建 2D 检测流水线
detection_pipeline = create_pipeline(
    grounding_dino_config="./weights/GroundingDINO_SwinT_OGC.py",
    grounding_dino_checkpoint="./weights/groundingdino_swint_ogc.pth",
    mobile_sam_checkpoint="./weights/mobile_sam.pt",
    device="cuda",
)

# Step 2: 创建 3D SLAM 流水线
slam_pipeline = create_object_slam_pipeline(
    detection_pipeline=detection_pipeline,
    voxel_size=0.05,  # 5cm 体素
    use_slam=False,   # 暂时不使用 SLAM
)

# Step 3: 处理 RGB-D 序列
import cv2
import numpy as np

for frame_id in range(num_frames):
    # 读取 RGB-D 图像
    rgb = cv2.imread(f"rgb/{frame_id:05d}.png")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    depth = cv2.imread(f"depth/{frame_id:05d}.png", cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32) / 1000.0
    
    # 相机参数
    intrinsics = np.array([
        [615.7, 0, 324.9],
        [0, 615.7, 241.8],
        [0, 0, 1]
    ])
    
    # 处理帧
    results = slam_pipeline.process_rgbd_frame(
        rgb_image=rgb,
        depth_image=depth,
        camera_intrinsics=intrinsics,
        classes=["chair", "table", "sofa"],
    )
    
    print(f"帧 {frame_id}: 检测到 {results['objects_detected']} 个对象")

# Step 4: 查看结果
print(f"总对象数：{len(slam_pipeline.object_map.objects)}")

# Step 5: 保存和可视化
slam_pipeline.save_reconstruction("./scene_graph.json")
slam_pipeline.visualize()
```

## 项目结构

```
open_vocab/
├── __init__.py              # 包入口
├── core.py                  # 主流水线（2D 检测）
├── slam_pipeline.py         # 3D SLAM 主流水线 ⭐
├── models/
│   ├── grounding_dino.py   # GroundingDINO 封装
│   ├── mobile_sam.py       # MobileSAM 封装
│   ├── clip_model.py       # CLIP 封装
│   ├── slam_3d.py          # 3D 数据结构 ⭐
│   └── multiview_fusion.py # 多视图融合 ⭐
├── utils/                   # 工具函数
└── configs/                 # 配置文件

demo.py                      # 2D 检测演示
example_3d_slam.py          # 3D SLAM 示例 ⭐
example_usage.py            # 使用示例
requirements.txt            # 依赖列表
```

## 核心 API

### 2D 检测 API

#### OpenVocabularyPipeline

```python
pipeline = create_pipeline(...)

# 检测指定类别
results = pipeline.detect_and_segment(
    image="image.jpg",
    classes=["person", "car"],
)

# 自由文本描述
results = pipeline.detect_with_caption(
    image="image.jpg",
    caption="red car",
)
```

### 3D SLAM API（新增）⭐

#### ObjectSLAMPipeline

```python
slam = create_object_slam_pipeline(detection_pipeline)

# 处理 RGB-D 帧
results = slam.process_rgbd_frame(
    rgb_image=rgb,
    depth_image=depth,
    camera_intrinsics=intrinsics,
    camera_pose=pose,  # 可选
    classes=["chair", "table"],
)

# 查询场景图
objects = slam.query_scene_graph("chairs near table")

# 可视化
slam.visualize()
```

#### 数据结构

```python
from open_vocab import VoxelGrid, ObjectMap, SceneGraph

# 体素网格
voxel_grid = VoxelGrid(voxel_size=0.05)

# 对象地图
object_map = ObjectMap(voxel_grid)
obj = object_map.add_object(
    label="chair",
    centroid=np.array([1.0, 2.0, 0.5]),
    bbox_3d=np.array([...]),
    semantic_features=features,
)

# 场景图
scene_graph = SceneGraph(object_map)
scene_graph.infer_spatial_relations()
```

## 使用示例

### 示例 1: 基础 2D 检测

```python
from open_vocab import create_pipeline

pipeline = create_pipeline(...)
results = pipeline.detect_and_segment(
    image="street.jpg",
    classes=["person", "car", "bicycle"],
)
```

### 示例 2: RGB-D 序列处理（3D）

```python
from open_vocab import create_object_slam_pipeline

slam = create_object_slam_pipeline(...)

for frame_id in range(100):
    results = slam.process_rgbd_frame(...)
    
slam.save_reconstruction("scene_graph.json")
```

### 示例 3: 场景图查询

```python
# 查询所有椅子
chairs = slam.query_scene_graph("chairs")

# 查询桌子附近的对象
near_table = slam.query_scene_graph("objects near table")
```

更多示例请查看 `example_3d_slam.py`

## 3D SLAM 流程说明

### 输入
- RGB-D 图像序列（彩色 + 深度）
- 相机内参
- 相机位姿（可选，可从 SLAM 系统获取）

### 处理流程
1. **2D 检测**：GroundingDINO + MobileSAM 检测对象
2. **3D 投影**：将 2D 掩码投影到 3D 空间
3. **数据关联**：匹配多帧中的同一对象
4. **特征融合**：融合多视角特征
5. **场景图构建**：推断对象间关系

### 输出
- 3D 对象地图（包含每个对象的 3D 位置、大小、语义特征）
- 场景图（对象间的空间和语义关系）
- 体素网格地图

## 数据集格式

### RGB-D 数据组织

```
rgbd_data/
├── rgb/           # RGB 图像
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
├── depth/         # 深度图（单位：毫米）
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
└── poses.txt      # 相机位姿（可选）
```

### 相机内参示例

```python
# RealSense D435
intrinsics = np.array([
    [615.7, 0, 324.9],
    [0, 615.7, 241.8],
    [0, 0, 1]
])
```

## 性能参考

在 RTX 5060 上的推理速度：

| 任务 | 时间 |
|------|------|
| 2D 检测（单帧） | ~150ms |
| 3D 投影（单帧） | ~50ms |
| 场景图更新 | ~20ms |
| **总计（每帧）** | **~220ms** |

## 常见问题

### Q: 如何适配自己的 RGB-D 相机？

需要提供相机内参矩阵和深度图格式。大多数 RGB-D 相机（RealSense、Kinect、Orbbec）都提供内参标定工具。

### Q: 没有 groundtruth 位姿怎么办？

可以：
1. 集成 ORB-SLAM3 或 DROID-SLAM 实时估计位姿
2. 使用简化的运动模型（如匀速假设）
3. 如果是静态场景，可以手动设置关键帧位姿

### Q: 如何处理大规模场景？

建议：
1. 使用更大的体素尺寸（如 10cm）
2. 实施关键帧选择策略
3. 定期优化场景图（移除远距离对象）

### Q: 场景图能保存为什么格式？

目前支持 JSON 格式，可以轻松转换为其他格式（如 PLY、PCD）。

## 依赖库

- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- transformers == 4.27.1
- groundingdino-py >= 0.4.0
- open_clip_torch >= 2.20.0
- opencv-python >= 4.8.0
- open3d >= 0.15
- scipy >= 1.9.0

## 相关项目

- [ConceptGraphs](https://concept-graphs.github.io/) - 开放词汇 3D 场景图
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)

## 许可证

本项目仅供学习和研究使用。
