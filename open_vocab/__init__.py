"""
Open Vocabulary Detection and Segmentation
开放词汇检测与分割工具包

整合 GroundingDINO + MobileSAM + CLIP
支持 2D 检测和 3D 物体级 SLAM
"""

# 2D 检测模块
from .core import OpenVocabularyPipeline, create_pipeline
from .models.grounding_dino import GroundingDINOPredictor, load_grounding_dino
from .models.mobile_sam import MobileSAMPredictor, load_mobile_sam
from .models.clip_model import CLIPPredictor, load_clip

# 3D SLAM 模块
from .slam_pipeline import ObjectSLAMPipeline, create_object_slam_pipeline
from .models.slam_3d import VoxelGrid, ObjectMap, SceneGraph, Object3D
from .models.multiview_fusion import MultiViewFusion, FrameObservation, create_frame_observation

__version__ = "0.2.0"
__all__ = [
    # 2D 检测
    "OpenVocabularyPipeline",
    "create_pipeline",
    "GroundingDINOPredictor",
    "MobileSAMPredictor",
    "CLIPPredictor",
    "load_grounding_dino",
    "load_mobile_sam",
    "load_clip",
    
    # 3D SLAM
    "ObjectSLAMPipeline",
    "create_object_slam_pipeline",
    "VoxelGrid",
    "ObjectMap",
    "SceneGraph",
    "Object3D",
    "MultiViewFusion",
    "FrameObservation",
    "create_frame_observation",
]
