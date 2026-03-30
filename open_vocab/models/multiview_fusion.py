"""
多视图融合模块

负责将 2D 检测结果投影到 3D 空间，并融合多帧观测
核心功能：
- RGB-D 深度处理
- 2D 掩码到 3D 点云投影
- 多视图特征融合
- 对象关联和数据关联
"""

import numpy as np
import torch
import cv2
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FrameObservation:
    """
    单帧观测数据
    
    Attributes:
        frame_id: 帧 ID
        timestamp: 时间戳
        rgb_image: RGB 图像
        depth_image: 深度图
        camera_pose: 相机位姿矩阵 [4x4]
        camera_intrinsics: 相机内参 [3x3]
        detections_2d: 2D 检测结果
        masks_2d: 2D 分割掩码
        clip_features: CLIP 特征
    """
    frame_id: int
    timestamp: float
    rgb_image: np.ndarray
    depth_image: np.ndarray
    camera_pose: np.ndarray  # 4x4 变换矩阵
    camera_intrinsics: np.ndarray  # 3x3 内参
    detections_2d: List[Dict]  # [{box, label, score}]
    masks_2d: List[np.ndarray]  # 每个对象的掩码
    clip_features: Optional[np.ndarray] = None  # 全局 CLIP 特征


class MultiViewFusion:
    """
    多视图融合器
    
    将多帧的 2D 检测结果融合到 3D 空间中
    """
    
    def __init__(self, min_depth: float = 0.1, max_depth: float = 10.0):
        """
        初始化多视图融合器
        
        Args:
            min_depth: 最小有效深度（米）
            max_depth: 最大有效深度（米）
        """
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.frame_buffer: Dict[int, FrameObservation] = {}
        
    def add_frame(self, observation: FrameObservation):
        """
        添加新的观测帧
        
        Args:
            observation: 帧观测数据
        """
        self.frame_buffer[observation.frame_id] = observation
    
    def project_mask_to_3d(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray
    ) -> np.ndarray:
        """
        将 2D 掩码投影到 3D 空间
        
        Args:
            mask: 2D 二值掩码 [H, W]
            depth: 深度图 [H, W]
            intrinsics: 相机内参 [3x3]
            extrinsics: 相机外参（世界坐标系下的位姿）[4x4]
            
        Returns:
            3D 点云 [N, 3]
        """
        # 获取掩码区域
        mask_indices = np.where(mask > 0)
        if len(mask_indices[0]) == 0:
            return np.array([]).reshape(0, 3)
        
        # 像素坐标
        u = mask_indices[1]  # x 坐标
        v = mask_indices[0]  # y 坐标
        
        # 获取对应深度
        depths = depth[v, u]
        
        # 过滤无效深度
        valid_mask = (depths > self.min_depth) & (depths < self.max_depth)
        if not np.any(valid_mask):
            return np.array([]).reshape(0, 3)
        
        u = u[valid_mask]
        v = v[valid_mask]
        depths = depths[valid_mask]
        
        # 反投影到相机坐标系
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        x_cam = (u - cx) * depths / fx
        y_cam = (v - cy) * depths / fy
        z_cam = depths
        
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
        
        # 转换到世界坐标系
        points_world = (extrinsics[:3, :3] @ points_cam.T + extrinsics[:3, 3:4]).T
        
        return points_world
    
    def compute_object_3d_bbox(
        self,
        points_3d: np.ndarray,
        expand_ratio: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从 3D 点云计算包围盒和质心
        
        Args:
            points_3d: 3D 点云 [N, 3]
            expand_ratio: 包围盒扩展比例
            
        Returns:
            centroid: 质心 [3]
            bbox_3d: 包围盒 [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        if len(points_3d) == 0:
            return np.zeros(3), np.zeros(6)
        
        # 计算质心
        centroid = np.mean(points_3d, axis=0)
        
        # 计算轴向包围盒 (AABB)
        min_corner = np.min(points_3d, axis=0)
        max_corner = np.max(points_3d, axis=0)
        
        # 扩展包围盒
        size = max_corner - min_corner
        center = (min_corner + max_corner) / 2
        size *= (1 + expand_ratio)
        
        min_corner = center - size / 2
        max_corner = center + size / 2
        
        bbox_3d = np.concatenate([min_corner, max_corner])
        
        return centroid, bbox_3d
    
    def associate_detections(
        self,
        current_objects: Dict[int, np.ndarray],  # {id: centroid}
        new_detections: List[np.ndarray],  # [centroid]
        iou_threshold: float = 0.3
    ) -> Tuple[Dict[int, int], List[int], List[int]]:
        """
        关联当前帧检测和已有对象
        
        使用 IoU 或距离进行数据关联
        
        Args:
            current_objects: 当前地图中的对象 {id: centroid}
            new_detections: 新检测到的对象 centroids
            iou_threshold: IoU 阈值
            
        Returns:
            associations: 关联关系 {new_id: existing_id}
            unmatched_new: 未匹配的新检测索引
            unmatched_existing: 未匹配的已有对象 ID
        """
        if len(new_detections) == 0:
            return {}, [], list(current_objects.keys())
        
        if len(current_objects) == 0:
            return {}, list(range(len(new_detections))), []
        
        # 构建代价矩阵（使用欧氏距离）
        n_current = len(current_objects)
        n_new = len(new_detections)
        
        cost_matrix = np.zeros((n_new, n_current))
        for i, new_centroid in enumerate(new_detections):
            for j, (obj_id, curr_centroid) in enumerate(current_objects.items()):
                cost_matrix[i, j] = np.linalg.norm(new_centroid - curr_centroid)
        
        # 匈牙利算法匹配
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        associations = {}
        matched_new = set()
        matched_existing = set()
        
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < iou_threshold:
                obj_id = list(current_objects.keys())[j]
                associations[i] = obj_id
                matched_new.add(i)
                matched_existing.add(obj_id)
        
        # 未匹配的
        unmatched_new = [i for i in range(n_new) if i not in matched_new]
        unmatched_existing = [
            obj_id for obj_id in current_objects.keys() 
            if obj_id not in matched_existing
        ]
        
        return associations, unmatched_new, unmatched_existing
    
    def fuse_multi_view_features(
        self,
        object_id: int,
        new_features: np.ndarray,
        view_count: int
    ) -> np.ndarray:
        """
        融合多视角特征
        
        使用移动平均融合来自不同视角的特征
        
        Args:
            object_id: 对象 ID
            new_features: 新观测的特征
            view_count: 已观测次数
            
        Returns:
            fused_features: 融合后的特征
        """
        alpha = 1.0 / (view_count + 1)
        
        # 从对象地图获取旧特征（这里简化处理）
        # 实际应该在 ObjectMap 中维护
        return new_features  # TODO: 实现完整的特征融合
    
    def process_frame(
        self,
        observation: FrameObservation,
        object_map,  # ObjectMap 类型
    ) -> Dict[int, int]:
        """
        处理单帧观测，更新对象地图
        
        Args:
            observation: 帧观测
            object_map: 对象地图
            
        Returns:
            associations: 关联关系 {detection_idx: object_id}
        """
        self.add_frame(observation)
        
        # 提取当前帧的对象位置
        current_centroids = {
            obj.id: obj.centroid 
            for obj in object_map.objects.values()
        }
        
        # 为每个检测生成 3D 位置
        new_centroids = []
        points_clouds = []
        
        for i, mask in enumerate(observation.masks_2d):
            # 投影到 3D
            points_3d = self.project_mask_to_3d(
                mask=mask,
                depth=observation.depth_image,
                intrinsics=observation.camera_intrinsics,
                extrinsics=observation.camera_pose
            )
            
            if len(points_3d) == 0:
                continue
            
            # 计算质心和包围盒
            centroid, bbox_3d = self.compute_object_3d_bbox(points_3d)
            new_centroids.append(centroid)
            points_clouds.append(points_3d)
        
        # 数据关联
        associations, unmatched_new, _ = self.associate_detections(
            current_centroids,
            new_centroids
        )
        
        # 更新已有对象或添加新对象
        for det_idx, obj_id in associations.items():
            obj = object_map.objects[obj_id]
            obj.update_from_observation(
                centroid=new_centroids[det_idx],
                bbox_3d=None,  # TODO: 传递 bbox_3d
                semantic_features=observation.clip_features if observation.clip_features is not None else np.zeros(512),
                frame_id=observation.frame_id
            )
            
            # 添加体素
            points = points_clouds[det_idx]
            for point in points:
                object_map.voxel_grid.update_voxel(
                    position=point,
                    occupied=True,
                    object_id=obj_id
                )
        
        # 添加新对象
        for det_idx in unmatched_new:
            obj = object_map.add_object(
                label=observation.detections_2d[det_idx]['label'],
                centroid=new_centroids[det_idx],
                bbox_3d=None,  # TODO: 传递 bbox_3d
                semantic_features=observation.clip_features if observation.clip_features is not None else np.zeros(512),
                confidence=observation.detections_2d[det_idx].get('score', 1.0),
                frame_id=observation.frame_id
            )
            
            # 添加体素
            for point in points_clouds[det_idx]:
                object_map.voxel_grid.update_voxel(
                    position=point,
                    occupied=True,
                    object_id=obj.id
                )
        
        return associations


def create_frame_observation(
    frame_id: int,
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    camera_pose: np.ndarray,
    camera_intrinsics: np.ndarray,
    detections: List[Dict],
    masks: List[np.ndarray],
    clip_features: Optional[np.ndarray] = None
) -> FrameObservation:
    """
    便捷函数：创建帧观测
    
    Args:
        frame_id: 帧 ID
        rgb_image: RGB 图像
        depth_image: 深度图
        camera_pose: 相机位姿 [4x4]
        camera_intrinsics: 相机内参 [3x3]
        detections: 检测结果列表
        masks: 掩码列表
        clip_features: CLIP 特征
        
    Returns:
        FrameObservation 实例
    """
    return FrameObservation(
        frame_id=frame_id,
        timestamp=0.0,  # 可以使用 time.time()
        rgb_image=rgb_image,
        depth_image=depth_image,
        camera_pose=camera_pose,
        camera_intrinsics=camera_intrinsics,
        detections_2d=detections,
        masks_2d=masks,
        clip_features=clip_features
    )
