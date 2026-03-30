"""
3D 物体级 SLAM 主流水线

整合感知、几何、跟踪和建图模块，实现完整的 3D 场景理解流程
"""

import numpy as np
import torch
import cv2
from typing import List, Dict, Optional, Tuple
import time

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

from .models.slam_3d import VoxelGrid, ObjectMap, SceneGraph
from .models.multiview_fusion import MultiViewFusion, FrameObservation, create_frame_observation


class ObjectSLAMPipeline:
    """
    3D 物体级 SLAM 流水线
    
    输入：RGB-D 图像序列 + 相机位姿（可选）
    输出：3D 对象地图 + 场景图
    
    流程：
    1. 从 RGB-D 图像中检测对象（GroundingDINO + MobileSAM）
    2. 估计相机位姿（如果有 SLAM 系统）
    3. 将 2D 检测投影到 3D
    4. 多视图融合和数据关联
    5. 构建场景图
    """
    
    def __init__(
        self,
        detection_pipeline,  # OpenVocabularyPipeline 类型
        voxel_size: float = 0.05,
        use_slam: bool = True,
    ):
        """
        初始化 3D SLAM 流水线
        
        Args:
            detection_pipeline: 2D 检测流水线（OpenVocabularyPipeline）
            voxel_size: 体素大小
            use_slam: 是否使用 SLAM 进行位姿估计
        """
        self.detection_pipeline = detection_pipeline
        self.voxel_size = voxel_size
        self.use_slam = use_slam
        
        # 初始化核心组件
        self.voxel_grid = VoxelGrid(voxel_size=voxel_size)
        self.object_map = ObjectMap(voxel_grid=self.voxel_grid)
        self.scene_graph = SceneGraph(object_map=self.object_map)
        self.multi_view_fusion = MultiViewFusion()
        
        # 状态变量
        self.current_frame_id = 0
        self.keyframes = []  # 关键帧列表
        self.camera_poses = {}  # {frame_id: pose_matrix}
        
        # SLAM 系统（可选）
        if self.use_slam:
            self.slam_system = self._initialize_slam()
        else:
            self.slam_system = None
    
    def _initialize_slam(self):
        """
        初始化 SLAM 系统
        
        TODO: 集成 ORB-SLAM3 或 DROID-SLAM
        """
        print("注意：SLAM 系统集成需要额外安装 ORB-SLAM3 或 DROID-SLAM")
        print("目前使用简化的位姿估计")
        return None
    
    def process_rgbd_frame(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        camera_intrinsics: np.ndarray,
        camera_pose: Optional[np.ndarray] = None,
        classes: Optional[List[str]] = None,
        caption: Optional[str] = None,
    ) -> Dict:
        """
        处理单帧 RGB-D 图像
        
        Args:
            rgb_image: RGB 图像 [H, W, 3]
            depth_image: 深度图 [H, W]
            camera_intrinsics: 相机内参 [3x3]
            camera_pose: 相机位姿 [4x4]（可选，如果为 None 则估计）
            classes: 要检测的类别列表
            caption: 自由文本描述
            
        Returns:
            results: 处理结果字典
        """
        frame_id = self.current_frame_id
        start_time = time.time()
        
        # Step 1: 估计或使用提供的相机位姿
        if camera_pose is None:
            if self.use_slam and self.slam_system:
                # TODO: 使用 SLAM 估计位姿
                camera_pose = np.eye(4)  # 简化处理
            else:
                camera_pose = np.eye(4)  # 默认单位矩阵
        
        self.camera_poses[frame_id] = camera_pose
        
        # Step 2: 2D 检测和分割
        print(f"[Frame {frame_id}] 执行 2D 检测...")
        if caption:
            detection_results = self.detection_pipeline.detect_with_caption(
                image=rgb_image,
                caption=caption,
            )
        else:
            detection_results = self.detection_pipeline.detect_and_segment(
                image=rgb_image,
                classes=classes,
            )
        
        boxes = detection_results['boxes']
        labels = detection_results['labels']
        scores = detection_results['scores']
        masks = detection_results.get('masks')
        
        if masks is None or len(masks) == 0:
            print(f"[Frame {frame_id}] 未检测到目标")
            self.current_frame_id += 1
            return {'frame_id': frame_id, 'objects_detected': 0}
        
        # Step 3: 提取 CLIP 特征（可选）
        clip_features = None
        # TODO: 为每个检测对象提取 CLIP 特征
        
        # Step 4: 创建帧观测
        detections_list = [
            {'label': label, 'score': score, 'box': box}
            for label, score, box in zip(labels, scores, boxes)
        ]
        
        observation = create_frame_observation(
            frame_id=frame_id,
            rgb_image=rgb_image,
            depth_image=depth_image,
            camera_pose=camera_pose,
            camera_intrinsics=camera_intrinsics,
            detections=detections_list,
            masks=masks,
            clip_features=clip_features
        )
        
        # Step 5: 多视图融合，更新对象地图
        print(f"[Frame {frame_id}] 融合多视图观测...")
        associations = self.multi_view_fusion.process_frame(
            observation=observation,
            object_map=self.object_map
        )
        
        # Step 6: 选择关键帧
        if self._is_keyframe(frame_id, camera_pose):
            self.keyframes.append(frame_id)
            print(f"[Frame {frame_id}] 添加为关键帧")
        
        # Step 7: 更新场景图
        if len(self.keyframes) > 0 and frame_id % 10 == 0:
            print(f"[Frame {frame_id}] 更新场景图...")
            self.scene_graph.infer_spatial_relations()
        
        processing_time = time.time() - start_time
        
        results = {
            'frame_id': frame_id,
            'objects_detected': len(labels),
            'objects_in_map': len(self.object_map.objects),
            'associations': associations,
            'is_keyframe': frame_id in self.keyframes,
            'processing_time': processing_time,
            'camera_pose': camera_pose,
        }
        
        print(f"[Frame {frame_id}] 完成，耗时：{processing_time:.3f}s")
        
        self.current_frame_id += 1
        return results
    
    def _is_keyframe(
        self,
        frame_id: int,
        current_pose: np.ndarray,
        min_distance: float = 0.5,
        min_rotation: float = 15.0
    ) -> bool:
        """
        判断是否为关键帧
        
        Args:
            frame_id: 当前帧 ID
            current_pose: 当前相机位姿
            min_distance: 最小平移距离（米）
            min_rotation: 最小旋转角度（度）
            
        Returns:
            是否是关键帧
        """
        if len(self.keyframes) == 0:
            return True
        
        # 获取上一个关键帧的位姿
        last_kf_id = self.keyframes[-1]
        last_pose = self.camera_poses[last_kf_id]
        
        # 计算相对运动
        rel_pose = np.linalg.inv(last_pose) @ current_pose
        translation = np.linalg.norm(rel_pose[:3, 3])
        
        # 计算旋转角度
        rotation_matrix = rel_pose[:3, :3]
        trace = np.trace(rotation_matrix)
        rotation_angle = np.abs(np.arccos((trace - 1) / 2)) * 180 / np.pi
        
        # 判断是否满足关键帧条件
        is_kf = (translation > min_distance) or (rotation_angle > min_rotation)
        
        return is_kf
    
    def get_object_cloud(self, object_id: int) -> np.ndarray:
        """
        获取对象的 3D 点云
        
        Args:
            object_id: 对象 ID
            
        Returns:
            点云数组 [N, 3]
        """
        obj = self.object_map.objects.get(object_id)
        if obj is None:
            return np.array([]).reshape(0, 3)
        
        # 从体素网格中提取对象对应的点
        points = []
        for voxel_idx in obj.voxel_indices:
            point = self.voxel_grid.voxel_to_world(voxel_idx)
            points.append(point)
        
        return np.array(points)
    
    def query_scene_graph(self, query_text: str) -> List[int]:
        """
        查询场景图
        
        Args:
            query_text: 查询文本
            
        Returns:
            匹配的对象 ID 列表
        """
        return self.scene_graph.query(query_text)
    
    def save_reconstruction(self, output_path: str):
        """
        保存重建结果
        
        Args:
            output_path: 输出路径
        """
        import json
        
        # 保存场景图为 JSON
        scene_graph_dict = self.scene_graph.to_dict()
        
        with open(output_path, 'w') as f:
            json.dump(scene_graph_dict, f, indent=2)
        
        print(f"重建结果已保存到：{output_path}")
    
    def visualize(self):
        """
        可视化重建结果（使用 Open3D）
        """
        if not HAS_OPEN3D:
            print("请安装 open3d: pip install open3d")
            return
        
        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="3D Object Map")
        
        # 添加体素点云
        pcd = self.voxel_grid.to_point_cloud()
        if len(pcd.points) > 0:
            vis.add_geometry(pcd)
        
        # 添加对象包围盒
        for obj in self.object_map.objects.values():
            if obj.bbox_3d is not None:
                bbox = self._create_3d_bbox(obj.bbox_3d)
                vis.add_geometry(bbox)
        
        # 运行可视化
        vis.run()
        vis.destroy_window()
    
    def _create_3d_bbox(self, bbox_3d: np.ndarray):  # -> o3d.geometry.LineSet
        """
        创建 3D 包围盒可视化
        
        Args:
            bbox_3d: [x_min, y_min, z_min, x_max, y_max, z_max]
            
        Returns:
            Open3D LineSet
        """
        if not HAS_OPEN3D:
            return None
        
        import open3d as o3d
        
        min_corner = bbox_3d[:3]
        max_corner = bbox_3d[3:]
        
        # 8 个顶点
        vertices = np.array([
            [min_corner[0], min_corner[1], min_corner[2]],
            [max_corner[0], min_corner[1], min_corner[2]],
            [max_corner[0], max_corner[1], min_corner[2]],
            [min_corner[0], max_corner[1], min_corner[2]],
            [min_corner[0], min_corner[1], max_corner[2]],
            [max_corner[0], min_corner[1], max_corner[2]],
            [max_corner[0], max_corner[1], max_corner[2]],
            [min_corner[0], max_corner[1], max_corner[2]],
        ])
        
        # 12 条边
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
            [0, 4], [1, 5], [2, 6], [3, 7],  # 垂直边
        ]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(vertices)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        return line_set


def create_object_slam_pipeline(
    detection_pipeline,
    voxel_size: float = 0.05,
    use_slam: bool = True,
) -> ObjectSLAMPipeline:
    """
    便捷函数：创建 3D 物体 SLAM 流水线
    
    Args:
        detection_pipeline: 2D 检测流水线
        voxel_size: 体素大小
        use_slam: 是否使用 SLAM
        
    Returns:
        ObjectSLAMPipeline 实例
    """
    pipeline = ObjectSLAMPipeline(
        detection_pipeline=detection_pipeline,
        voxel_size=voxel_size,
        use_slam=use_slam,
    )
    return pipeline
