"""
3D 数据结构模块

提供 3D 物体级 SLAM 所需的核心数据结构：
- VoxelGrid: 体素网格表示
- Object3D: 3D 对象实例
- ObjectMap: 对象地图
- SceneGraph: 3D 场景图
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
import open3d as o3d


@dataclass
class Voxel:
    """
    体素单元
    
    Attributes:
        position: 体素中心位置 [x, y, z]
        occupied: 是否被占据
        semantic_features: 语义特征向量 (CLIP 特征)
        object_id: 关联的对象 ID（如果有）
        color: 平均颜色 [R, G, B]
        observations: 观测次数
    """
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    occupied: bool = False
    semantic_features: Optional[np.ndarray] = None
    object_id: Optional[int] = None
    color: np.ndarray = field(default_factory=lambda: np.zeros(3))
    observations: int = 0


class VoxelGrid:
    """
    体素网格地图
    
    用于存储环境的 3D 体素表示，支持增量式更新和查询
    
    Attributes:
        voxel_size: 体素大小（米）
        grid_shape: 网格形状 [H, W, D]
        origin: 网格原点坐标
        voxels: 体素字典 {(x, y, z): Voxel}
    """
    
    def __init__(self, voxel_size: float = 0.05, resolution: int = 256):
        """
        初始化体素网格
        
        Args:
            voxel_size: 体素大小（米），默认 5cm
            resolution: 网格分辨率，默认 256^3
        """
        self.voxel_size = voxel_size
        self.resolution = resolution
        self.origin = np.zeros(3)  # 网格原点
        self.voxels: Dict[Tuple[int, int, int], Voxel] = {}  # 体素字典
        
    def world_to_voxel(self, position: np.ndarray) -> Tuple[int, int, int]:
        """
        将世界坐标转换为体素索引
        
        Args:
            position: 世界坐标 [x, y, z]
            
        Returns:
            体素索引 (vx, vy, vz)
        """
        relative_pos = position - self.origin
        voxel_idx = np.floor(relative_pos / self.voxel_size).astype(int)
        return tuple(voxel_idx)
    
    def voxel_to_world(self, voxel_idx: Tuple[int, int, int]) -> np.ndarray:
        """
        将体素索引转换为世界坐标
        
        Args:
            voxel_idx: 体素索引 (vx, vy, vz)
            
        Returns:
            体素中心的世界坐标 [x, y, z]
        """
        return np.array(voxel_idx) * self.voxel_size + self.origin
    
    def update_voxel(
        self,
        position: np.ndarray,
        occupied: bool = True,
        semantic_features: Optional[np.ndarray] = None,
        color: Optional[np.ndarray] = None,
        object_id: Optional[int] = None,
    ):
        """
        更新体素信息
        
        Args:
            position: 世界坐标
            occupied: 是否占据
            semantic_features: 语义特征
            color: 颜色
            object_id: 对象 ID
        """
        voxel_idx = self.world_to_voxel(position)
        
        # 检查边界
        if not all(0 <= idx < self.resolution for idx in voxel_idx):
            return
        
        # 创建或更新体素
        if voxel_idx not in self.voxels:
            self.voxels[voxel_idx] = Voxel()
        
        voxel = self.voxels[voxel_idx]
        voxel.position = self.voxel_to_world(voxel_idx)
        voxel.occupied = occupied
        voxel.observations += 1
        
        if semantic_features is not None:
            # 移动平均更新特征
            alpha = 1.0 / voxel.observations
            if voxel.semantic_features is None:
                voxel.semantic_features = semantic_features.copy()
            else:
                voxel.semantic_features = (1 - alpha) * voxel.semantic_features + alpha * semantic_features
        
        if color is not None:
            voxel.color = color
        
        if object_id is not None:
            voxel.object_id = object_id
    
    def get_occupied_voxels(self) -> List[Voxel]:
        """获取所有被占据的体素"""
        return [v for v in self.voxels.values() if v.occupied]
    
    def to_point_cloud(self) -> o3d.geometry.PointCloud:
        """
        转换为 Open3D 点云
        
        Returns:
            Open3D 点云对象
        """
        points = []
        colors = []
        
        for voxel in self.get_occupied_voxels():
            points.append(voxel.position)
            colors.append(voxel.color)
        
        if len(points) == 0:
            return o3d.geometry.PointCloud()
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        
        return pcd


@dataclass
class Object3D:
    """
    3D 对象实例
    
    Attributes:
        id: 对象唯一标识符
        label: 对象类别标签
        confidence: 检测置信度
        bbox_3d: 3D 包围盒 [x_min, y_min, z_min, x_max, y_max, z_max]
        centroid: 3D 质心位置 [x, y, z]
        semantic_features: CLIP 语义特征
        voxel_indices: 关联的体素索引集合
        observations: 观测次数（多视图）
        view_features: 各视角的特征列表
        first_seen: 首次观测的帧 ID
        last_seen: 最后观测的帧 ID
    """
    id: int
    label: str
    confidence: float = 1.0
    bbox_3d: Optional[np.ndarray] = None
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    semantic_features: Optional[np.ndarray] = None
    voxel_indices: Set[Tuple[int, int, int]] = field(default_factory=set)
    observations: int = 0
    view_features: List[np.ndarray] = field(default_factory=list)
    first_seen: int = 0
    last_seen: int = 0
    
    def update_from_observation(
        self,
        centroid: np.ndarray,
        bbox_3d: np.ndarray,
        semantic_features: np.ndarray,
        frame_id: int,
    ):
        """
        从新的观测更新对象
        
        Args:
            centroid: 观测到的质心
            bbox_3d: 观测到的 3D 框
            semantic_features: 观测的语义特征
            frame_id: 帧 ID
        """
        self.observations += 1
        self.last_seen = frame_id
        
        # 移动平均更新质心
        alpha = 1.0 / self.observations
        self.centroid = (1 - alpha) * self.centroid + alpha * centroid
        
        # 更新包围盒（取并集）
        if self.bbox_3d is not None:
            self.bbox_3d = np.minimum(self.bbox_3d[:3], bbox_3d[:3])
            self.bbox_3d = np.concatenate([
                self.bbox_3d[:3],
                np.maximum(self.bbox_3d[3:], bbox_3d[3:])
            ])
        else:
            self.bbox_3d = bbox_3d
        
        # 更新语义特征
        if self.semantic_features is None:
            self.semantic_features = semantic_features.copy()
        else:
            self.semantic_features = (1 - alpha) * self.semantic_features + alpha * semantic_features
    
    def get_volume(self) -> float:
        """计算对象体积（立方米）"""
        if self.bbox_3d is None:
            return 0.0
        dims = self.bbox_3d[3:] - self.bbox_3d[:3]
        return np.prod(dims)
    
    def get_size(self) -> np.ndarray:
        """获取对象尺寸 [长，宽，高]"""
        if self.bbox_3d is None:
            return np.zeros(3)
        return self.bbox_3d[3:] - self.bbox_3d[:3]


class ObjectMap:
    """
    3D 对象地图
    
    存储和管理所有检测到的 3D 对象实例
    
    Attributes:
        objects: 对象字典 {id: Object3D}
        next_id: 下一个对象 ID
        voxel_grid: 关联的体素网格
    """
    
    def __init__(self, voxel_grid: Optional[VoxelGrid] = None):
        """
        初始化对象地图
        
        Args:
            voxel_grid: 关联的体素网格
        """
        self.objects: Dict[int, Object3D] = {}
        self.next_id = 0
        self.voxel_grid = voxel_grid if voxel_grid else VoxelGrid()
    
    def add_object(
        self,
        label: str,
        centroid: np.ndarray,
        bbox_3d: np.ndarray,
        semantic_features: np.ndarray,
        confidence: float = 1.0,
        frame_id: int = 0,
    ) -> Object3D:
        """
        添加新对象
        
        Args:
            label: 对象标签
            centroid: 3D 质心
            bbox_3d: 3D 包围盒
            semantic_features: 语义特征
            confidence: 置信度
            frame_id: 帧 ID
            
        Returns:
            新创建的对象实例
        """
        obj = Object3D(
            id=self.next_id,
            label=label,
            confidence=confidence,
            bbox_3d=bbox_3d,
            centroid=centroid.copy(),
            semantic_features=semantic_features.copy(),
            first_seen=frame_id,
            last_seen=frame_id,
        )
        
        self.objects[self.next_id] = obj
        self.next_id += 1
        
        return obj
    
    def find_nearest_object(
        self,
        position: np.ndarray,
        threshold: float = 1.0
    ) -> Optional[Object3D]:
        """
        查找最近的对象
        
        Args:
            position: 查询位置
            threshold: 距离阈值（米）
            
        Returns:
            最近的对象实例，如果没有则返回 None
        """
        min_dist = float('inf')
        nearest_obj = None
        
        for obj in self.objects.values():
            dist = np.linalg.norm(obj.centroid - position)
            if dist < min_dist and dist < threshold:
                min_dist = dist
                nearest_obj = obj
        
        return nearest_obj
    
    def get_objects_in_view(
        self,
        camera_position: np.ndarray,
        camera_direction: np.ndarray,
        fov: float = 60.0,
        max_distance: float = 10.0
    ) -> List[Object3D]:
        """
        获取视野内的对象
        
        Args:
            camera_position: 相机位置
            camera_direction: 相机方向（单位向量）
            fov: 视场角（度）
            max_distance: 最大距离
            
        Returns:
            视野内的对象列表
        """
        visible_objects = []
        
        for obj in self.objects.values():
            # 计算相对位置
            to_object = obj.centroid - camera_position
            distance = np.linalg.norm(to_object)
            
            if distance > max_distance:
                continue
            
            # 计算视角
            to_object_normalized = to_object / distance
            cos_angle = np.dot(camera_direction, to_object_normalized)
            angle_threshold = np.cos(np.radians(fov / 2))
            
            if cos_angle > angle_threshold:
                visible_objects.append(obj)
        
        return visible_objects


@dataclass
class SceneGraphNode:
    """
    场景图节点
    
    Attributes:
        object_id: 关联的对象 ID
        attributes: 对象属性字典
    """
    object_id: int
    attributes: Dict[str, any] = field(default_factory=dict)


@dataclass
class SceneGraphEdge:
    """
    场景图边（对象间关系）
    
    Attributes:
        source_id: 源对象 ID
        target_id: 目标对象 ID
        relation_type: 关系类型（如 "on", "in", "near"）
        confidence: 关系置信度
        attributes: 关系属性
    """
    source_id: int
    target_id: int
    relation_type: str
    confidence: float = 1.0
    attributes: Dict[str, any] = field(default_factory=dict)


class SceneGraph:
    """
    3D 场景图
    
    表示对象间的空间和语义关系
    
    Attributes:
        nodes: 节点字典 {id: SceneGraphNode}
        edges: 边列表
        object_map: 关联的对象地图
    """
    
    def __init__(self, object_map: ObjectMap):
        """
        初始化场景图
        
        Args:
            object_map: 关联的对象地图
        """
        self.nodes: Dict[int, SceneGraphNode] = {}
        self.edges: List[SceneGraphEdge] = []
        self.object_map = object_map
    
    def add_node(self, object_id: int, attributes: Optional[Dict] = None):
        """
        添加场景图节点
        
        Args:
            object_id: 对象 ID
            attributes: 节点属性
        """
        self.nodes[object_id] = SceneGraphNode(
            object_id=object_id,
            attributes=attributes or {}
        )
    
    def add_edge(
        self,
        source_id: int,
        target_id: int,
        relation_type: str,
        confidence: float = 1.0,
        attributes: Optional[Dict] = None
    ):
        """
        添加场景图边
        
        Args:
            source_id: 源对象 ID
            target_id: 目标对象 ID
            relation_type: 关系类型
            confidence: 置信度
            attributes: 边属性
        """
        edge = SceneGraphEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            confidence=confidence,
            attributes=attributes or {}
        )
        self.edges.append(edge)
    
    def infer_spatial_relations(self, distance_threshold: float = 2.0):
        """
        推断对象间的空间关系
        
        自动检测如 "on", "in", "near", "left_of", "right_of" 等关系
        
        Args:
            distance_threshold: "near"关系的距离阈值
        """
        self.edges.clear()
        
        objects = list(self.object_map.objects.values())
        
        for i, obj1 in enumerate(objects):
            # 确保节点存在
            if obj1.id not in self.nodes:
                self.add_node(obj1.id)
            
            for j, obj2 in enumerate(objects):
                if i >= j:
                    continue
                
                # 确保节点存在
                if obj2.id not in self.nodes:
                    self.add_node(obj2.id)
                
                # 计算空间关系
                self._infer_relation(obj1, obj2, distance_threshold)
    
    def _infer_relation(
        self,
        obj1: Object3D,
        obj2: Object3D,
        distance_threshold: float
    ):
        """
        推断两个对象间的关系
        
        Args:
            obj1: 对象 1
            obj2: 对象 2
            distance_threshold: 距离阈值
        """
        if obj1.bbox_3d is None or obj2.bbox_3d is None:
            return
        
        # 垂直关系（on/under）
        if abs(obj1.centroid[0] - obj2.centroid[0]) < 0.5 and \
           abs(obj1.centroid[1] - obj2.centroid[1]) < 0.5:
            if obj1.centroid[2] > obj2.centroid[2] + 0.5:
                self.add_edge(obj1.id, obj2.id, "above", 0.8)
            elif obj2.centroid[2] > obj1.centroid[2] + 0.5:
                self.add_edge(obj2.id, obj1.id, "above", 0.8)
        
        # 水平关系（near）
        distance = np.linalg.norm(obj1.centroid - obj2.centroid)
        if distance < distance_threshold:
            self.add_edge(obj1.id, obj2.id, "near", 1.0 - distance / distance_threshold)
        
        # 左右关系
        if obj1.centroid[0] < obj2.centroid[0] - 0.5:
            self.add_edge(obj1.id, obj2.id, "left_of", 0.7)
        elif obj1.centroid[0] > obj2.centroid[0] + 0.5:
            self.add_edge(obj1.id, obj2.id, "right_of", 0.7)
    
    def query(self, query_text: str) -> List[int]:
        """
        查询场景图
        
        Args:
            query_text: 查询文本，如 "find all chairs near table"
            
        Returns:
            匹配的对象 ID 列表
        """
        # TODO: 实现基于 CLIP 的语义查询
        # 这里简化为返回所有对象
        return list(self.nodes.keys())
    
    def to_dict(self) -> Dict:
        """
        转换为字典格式
        
        Returns:
            场景图的字典表示
        """
        return {
            "nodes": [
                {
                    "id": node.object_id,
                    "label": self.object_map.objects[node.object_id].label,
                    "centroid": self.object_map.objects[node.object_id].centroid.tolist(),
                    "attributes": node.attributes
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "relation": edge.relation_type,
                    "confidence": edge.confidence
                }
                for edge in self.edges
            ]
        }
