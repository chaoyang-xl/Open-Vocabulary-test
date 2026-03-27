"""
MobileSAM 模型加载和推理模块
"""
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Union
import cv2


class MobileSAMPredictor:
    """MobileSAM 分割模型封装"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.predictor = None
        
    def load_model(self):
        """加载 MobileSAM 模型"""
        try:
            # 尝试导入 MobileSAM
            try:
                from mobile_sam import sam_model_registry, SamPredictor
            except ImportError:
                # 如果 MobileSAM 未安装，使用原版 SAM
                print("MobileSAM 未安装，尝试使用原版 SAM...")
                from segment_anything import sam_model_registry, SamPredictor
            
            # 加载模型
            sam = sam_model_registry["vit_t"](checkpoint=self.checkpoint_path)
            sam.to(device=self.device)
            
            self.model = sam
            self.predictor = SamPredictor(sam)
            
            print(f"✓ MobileSAM 模型加载成功")
            return True
            
        except Exception as e:
            print(f"✗ MobileSAM 模型加载失败: {e}")
            raise RuntimeError(f"MobileSAM 模型加载失败: {e}")
    
    def set_image(self, image: np.ndarray):
        """
        设置输入图像
        
        Args:
            image: RGB 格式的 numpy 图像
        """
        if self.predictor is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        self.predictor.set_image(image)
    
    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        multimask_output: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        执行分割预测
        
        Args:
            point_coords: 点提示坐标 [N, 2]
            point_labels: 点标签 (1=前景, 0=背景) [N]
            box: 框提示 [4] (x1, y1, x2, y2)
            multimask_output: 是否输出多个 mask
            
        Returns:
            masks: 分割掩码 [N, H, W]
            scores: mask 置信度分数 [N]
            logits: 原始 logits [N, H, W]
        """
        if self.predictor is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask_output,
        )
        
        return masks, scores, logits
    
    def predict_from_boxes(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[np.ndarray]:
        """
        从检测框生成分割掩码
        
        Args:
            image: RGB 格式的 numpy 图像
            boxes: 检测框 [N, 4] (x1, y1, x2, y2)
            
        Returns:
            masks: 分割掩码列表 [N, H, W]
        """
        self.set_image(image)
        
        masks_list = []
        for box in boxes:
            masks, scores, _ = self.predict(
                box=box,
                multimask_output=False
            )
            # 选择得分最高的 mask
            best_mask = masks[scores.argmax()]
            masks_list.append(best_mask)
        
        return masks_list
    
    def predict_from_points(
        self,
        image: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
    ) -> np.ndarray:
        """
        从点提示生成分割掩码
        
        Args:
            image: RGB 格式的 numpy 图像
            point_coords: 点坐标 [N, 2]
            point_labels: 点标签 [N]
            
        Returns:
            mask: 分割掩码 [H, W]
        """
        self.set_image(image)
        
        masks, scores, _ = self.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )
        
        # 返回得分最高的 mask
        return masks[scores.argmax()]


def load_mobile_sam(
    checkpoint_path: str,
    device: str = "cuda"
) -> MobileSAMPredictor:
    """
    便捷函数：加载 MobileSAM 模型
    
    Args:
        checkpoint_path: 模型权重文件路径
        device: 运行设备
        
    Returns:
        MobileSAMPredictor 实例
    """
    predictor = MobileSAMPredictor(
        checkpoint_path=checkpoint_path,
        device=device
    )
    predictor.load_model()
    return predictor
