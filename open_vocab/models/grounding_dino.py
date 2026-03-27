"""
GroundingDINO 模型加载和推理模块
"""
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import cv2


class GroundingDINOPredictor:
    """GroundingDINO 开放词汇检测模型封装"""
    
    def __init__(
        self,
        model_config_path: str,
        model_checkpoint_path: str,
        device: str = "cuda",
    ):
        self.device = device
        self.model = None
        self.model_config_path = model_config_path
        self.model_checkpoint_path = model_checkpoint_path
        
    def load_model(self):
        """加载 GroundingDINO 模型"""
        try:
            from groundingdino.util.inference import Model
            
            self.model = Model(
                model_config_path=self.model_config_path,
                model_checkpoint_path=self.model_checkpoint_path,
                device=self.device
            )
            print(f"✓ GroundingDINO 模型加载成功")
            return True
        except Exception as e:
            print(f"✗ GroundingDINO 模型加载失败: {e}")
            raise RuntimeError(f"GroundingDINO 模型加载失败: {e}")
    
    def predict(
        self,
        image: np.ndarray,
        classes: List[str],
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        执行开放词汇目标检测
        
        Args:
            image: BGR 格式的 numpy 图像
            classes: 要检测的类别列表
            box_threshold: 检测框置信度阈值
            text_threshold: 文本匹配阈值
            
        Returns:
            boxes: 检测框坐标 [N, 4] (x1, y1, x2, y2)
            scores: 置信度分数 [N]
            labels: 类别标签列表 [N]
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        # 构建文本提示
        captions = ". ".join(classes)
        
        # 执行检测
        detections = self.model.predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        
        # 提取结果
        boxes = detections.xyxy
        scores = detections.confidence
        class_ids = detections.class_id
        
        # 转换为标签列表
        labels = [classes[class_id] if class_id is not None else "unknown" 
                  for class_id in class_ids]
        
        return boxes, scores, labels
    
    def predict_with_caption(
        self,
        image: np.ndarray,
        caption: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        使用自由文本描述进行检测
        
        Args:
            image: BGR 格式的 numpy 图像
            caption: 文本描述，如 "cat and dog"
            box_threshold: 检测框置信度阈值
            text_threshold: 文本匹配阈值
            
        Returns:
            boxes, scores, phrases
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        # groundingdino-py 的 Model 类 - 使用 predict_with_classes 方法
        # 将 caption 作为单个类别处理
        detections = self.model.predict_with_classes(
            image=image,
            classes=[caption],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        
        # 提取结果
        boxes = detections.xyxy
        scores = detections.confidence
        class_ids = detections.class_id
        
        # 转换为标签列表
        phrases = [caption if class_id == 0 else "unknown" for class_id in class_ids]
        
        # 确保 boxes 是 tensor
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.tensor(boxes) if boxes is not None else torch.tensor([])
        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores) if scores is not None else torch.tensor([])
        
        return boxes, scores, phrases


def load_grounding_dino(
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda"
) -> GroundingDINOPredictor:
    """
    便捷函数：加载 GroundingDINO 模型
    
    Args:
        config_path: 模型配置文件路径
        checkpoint_path: 模型权重文件路径
        device: 运行设备
        
    Returns:
        GroundingDINOPredictor 实例
    """
    predictor = GroundingDINOPredictor(
        model_config_path=config_path,
        model_checkpoint_path=checkpoint_path,
        device=device
    )
    predictor.load_model()
    return predictor
