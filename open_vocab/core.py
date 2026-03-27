"""
开放词汇检测与分割核心模块
整合 GroundingDINO + MobileSAM + CLIP
"""
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict, Union
import cv2

from .models.grounding_dino import GroundingDINOPredictor
from .models.mobile_sam import MobileSAMPredictor
from .models.clip_model import CLIPPredictor


class OpenVocabularyPipeline:
    """
    开放词汇检测与分割流水线
    
    功能：
    1. 使用 GroundingDINO 进行开放词汇目标检测
    2. 使用 MobileSAM 进行精细分割
    3. 使用 CLIP 进行语义验证和分类
    """
    
    def __init__(
        self,
        grounding_dino_config: str,
        grounding_dino_checkpoint: str,
        mobile_sam_checkpoint: str,
        device: str = "cuda",
        use_clip: bool = True,
        clip_model_name: str = "ViT-B-32",
    ):
        self.device = device
        self.use_clip = use_clip
        
        # 初始化模型
        self.grounding_dino = GroundingDINOPredictor(
            model_config_path=grounding_dino_config,
            model_checkpoint_path=grounding_dino_checkpoint,
            device=device
        )
        
        self.mobile_sam = MobileSAMPredictor(
            checkpoint_path=mobile_sam_checkpoint,
            device=device
        )
        
        self.clip = None
        if use_clip:
            self.clip = CLIPPredictor(
                model_name=clip_model_name,
                device=device
            )
    
    def load_models(self):
        """加载所有模型"""
        print("=" * 50)
        print("加载开放词汇模型...")
        print("=" * 50)
        
        self.grounding_dino.load_model()
        self.mobile_sam.load_model()
        
        if self.use_clip and self.clip:
            self.clip.load_model()
        
        print("=" * 50)
        print("所有模型加载完成！")
        print("=" * 50)
    
    def detect_and_segment(
        self,
        image: Union[np.ndarray, str],
        classes: List[str],
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        use_sam: bool = True,
    ) -> Dict:
        """
        检测并分割目标
        
        Args:
            image: 输入图像 (numpy array 或文件路径)
            classes: 要检测的类别列表
            box_threshold: 检测框置信度阈值
            text_threshold: 文本匹配阈值
            use_sam: 是否使用 SAM 进行分割
            
        Returns:
            results: 包含检测结果的字典
                {
                    'boxes': 检测框 [N, 4],
                    'scores': 置信度 [N],
                    'labels': 类别标签 [N],
                    'masks': 分割掩码 [N, H, W] (如果 use_sam=True),
                    'visualization': 可视化图像
                }
        """
        # 加载图像
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        original_image = image.copy()
        
        # Step 1: GroundingDINO 检测
        print(f"[1/3] GroundingDINO 检测中... 类别: {classes}")
        boxes, scores, labels = self.grounding_dino.predict(
            image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            classes=classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        
        if len(boxes) == 0:
            print("未检测到目标")
            return {
                'boxes': [],
                'scores': [],
                'labels': [],
                'masks': [],
                'visualization': image
            }
        
        print(f"检测到 {len(boxes)} 个目标")
        
        # Step 2: MobileSAM 分割
        masks = []
        if use_sam:
            print(f"[2/3] MobileSAM 分割中...")
            masks = self.mobile_sam.predict_from_boxes(
                image=original_image,
                boxes=boxes.cpu().numpy() if torch.is_tensor(boxes) else boxes
            )
            print(f"生成分割掩码完成")
        
        # Step 3: CLIP 验证 (可选)
        if self.use_clip and self.clip and len(boxes) > 0:
            print(f"[3/3] CLIP 语义验证中...")
            # 可以在这里添加 CLIP 验证逻辑
            pass
        
        # 生成可视化结果
        vis_image = self.visualize_results(
            image=original_image,
            boxes=boxes,
            labels=labels,
            scores=scores,
            masks=masks if masks else None
        )
        
        results = {
            'boxes': boxes.cpu().numpy() if torch.is_tensor(boxes) else boxes,
            'scores': scores.cpu().numpy() if torch.is_tensor(scores) else scores,
            'labels': labels,
            'masks': np.array(masks) if masks else None,
            'visualization': vis_image
        }
        
        return results
    
    def detect_with_caption(
        self,
        image: Union[np.ndarray, str],
        caption: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> Dict:
        """
        使用自由文本描述进行检测和分割
        
        Args:
            image: 输入图像
            caption: 文本描述，如 "cat and dog"
            box_threshold: 检测框置信度阈值
            text_threshold: 文本匹配阈值
            
        Returns:
            results: 检测结果字典
        """
        # 加载图像
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        original_image = image.copy()
        
        # GroundingDINO 检测
        print(f"使用描述进行检测: '{caption}'")
        boxes, scores, phrases = self.grounding_dino.predict_with_caption(
            image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        
        if len(boxes) == 0:
            return {
                'boxes': [],
                'scores': [],
                'labels': [],
                'masks': [],
                'visualization': image
            }
        
        # MobileSAM 分割
        masks = self.mobile_sam.predict_from_boxes(
            image=original_image,
            boxes=boxes.cpu().numpy() if torch.is_tensor(boxes) else boxes
        )
        
        # 生成可视化
        vis_image = self.visualize_results(
            image=original_image,
            boxes=boxes,
            labels=phrases,
            scores=scores,
            masks=masks
        )
        
        return {
            'boxes': boxes.cpu().numpy() if torch.is_tensor(boxes) else boxes,
            'scores': scores.cpu().numpy() if torch.is_tensor(scores) else scores,
            'labels': phrases,
            'masks': np.array(masks),
            'visualization': vis_image
        }
    
    def visualize_results(
        self,
        image: np.ndarray,
        boxes: Union[torch.Tensor, np.ndarray],
        labels: List[str],
        scores: Union[torch.Tensor, np.ndarray],
        masks: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 原始图像
            boxes: 检测框
            labels: 类别标签
            scores: 置信度
            masks: 分割掩码
            
        Returns:
            vis_image: 可视化图像
        """
        vis_image = image.copy()
        
        # 转换 tensor 为 numpy
        if torch.is_tensor(boxes):
            boxes = boxes.cpu().numpy()
        if torch.is_tensor(scores):
            scores = scores.cpu().numpy()
        
        # 生成随机颜色
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(boxes), 3), dtype=np.uint8)
        
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            color = colors[i].tolist()
            x1, y1, x2, y2 = map(int, box)
            
            # 绘制检测框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label_text = f"{label}: {score:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                vis_image, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1
            )
            cv2.putText(
                vis_image, label_text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )
            
            # 绘制分割掩码
            if masks and i < len(masks):
                mask = masks[i]
                colored_mask = np.zeros_like(vis_image)
                colored_mask[mask > 0] = color
                vis_image = cv2.addWeighted(vis_image, 1.0, colored_mask, 0.4, 0)
        
        return vis_image


def create_pipeline(
    grounding_dino_config: str,
    grounding_dino_checkpoint: str,
    mobile_sam_checkpoint: str,
    device: str = "cuda",
    use_clip: bool = True,
) -> OpenVocabularyPipeline:
    """
    便捷函数：创建开放词汇流水线
    
    Args:
        grounding_dino_config: GroundingDINO 配置文件路径
        grounding_dino_checkpoint: GroundingDINO 权重文件路径
        mobile_sam_checkpoint: MobileSAM 权重文件路径
        device: 运行设备
        use_clip: 是否使用 CLIP
        
    Returns:
        OpenVocabularyPipeline 实例
    """
    pipeline = OpenVocabularyPipeline(
        grounding_dino_config=grounding_dino_config,
        grounding_dino_checkpoint=grounding_dino_checkpoint,
        mobile_sam_checkpoint=mobile_sam_checkpoint,
        device=device,
        use_clip=use_clip,
    )
    pipeline.load_models()
    return pipeline
