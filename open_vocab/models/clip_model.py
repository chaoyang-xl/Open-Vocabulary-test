"""
CLIP 模型加载和推理模块
用于图像-文本特征对齐和相似度计算
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Union, Optional, Tuple


class CLIPPredictor:
    """CLIP 模型封装"""
    
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cuda",
    ):
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        
    def load_model(self):
        """加载 CLIP 模型"""
        try:
            import open_clip
            
            # 加载模型和预处理
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name=self.model_name,
                pretrained=self.pretrained,
                device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            
            # 设置为评估模式
            self.model.eval()
            
            print(f"✓ CLIP 模型加载成功: {self.model_name} ({self.pretrained})")
            return True
            
        except Exception as e:
            print(f"✗ CLIP 模型加载失败: {e}")
            raise RuntimeError(f"CLIP 模型加载失败: {e}")
    
    @torch.no_grad()
    def encode_image(
        self,
        images: Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        编码图像
        
        Args:
            images: 输入图像或图像列表
            normalize: 是否对特征进行 L2 归一化
            
        Returns:
            image_features: 图像特征 [N, D]
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        # 处理不同类型的输入
        if isinstance(images, (Image.Image, np.ndarray)):
            images = [images]
        
        if isinstance(images, list):
            # 预处理图像
            if isinstance(images[0], np.ndarray):
                images = [Image.fromarray(img) for img in images]
            images = torch.stack([self.preprocess(img) for img in images])
        
        images = images.to(self.device)
        
        # 提取特征
        image_features = self.model.encode_image(images)
        
        if normalize:
            image_features = F.normalize(image_features, dim=-1)
        
        return image_features
    
    @torch.no_grad()
    def encode_text(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        编码文本
        
        Args:
            texts: 输入文本或文本列表
            normalize: 是否对特征进行 L2 归一化
            
        Returns:
            text_features: 文本特征 [N, D]
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        if isinstance(texts, str):
            texts = [texts]
        
        # 分词
        text_tokens = self.tokenizer(texts).to(self.device)
        
        # 提取特征
        text_features = self.model.encode_text(text_tokens)
        
        if normalize:
            text_features = F.normalize(text_features, dim=-1)
        
        return text_features
    
    def compute_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算图像-文本相似度
        
        Args:
            image_features: 图像特征 [N, D]
            text_features: 文本特征 [M, D]
            
        Returns:
            similarity: 相似度矩阵 [N, M]
        """
        # 计算余弦相似度
        similarity = (image_features @ text_features.T) * 100
        return similarity
    
    def classify(
        self,
        images: Union[Image.Image, List[Image.Image]],
        class_names: List[str],
        templates: Optional[List[str]] = None,
    ) -> Tuple[List[str], torch.Tensor]:
        """
        使用 CLIP 进行零样本分类
        
        Args:
            images: 输入图像
            class_names: 类别名称列表
            templates: 文本模板列表，如 ["a photo of a {}", "a picture of a {}"]
            
        Returns:
            predicted_labels: 预测的类别标签
            probs: 各类别的概率
        """
        if templates is None:
            templates = ["a photo of a {}"]
        
        # 构建所有文本提示
        all_texts = []
        for classname in class_names:
            for template in templates:
                all_texts.append(template.format(classname))
        
        # 编码图像和文本
        image_features = self.encode_image(images)
        text_features = self.encode_text(all_texts)
        
        # 计算相似度
        similarity = self.compute_similarity(image_features, text_features)
        
        # 对每个类别取平均（如果使用多个模板）
        num_templates = len(templates)
        similarity_per_class = similarity.reshape(-1, len(class_names), num_templates).mean(dim=2)
        
        # 获取预测结果
        probs = similarity_per_class.softmax(dim=-1)
        predicted_indices = similarity_per_class.argmax(dim=-1)
        predicted_labels = [class_names[idx] for idx in predicted_indices]
        
        return predicted_labels, probs


def load_clip(
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str = "cuda"
) -> CLIPPredictor:
    """
    便捷函数：加载 CLIP 模型
    
    Args:
        model_name: 模型名称
        pretrained: 预训练权重来源
        device: 运行设备
        
    Returns:
        CLIPPredictor 实例
    """
    predictor = CLIPPredictor(
        model_name=model_name,
        pretrained=pretrained,
        device=device
    )
    predictor.load_model()
    return predictor
