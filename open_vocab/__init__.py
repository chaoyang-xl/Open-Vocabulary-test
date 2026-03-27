"""
Open Vocabulary Detection and Segmentation
开放词汇检测与分割工具包

整合 GroundingDINO + MobileSAM + CLIP
"""

from .core import OpenVocabularyPipeline, create_pipeline
from .models.grounding_dino import GroundingDINOPredictor, load_grounding_dino
from .models.mobile_sam import MobileSAMPredictor, load_mobile_sam
from .models.clip_model import CLIPPredictor, load_clip

__version__ = "0.1.0"
__all__ = [
    "OpenVocabularyPipeline",
    "create_pipeline",
    "GroundingDINOPredictor",
    "MobileSAMPredictor", 
    "CLIPPredictor",
    "load_grounding_dino",
    "load_mobile_sam",
    "load_clip",
]
