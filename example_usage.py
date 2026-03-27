"""
开放词汇检测与分割使用示例
展示如何在代码中集成使用
"""
import cv2
import numpy as np
from pathlib import Path

from open_vocab import create_pipeline


def example_basic_detection():
    """基础检测示例"""
    print("=" * 60)
    print("示例 1: 基础目标检测与分割")
    print("=" * 60)
    
    # 创建流水线（模型会自动加载）
    pipeline = create_pipeline(
        grounding_dino_config="./weights/GroundingDINO_SwinT_OGC.py",
        grounding_dino_checkpoint="./weights/groundingdino_swint_ogc.pth",
        mobile_sam_checkpoint="./weights/mobile_sam.pt",
        device="cuda",
        use_clip=True,
    )
    
    # 读取图像
    image_path = "your_image.jpg"  # 替换为你的图像路径
    
    # 检测指定类别
    results = pipeline.detect_and_segment(
        image=image_path,
        classes=["person", "car", "bicycle", "dog"],
        box_threshold=0.35,
        text_threshold=0.25,
        use_sam=True,
    )
    
    # 保存可视化结果
    output = cv2.cvtColor(results['visualization'], cv2.COLOR_RGB2BGR)
    cv2.imwrite("output_basic.jpg", output)
    
    print(f"检测到 {len(results['boxes'])} 个目标")
    for label, score in zip(results['labels'], results['scores']):
        print(f"  - {label}: {score:.3f}")


def example_free_text_detection():
    """自由文本描述检测示例"""
    print("\n" + "=" * 60)
    print("示例 2: 使用自由文本描述检测")
    print("=" * 60)
    
    pipeline = create_pipeline(
        grounding_dino_config="./weights/GroundingDINO_SwinT_OGC.py",
        grounding_dino_checkpoint="./weights/groundingdino_swint_ogc.pth",
        mobile_sam_checkpoint="./weights/mobile_sam.pt",
        device="cuda",
    )
    
    image_path = "your_image.jpg"
    
    # 使用自然语言描述
    results = pipeline.detect_with_caption(
        image=image_path,
        caption="red car on the street",
        box_threshold=0.3,
        text_threshold=0.2,
    )
    
    output = cv2.cvtColor(results['visualization'], cv2.COLOR_RGB2BGR)
    cv2.imwrite("output_caption.jpg", output)
    
    print(f"检测到 {len(results['boxes'])} 个目标")


def example_batch_processing():
    """批量处理示例"""
    print("\n" + "=" * 60)
    print("示例 3: 批量图像处理")
    print("=" * 60)
    
    pipeline = create_pipeline(
        grounding_dino_config="./weights/GroundingDINO_SwinT_OGC.py",
        grounding_dino_checkpoint="./weights/groundingdino_swint_ogc.pth",
        mobile_sam_checkpoint="./weights/mobile_sam.pt",
        device="cuda",
    )
    
    # 图像目录
    image_dir = Path("./images")
    output_dir = Path("./outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 类别列表
    classes = ["person", "vehicle", "animal"]
    
    # 批量处理
    for image_path in image_dir.glob("*.jpg"):
        print(f"处理: {image_path.name}")
        
        results = pipeline.detect_and_segment(
            image=str(image_path),
            classes=classes,
            box_threshold=0.35,
            text_threshold=0.25,
        )
        
        # 保存结果
        output_path = output_dir / f"result_{image_path.name}"
        output = cv2.cvtColor(results['visualization'], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), output)


def example_custom_classes():
    """自定义类别示例"""
    print("\n" + "=" * 60)
    print("示例 4: 检测任意自定义类别")
    print("=" * 60)
    
    pipeline = create_pipeline(
        grounding_dino_config="./weights/GroundingDINO_SwinT_OGC.py",
        grounding_dino_checkpoint="./weights/groundingdino_swint_ogc.pth",
        mobile_sam_checkpoint="./weights/mobile_sam.pt",
        device="cuda",
    )
    
    image_path = "your_image.jpg"
    
    # 可以检测任意类别，无需训练
    custom_classes = [
        "coffee cup",
        "laptop computer", 
        "wooden chair",
        "potted plant",
        "whiteboard marker"
    ]
    
    results = pipeline.detect_and_segment(
        image=image_path,
        classes=custom_classes,
        box_threshold=0.3,
        text_threshold=0.2,
    )
    
    output = cv2.cvtColor(results['visualization'], cv2.COLOR_RGB2BGR)
    cv2.imwrite("output_custom.jpg", output)
    
    print("自定义类别检测结果:")
    for label, score in zip(results['labels'], results['scores']):
        print(f"  - {label}: {score:.3f}")


def example_access_masks():
    """访问分割掩码示例"""
    print("\n" + "=" * 60)
    print("示例 5: 访问和使用分割掩码")
    print("=" * 60)
    
    pipeline = create_pipeline(
        grounding_dino_config="./weights/GroundingDINO_SwinT_OGC.py",
        grounding_dino_checkpoint="./weights/groundingdino_swint_ogc.pth",
        mobile_sam_checkpoint="./weights/mobile_sam.pt",
        device="cuda",
    )
    
    image_path = "your_image.jpg"
    
    results = pipeline.detect_and_segment(
        image=image_path,
        classes=["person"],
        box_threshold=0.35,
        text_threshold=0.25,
    )
    
    # 读取原始图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 遍历每个检测到的目标
    for i, (box, label, score, mask) in enumerate(
        zip(results['boxes'], results['labels'], results['scores'], results['masks'])
    ):
        print(f"目标 {i+1}: {label} (置信度: {score:.3f})")
        print(f"  检测框: {box}")
        print(f"  掩码形状: {mask.shape}")
        print(f"  掩码像素数: {mask.sum()}")
        
        # 提取目标区域
        masked_image = image.copy()
        masked_image[~mask] = [0, 0, 0]  # 背景变黑
        
        # 保存提取的目标
        output = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"extracted_object_{i+1}.jpg", output)


if __name__ == "__main__":
    print("开放词汇检测与分割 - 使用示例")
    print("=" * 60)
    print("\n使用前请确保:")
    print("1. 已安装依赖: pip install -r requirements.txt")
    print("2. 已下载模型权重到 ./weights/ 目录")
    print("3. 替换示例中的图像路径为你的实际图像")
    print("\n可用的示例函数:")
    print("  - example_basic_detection(): 基础检测")
    print("  - example_free_text_detection(): 自由文本检测")
    print("  - example_batch_processing(): 批量处理")
    print("  - example_custom_classes(): 自定义类别")
    print("  - example_access_masks(): 访问分割掩码")
    print("\n取消注释下方对应的函数调用来运行示例")
    
    # 取消注释你想运行的示例:
    # example_basic_detection()
    # example_free_text_detection()
    # example_batch_processing()
    # example_custom_classes()
    # example_access_masks()
