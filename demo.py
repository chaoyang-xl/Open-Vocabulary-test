"""
开放词汇检测与分割演示脚本
"""
import os
import sys
import cv2
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from open_vocab import create_pipeline


def download_models():
    """下载所需的模型权重"""
    import urllib.request
    import shutil
    
    model_dir = Path("./weights")
    model_dir.mkdir(exist_ok=True)
    
    # 模型下载链接
    models = {
        "groundingdino_swint_ogc.pth": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        "mobile_sam.pt": "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
    }
    
    config_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    
    print("检查模型文件...")
    
    # 下载配置文件
    config_path = model_dir / "GroundingDINO_SwinT_OGC.py"
    if not config_path.exists():
        print(f"下载配置文件...")
        try:
            urllib.request.urlretrieve(config_url, config_path)
            print(f"✓ 配置文件已下载")
        except Exception as e:
            print(f"✗ 配置文件下载失败: {e}")
            print("请手动下载配置文件到 ./weights/ 目录")
    
    # 下载模型权重
    for model_name, url in models.items():
        model_path = model_dir / model_name
        if not model_path.exists():
            print(f"下载 {model_name}...")
            print(f"请手动从以下链接下载：")
            print(f"  {url}")
            print(f"并保存到: {model_path}")
        else:
            print(f"✓ {model_name} 已存在")
    
    return model_dir


def main():
    parser = argparse.ArgumentParser(description="开放词汇检测与分割演示")
    parser.add_argument("--image", "-i", type=str, help="输入图像路径")
    parser.add_argument("--classes", "-c", type=str, nargs="+", 
                        default=["person", "car", "dog", "cat"],
                        help="要检测的类别列表")
    parser.add_argument("--caption", type=str, help="使用自由文本描述检测")
    parser.add_argument("--output", "-o", type=str, default="output.jpg",
                        help="输出图像路径")
    parser.add_argument("--box_threshold", type=float, default=0.35,
                        help="检测框置信度阈值")
    parser.add_argument("--text_threshold", type=float, default=0.25,
                        help="文本匹配阈值")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="运行设备")
    parser.add_argument("--no_sam", action="store_true",
                        help="不使用 SAM 进行分割")
    parser.add_argument("--download", action="store_true",
                        help="下载模型权重")
    parser.add_argument("--classes_file", type=str, 
                        help="类别文件路径")
    
    args = parser.parse_args()
    
    if args.classes_file:
        with open(args.classes_file, "r") as f:
            # 读取类别文件 读取所有行 去掉空格和换行符
            args.classes = [line.strip() for line in f.readlines()]
            
    
    # 下载模型
    if args.download:
        download_models()
        return
    
    # 检查模型文件
    model_dir = Path("./weights")
    dino_config = model_dir / "GroundingDINO_SwinT_OGC.py"
    dino_checkpoint = model_dir / "groundingdino_swint_ogc.pth"
    sam_checkpoint = model_dir / "mobile_sam.pt"
    
    if not dino_checkpoint.exists() or not sam_checkpoint.exists():
        print("模型文件不存在，请先运行: python demo.py --download")
        print("然后手动下载模型权重到 ./weights/ 目录")
        return
    
    # 创建流水线
    print("\n初始化开放词汇检测流水线...")
    pipeline = create_pipeline(
        grounding_dino_config=str(dino_config),
        grounding_dino_checkpoint=str(dino_checkpoint),
        mobile_sam_checkpoint=str(sam_checkpoint),
        device=args.device,
        use_clip=True,
    )
    
    # 读取图像或使用默认图像
    if args.image:
        image_path = args.image
        if not os.path.exists(image_path):
            print(f"错误: 图像不存在 {image_path}")
            return
    else:
        # 创建一个测试图像
        print("未提供图像，创建测试图像...")
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        image_path = "test_image.jpg"
        cv2.imwrite(image_path, test_image)
        print(f"测试图像已保存到 {image_path}")
    
    # 执行检测
    print(f"\n处理图像: {image_path}")
    print("-" * 50)
    
    if args.caption:
        # 使用自由文本描述
        results = pipeline.detect_with_caption(
            image=image_path,
            caption=args.caption,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
    else:
        # 使用类别列表
        print(f"检测类别: {args.classes}")
        results = pipeline.detect_and_segment(
            image=image_path,
            classes=args.classes,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            use_sam=not args.no_sam,
        )
    
    # 保存结果
    output_image = cv2.cvtColor(results['visualization'], cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, output_image)
    print(f"\n结果已保存到: {args.output}")
    
    # 打印检测结果
    print("\n检测结果:")
    print("-" * 50)
    for i, (label, score) in enumerate(zip(results['labels'], results['scores'])):
        print(f"  {i+1}. {label}: {score:.3f}")
    
    print("\n完成!")


if __name__ == "__main__":
    import numpy as np
    main()
