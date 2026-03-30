"""
3D 物体级 SLAM 使用示例

演示如何使用 ObjectSLAMPipeline 从 RGB-D 图像序列构建 3D 场景图
"""

import numpy as np
import cv2
from pathlib import Path
import open3d as o3d


def example_rgbd_sequence_processing():
    """
    示例 1: 处理 RGB-D 图像序列
    
    模拟 ConceptGraphs 的流程，从 RGB-D 序列构建对象地图
    """
    print("=" * 60)
    print("示例 1: RGB-D 序列处理与 3D 场景图构建")
    print("=" * 60)
    
    from open_vocab import create_pipeline, create_object_slam_pipeline
    
    # Step 1: 创建 2D 检测流水线
    print("\n[1/4] 初始化 2D 检测流水线...")
    detection_pipeline = create_pipeline(
        grounding_dino_config="./weights/GroundingDINO_SwinT_OGC.py",
        grounding_dino_checkpoint="./weights/groundingdino_swint_ogc.pth",
        mobile_sam_checkpoint="./weights/mobile_sam.pt",
        device="cuda",
        use_clip=True,
    )
    
    # Step 2: 创建 3D SLAM 流水线
    print("[2/4] 初始化 3D SLAM 流水线...")
    slam_pipeline = create_object_slam_pipeline(
        detection_pipeline=detection_pipeline,
        voxel_size=0.05,  # 5cm 体素
        use_slam=False,   # 暂时不使用 SLAM，使用提供的位姿
    )
    
    # Step 3: 加载 RGB-D 数据
    # 这里假设有组织好的数据格式
    # 实际使用时需要替换为你的数据路径
    rgb_dir = Path("./rgbd_data/rgb")
    depth_dir = Path("./rgbd_data/depth")
    poses_file = Path("./rgbd_data/poses.txt")
    
    # 相机内参（示例为 RealSense D435）
    camera_intrinsics = np.array([
        [615.7, 0, 324.9],
        [0, 615.7, 241.8],
        [0, 0, 1]
    ])
    
    # Step 4: 处理每一帧
    print("[3/4] 处理 RGB-D 序列...")
    
    # 模拟处理多帧
    for frame_id in range(10):  # 处理前 10 帧
        # 读取 RGB 和深度图像
        rgb_path = rgb_dir / f"{frame_id:05d}.png"
        depth_path = depth_dir / f"{frame_id:05d}.png"
        
        if not rgb_path.exists() or not depth_path.exists():
            print(f"帧 {frame_id}: 文件不存在，跳过")
            continue
        
        rgb_image = cv2.imread(str(rgb_path))
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        depth_image = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        depth_image = depth_image.astype(np.float32) / 1000.0  # 转换为米
        
        # 读取相机位姿（如果有）
        camera_pose = None
        if poses_file.exists():
            # TODO: 从文件读取位姿
            pass
        
        # 处理帧
        results = slam_pipeline.process_rgbd_frame(
            rgb_image=rgb_image,
            depth_image=depth_image,
            camera_intrinsics=camera_intrinsics,
            camera_pose=camera_pose,
            classes=["chair", "table", "sofa", "monitor"],  # 要检测的类别
        )
        
        print(f"帧 {frame_id}: 检测到 {results['objects_detected']} 个对象，"
              f"地图中共有 {results['objects_in_map']} 个对象")
    
    # Step 5: 查看结果
    print("[4/4] 查看重建结果...")
    print(f"总对象数：{len(slam_pipeline.object_map.objects)}")
    print(f"关键帧数：{len(slam_pipeline.keyframes)}")
    
    # 打印每个对象的信息
    for obj_id, obj in slam_pipeline.object_map.objects.items():
        print(f"\n对象 {obj_id}:")
        print(f"  类别：{obj.label}")
        print(f"  质心：{obj.centroid}")
        print(f"  观测次数：{obj.observations}")
    
    # Step 6: 保存和可视化
    print("\n保存重建结果...")
    slam_pipeline.save_reconstruction("./scene_graph.json")
    
    print("可视化 3D 场景...")
    slam_pipeline.visualize()


def example_single_frame_3d():
    """
    示例 2: 单帧 RGB-D 的 3D 投影
    
    演示如何将单帧的 2D 检测结果投影到 3D 空间
    """
    print("\n" + "=" * 60)
    print("示例 2: 单帧 RGB-D 的 3D 投影")
    print("=" * 60)
    
    from open_vocab import create_pipeline, VoxelGrid, ObjectMap
    from open_vocab.models.multiview_fusion import MultiViewFusion, create_frame_observation
    
    # 创建检测流水线
    pipeline = create_pipeline(
        grounding_dino_config="./weights/GroundingDINO_SwinT_OGC.py",
        grounding_dino_checkpoint="./weights/groundingdino_swint_ogc.pth",
        mobile_sam_checkpoint="./weights/mobile_sam.pt",
        device="cuda",
    )
    
    # 读取 RGB-D 图像
    rgb_image = cv2.imread("your_rgb_image.jpg")
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    
    depth_image = cv2.imread("your_depth_image.png", cv2.IMREAD_UNCHANGED)
    depth_image = depth_image.astype(np.float32) / 1000.0
    
    # 相机参数
    intrinsics = np.array([
        [615.7, 0, 324.9],
        [0, 615.7, 241.8],
        [0, 0, 1]
    ])
    pose = np.eye(4)  # 单位矩阵
    
    # 执行 2D 检测
    results = pipeline.detect_and_segment(
        image=rgb_image,
        classes=["object"],
    )
    
    # 创建帧观测
    detections = [
        {'label': label, 'score': score, 'box': box}
        for label, score, box in zip(results['labels'], results['scores'], results['boxes'])
    ]
    
    observation = create_frame_observation(
        frame_id=0,
        rgb_image=rgb_image,
        depth_image=depth_image,
        camera_pose=pose,
        camera_intrinsics=intrinsics,
        detections=detections,
        masks=results['masks'],
    )
    
    # 多视图融合器
    fusion = MultiViewFusion()
    
    # 为每个掩码生成 3D 点云
    for i, mask in enumerate(observation.masks_2d):
        points_3d = fusion.project_mask_to_3d(
            mask=mask,
            depth=depth_image,
            intrinsics=intrinsics,
            extrinsics=pose
        )
        
        print(f"\n对象 {i}: {detections[i]['label']}")
        print(f"  3D 点数：{len(points_3d)}")
        
        if len(points_3d) > 0:
            centroid, bbox = fusion.compute_object_3d_bbox(points_3d)
            print(f"  质心：{centroid}")
            print(f"  包围盒：{bbox}")
            
            # 可视化点云
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3d)
            o3d.visualization.show_geometries([pcd])


def example_scene_graph_query():
    """
    示例 3: 场景图查询
    
    演示如何查询构建好的场景图
    """
    print("\n" + "=" * 60)
    print("示例 3: 场景图查询")
    print("=" * 60)
    
    from open_vocab import create_object_slam_pipeline
    
    # 假设已经有了重建好的场景
    # 加载已有的场景图
    slam_pipeline = None  # TODO: 加载已保存的场景
    
    if slam_pipeline is None:
        print("需要先运行示例 1 来构建场景")
        return
    
    # 查询场景图
    print("\n查询所有对象:")
    all_objects = slam_pipeline.query_scene_graph("all objects")
    print(f"共找到 {len(all_objects)} 个对象")
    
    print("\n查询椅子:")
    chairs = slam_pipeline.query_scene_graph("chairs")
    print(f"找到 {len(chairs)} 个椅子")
    
    print("\n查询桌子附近的对象:")
    near_table = slam_pipeline.query_scene_graph("objects near table")
    print(f"找到 {len(near_table)} 个对象")
    
    # 查看空间关系
    print("\n场景图关系:")
    for edge in slam_pipeline.scene_graph.edges:
        src_obj = slam_pipeline.object_map.objects[edge.source_id]
        tgt_obj = slam_pipeline.object_map.objects[edge.target_id]
        print(f"  {src_obj.label} --[{edge.relation_type}]--> {tgt_obj.label}")


def example_custom_dataset():
    """
    示例 4: 使用自定义数据集
    
    演示如何适配自己的 RGB-D 数据集格式
    """
    print("\n" + "=" * 60)
    print("示例 4: 自定义数据集适配")
    print("=" * 60)
    
    from open_vocab import create_pipeline, create_object_slam_pipeline
    
    # 创建流水线
    detection_pipeline = create_pipeline(
        grounding_dino_config="./weights/GroundingDINO_SwinT_OGC.py",
        grounding_dino_checkpoint="./weights/groundingdino_swint_ogc.pth",
        mobile_sam_checkpoint="./weights/mobile_sam.pt",
        device="cuda",
    )
    
    slam_pipeline = create_object_slam_pipeline(
        detection_pipeline=detection_pipeline,
        voxel_size=0.05,
        use_slam=False,
    )
    
    # 根据你的数据集格式调整
    # 以下是伪代码示例
    
    # dataset = YourCustomDataset("/path/to/dataset")
    # 
    # for frame in dataset:
    #     rgb = frame.get_rgb()
    #     depth = frame.get_depth()
    #     intrinsics = frame.get_intrinsics()
    #     pose = frame.get_pose()  # 如果有 groundtruth 位姿
    #     
    #     results = slam_pipeline.process_rgbd_frame(
    #         rgb_image=rgb,
    #         depth_image=depth,
    #         camera_intrinsics=intrinsics,
    #         camera_pose=pose,
    #         classes=["your_classes"],
    #     )
    
    print("请根据你的数据集格式修改此示例代码")


if __name__ == "__main__":
    print("3D 物体级 SLAM - 使用示例")
    print("=" * 60)
    print("\n可用的示例:")
    print("  1. example_rgbd_sequence_processing() - RGB-D 序列处理")
    print("  2. example_single_frame_3d() - 单帧 3D 投影")
    print("  3. example_scene_graph_query() - 场景图查询")
    print("  4. example_custom_dataset() - 自定义数据集")
    print("\n取消注释对应的函数调用来运行示例\n")
    
    # 运行示例（取消注释你想运行的示例）
    # example_rgbd_sequence_processing()
    # example_single_frame_3d()
    # example_scene_graph_query()
    # example_custom_dataset()
