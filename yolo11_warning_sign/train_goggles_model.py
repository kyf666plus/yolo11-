"""
YOLO11n 护目镜检测模型训练脚本
使用CPU进行训练
"""
from ultralytics import YOLO
import torch
import os
from pathlib import Path


def train_goggles_model(
        data_yaml="D:\\数据集标注\\护目镜\\dataset\\data.yaml",
        epochs=100,
        batch_size=16,
        img_size=640,
        device='cpu',
        project='goggles_detection',
        name='yolo11n_goggles'
):
    """
    训练护目镜检测模型

    Args:
        data_yaml: 数据集配置文件路径
        epochs: 训练轮数
        batch_size:批次大小
        img_size: 图片尺寸
        device: 设备 ('cpu' 或 'cuda')
        project: 项目名称
        name: 实验名称
    """

    print("🚀 护目镜检测模型训练")
    print("=" * 60)

    # 检查数据集配置文件
    if not os.path.exists(data_yaml):
        print(f"❌ 错误: 找不到数据集配置文件 {data_yaml}")
        print("请先运行 split_goggles_dataset.py 划分数据集！")
        return None

    print(f"📊 数据集配置: {data_yaml}")
    print(f"🖥️  设备: {device.upper()}")
    print(f"📷 图片尺寸: {img_size}x{img_size}")
    print(f"📦 批次大小: {batch_size}")
    print(f"🔄 训练轮数: {epochs}")
    print("=" * 60)

    # 加载YOLO11n模型
    print("\n📥 加载 YOLO11n 预训练模型...")
    model = YOLO('yolo11n.pt')  # 会自动下载预训练权重

    print(f"✅ 模型加载成功")
    print(f"📌 模型架构: YOLO11n")
    print(f"🎯 检测类别: 1 (护目镜)")

    # 开始训练
    print(f"\n🎓 开始训练...\n")

    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            project=project,
            name=name,
            patience=50,  # 早停耐心值
            save=True,  # 保存检查点
            save_period=10,  # 每10个epoch保存一次
            plots=True,  # 生成训练图表
            verbose=True,
            # 优化设置（适合CPU训练）
            workers=4,  # 数据加载线程
            optimizer='SGD',
            lr0=0.01,  # 初始学习率
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            # 数据增强
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
        )

        print("\n" + "=" * 60)
        print("✅ 训练完成！")
        print("=" * 60)

        # 显示训练结果路径
        save_dir = Path(project) / name
        print(f"\n📁 训练结果保存在: {save_dir}")
        print(f"   - 最佳模型: {save_dir / 'weights' / 'best.pt'}")
        print(f"   - 最后模型: {save_dir / 'weights' / 'last.pt'}")
        print(f"   - 训练曲线: {save_dir / 'results.png'}")
        print(f"   - 混淆矩阵: {save_dir / 'confusion_matrix.png'}")

        return str(save_dir / 'weights' / 'best.pt')

    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("\n" + "🔥" * 30)
    print("  YOLO11n 护目镜检测模型训练")
    print("🔥" * 30 + "\n")

    # 训练模型
    best_model_path = train_goggles_model(
        epochs=100,  # 可以根据需要调整
        batch_size=8,  # CPU训练建议使用较小的batch size
        img_size=640,
        device='cpu'
    )

    if best_model_path:
        print(f"\n✨ 训练成功完成！")
        print(f"📌 最佳模型路径: {best_model_path}")
        print(f"\n接下来可以运行测试脚本:")
        print(f"   python test_goggles_model.py")
    else:
        print(f"\n❌ 训练失败，请检查错误信息")