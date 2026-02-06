from ultralytics import YOLO
import yaml
import os
from pathlib import Path
import matplotlib.pyplot as plt


def train_face_mask_model():
    """训练防护面罩检测模型"""

    # 数据集配置文件路径
    dataset_yaml = r"D:\数据集标注\防护面罩_yolo\face_mask_dataset.yaml"

    # 检查数据集配置文件是否存在
    if not Path(dataset_yaml).exists():
        print(f"❌ 数据集配置文件不存在: {dataset_yaml}")
        print("请先运行 prepare_face_mask_dataset.py 准备数据集")
        return None

    print(f"✅ 使用数据集配置: {dataset_yaml}")

    # 加载预训练模型
    print("🔄 加载预训练模型...")
    model = YOLO('yolov8n.pt')  # 使用nano版本，速度快

    # 训练参数配置
    training_config = {
        'data': dataset_yaml,
        'epochs': 100,  # 训练轮数
        'imgsz': 640,  # 输入图像尺寸
        'batch': 16,  # 批次大小
        'name': 'face_mask_detection',  # 实验名称
        'project': 'runs/train',  # 项目文件夹
        'save': True,  # 保存检查点
        'plots': True,  # 生成训练图表
        'val': True,  # 验证
        'patience': 50,  # 早停耐心值
        'device': 'cpu',  # 设备 ('cpu' 或 'cuda')
        'workers': 4,  # 数据加载器工作进程数
        'cache': False,  # 缓存数据集
        'optimizer': 'AdamW',  # 优化器
        'lr0': 0.01,  # 初始学习率
        'weight_decay': 0.0005,  # 权重衰减
        'warmup_epochs': 3,  # 预热轮数
        'box': 7.5,  # 边界框损失权重
        'cls': 0.5,  # 分类损失权重
        'dfl': 1.5,  # DFL损失权重
    }

    print("\n" + "=" * 60)
    print("🚀 开始训练防护面罩检测模型")
    print("=" * 60)
    print(f"📊 训练参数:")
    for key, value in training_config.items():
        print(f"   {key}: {value}")
    print("=" * 60)

    try:
        # 开始训练
        results = model.train(**training_config)

        print("\n" + "=" * 60)
        print("✅ 训练完成！")
        print("=" * 60)

        # 获取最佳模型路径
        best_model_path = Path('runs/train/face_mask_detection/weights/best.pt')
        last_model_path = Path('runs/train/face_mask_detection/weights/last.pt')

        print(f"📁 模型保存位置:")
        print(f"   最佳模型: {best_model_path}")
        print(f"   最后模型: {last_model_path}")

        # 显示训练结果
        results_dir = Path('runs/train/face_mask_detection')
        print(f"📊 训练结果文件夹: {results_dir}")

        # 检查结果文件
        result_files = [
            'results.png',  # 训练曲线
            'confusion_matrix.png',  # 混淆矩阵
            'val_batch0_pred.jpg',  # 验证预测示例
            'train_batch0.jpg',  # 训练批次示例
        ]

        print(f"📈 生成的结果文件:")
        for file in result_files:
            file_path = results_dir / file
            if file_path.exists():
                print(f"   ✅ {file}")
            else:
                print(f"   ❌ {file} (未生成)")

        return str(best_model_path)

    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_model(model_path, dataset_yaml):
    """验证模型性能"""
    print(f"\n🔍 验证模型性能...")

    try:
        # 加载训练好的模型
        model = YOLO(model_path)

        # 在验证集上评估
        results = model.val(data=dataset_yaml)

        print(f"📊 验证结果:")
        print(f"   mAP50: {results.box.map50:.4f}")
        print(f"   mAP50-95: {results.box.map:.4f}")
        print(f"   精确度: {results.box.mp:.4f}")
        print(f"   召回率: {results.box.mr:.4f}")

        return results

    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return None


if __name__ == "__main__":
    # 训练模型
    best_model_path = train_face_mask_model()

    if best_model_path:
        print(f"\n🎉 训练成功完成！")
        print(f"🎯 最佳模型路径: {best_model_path}")

        # 验证模型
        dataset_yaml = r"D:\数据集标注\防护面罩_yolo\face_mask_dataset.yaml"
        validate_model(best_model_path, dataset_yaml)

        print(f"\n📝 下一步:")
        print(f"1. 查看训练结果: runs/train/face_mask_detection/")
        print(f"2. 测试模型: python test_face_mask_model.py")
        print(f"3. 集成到系统: 修改 app.py 配置")
    else:
        print(f"❌ 训练失败，请检查数据集和配置")