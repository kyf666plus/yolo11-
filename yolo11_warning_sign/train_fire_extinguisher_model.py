from ultralytics import YOLO
import torch
from pathlib import Path
import yaml


def train_fire_extinguisher_model():
    """训练灭火器检测模型"""

    print("🚒 开始训练灭火器检测模型")
    print("=" * 60)

    # 检查GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  使用设备: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    # 数据集配置文件路径
    data_config = r"D:\数据集标注\灭火器_yolo\fire_extinguisher.yaml"

    # 检查配置文件是否存在
    if not Path(data_config).exists():
        print(f"❌ 数据集配置文件不存在: {data_config}")
        print("请先运行数据集划分脚本！")
        return False

    # 加载预训练模型
    print("\n📥 加载预训练模型...")
    model = YOLO('yolo11n.pt')  # 使用YOLOv11 nano版本，速度快

    # 训练参数
    train_args = {
        'data': data_config,  # 数据集配置文件
        'epochs': 100,  # 训练轮数
        'imgsz': 640,  # 图片尺寸
        'batch': 16,  # 批次大小
        'device': device,  # 设备
        'workers': 4,  # 数据加载线程数
        'project': 'runs/detect',  # 项目目录
        'name': 'fire_extinguisher_detection',  # 实验名称
        'exist_ok': True,  # 允许覆盖现有实验
        'pretrained': True,  # 使用预训练权重
        'optimizer': 'AdamW',  # 优化器
        'lr0': 0.01,  # 初始学习率
        'lrf': 0.1,  # 最终学习率因子
        'momentum': 0.937,  # 动量
        'weight_decay': 0.0005,  # 权重衰减
        'warmup_epochs': 3,  # 预热轮数
        'warmup_momentum': 0.8,  # 预热动量
        'box': 7.5,  # 边界框损失权重
        'cls': 0.5,  # 分类损失权重
        'dfl': 1.5,  # DFL损失权重
        'save': True,  # 保存检查点
        'save_period': 10,  # 保存周期
        'cache': False,  # 不缓存图片到内存
        'close_mosaic': 10,  # 最后10轮关闭马赛克增强
        'resume': False,  # 不恢复训练
        'amp': True,  # 自动混合精度
        'fraction': 1.0,  # 使用全部数据
        'profile': False,  # 不进行性能分析
        'freeze': None,  # 不冻结层
        'multi_scale': False,  # 不使用多尺度训练
        'overlap_mask': True,  # 重叠掩码
        'mask_ratio': 4,  # 掩码比例
        'dropout': 0.0,  # Dropout
        'val': True,  # 验证
        'plots': True,  # 生成图表
        'verbose': True  # 详细输出
    }

    print("\n🏋️ 开始训练...")
    print("训练参数:")
    for key, value in train_args.items():
        print(f"   {key}: {value}")

    try:
        # 开始训练
        results = model.train(**train_args)

        print("\n✅ 训练完成！")
        print(f"📁 模型保存路径: runs/detect/fire_extinguisher_detection/weights/")
        print(f"🏆 最佳模型: runs/detect/fire_extinguisher_detection/weights/best.pt")
        print(f"📊 训练结果: runs/detect/fire_extinguisher_detection/")

        return True

    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    train_fire_extinguisher_model()