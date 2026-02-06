# train_model.py
from ultralytics import YOLO
from pathlib import Path


def train_going_down_well_model():
    """训练下井动作检测模型"""

    print("🚀 开始训练下井动作检测模型...")

    # 数据集配置文件路径
    data_yaml = r"D:\数据集标注\下井动作_yolo_format\data.yaml"

    # 加载预训练模型
    model = YOLO('yolo11n.pt')  # 使用nano模型，速度快

    # 训练参数
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        name='going_down_well_detection',
        patience=20,
        save=True,
        device='cpu',
        workers=4,
        project='runs/detect',

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

    print("\n✅ 训练完成！")
    print(f"📁 模型保存在: runs/detect/going_down_well_detection/weights/best.pt")

    return results


if __name__ == "__main__":
    train_going_down_well_model()