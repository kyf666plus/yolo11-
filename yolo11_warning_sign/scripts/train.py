# scripts/train.py
from ultralytics import YOLO
import os


def train_model():
    """训练YOLO11模型"""

    # 加载预训练模型
    model = YOLO('yolo11n.pt')  # 使用nano版本，速度快
    # 如果需要更高精度，可以使用: yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt

    # 训练参数
    results = model.train(
        data='datasets/warning_sign_split.yaml',  # 数据集配置文件
        epochs=100,  # 训练轮数
        imgsz=640,  # 图片大小
        batch=16,  # 批次大小（根据显存调整）
        device='cpu',  # 使用GPU 0，如果没有GPU使用'cpu'
        workers=4,  # 数据加载线程数
        project='runs/train',  # 保存路径
        name='warning_sign',  # 实验名称
        patience=50,  # 早停patience
        save=True,  # 保存检查点
        plots=True,  # 保存训练图表

        # 数据增强参数
        hsv_h=0.015,  # HSV色调增强
        hsv_s=0.7,  # HSV饱和度增强
        hsv_v=0.4,  # HSV明度增强
        degrees=0.0,  # 旋转角度
        translate=0.1,  # 平移
        scale=0.5,  # 缩放
        shear=0.0,  # 剪切
        perspective=0.0,  # 透视变换
        flipud=0.0,  # 上下翻转概率
        fliplr=0.5,  # 左右翻转概率
        mosaic=1.0,  # 马赛克增强
    )

    print("训练完成！")
    print(f"最佳模型保存在: {results.save_dir}")


if __name__ == "__main__":
    train_model()