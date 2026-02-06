# scripts/split_dataset.py
import os
import shutil
from pathlib import Path
import random


def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.8):
    """
    将数据集划分为训练集和验证集

    Args:
        images_dir: 图片目录
        labels_dir: 标注目录
        output_dir: 输出目录
        train_ratio: 训练集比例
    """
    # 创建输出目录
    output_path = Path(output_dir)
    train_images = output_path / 'images' / 'train'
    train_labels = output_path / 'labels' / 'train'
    val_images = output_path / 'images' / 'val'
    val_labels = output_path / 'labels' / 'val'

    for dir_path in [train_images, train_labels, val_images, val_labels]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # 获取所有图片文件
    images_path = Path(images_dir)
    image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png')) + list(images_path.glob('*.jpeg'))

    # 打乱数据
    random.shuffle(image_files)

    # 计算划分点
    split_idx = int(len(image_files) * train_ratio)

    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    print(f"总共 {len(image_files)} 张图片")
    print(f"训练集: {len(train_files)} 张")
    print(f"验证集: {len(val_files)} 张")

    # 复制训练集
    for img_file in train_files:
        # 复制图片
        shutil.copy(img_file, train_images / img_file.name)

        # 复制对应的标注文件
        label_file = Path(labels_dir) / f"{img_file.stem}.txt"
        if label_file.exists():
            shutil.copy(label_file, train_labels / label_file.name)

    # 复制验证集
    for img_file in val_files:
        # 复制图片
        shutil.copy(img_file, val_images / img_file.name)

        # 复制对应的标注文件
        label_file = Path(labels_dir) / f"{img_file.stem}.txt"
        if label_file.exists():
            shutil.copy(label_file, val_labels / label_file.name)

    print("数据集划分完成！")


if __name__ == "__main__":
    images_dir = r"D:\数据集标注\警示标志\images"
    labels_dir = r"D:\数据集标注\警示标志\labels"
    output_dir = r"datasets/warning_sign_split"

    split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.8)