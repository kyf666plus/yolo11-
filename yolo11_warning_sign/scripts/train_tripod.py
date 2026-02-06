import os
import shutil
import random
from pathlib import Path
import yaml
from ultralytics import YOLO
import torch


def read_classes_file(classes_file_path):
    """读取classes.txt文件"""
    classes = []
    if os.path.exists(classes_file_path):
        try:
            with open(classes_file_path, 'r', encoding='utf-8') as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
            print(f"读取到 {len(classes)} 个类别: {classes}")
        except Exception as e:
            print(f"读取类别文件出错: {e}")
    else:
        print(f"类别文件不存在: {classes_file_path}")

    return classes


def split_tripod_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    划分三脚架数据集
    """
    print("开始划分三脚架数据集...")

    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)

    # 检查输入目录是否存在
    if not images_path.exists():
        print(f"错误: 图片目录不存在 {images_dir}")
        return None

    if not labels_path.exists():
        print(f"错误: 标签目录不存在 {labels_dir}")
        return None

    # 创建输出目录结构
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            (output_path / split / subdir).mkdir(parents=True, exist_ok=True)

    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(list(images_path.glob(f'*{ext}')))
        image_files.extend(list(images_path.glob(f'*{ext.upper()}')))

    # 过滤出有对应标注文件的图片
    valid_images = []
    for img_file in image_files:
        # 在labels目录中查找对应的标注文件
        label_file = labels_path / (img_file.stem + '.txt')
        if label_file.exists():
            valid_images.append(img_file)

    print(f"找到 {len(valid_images)} 个有效的三脚架图片-标注对")

    if len(valid_images) == 0:
        print("错误: 没有找到有效的图片-标注对!")
        return None

    # 随机打乱
    random.shuffle(valid_images)

    # 计算划分点
    total = len(valid_images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    # 划分数据
    splits = {
        'train': valid_images[:train_end],
        'val': valid_images[train_end:val_end],
        'test': valid_images[val_end:]
    }

    # 复制文件
    for split_name, files in splits.items():
        print(f"处理 {split_name} 集: {len(files)} 个文件")

        for img_file in files:
            # 复制图片
            dst_img = output_path / split_name / 'images' / img_file.name
            shutil.copy2(img_file, dst_img)

            # 复制标注（从labels目录）
            label_file = labels_path / (img_file.stem + '.txt')
            dst_label = output_path / split_name / 'labels' / (img_file.stem + '.txt')
            shutil.copy2(label_file, dst_label)

    print(f"三脚架数据集划分完成:")
    print(f"  训练集: {len(splits['train'])} 个样本")
    print(f"  验证集: {len(splits['val'])} 个样本")
    print(f"  测试集: {len(splits['test'])} 个样本")

    return splits


def create_tripod_yaml(dataset_path, class_names, yaml_path):
    """
    创建三脚架数据集的YAML配置文件
    """
    config = {
        'path': str(dataset_path),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }

    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"三脚架配置文件已创建: {yaml_path}")


def train_tripod_model():
    """
    训练三脚架检测模型
    """
    print("\n开始训练三脚架检测模型")
    print("=" * 50)

    # 数据集路径 - 修正为正确的路径结构
    images_dir = r'D:\数据集标注\三脚架\images'
    labels_dir = r'D:\数据集标注\三脚架\labels'
    classes_file = r'D:\数据集标注\三脚架\classes.txt'
    output_dir = 'datasets/tripod_split'
    yaml_path = 'datasets/tripod.yaml'

    # 检查源目录
    if not os.path.exists(images_dir):
        print(f"错误: 图片目录不存在 {images_dir}")
        return None

    if not os.path.exists(labels_dir):
        print(f"错误: 标签目录不存在 {labels_dir}")
        return None

    # 设置随机种子
    random.seed(42)

    # 读取类别文件
    classes = read_classes_file(classes_file)
    if not classes:
        print("警告: 无法读取类别文件，使用默认类别名")
        classes = ['tripod']

    # 划分数据集
    splits = split_tripod_dataset(images_dir, labels_dir, output_dir)
    if splits is None:
        return None

    # 创建配置文件
    create_tripod_yaml(os.path.abspath(output_dir), classes, yaml_path)

    # 加载预训练模型
    model = YOLO('yolov8n.pt')

    # CPU优化的训练参数
    train_args = {
        'data': yaml_path,
        'epochs': 50,  # CPU训练减少轮数
        'imgsz': 416,  # 减小图像尺寸以适应CPU
        'batch': 4,  # CPU使用小批次
        'name': 'tripod_detector',
        'project': 'runs/train',
        'save': True,
        'save_period': 10,
        'cache': False,
        'device': 'cpu',
        'workers': 2,  # CPU使用较少worker
        'patience': 20,
        'optimizer': 'SGD',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'label_smoothing': 0.0,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'verbose': True
    }

    try:
        print("开始训练三脚架模型...")
        print("注意: CPU训练速度较慢，请耐心等待")

        results = model.train(**train_args)

        print("三脚架模型训练完成!")
        print(f"最佳权重保存在: runs/train/tripod_detector/weights/best.pt")

        return results

    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        return None


def main():
    print("三脚架检测模型训练程序")
    print("使用CPU进行训练")
    print("=" * 60)

    # 训练模型
    results = train_tripod_model()

    if results:
        print("\n训练成功完成!")
        print("可以在以下位置找到结果:")
        print("- 训练日志: runs/train/tripod_detector/")
        print("- 模型权重: runs/train/tripod_detector/weights/")
        print("- 训练图表: runs/train/tripod_detector/results.png")
    else:
        print("\n训练失败!")


if __name__ == "__main__":
    main()