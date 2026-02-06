import os
import shutil
import random
from pathlib import Path
import yaml


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


def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    划分数据集为训练集、验证集和测试集
    """
    # 确保比例总和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例总和必须为1"

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

    print(f"找到 {len(valid_images)} 个有效的图片-标注对")

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

    print(f"数据集划分完成:")
    print(f"  训练集: {len(splits['train'])} 个样本")
    print(f"  验证集: {len(splits['val'])} 个样本")
    print(f"  测试集: {len(splits['test'])} 个样本")

    return splits


def create_yaml_config(dataset_path, class_names, yaml_path):
    """
    创建YOLO格式的yaml配置文件
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

    print(f"配置文件已创建: {yaml_path}")


def main():
    # 设置随机种子
    random.seed(42)

    # 数据集配置
    datasets_config = {
        'tripod': {
            'images_dir': r'D:\数据集标注\三脚架\images',
            'labels_dir': r'D:\数据集标注\三脚架\labels',
            'classes_file': r'D:\数据集标注\三脚架\classes.txt',
            'output': 'datasets/tripod_split'
        },
        'safety_barrier': {
            'images_dir': r'D:\数据集标注\安全防护栏\images',
            'labels_dir': r'D:\数据集标注\安全防护栏\labels',
            'classes_file': r'D:\数据集标注\安全防护栏\classes.txt',
            'output': 'datasets/safety_barrier_split'
        }
    }

    for dataset_name, config in datasets_config.items():
        print(f"\n处理数据集: {dataset_name}")
        print("=" * 50)

        # 检查源目录是否存在
        if not os.path.exists(config['images_dir']):
            print(f"警告: 图片目录不存在 {config['images_dir']}")
            continue

        if not os.path.exists(config['labels_dir']):
            print(f"警告: 标签目录不存在 {config['labels_dir']}")
            continue

        # 读取类别文件
        classes = read_classes_file(config['classes_file'])
        if not classes:
            print(f"警告: 无法读取类别文件，使用默认类别名")
            classes = [dataset_name.replace('_', ' ')]

        # 划分数据集
        splits = split_dataset(
            images_dir=config['images_dir'],
            labels_dir=config['labels_dir'],
            output_dir=config['output'],
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1
        )

        if splits is None:
            continue

        # 创建yaml配置文件
        yaml_path = f"datasets/{dataset_name}.yaml"
        create_yaml_config(
            dataset_path=os.path.abspath(config['output']),
            class_names=classes,
            yaml_path=yaml_path
        )


if __name__ == "__main__":
    main()