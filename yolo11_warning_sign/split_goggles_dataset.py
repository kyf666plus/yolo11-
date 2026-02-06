"""
护目镜数据集划分脚本
将数据集按照 train:val:test = 7:2:1 的比例划分
"""
import os
import shutil
import random
from pathlib import Path


def split_goggles_dataset(
        images_dir="D:\\数据集标注\\护目镜\\images",
        labels_dir="D:\\数据集标注\\护目镜\\labels",
        output_dir="D:\\数据集标注\\护目镜\\dataset",
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42
):
    """
    划分护目镜数据集

    Args:
        images_dir: 图片文件夹路径
        labels_dir: 标签文件夹路径
        output_dir: 输出文件夹路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    """
    random.seed(seed)

    # 创建输出目录结构
    output_path = Path(output_dir)
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # 获取所有图片文件
    images_path = Path(images_dir)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(images_path.glob(ext))

    if not image_files:
        print(f"❌ 在 {images_dir} 中没有找到图片文件！")
        return

    print(f"📊 找到 {len(image_files)} 张图片")

    # 检查对应的标签文件
    valid_pairs = []
    labels_path = Path(labels_dir)

    for img_file in image_files:
        label_file = labels_path / f"{img_file.stem}.txt"
        if label_file.exists():
            valid_pairs.append((img_file, label_file))
        else:
            print(f"⚠️  警告: 图片 {img_file.name} 没有对应的标签文件")

    print(f"✅ 找到 {len(valid_pairs)} 对有效的图片-标签对")

    if not valid_pairs:
        print("❌ 没有找到有效的图片-标签对！")
        return

    # 随机打乱数据
    random.shuffle(valid_pairs)

    # 计算划分点
    total = len(valid_pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_pairs = valid_pairs[:train_end]
    val_pairs = valid_pairs[train_end:val_end]
    test_pairs = valid_pairs[val_end:]

    print(f"\n📁 数据集划分:")
    print(f"   训练集: {len(train_pairs)} ({len(train_pairs) / total * 100:.1f}%)")
    print(f"   验证集: {len(val_pairs)} ({len(val_pairs) / total * 100:.1f}%)")
    print(f"   测试集: {len(test_pairs)} ({len(test_pairs) / total * 100:.1f}%)")

    # 复制文件
    def copy_files(pairs, split_name):
        print(f"\n📋 正在复制 {split_name} 数据...")
        for img_file, label_file in pairs:
            # 复制图片
            dst_img = output_path / split_name / 'images' / img_file.name
            shutil.copy2(img_file, dst_img)

            # 复制标签
            dst_label = output_path / split_name / 'labels' / label_file.name
            shutil.copy2(label_file, dst_label)
        print(f"✅ {split_name} 数据复制完成")

    copy_files(train_pairs, 'train')
    copy_files(val_pairs, 'val')
    copy_files(test_pairs, 'test')

    # 创建 data.yaml 配置文件
    yaml_content = f"""# 护目镜检测数据集配置
path: {output_dir}  # 数据集根目录
train: train/images  # 训练集图片路径（相对于path）
val: val/images      # 验证集图片路径
test: test/images    # 测试集图片路径

# 类别
names:
  0: goggles  # 护目镜

# 类别数量
nc: 1
"""

    yaml_file = output_path / 'data.yaml'
    with open(yaml_file, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"\n✅ 数据集配置文件已创建: {yaml_file}")
    print(f"\n🎉 数据集划分完成！")
    print(f"📂 输出目录: {output_dir}")
    print("\n目录结构:")
    print("dataset/")
    print("├── data.yaml")
    print("├── train/")
    print("│   ├── images/")
    print("│   └── labels/")
    print("├── val/")
    print("│   ├── images/")
    print("│   └── labels/")
    print("└── test/")
    print("    ├── images/")
    print("    └── labels/")

    return str(yaml_file)


if __name__ == "__main__":
    print("🚀 开始护目镜数据集划分...\n")

    # 执行数据集划分
    yaml_path = split_goggles_dataset()

    if yaml_path:
        print(f"\n✨ 接下来可以使用以下命令训练模型:")
        print(f"   python train_goggles_model.py")