import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_fire_extinguisher_dataset():
    """划分灭火器数据集为训练集、验证集和测试集"""
    # 数据集路径
    images_dir = Path(r"D:\数据集标注\灭火器\images")
    labels_dir = Path(r"D:\数据集标注\灭火器\labels")

    # 输出路径
    output_dir = Path(r"D:\数据集标注\灭火器_yolo")

    # 创建输出目录结构
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    print("🚒 开始划分灭火器数据集...")
    print(f"📁 原始图片目录: {images_dir}")
    print(f"📁 原始标签目录: {labels_dir}")
    print(f"📁 输出目录: {output_dir}")
    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(f'*{ext}')))
        image_files.extend(list(images_dir.glob(f'*{ext.upper()}')))

    print(f"📊 找到 {len(image_files)} 张图片")

    if len(image_files) == 0:
        print("❌ 没有找到图片文件！")
        return False

    # 检查对应的标签文件
    valid_pairs = []
    missing_labels = []

    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            valid_pairs.append((img_path, label_path))
        else:
            missing_labels.append(img_path.name)

    print(f"✅ 有效的图片-标签对: {len(valid_pairs)}")
    if missing_labels:
        print(f"⚠️  缺少标签的图片: {len(missing_labels)} 个")
        print("   前5个:", missing_labels[:5])

    if len(valid_pairs) == 0:
        print("❌ 没有找到有效的图片-标签对！")
        return False

    # 随机打乱数据
    random.seed(42)
    random.shuffle(valid_pairs)

    # 划分数据集 (70% 训练, 20% 验证, 10% 测试)
    train_pairs, temp_pairs = train_test_split(valid_pairs, test_size=0.3, random_state=42)
    val_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.33, random_state=42)  # 0.33 * 0.3 ≈ 0.1

    print(f"📊 数据集划分:")
    print(f"   训练集: {len(train_pairs)} 对 ({len(train_pairs) / len(valid_pairs) * 100:.1f}%)")
    print(f"   验证集: {len(val_pairs)} 对 ({len(val_pairs) / len(valid_pairs) * 100:.1f}%)")
    print(f"   测试集: {len(test_pairs)} 对 ({len(test_pairs) / len(valid_pairs) * 100:.1f}%)")

    # 复制文件到对应目录
    def copy_files(pairs, split_name):
        print(f"\n📋 复制{split_name}文件...")
        for i, (img_path, label_path) in enumerate(pairs):
            # 复制图片
            dst_img = output_dir / 'images' / split_name / img_path.name
            shutil.copy2(img_path, dst_img)

            # 复制标签
            dst_label = output_dir / 'labels' / split_name / label_path.name
            shutil.copy2(label_path, dst_label)
            if (i + 1) % 50 == 0 or i == len(pairs) - 1:
                print(f"   已复制: {i + 1}/{len(pairs)}")

    copy_files(train_pairs, 'train')
    copy_files(val_pairs, 'val')
    copy_files(test_pairs, 'test')

    # 创建数据集配置文件
    config_content = f"""# 灭火器检测数据集配置
path: {output_dir.as_posix()}  # 数据集根目录
train: images/train  # 训练图片路径
val: images/val      # 验证图片路径
test: images/test    # 测试图片路径

# 类别数量
nc: 1

# 类别名称
names:
  0: fire_extinguisher
"""

    config_path = output_dir / 'fire_extinguisher.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)

    print(f"\n✅ 数据集划分完成！")
    print(f"📁 输出目录: {output_dir}")
    print(f"📄 配置文件: {config_path}")
    print(f"🎯 可以开始训练模型了！")

    return True


if __name__ == "__main__":
    split_fire_extinguisher_dataset()