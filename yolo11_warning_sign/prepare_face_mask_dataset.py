import os
import shutil
from pathlib import Path
import random


def prepare_face_mask_dataset():
    """准备防护面罩数据集"""
    # 源数据路径
    source_images = Path(r"D:\数据集标注\防护面罩\images")
    source_labels = Path(r"D:\数据集标注\防护面罩\labels")

    # 目标数据集路径
    dataset_root = Path(r"D:\数据集标注\防护面罩_yolo")

    # 创建目标文件夹结构
    train_images = dataset_root / "images" / "train"
    val_images = dataset_root / "images" / "val"
    test_images = dataset_root / "images" / "test"
    train_labels = dataset_root / "labels" / "train"
    val_labels = dataset_root / "labels" / "val"
    test_labels = dataset_root / "labels" / "test"

    # 创建所有必要的文件夹
    for folder in [train_images, val_images, test_images, train_labels, val_labels, test_labels]:
        folder.mkdir(parents=True, exist_ok=True)

    print(f"✅ 创建数据集文件夹结构: {dataset_root}")

    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(list(source_images.glob(f'*{ext}')))
        image_files.extend(list(source_images.glob(f'*{ext.upper()}')))

    if not image_files:
        print(f"❌ 在 {source_images} 中没有找到图片文件")
        return False

    print(f"📊 找到 {len(image_files)} 张图片")

    # 检查对应的标签文件
    valid_pairs = []
    for img_file in image_files:
        label_file = source_labels / f"{img_file.stem}.txt"
        if label_file.exists():
            valid_pairs.append((img_file, label_file))
        else:
            print(f"⚠️  缺少标签文件: {label_file}")

    print(f"✅ 有效的图片-标签对: {len(valid_pairs)}")

    if len(valid_pairs) == 0:
        print("❌ 没有找到有效的图片-标签对")
        return False

    # 随机打乱数据
    random.shuffle(valid_pairs)

    # 按比例分割数据集 (70% 训练, 20% 验证, 10% 测试)
    total = len(valid_pairs)
    train_count = int(total * 0.7)
    val_count = int(total * 0.2)
    test_count = total - train_count - val_count

    train_pairs = valid_pairs[:train_count]
    val_pairs = valid_pairs[train_count:train_count + val_count]
    test_pairs = valid_pairs[train_count + val_count:]

    print(f"📊 数据集分割:")
    print(f"   训练集: {len(train_pairs)} 张")
    print(f"   验证集: {len(val_pairs)} 张")
    print(f"   测试集: {len(test_pairs)} 张")

    # 复制文件到对应文件夹
    def copy_files(pairs, img_dest, label_dest, split_name):
        print(f"🔄 复制 {split_name} 数据...")
        for img_file, label_file in pairs:
            # 复制图片
            shutil.copy2(img_file, img_dest / img_file.name)
            # 复制标签
            shutil.copy2(label_file, label_dest / label_file.name)

    copy_files(train_pairs, train_images, train_labels, "训练集")
    copy_files(val_pairs, val_images, val_labels, "验证集")
    copy_files(test_pairs, test_images, test_labels, "测试集")

    # 创建数据集配置文件
    yaml_content = f"""# 防护面罩检测数据集配置
path: {dataset_root.as_posix()}
train: images/train
val: images/val
test: images/test

# 类别数量
nc: 1

# 类别名称
names:
  0: face_mask  # 防护面罩
"""

    yaml_path = dataset_root / "face_mask_dataset.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"✅ 创建数据集配置文件: {yaml_path}")
    print(f"✅ 数据集准备完成！")

    return str(yaml_path)


if __name__ == "__main__":
    yaml_path = prepare_face_mask_dataset()
    if yaml_path:
        print(f"\n🎯 下一步可以使用以下配置文件训练模型:")
        print(f"   {yaml_path}")