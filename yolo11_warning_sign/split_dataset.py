# split_dataset.py
import os
import shutil
from pathlib import Path
import random
from sklearn.model_selection import train_test_split


def split_dataset():
    """将视频和标注文件划分为训练集、验证集、测试集"""

    # 数据集路径
    source_dir = Path(r"D:\数据集标注\下井视频\【视频标注】有限空间下井")
    output_dir = Path(r"D:\数据集标注\下井动作_yolo")

    # 创建输出目录结构
    for split in ['train', 'val', 'test']:
        (output_dir / 'videos' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'annotations' / split).mkdir(parents=True, exist_ok=True)

    print("🎬 开始划分下井动作检测数据集...")
    print(f"📁 源数据目录: {source_dir}")
    print(f"📁 输出目录: {output_dir}")

    # 获取所有视频文件和JSON文件
    video_files = list(source_dir.glob("*.mp4"))
    json_files = list(source_dir.glob("*.json"))

    print(f"📹 找到 {len(video_files)} 个视频文件")
    print(f"📄 找到 {len(json_files)} 个JSON标注文件")

    if len(video_files) == 0:
        print("❌ 没有找到视频文件！")
        return False

    # 创建文件对（视频+对应的JSON）
    file_pairs = []
    video_dict = {vf.stem: vf for vf in video_files}
    json_dict = {jf.stem: jf for jf in json_files}

    for stem in video_dict.keys():
        if stem in json_dict:
            file_pairs.append({
                'stem': stem,
                'video': video_dict[stem],
                'json': json_dict[stem]
            })
        else:
            print(f"⚠️  视频文件 {stem} 没有对应的JSON标注")

    print(f"✅ 找到 {len(file_pairs)} 对匹配的文件")

    if len(file_pairs) == 0:
        print("❌ 没有找到匹配的文件对！")
        return False

    # 随机打乱
    random.seed(42)
    random.shuffle(file_pairs)

    # 划分数据集 (70% 训练, 20% 验证, 10% 测试)
    train_pairs, temp_pairs = train_test_split(file_pairs, test_size=0.3, random_state=42)
    val_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.33, random_state=42)

    print(f"\n📊 数据集划分:")
    print(f"   训练集: {len(train_pairs)} 对 ({len(train_pairs) / len(file_pairs) * 100:.1f}%)")
    print(f"   验证集: {len(val_pairs)} 对 ({len(val_pairs) / len(file_pairs) * 100:.1f}%)")
    print(f"   测试集: {len(test_pairs)} 对 ({len(test_pairs) / len(file_pairs) * 100:.1f}%)")

    # 复制文件到对应目录
    copy_files_to_split(train_pairs, output_dir, 'train')
    copy_files_to_split(val_pairs, output_dir, 'val')
    copy_files_to_split(test_pairs, output_dir, 'test')

    # 创建数据集配置文件
    create_config(output_dir, len(train_pairs), len(val_pairs), len(test_pairs))

    print(f"\n✅ 数据集划分完成！")
    print(f"📁 输出目录: {output_dir}")
    return True


def copy_files_to_split(file_pairs, output_dir, split_name):
    """复制文件到指定的划分目录"""
    print(f"\n📋 复制{split_name}数据...")

    video_dir = output_dir / 'videos' / split_name
    json_dir = output_dir / 'annotations' / split_name

    for i, pair in enumerate(file_pairs):
        try:
            # 复制视频文件
            video_dest = video_dir / pair['video'].name
            shutil.copy2(pair['video'], video_dest)

            # 复制JSON文件
            json_dest = json_dir / pair['json'].name
            shutil.copy2(pair['json'], json_dest)

            if (i + 1) % 5 == 0 or i == len(file_pairs) - 1:
                print(f"   已复制: {i + 1}/{len(file_pairs)}")
        except Exception as e:
            print(f"   ❌ 复制第{i + 1}对文件失败: {e}")


def create_config(output_dir, train_count, val_count, test_count):
    """创建数据集配置文件"""
    total_count = train_count + val_count + test_count

    config_content = f"""# 下井动作检测数据集配置
path: {output_dir.as_posix()}

train_videos: videos/train
val_videos: videos/val
test_videos: videos/test

train_annotations: annotations/train
val_annotations: annotations/val
test_annotations: annotations/test

nc: 1
names:
  0: going_down_well
"""

    config_path = output_dir / 'dataset_config.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)

    print(f"📄 配置文件已创建: {config_path}")

    # 创建统计文件
    stats_content = f"""# 数据集划分统计

总文件对数: {total_count}

训练集: {train_count} 对 ({train_count / total_count * 100:.1f}%)
验证集: {val_count} 对 ({val_count / total_count * 100:.1f}%)
测试集: {test_count} 对 ({test_count / total_count * 100:.1f}%)
"""

    stats_path = output_dir / 'split_stats.txt'
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(stats_content)

    print(f"📊 统计文件已创建: {stats_path}")


if __name__ == "__main__":
    split_dataset()