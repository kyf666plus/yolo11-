# extract_frames_yolo_fixed_v2.py
import os
import json
import cv2
from pathlib import Path
import numpy as np


def extract_frames_and_convert():
    """从视频中提取帧并转换标注为YOLO格式"""

    dataset_dir = Path(r"D:\数据集标注\下井动作_yolo")
    yolo_dir = Path(r"D:\数据集标注\下井动作_yolo_format")

    # 创建YOLO格式目录
    for split in ['train', 'val', 'test']:
        (yolo_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    print("🎬 开始提取帧并转换为YOLO格式...")

    # 处理每个划分
    for split in ['train', 'val', 'test']:
        print(f"\n📋 处理{split}集...")

        video_dir = dataset_dir / 'videos' / split
        json_dir = dataset_dir / 'annotations' / split

        video_files = list(video_dir.glob("*.mp4"))
        total_frames = 0
        total_saved = 0

        for video_file in video_files:
            json_file = json_dir / f"{video_file.stem}.json"

            if not json_file.exists():
                print(f"   ⚠️  未找到标注文件: {json_file.name}")
                continue

            frames, saved = process_video_fixed(video_file, json_file, yolo_dir, split)
            total_frames += frames
            total_saved += saved

            if frames > 0:
                print(f"   ✅ {video_file.name}: 提取 {frames} 帧, 保存 {saved} 帧")
            else:
                print(f"   ⚠️  {video_file.name}: 提取 0 帧")

        print(f"   📊 {split}集总共提取: {total_frames} 帧, 成功保存: {total_saved} 帧")

    # 创建YOLO配置文件
    create_yolo_config(yolo_dir)

    print(f"\n✅ 数据集转换完成！")
    print(f"📁 YOLO格式数据集: {yolo_dir}")


def process_video_fixed(video_file, json_file, output_dir, split):
    """处理单个视频和标注（修复版）"""
    try:
        # 读取JSON标注
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # JSON是一个列表，需要找到对应的视频
        video_stem = video_file.stem
        sequence = None

        # 如果data是列表，遍历查找匹配的视频
        if isinstance(data, list):
            for item in data:
                file_upload = item.get('file_upload', '')
                # 匹配文件名
                if video_stem in file_upload or file_upload.replace('.mp4', '') == video_stem:
                    # 找到了对应的视频，提取sequence
                    if 'annotations' in item and item['annotations']:
                        annotation = item['annotations'][0]
                        if 'result' in annotation and annotation['result']:
                            result = annotation['result'][0]
                            if 'value' in result and 'sequence' in result['value']:
                                sequence = result['value']['sequence']
                                break
        # 如果data是字典（旧格式兼容）
        elif isinstance(data, dict):
            if 'annotations' in data and data['annotations']:
                annotation = data['annotations'][0]
                if 'result' in annotation and annotation['result']:
                    result = annotation['result'][0]
                    if 'value' in result and 'sequence' in result['value']:
                        sequence = result['value']['sequence']

        if not sequence:
            return 0, 0

        # 打开视频
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            return 0, 0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_count = 0
        saved_count = 0

        # 处理每个标注帧
        for ann in sequence:
            if not ann.get('enabled', True):
                continue

            frame_num = ann.get('frame', 0)
            if frame_num <= 0:
                continue

            # 读取帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1

            # 转换标注为YOLO格式
            yolo_ann = convert_to_yolo(ann, width, height)
            if not yolo_ann:
                continue

            # 保存图片 - 使用绝对路径并检查结果
            img_name = f"{video_file.stem}_frame_{frame_num:06d}.jpg"
            img_path = output_dir / 'images' / split / img_name

            # 🔧 关键修复：使用绝对路径并添加编码参数
            abs_img_path = str(img_path.absolute())

            # 方法1：使用 imencode + 文件写入（更可靠）
            try:
                # 编码为jpg
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                result, encimg = cv2.imencode('.jpg', frame, encode_param)

                if result:
                    # 写入文件
                    with open(abs_img_path, 'wb') as f:
                        f.write(encimg)

                    # 验证文件是否存在
                    if img_path.exists() and img_path.stat().st_size > 0:
                        saved_count += 1
                    else:
                        print(f"      ⚠️  图片保存失败: {img_name}")
                else:
                    print(f"      ⚠️  图片编码失败: {img_name}")

            except Exception as e:
                print(f"      ❌ 保存图片异常 {img_name}: {e}")
                # 尝试备用方法
                try:
                    success = cv2.imwrite(abs_img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if success and img_path.exists():
                        saved_count += 1
                except:
                    pass

            # 保存标注
            label_name = f"{video_file.stem}_frame_{frame_num:06d}.txt"
            label_path = output_dir / 'labels' / split / label_name
            with open(label_path, 'w') as f:
                f.write(yolo_ann)

        cap.release()
        return frame_count, saved_count

    except Exception as e:
        print(f"   ❌ 处理失败 {video_file.name}: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0


def convert_to_yolo(annotation, img_width, img_height):
    """将标注转换为YOLO格式"""
    try:
        x = annotation.get('x', 0)  # 百分比
        y = annotation.get('y', 0)
        width = annotation.get('width', 0)
        height = annotation.get('height', 0)

        if width <= 0 or height <= 0:
            return None

        # 转换为像素
        x_pixel = x * img_width / 100
        y_pixel = y * img_height / 100
        w_pixel = width * img_width / 100
        h_pixel = height * img_height / 100

        # 转换为YOLO格式（中心点坐标 + 归一化宽高）
        center_x = (x_pixel + w_pixel / 2) / img_width
        center_y = (y_pixel + h_pixel / 2) / img_height
        norm_w = w_pixel / img_width
        norm_h = h_pixel / img_height

        # 确保在[0,1]范围内
        center_x = max(0, min(1, center_x))
        center_y = max(0, min(1, center_y))
        norm_w = max(0, min(1, norm_w))
        norm_h = max(0, min(1, norm_h))

        return f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}"

    except Exception as e:
        return None


def create_yolo_config(output_dir):
    """创建YOLO配置文件"""
    config_content = f"""path: {output_dir.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: going_down_well
"""

    config_path = output_dir / 'data.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)

    print(f"📄 YOLO配置文件已创建: {config_path}")


if __name__ == "__main__":
    extract_frames_and_convert()