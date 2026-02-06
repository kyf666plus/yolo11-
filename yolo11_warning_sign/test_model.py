# test_model.py
from ultralytics import YOLO
from pathlib import Path
import cv2
import json
import numpy as np
from collections import defaultdict


def test_model_performance():
    """测试模型性能"""

    print("🧪 开始测试模型性能...")

    # 加载训练好的模型
    model_path = "D:/集成/yolo11_warning_sign/runs/detect/runs/detect/going_down_well_detection2/weights/best.pt"
    model = YOLO(model_path)

    # 测试集路径
    test_images = Path(r"D:\数据集标注\下井动作_yolo_format\images\test")
    test_labels = Path(r"D:\数据集标注\下井动作_yolo_format\labels\test")

    # 在测试集上验证
    print("\n📊 在测试集上评估...")
    metrics = model.val(
        data=r"D:\数据集标注\下井动作_yolo_format\data.yaml",
        split='test',
        save_json=True,
        save_hybrid=True
    )

    print(f"\n📈 测试集性能指标:")
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall: {metrics.box.mr:.4f}")
    # 在测试视频上推理
    print("\n🎥 在测试视频上推理...")
    test_videos = Path(r"D:\数据集标注\下井动作_yolo\videos\test")
    output_dir = Path("runs/detect/test_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    for video_file in test_videos.glob("*.mp4"):
        print(f"\n   处理视频: {video_file.name}")

        # 推理
        results = model.predict(
            source=str(video_file),
            save=True,
            project=str(output_dir),
            name=video_file.stem,
            conf=0.25,
            iou=0.45
        )

        print(f"   ✅ 结果保存在: {output_dir / video_file.stem}")
    print(f"\n✅ 测试完成！")
    print(f"📁 结果保存在: {output_dir}")

    return metrics


if __name__ == "__main__":
    test_model_performance()


