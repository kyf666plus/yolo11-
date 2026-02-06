"""
护目镜检测模型测试和评估脚本
计算准确率、召回率、mAP等指标
"""
from ultralytics import YOLO
import os
from pathlib import Path
import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import json


def test_goggles_model(
        model_path="D:/集成/yolo11_warning_sign/runs/detect/goggles_detection/yolo11n_goggles/weights/best.pt",
        data_yaml="D:\\数据集标注\\护目镜\\dataset\\data.yaml",
        conf_threshold=0.25,
        iou_threshold=0.45,
        device='cpu'
):
    """
    测试护目镜检测模型并计算各项指标

    Args:
        model_path: 训练好的模型路径
        data_yaml: 数据集配置文件
        conf_threshold: 置信度阈值
        iou_threshold: IOU阈值
        device: 设备
    """

    print("🧪 护目镜检测模型测试与评估")
    print("=" * 70)

    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到模型文件 {model_path}")
        print("请先运行 train_goggles_model.py 训练模型！")
        return

    # 检查数据集配置
    if not os.path.exists(data_yaml):
        print(f"❌ 错误: 找不到数据集配置文件 {data_yaml}")
        return

    print(f"📦 模型路径: {model_path}")
    print(f"📊 数据集配置: {data_yaml}")
    print(f"🎯 置信度阈值: {conf_threshold}")
    print(f"📐 IOU阈值: {iou_threshold}")
    print(f"🖥️  设备: {device.upper()}")
    print("=" * 70)

    # 加载模型
    print("\n📥 加载训练好的模型...")
    model = YOLO(model_path)
    print("✅ 模型加载成功\n")

    # 在验证集上评估
    print("🔍 在验证集上评估模型性能...")
    print("-" * 70)

    val_results = model.val(
        data=data_yaml,
        split='val',
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        plots=True,
        save_json=True
    )

    # 在测试集上评估
    print("\n" + "=" * 70)
    print("🧪 在测试集上评估模型性能...")
    print("-" * 70)

    test_results = model.val(
        data=data_yaml,
        split='test',
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        plots=True,
        save_json=True
    )

    # 打印详细结果
    print("\n" + "=" * 70)
    print("📊 评估结果汇总")
    print("=" * 70)

    def print_metrics(results, dataset_name):
        print(f"\n【{dataset_name}】")
        print("-" * 70)

        # mAP指标
        map50 = results.box.map50 if hasattr(results.box, 'map50') else 0
        map50_95 = results.box.map if hasattr(results.box, 'map') else 0

        # 精确率和召回率
        precision = results.box.mp if hasattr(results.box, 'mp') else 0
        recall = results.box.mr if hasattr(results.box, 'mr') else 0

        print(f"📈 mAP@0.5        : {map50:.4f} ({map50 * 100:.2f}%)")
        print(f"📈 mAP@0.5:0.95   : {map50_95:.4f} ({map50_95 * 100:.2f}%)")
        print(f"🎯 精确率 (Precision): {precision:.4f} ({precision * 100:.2f}%)")
        print(f"🎯 召回率 (Recall)   : {recall:.4f} ({recall * 100:.2f}%)")

        # F1分数
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            print(f"🎯 F1分数         : {f1:.4f} ({f1 * 100:.2f}%)")

        # 其他统计信息
        if hasattr(results, 'speed'):
            preprocess_time = results.speed.get('preprocess', 0)
            inference_time = results.speed.get('inference', 0)
            postprocess_time = results.speed.get('postprocess', 0)
            total_time = preprocess_time + inference_time + postprocess_time

            print(f"\n⏱️  推理速度:")
            print(f"   - 预处理   : {preprocess_time:.2f} ms")
            print(f"   - 推理     : {inference_time:.2f} ms")
            print(f"   - 后处理   : {postprocess_time:.2f} ms")
            print(f"   - 总耗时   : {total_time:.2f} ms")
            if total_time > 0:
                fps = 1000 / total_time
                print(f"   - FPS      : {fps:.2f}")

    print_metrics(val_results, "验证集 (Validation Set)")
    print_metrics(test_results, "测试集 (Test Set)")

    # 保存结果到文件
    results_dict = {
        'validation': {
            'mAP@0.5': float(val_results.box.map50) if hasattr(val_results.box, 'map50') else 0,
            'mAP@0.5:0.95': float(val_results.box.map) if hasattr(val_results.box, 'map') else 0,
            'precision': float(val_results.box.mp) if hasattr(val_results.box, 'mp') else 0,
            'recall': float(val_results.box.mr) if hasattr(val_results.box, 'mr') else 0,
        },
        'test': {
            'mAP@0.5': float(test_results.box.map50) if hasattr(test_results.box, 'map50') else 0,
            'mAP@0.5:0.95': float(test_results.box.map) if hasattr(test_results.box, 'map') else 0,
            'precision': float(test_results.box.mp) if hasattr(test_results.box, 'mp') else 0,
            'recall': float(test_results.box.mr) if hasattr(test_results.box, 'mr') else 0,
        }
    }

    # 计算F1分数
    for split in ['validation', 'test']:
        p = results_dict[split]['precision']
        r = results_dict[split]['recall']
        if p + r > 0:
            results_dict[split]['f1_score'] = 2 * (p * r) / (p + r)
        else:
            results_dict[split]['f1_score'] = 0

    # 保存到JSON文件
    output_dir = Path(model_path).parent.parent
    results_file = output_dir / 'evaluation_results.json'

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"💾 评估结果已保存到: {results_file}")

    # 给出性能评价
    print("\n" + "=" * 70)
    print("📝 模型性能评价")
    print("=" * 70)

    test_map50 = results_dict['test']['mAP@0.5']
    test_precision = results_dict['test']['precision']
    test_recall = results_dict['test']['recall']

    if test_map50 >= 0.9:
        performance = "🌟 优秀"
    elif test_map50 >= 0.8:
        performance = "✅ 良好"
    elif test_map50 >= 0.7:
        performance = "⚠️  一般"
    else:
        performance = "❌ 需要改进"

    print(f"\n总体评价: {performance}")
    print(f"\n建议:")

    if test_map50 < 0.7:
        print("  • 模型性能较低，建议:")
        print("    - 增加训练数据量")
        print("    - 增加训练轮数")
        print("    - 检查数据标注质量")
        print("    - 尝试数据增强")

    if test_precision < 0.8:
        print("  • 精确率偏低，存在较多误检，建议:")
        print("    - 提高置信度阈值")
        print("    - 增加负样本训练")

    if test_recall < 0.8:
        print("  • 召回率偏低，存在漏检，建议:")
        print("    - 降低置信度阈值")
        print("    - 增加困难样本训练")
        print("    - 检查是否有未标注的目标")

    if test_precision >= 0.8 and test_recall >= 0.8 and test_map50 >= 0.8:
        print("  • ✨ 模型性能良好，可以投入使用！")
        print("  • 建议进行实际场景测试以验证泛化能力")

    print("\n" + "=" * 70)
    print("✅ 测试评估完成！")
    print("=" * 70)

    return results_dict


if __name__ == "__main__":
    print("\n" + "🧪" * 35)
    print("  护目镜检测模型测试与评估")
    print("🧪" * 35 + "\n")

    # 执行测试
    results = test_goggles_model(
        conf_threshold=0.25,
        iou_threshold=0.45,
        device='cpu'
    )

    if results:
        print("\n✨ 测试完成！检查上方的详细评估结果。")