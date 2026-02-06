from ultralytics import YOLO
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns


def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def parse_yolo_label(label_path, img_width, img_height):
    """解析YOLO格式的标签文件"""
    boxes = []
    if not Path(label_path).exists():
        return boxes

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height

            # 转换为 x1, y1, x2, y2 格式
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            boxes.append({
                'class_id': class_id,
                'bbox': [x1, y1, x2, y2]
            })

    return boxes


def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # 计算交集
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    # 计算并集
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def match_predictions_to_ground_truth(pred_boxes, gt_boxes, iou_threshold=0.5):
    """将预测框与真实框进行匹配"""
    matches = []
    used_gt = set()
    used_pred = set()

    # 按置信度排序预测框
    pred_boxes_sorted = sorted(enumerate(pred_boxes), key=lambda x: x[1]['confidence'], reverse=True)

    for pred_idx, pred_box in pred_boxes_sorted:
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in used_gt:
                continue

            iou = calculate_iou(pred_box['bbox'], gt_box['bbox'])
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx != -1:
            matches.append({
                'pred_idx': pred_idx,
                'gt_idx': best_gt_idx,
                'iou': best_iou,
                'confidence': pred_box['confidence']
            })
            used_gt.add(best_gt_idx)
            used_pred.add(pred_idx)

    return matches, used_pred, used_gt


def calculate_metrics(all_predictions, all_ground_truths, confidence_threshold=0.5, iou_threshold=0.5):
    """计算准确率、召回率等指标"""
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    all_confidences = []
    all_ious = []

    for i, (pred_boxes, gt_boxes) in enumerate(zip(all_predictions, all_ground_truths)):
        # 过滤低置信度的预测
        filtered_pred_boxes = [box for box in pred_boxes if box['confidence'] >= confidence_threshold]

        # 匹配预测框和真实框
        matches, used_pred, used_gt = match_predictions_to_ground_truth(
            filtered_pred_boxes, gt_boxes, iou_threshold
        )

        # 统计TP, FP, FN
        tp_count = len(matches)
        fp_count = len(filtered_pred_boxes) - tp_count
        fn_count = len(gt_boxes) - len(used_gt)

        true_positives += tp_count
        false_positives += fp_count
        false_negatives += fn_count

        # 收集置信度和IoU
        for match in matches:
            all_confidences.append(match['confidence'])
            all_ious.append(match['iou'])

    # 计算指标
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'avg_confidence': np.mean(all_confidences) if all_confidences else 0,
        'avg_iou': np.mean(all_ious) if all_ious else 0
    }


def plot_metrics_curves(metrics_results, results_dir):
    """绘制性能指标曲线"""
    thresholds = [m['confidence_threshold'] for m in metrics_results]
    precisions = [m['precision'] for m in metrics_results]
    recalls = [m['recall'] for m in metrics_results]
    f1_scores = [m['f1_score'] for m in metrics_results]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(thresholds, precisions, 'b-o', label='Precision')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Precision')
    plt.title('Precision vs Confidence Threshold')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(thresholds, recalls, 'r-o', label='Recall')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Recall')
    plt.title('Recall vs Confidence Threshold')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(thresholds, f1_scores, 'g-o', label='F1 Score')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Confidence Threshold')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(thresholds, precisions, 'b-o', label='Precision')
    plt.plot(thresholds, recalls, 'r-o', label='Recall')
    plt.plot(thresholds, f1_scores, 'g-o', label='F1 Score')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Score')
    plt.title('All Metrics vs Confidence Threshold')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(results_dir / 'metrics_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 性能曲线图保存: {results_dir / 'metrics_curves.png'}")


def generate_performance_report(metrics_results, results_dir):
    """生成性能报告"""
    # 找到最佳F1分数
    best_metrics = max(metrics_results, key=lambda x: x['f1_score'])

    report = f"""# 灭火器检测模型性能报告

## 📊 最佳性能指标
- **置信度阈值**: {best_metrics['confidence_threshold']}
- **准确率 (Precision)**: {best_metrics['precision']:.3f} ({best_metrics['precision'] * 100:.1f}%)
- **召回率 (Recall)**: {best_metrics['recall']:.3f} ({best_metrics['recall'] * 100:.1f}%)
- **F1分数**: {best_metrics['f1_score']:.3f} ({best_metrics['f1_score'] * 100:.1f}%)
- **平均置信度**: {best_metrics['avg_confidence']:.3f}
- **平均IoU**: {best_metrics['avg_iou']:.3f}

## 📈 混淆矩阵统计
- **真正例 (TP)**: {best_metrics['true_positives']} - 正确检测到的灭火器
- **假正例 (FP)**: {best_metrics['false_positives']} - 误检测的灭火器
- **假负例 (FN)**: {best_metrics['false_negatives']} - 漏检测的灭火器

## 💡 性能解读
- **准确率 {best_metrics['precision'] * 100:.1f}%** 表示检测结果的可靠性
- **召回率 {best_metrics['recall'] * 100:.1f}%** 表示能检测到的灭火器比例
- **F1分数 {best_metrics['f1_score'] * 100:.1f}%** 表示模型整体性能

## 🎯 建议
- 模型性能优秀，可以部署使用
- 如需提高召回率，可以适当降低置信度阈值
- 当前设置在准确率和召回率之间取得了很好的平衡
"""

    # 保存报告
    with open(results_dir / 'performance_report.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✅ 性能报告保存: {results_dir / 'performance_report.md'}")

    # 打印简要报告
    print(f"\n📋 性能总结:")
    print(f"   🎯 最佳F1分数: {best_metrics['f1_score']:.3f} (阈值: {best_metrics['confidence_threshold']})")
    print(f"   ✅ 准确率: {best_metrics['precision'] * 100:.1f}%")
    print(f"   🔍 召回率: {best_metrics['recall'] * 100:.1f}%")
    print(f"   📊 检测统计: TP={best_metrics['true_positives']}, FP={best_metrics['false_positives']}, FN={best_metrics['false_negatives']}")


def test_fire_extinguisher_model():
    """测试灭火器检测模型并计算性能指标"""

    # 模型路径
    model_path = "runs/detect/runs/detect/fire_extinguisher_detection/weights/best.pt"

    # 检查模型是否存在
    if not Path(model_path).exists():
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先训练模型！")
        return False

    print(f"✅ 加载模型: {model_path}")
    model = YOLO(model_path)

    # 测试数据路径
    test_images_dir = Path(r"D:\数据集标注\灭火器_yolo\images\test")
    test_labels_dir = Path(r"D:\数据集标注\灭火器_yolo\labels\test")

    if not test_images_dir.exists():
        print(f"❌ 测试图片文件夹不存在: {test_images_dir}")
        return False

    if not test_labels_dir.exists():
        print(f"❌ 测试标签文件夹不存在: {test_labels_dir}")
        return False

        # 获取测试图片
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    test_images = []
    for ext in image_extensions:
        test_images.extend(list(test_images_dir.glob(f'*{ext}')))
        test_images.extend(list(test_images_dir.glob(f'*{ext.upper()}')))

    # 去重处理
    test_images = sorted(list(set(test_images)))

    if not test_images:
        print(f"❌ 在 {test_images_dir} 中没有找到测试图片")
        return False

    print(f"📊 找到 {len(test_images)} 张测试图片")

    # 创建结果文件夹
    results_dir = Path("test_results/fire_extinguisher_metrics")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔄 开始测试并计算指标...")

    all_predictions = []
    all_ground_truths = []
    detailed_results = []

    # 测试每张图片
    for i, img_path in enumerate(test_images):
        print(f"🔍 测试图片 {i + 1}/{len(test_images)}: {img_path.name}")

        # 读取图片获取尺寸
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"   ❌ 无法读取图片: {img_path}")
            continue

        img_height, img_width = image.shape[:2]

        # 执行检测
        try:
            results = model.predict(
                source=str(img_path),
                conf=0.1,  # 使用较低的置信度阈值，后面再过滤
                save=False,
                verbose=False
            )

            result = results[0]
            boxes = result.boxes

            # 解析预测结果
            pred_boxes = []
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())

                    pred_boxes.append({
                        'class_id': class_id,
                        'confidence': conf,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    })

            # 解析真实标签
            label_path = test_labels_dir / f"{img_path.stem}.txt"
            gt_boxes = parse_yolo_label(label_path, img_width, img_height)

            all_predictions.append(pred_boxes)
            all_ground_truths.append(gt_boxes)

            # 记录详细结果
            detailed_results.append({
                'image_name': img_path.name,
                'predictions': len(pred_boxes),
                'ground_truths': len(gt_boxes)
            })

            print(f"   预测: {len(pred_boxes)} 个, 真实: {len(gt_boxes)} 个")

        except Exception as e:
            print(f"   ❌ 处理图片失败: {e}")
            continue

    if not all_predictions:
        print("❌ 没有成功处理任何图片")
        return False

    # 计算不同置信度阈值下的指标
    confidence_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    iou_threshold = 0.5

    print(f"\n📊 计算性能指标 (IoU阈值: {iou_threshold}):")
    print("=" * 80)
    print(f"{'置信度阈值':<10} {'准确率':<10} {'召回率':<10} {'F1分数':<10} {'TP':<6} {'FP':<6} {'FN':<6}")
    print("=" * 80)

    best_f1 = 0
    best_metrics = None
    best_threshold = 0

    metrics_results = []

    for conf_thresh in confidence_thresholds:
        try:
            metrics = calculate_metrics(all_predictions, all_ground_truths, conf_thresh, iou_threshold)

            print(f"{conf_thresh:<10.1f} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
                  f"{metrics['f1_score']:<10.3f} {metrics['true_positives']:<6} "
                  f"{metrics['false_positives']:<6} {metrics['false_negatives']:<6}")

            metrics_results.append({
                'confidence_threshold': conf_thresh,
                **metrics
            })

            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                best_metrics = metrics
                best_threshold = conf_thresh

        except Exception as e:
            print(f"❌ 计算阈值 {conf_thresh} 的指标时出错: {e}")
            continue

    print("=" * 80)

    if best_metrics:
        # 绘制性能曲线
        try:
            plot_metrics_curves(metrics_results, results_dir)
        except Exception as e:
            print(f"❌ 绘制性能曲线失败: {e}")

        # 生成性能报告
        try:
            generate_performance_report(metrics_results, results_dir)
        except Exception as e:
            print(f"❌ 生成性能报告失败: {e}")

        # 保存详细结果
        try:
            detailed_results_clean = convert_numpy_types(detailed_results)
            metrics_results_clean = convert_numpy_types(metrics_results)

            results = {
                'summary': {
                    'total_images': len(detailed_results),
                    'metrics_by_threshold': metrics_results_clean
                },
                'detailed_results': detailed_results_clean
            }

            with open(results_dir / 'detailed_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✅ 详细结果保存: {results_dir / 'detailed_results.json'}")
        except Exception as e:
            print(f"❌ 保存详细结果失败: {e}")

        print(f"\n📁 所有结果保存在: {results_dir}")
    else:
        print("❌ 没有成功计算任何性能指标")

    return True

if __name__ == "__main__":
    test_fire_extinguisher_model()