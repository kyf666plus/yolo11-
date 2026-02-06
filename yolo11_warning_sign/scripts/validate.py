# scripts/validate.py
from ultralytics import YOLO
import json
from pathlib import Path


def validate_model(model_path, data_yaml='datasets/warning_sign_split.yaml', save_json=True):
    """
    验证模型性能，输出详细指标

    Args:
        model_path: 模型路径
        data_yaml: 数据集配置文件
        save_json: 是否保存JSON格式的结果
    """

    # 加载训练好的模型
    model = YOLO(model_path)

    # 在验证集上评估
    metrics = model.val(
        data=data_yaml,
        imgsz=640,
        batch=16,
        device=0,
        conf=0.001,  # 降低置信度阈值以获得完整的PR曲线
        iou=0.6,  # IoU阈值
        save_json=save_json,
        plots=True,  # 生成混淆矩阵等图表
    )

    # 提取详细指标
    print("\n" + "=" * 60)
    print("模型性能评估报告")
    print("=" * 60)

    # 总体指标
    print("\n【总体指标】")
    print(f"类别数量: {metrics.box.nc}")
    print(f"图片数量: {metrics.box.n}")

    # mAP指标
    print("\n【mAP (Mean Average Precision)】")
    print(f"mAP@0.5      : {metrics.box.map50:.4f}  (IoU=0.5时的平均精度)")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}  (IoU从0.5到0.95的平均精度)")
    print(f"mAP@0.75     : {metrics.box.map75:.4f}  (IoU=0.75时的平均精度)")

    # 精确率 (Precision)
    print("\n【精确率 Precision】")
    print(f"Precision: {metrics.box.mp:.4f}")
    print("含义: 在所有预测为正例的样本中，真正为正例的比例")
    print("公式: TP / (TP + FP)")
    print("解释: 精确率高说明误检（False Positive）少")

    # 召回率 (Recall)
    print("\n【召回率 Recall】")
    print(f"Recall: {metrics.box.mr:.4f}")
    print("含义: 在所有真实正例中，被正确预测为正例的比例")
    print("公式: TP / (TP + FN)")
    print("解释: 召回率高说明漏检（False Negative）少")

    # F1分数
    if metrics.box.mp > 0 and metrics.box.mr > 0:
        f1_score = 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr)
        print("\n【F1分数】")
        print(f"F1-Score: {f1_score:.4f}")
        print("含义: 精确率和召回率的调和平均数")
        print("公式: 2 × (Precision × Recall) / (Precision + Recall)")
        print("解释: F1分数综合考虑了精确率和召回率")

    # 按类别的详细指标
    print("\n【各类别详细指标】")
    print(f"{'类别':<15} {'Precision':<12} {'Recall':<12} {'mAP@0.5':<12} {'mAP@0.5:0.95':<15}")
    print("-" * 70)

    # 获取类别名称
    class_names = model.names

    # 遍历每个类别
    for i in range(len(metrics.box.ap_class_index)):
        class_id = metrics.box.ap_class_index[i]
        class_name = class_names[int(class_id)]
        precision = metrics.box.p[i] if i < len(metrics.box.p) else 0
        recall = metrics.box.r[i] if i < len(metrics.box.r) else 0
        ap50 = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0
        ap = metrics.box.ap[i] if i < len(metrics.box.ap) else 0

        print(f"{class_name:<15} {precision:<12.4f} {recall:<12.4f} {ap50:<12.4f} {ap:<15.4f}")

    # 速度指标
    print("\n【推理速度】")
    print(f"预处理时间: {metrics.speed['preprocess']:.2f} ms")
    print(f"推理时间: {metrics.speed['inference']:.2f} ms")
    print(f"后处理时间: {metrics.speed['postprocess']:.2f} ms")

    print("\n" + "=" * 60)

    # 保存详细报告到文件
    report_path = Path(model_path).parent.parent / 'validation_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("模型性能评估报告\n")
        f.write("=" * 60 + "\n\n")

        f.write("【总体指标】\n")
        f.write(f"类别数量: {metrics.box.nc}\n")
        f.write(f"图片数量: {metrics.box.n}\n\n")

        f.write("【mAP指标】\n")
        f.write(f"mAP@0.5      : {metrics.box.map50:.4f}\n")
        f.write(f"mAP@0.5:0.95 : {metrics.box.map:.4f}\n")
        f.write(f"mAP@0.75     : {metrics.box.map75:.4f}\n\n")

        f.write("【精确率与召回率】\n")
        f.write(f"Precision: {metrics.box.mp:.4f}\n")
        f.write(f"Recall: {metrics.box.mr:.4f}\n")

        if metrics.box.mp > 0 and metrics.box.mr > 0:
            f1_score = 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr)
            f.write(f"F1-Score: {f1_score:.4f}\n\n")

        f.write("【各类别详细指标】\n")
        f.write(f"{'类别':<15} {'Precision':<12} {'Recall':<12} {'mAP@0.5':<12} {'mAP@0.5:0.95':<15}\n")
        f.write("-" * 70 + "\n")

        for i in range(len(metrics.box.ap_class_index)):
            class_id = metrics.box.ap_class_index[i]
            class_name = class_names[int(class_id)]
            precision = metrics.box.p[i] if i < len(metrics.box.p) else 0
            recall = metrics.box.r[i] if i < len(metrics.box.r) else 0
            ap50 = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0
            ap = metrics.box.ap[i] if i < len(metrics.box.ap) else 0

            f.write(f"{class_name:<15} {precision:<12.4f} {recall:<12.4f} {ap50:<12.4f} {ap:<15.4f}\n")

        f.write("\n【推理速度】\n")
        f.write(f"预处理时间: {metrics.speed['preprocess']:.2f} ms\n")
        f.write(f"推理时间: {metrics.speed['inference']:.2f} ms\n")
        f.write(f"后处理时间: {metrics.speed['postprocess']:.2f} ms\n")

    print(f"\n详细报告已保存至: {report_path}")

    # 如果需要保存JSON格式
    if save_json:
        json_report = {
            "overall": {
                "num_classes": int(metrics.box.nc),
                "num_images": int(metrics.box.n),
                "mAP@0.5": float(metrics.box.map50),
                "mAP@0.5:0.95": float(metrics.box.map),
                "mAP@0.75": float(metrics.box.map75),
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
            },
            "per_class": [],
            "speed": {
                "preprocess_ms": float(metrics.speed['preprocess']),
                "inference_ms": float(metrics.speed['inference']),
                "postprocess_ms": float(metrics.speed['postprocess'])
            }
        }

        # 添加F1分数
        if metrics.box.mp > 0 and metrics.box.mr > 0:
            json_report["overall"]["f1_score"] = float(
                2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr))

        # 添加各类别指标
        for i in range(len(metrics.box.ap_class_index)):
            class_id = metrics.box.ap_class_index[i]
            class_name = class_names[int(class_id)]

            class_metrics = {
                "class_name": class_name,
                "class_id": int(class_id),
                "precision": float(metrics.box.p[i]) if i < len(metrics.box.p) else 0,
                "recall": float(metrics.box.r[i]) if i < len(metrics.box.r) else 0,
                "mAP@0.5": float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else 0,
                "mAP@0.5:0.95": float(metrics.box.ap[i]) if i < len(metrics.box.ap) else 0,
            }
            json_report["per_class"].append(class_metrics)

        json_path = Path(model_path).parent.parent / 'validation_report.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=4, ensure_ascii=False)

        print(f"JSON报告已保存至: {json_path}")

    return metrics


def compare_models(model_paths, data_yaml='datasets/warning_sign_split.yaml'):
    """
    比较多个模型的性能

    Args:
        model_paths: 模型路径列表
        data_yaml: 数据集配置文件
    """
    results = []

    print("\n" + "=" * 80)
    print("多模型性能对比")
    print("=" * 80 + "\n")

    for model_path in model_paths:
        print(f"正在评估模型: {model_path}")
        model = YOLO(model_path)
        metrics = model.val(data=data_yaml, imgsz=640, batch=16, device=0, verbose=False)

        f1 = 0
        if metrics.box.mp > 0 and metrics.box.mr > 0:
            f1 = 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr)

        results.append({
            'model': Path(model_path).stem,
            'precision': metrics.box.mp,
            'recall': metrics.box.mr,
            'f1': f1,
            'mAP50': metrics.box.map50,
            'mAP50_95': metrics.box.map
        })

    # 打印对比表格
    print(f"\n{'模型':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'mAP@0.5':<12} {'mAP@0.5:0.95':<15}")
    print("-" * 95)

    for result in results:
        print(f"{result['model']:<20} {result['precision']:<12.4f} {result['recall']:<12.4f} "
              f"{result['f1']:<12.4f} {result['mAP50']:<12.4f} {result['mAP50_95']:<15.4f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # 验证单个模型
    model_path = 'runs/train/warning_sign/weights/best.pt'
    validate_model(model_path, save_json=True)

    # 如果要比较多个模型，取消下面的注释
    # model_paths = [
    #     'runs/train/warning_sign/weights/best.pt',
    #     'runs/train/warning_sign2/weights/best.pt',
    # ]
    # compare_models(model_paths)