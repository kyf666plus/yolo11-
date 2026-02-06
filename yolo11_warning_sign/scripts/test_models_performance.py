import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score
import pandas as pd

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class ModelTester:
    def __init__(self, model_path, model_name, class_names):
        self.model_path = model_path
        self.model_name = model_name
        self.class_names = class_names
        self.model = None
        self.load_model()

    def load_model(self):
        """加载模型"""
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ 成功加载模型: {self.model_name}")
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            self.model = None

    def parse_yolo_label(self, label_path, img_width, img_height):
        """解析YOLO格式标签"""
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        center_x = float(parts[1]) * img_width
                        center_y = float(parts[2]) * img_height
                        width = float(parts[3]) * img_width
                        height = float(parts[4]) * img_height

                        # 转换为左上角和右下角坐标
                        x1 = center_x - width / 2
                        y1 = center_y - height / 2
                        x2 = center_x + width / 2
                        y2 = center_y + height / 2

                        boxes.append([x1, y1, x2, y2, class_id])
        return boxes

    def calculate_iou(self, box1, box2):
        """计算两个边界框的IoU"""
        x1_1, y1_1, x2_1, y2_1 = box1[:4]
        x1_2, y1_2, x2_2, y2_2 = box2[:4]

        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def evaluate_on_test_set(self, test_images_dir, test_labels_dir, conf_threshold=0.25, iou_threshold=0.5):
        """在测试集上评估模型"""
        if self.model is None:
            print("❌ 模型未加载，无法进行评估")
            return None

        print(f"\n{'=' * 60}")
        print(f"开始评估模型: {self.model_name}")
        print(f"{'=' * 60}")
        print(f"测试图片目录: {test_images_dir}")
        print(f"测试标签目录: {test_labels_dir}")
        print(f"置信度阈值: {conf_threshold}")
        print(f"IoU阈值: {iou_threshold}")

        # 获取所有测试图片
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        test_images = []

        test_images_path = Path(test_images_dir)
        if not test_images_path.exists():
            print(f"❌ 测试图片目录不存在: {test_images_dir}")
            return None

        for ext in image_extensions:
            test_images.extend(list(test_images_path.glob(f'*{ext}')))
            test_images.extend(list(test_images_path.glob(f'*{ext.upper()}')))

        if len(test_images) == 0:
            print("❌ 没有找到测试图片")
            return None

        print(f"找到 {len(test_images)} 张测试图片")

        # 统计变量
        total_gt_boxes = 0  # 总的真实框数量
        total_pred_boxes = 0  # 总的预测框数量
        true_positives = 0  # 真正例
        false_positives = 0  # 假正例
        false_negatives = 0  # 假负例

        all_confidences = []
        all_labels = []  # 1表示正确检测，0表示错误检测

        results_detail = []

        print(f"\n开始处理图片...")
        for idx, img_path in enumerate(test_images):
            if (idx + 1) % 10 == 0:
                print(f"处理进度: {idx + 1}/{len(test_images)}")

            # 读取图片
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"⚠️  无法读取图片: {img_path.name}")
                continue

            img_height, img_width = img.shape[:2]

            # 获取对应的标签文件
            label_path = Path(test_labels_dir) / (img_path.stem + '.txt')
            gt_boxes = self.parse_yolo_label(label_path, img_width, img_height)
            total_gt_boxes += len(gt_boxes)

            # 模型预测
            try:
                results = self.model(img, conf=conf_threshold, device='cpu', verbose=False)

                pred_boxes = []
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        conf = boxes.conf[i].cpu().numpy()
                        cls = int(boxes.cls[i].cpu().numpy())
                        pred_boxes.append([x1, y1, x2, y2, cls, conf])

                total_pred_boxes += len(pred_boxes)

                # 匹配预测框和真实框
                matched_gt = set()
                matched_pred = set()

                for i, pred_box in enumerate(pred_boxes):
                    best_iou = 0
                    best_gt_idx = -1

                    for j, gt_box in enumerate(gt_boxes):
                        if j in matched_gt:
                            continue

                        # 检查类别是否匹配
                        if pred_box[4] == gt_box[4]:
                            iou = self.calculate_iou(pred_box, gt_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = j

                    # 记录置信度和标签
                    all_confidences.append(pred_box[5])

                    if best_iou >= iou_threshold and best_gt_idx != -1:
                        # 真正例
                        true_positives += 1
                        matched_gt.add(best_gt_idx)
                        matched_pred.add(i)
                        all_labels.append(1)
                    else:
                        # 假正例
                        false_positives += 1
                        all_labels.append(0)

                # 未匹配的真实框为假负例
                false_negatives += len(gt_boxes) - len(matched_gt)

                # 记录详细结果
                results_detail.append({
                    'image': img_path.name,
                    'gt_boxes': len(gt_boxes),
                    'pred_boxes': len(pred_boxes),
                    'matched': len(matched_gt),
                    'tp': len(matched_gt),
                    'fp': len(pred_boxes) - len(matched_pred),
                    'fn': len(gt_boxes) - len(matched_gt)
                })

            except Exception as e:
                print(f"⚠️  处理图片 {img_path.name} 时出错: {e}")
                continue

        print(f"✅ 图片处理完成")

        # 计算指标
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # 计算AP (Average Precision)
        ap = 0
        if len(all_confidences) > 0 and len(set(all_labels)) > 1:
            try:
                ap = average_precision_score(all_labels, all_confidences)
            except:
                ap = 0

        results = {
            'model_name': self.model_name,
            'total_test_images': len(test_images),
            'total_gt_boxes': total_gt_boxes,
            'total_pred_boxes': total_pred_boxes,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'average_precision': ap,
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'details': results_detail
        }

        return results

    def print_results(self, results):
        """打印评估结果"""
        if results is None:
            return

        print(f"\n{'=' * 60}")
        print(f"📊 模型评估结果: {results['model_name']}")
        print(f"{'=' * 60}")
        print(f"测试图片数量: {results['total_test_images']}")
        print(f"真实框总数: {results['total_gt_boxes']}")
        print(f"预测框总数: {results['total_pred_boxes']}")
        print(f"-" * 60)
        print(f"真正例 (TP): {results['true_positives']}")
        print(f"假正例 (FP): {results['false_positives']}")
        print(f"假负例 (FN): {results['false_negatives']}")
        print(f"-" * 60)
        print(f"✨ 精确率 (Precision): {results['precision']:.4f} ({results['precision'] * 100:.2f}%)")
        print(f"✨ 召回率 (Recall): {results['recall']:.4f} ({results['recall'] * 100:.2f}%)")
        print(f"✨ F1分数: {results['f1_score']:.4f} ({results['f1_score'] * 100:.2f}%)")
        print(f"✨ 平均精度 (AP): {results['average_precision']:.4f} ({results['average_precision'] * 100:.2f}%)")
        print(f"-" * 60)
        print(f"置信度阈值: {results['conf_threshold']}")
        print(f"IoU阈值: {results['iou_threshold']}")
        print(f"{'=' * 60}")


def save_results_to_file(results_list, output_file):
    """保存结果到JSON文件"""
    # 移除details字段以减小文件大小（可选）
    simplified_results = []
    for result in results_list:
        simplified = result.copy()
        # 保留details但只保留前10条
        if 'details' in simplified and len(simplified['details']) > 10:
            simplified['details'] = simplified['details'][:10]
            simplified['details_note'] = f"仅显示前10条，共{len(result['details'])}条"
        simplified_results.append(simplified)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(simplified_results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 详细结果已保存到: {output_file}")


def create_comparison_chart(results_list, output_dir):
    """创建对比图表"""
    if len(results_list) == 0:
        print("⚠️  没有结果可以生成图表")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 准备数据
    model_names = [r['model_name'] for r in results_list]
    precisions = [r['precision'] for r in results_list]
    recalls = [r['recall'] for r in results_list]
    f1_scores = [r['f1_score'] for r in results_list]
    aps = [r['average_precision'] for r in results_list]

    # 创建对比图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 设置颜色
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    # 精确率对比
    bars1 = ax1.bar(range(len(model_names)), precisions, color=colors[:len(model_names)], alpha=0.8, edgecolor='black')
    ax1.set_title('精确率 (Precision) 对比', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('精确率', fontsize=12)
    ax1.set_ylim(0, 1.1)
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=15, ha='right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    for i, (bar, val) in enumerate(zip(bars1, precisions)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{val:.3f}\n({val * 100:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 召回率对比
    bars2 = ax2.bar(range(len(model_names)), recalls, color=colors[:len(model_names)], alpha=0.8, edgecolor='black')
    ax2.set_title('召回率 (Recall) 对比', fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('召回率', fontsize=12)
    ax2.set_ylim(0, 1.1)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=15, ha='right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    for i, (bar, val) in enumerate(zip(bars2, recalls)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{val:.3f}\n({val * 100:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # F1分数对比
    bars3 = ax3.bar(range(len(model_names)), f1_scores, color=colors[:len(model_names)], alpha=0.8, edgecolor='black')
    ax3.set_title('F1分数对比', fontsize=16, fontweight='bold', pad=20)
    ax3.set_ylabel('F1分数', fontsize=12)
    ax3.set_ylim(0, 1.1)
    ax3.set_xticks(range(len(model_names)))
    ax3.set_xticklabels(model_names, rotation=15, ha='right')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    for i, (bar, val) in enumerate(zip(bars3, f1_scores)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{val:.3f}\n({val * 100:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 平均精度对比
    bars4 = ax4.bar(range(len(model_names)), aps, color=colors[:len(model_names)], alpha=0.8, edgecolor='black')
    ax4.set_title('平均精度 (AP) 对比', fontsize=16, fontweight='bold', pad=20)
    ax4.set_ylabel('平均精度', fontsize=12)
    ax4.set_ylim(0, 1.1)
    ax4.set_xticks(range(len(model_names)))
    ax4.set_xticklabels(model_names, rotation=15, ha='right')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    for i, (bar, val) in enumerate(zip(bars4, aps)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{val:.3f}\n({val * 100:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'models_comparison.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"📊 对比图表已保存到: {chart_path}")
    plt.close()


def create_summary_table(results_list, output_dir):
    """创建汇总表格"""
    if len(results_list) == 0:
        return

    # 准备表格数据
    table_data = []
    for result in results_list:
        table_data.append({
            '模型名称': result['model_name'],
            '测试图片': result['total_test_images'],
            '真实框': result['total_gt_boxes'],
            '预测框': result['total_pred_boxes'],
            'TP': result['true_positives'],
            'FP': result['false_positives'],
            'FN': result['false_negatives'],
            '精确率': f"{result['precision']:.4f}",
            '召回率': f"{result['recall']:.4f}",
            'F1分数': f"{result['f1_score']:.4f}",
            'AP': f"{result['average_precision']:.4f}"
        })

    df = pd.DataFrame(table_data)

    # 保存为CSV
    csv_path = os.path.join(output_dir, 'performance_summary.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"📋 性能汇总表已保存到: {csv_path}")

    # 创建表格图片
    fig, ax = plt.subplots(figsize=(16, len(results_list) * 0.8 + 2))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 设置表头样式
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 设置交替行颜色
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')

    plt.title('模型性能汇总表', fontsize=16, fontweight='bold', pad=20)

    table_path = os.path.join(output_dir, 'performance_table.png')
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    print(f"📊 性能表格图已保存到: {table_path}")
    plt.close()


def main():
    print("=" * 60)
    print("🚀 模型性能测试程序")
    print("=" * 60)

    # 模型配置 - 修改后的路径
    models_config = [
        {
            'model_path': 'runs/detect/runs/train/tripod_detector/weights/best.pt',
            'model_name': '三脚架检测器',
            'class_names': ['tripod'],
            'test_images_dir': 'datasets/tripod_split/test/images',
            'test_labels_dir': 'datasets/tripod_split/test/labels'
        },
        {
            'model_path': 'runs/detect/runs/train/safety_barrier_detector/weights/best.pt',
            'model_name': '安全防护栏检测器',
            'class_names': ['safety_barrier'],
            'test_images_dir': 'datasets/safety_barrier_split/test/images',
            'test_labels_dir': 'datasets/safety_barrier_split/test/labels'
        }
    ]

    all_results = []

    # 测试每个模型
    for config in models_config:
        print(f"\n{'=' * 60}")
        print(f"🔍 准备测试模型: {config['model_name']}")
        print(f"{'=' * 60}")

        # 检查模型文件是否存在
        if not os.path.exists(config['model_path']):
            print(f"❌ 模型文件不存在: {config['model_path']}")
            print(f"   请先训练该模型")
            continue

        # 检查测试数据是否存在
        if not os.path.exists(config['test_images_dir']):
            print(f"❌ 测试图片目录不存在: {config['test_images_dir']}")
            print(f"   请先运行数据集划分脚本")
            continue

        # 创建测试器
        tester = ModelTester(
            model_path=config['model_path'],
            model_name=config['model_name'],
            class_names=config['class_names']
        )

        # 评估模型
        results = tester.evaluate_on_test_set(
            test_images_dir=config['test_images_dir'],
            test_labels_dir=config['test_labels_dir'],
            conf_threshold=0.25,
            iou_threshold=0.5
        )

        if results:
            tester.print_results(results)
            all_results.append(results)

    # 保存和可视化结果
    if all_results:
        print(f"\n{'=' * 60}")
        print("📁 保存测试结果")
        print(f"{'=' * 60}")

        output_dir = 'evaluation_results'
        os.makedirs(output_dir, exist_ok=True)

        # 保存JSON结果
        save_results_to_file(all_results, os.path.join(output_dir, 'detailed_results.json'))

        # 创建对比图表
        create_comparison_chart(all_results, output_dir)

        # 创建汇总表格
        create_summary_table(all_results, output_dir)

        print(f"\n{'=' * 60}")
        print("✅ 所有模型测试完成!")
        print(f"{'=' * 60}")
        print("📂 结果文件位置:")
        print(f"   - 详细结果JSON: {output_dir}/detailed_results.json")
        print(f"   - 性能对比图: {output_dir}/models_comparison.png")
        print(f"   - 性能汇总CSV: {output_dir}/performance_summary.csv")
        print(f"   - 性能表格图: {output_dir}/performance_table.png")
        print(f"{'=' * 60}")
    else:
        print("\n❌ 没有成功测试任何模型")
        print("请检查:")
        print("1. 模型是否已训练完成")
        print("2. 数据集是否已正确划分")
        print("3. 路径配置是否正确")

if __name__ == "__main__":
    main()