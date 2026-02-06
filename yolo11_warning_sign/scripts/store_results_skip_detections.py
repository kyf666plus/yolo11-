# scripts/store_results_minimal.py
import sys

sys.path.append('.')

import pandas as pd
import yaml
import json
import os
from pathlib import Path
from datetime import datetime
from database.db_manager import DatabaseManager


def store_training_results_to_db(
        results_dir="runs/detect/runs/train/warning_sign",
        project_name="警示标志检测",
        scene_type="warning_sign"
):
    """
    存储训练结果到数据库（方案3：最简化存储）
    """

    db = DatabaseManager()
    results_path = Path(results_dir).resolve()

    try:
        print("🗄️  开始存储训练结果到数据库...")
        print(f"📁 结果目录: {results_dir}")
        print(f"📁 绝对路径: {results_path}")
        print("ℹ️  注意：使用最简化存储方案，只存储关键路径")
        print("-" * 60)

        # 1. 检查必要文件
        model_file = results_path / "weights" / "best.pt"
        csv_file = results_path / "results.csv"
        args_file = results_path / "args.yaml"

        print("🔍 检查文件存在性:")
        print(f"   模型文件: {model_file} - {'✅' if model_file.exists() else '❌'}")
        print(f"   结果文件: {csv_file} - {'✅' if csv_file.exists() else '❌'}")
        print(f"   配置文件: {args_file} - {'✅' if args_file.exists() else '❌'}")

        if not model_file.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_file}")

        if not csv_file.exists():
            raise FileNotFoundError(f"结果文件不存在: {csv_file}")

        print(f"✅ 找到必要文件")

        # 2. 读取训练参数
        training_args = {}
        if args_file.exists():
            with open(args_file, 'r', encoding='utf-8') as f:
                training_args = yaml.safe_load(f)
            print(f"✅ 读取训练参数: {len(training_args)} 个参数")
            print(f"   训练轮数: {training_args.get('epochs', 'N/A')}")
            print(f"   图片尺寸: {training_args.get('imgsz', 'N/A')}")
            print(f"   批次大小: {training_args.get('batch', 'N/A')}")
        else:
            print("⚠️  未找到args.yaml，使用默认参数")
            training_args = {"epochs": 100, "imgsz": 640, "batch": 16}

        # 3. 读取CSV训练结果
        df = pd.read_csv(csv_file)
        print(f"✅ 读取训练结果: {len(df)} 轮训练")

        # 获取最后一轮的指标
        last_epoch = df.iloc[-1]

        # 提取关键指标
        training_metrics = {
            'total_epochs': len(df),
            'final_epoch': int(last_epoch['epoch']),
            'total_training_time': float(last_epoch['time']),

            # 最终性能指标
            'precision': float(last_epoch['metrics/precision(B)']),
            'recall': float(last_epoch['metrics/recall(B)']),
            'mAP@0.5': float(last_epoch['metrics/mAP50(B)']),
            'mAP@0.5:0.95': float(last_epoch['metrics/mAP50-95(B)']),

            # 最终损失
            'final_train_box_loss': float(last_epoch['train/box_loss']),
            'final_train_cls_loss': float(last_epoch['train/cls_loss']),
            'final_train_dfl_loss': float(last_epoch['train/dfl_loss']),
            'final_val_box_loss': float(last_epoch['val/box_loss']),
            'final_val_cls_loss': float(last_epoch['val/cls_loss']),
            'final_val_dfl_loss': float(last_epoch['val/dfl_loss']),

            # 学习率
            'final_learning_rate': float(last_epoch['lr/pg0']),
        }

        # 计算F1分数
        p, r = training_metrics['precision'], training_metrics['recall']
        if p > 0 and r > 0:
            training_metrics['f1_score'] = 2 * (p * r) / (p + r)
        else:
            training_metrics['f1_score'] = 0.0

        # 计算最佳指标（整个训练过程中的最高值）
        training_metrics['best_mAP50'] = float(df['metrics/mAP50(B)'].max())
        training_metrics['best_mAP50_95'] = float(df['metrics/mAP50-95(B)'].max())
        training_metrics['best_precision'] = float(df['metrics/precision(B)'].max())
        training_metrics['best_recall'] = float(df['metrics/recall(B)'].max())

        print(f"✅ 提取训练指标完成")
        print(f"   最终精确率: {training_metrics['precision']:.4f}")
        print(f"   最终召回率: {training_metrics['recall']:.4f}")
        print(f"   最终mAP@0.5: {training_metrics['mAP@0.5']:.4f}")
        print(f"   最终F1分数: {training_metrics['f1_score']:.4f}")

        # 4. 创建项目记录 (projects表)
        classes = {"0": "warning_sign"}

        project_id = db.create_project(
            name=project_name,
            scene_type=scene_type,
            model_path=str(model_file),
            classes=classes
        )

        print(f"✅ 项目已创建，ID: {project_id}")

        # 5. 创建训练任务记录 (detection_tasks表)
        task_id, task_uuid = db.create_detection_task(
            project_id=project_id,
            file_name="training_validation",
            file_type="image",
            file_path=training_args.get('data', 'datasets/warning_sign_split.yaml'),
            confidence=0.25
        )

        # 6. 更新任务状态为已完成
        processing_time = training_metrics['total_training_time']
        db.update_task_status(task_id, 'completed', processing_time=processing_time)

        print(f"✅ 训练任务已创建，ID: {task_id}, 处理时间: {processing_time:.2f}秒")

        # 7. 准备类别统计（使用实际的训练指标）
        class_counts = {
            "warning_sign": {
                "precision": training_metrics['precision'],
                "recall": training_metrics['recall'],
                "mAP@0.5": training_metrics['mAP@0.5'],
                "mAP@0.5:0.95": training_metrics['mAP@0.5:0.95'],
                "f1_score": training_metrics['f1_score'],
                "best_precision": training_metrics['best_precision'],
                "best_recall": training_metrics['best_recall'],
                "best_mAP@0.5": training_metrics['best_mAP50'],
                "best_mAP@0.5:0.95": training_metrics['best_mAP50_95'],
                "final_box_loss": training_metrics['final_val_box_loss'],
                "final_cls_loss": training_metrics['final_val_cls_loss'],
                "final_dfl_loss": training_metrics['final_val_dfl_loss'],
                "training_epochs": training_metrics['total_epochs'],
                "training_time_seconds": training_metrics['total_training_time']
            }
        }

        # 估算检测数量（基于验证集大小）
        estimated_val_images = training_args.get('batch', 16) * 10  # 估算值
        total_detections = estimated_val_images

        # 8. 创建最简化的结果URL信息（方案3）
        # 只存储基础目录和相对路径，总字符数控制在200以内
        minimal_result_info = {
            "base_dir": results_dir,  # 基础目录
            "model": "weights/best.pt",  # 相对路径
            "results": "results.csv",
            "config": "args.yaml" if args_file.exists() else None,
            "note": "完整文件列表见本地报告"
        }

        # 转换为JSON字符串并检查长度
        result_url_json = json.dumps(minimal_result_info, ensure_ascii=False)
        print(f"📏 result_url 长度: {len(result_url_json)} 字符 (限制: 500)")

        if len(result_url_json) > 500:
            # 如果还是太长，进一步简化
            minimal_result_info = {
                "dir": results_dir,
                "model": "weights/best.pt",
                "csv": "results.csv"
            }
            result_url_json = json.dumps(minimal_result_info, ensure_ascii=False)
            print(f"📏 简化后长度: {len(result_url_json)} 字符")

        # 9. 存储检测结果 (detection_results表)
        result_id = db.create_detection_result(
            task_id=task_id,
            width=training_args.get('imgsz', 640),
            height=training_args.get('imgsz', 640),
            total_frames=None,  # 图片训练不需要帧数
            total_detections=total_detections,
            class_counts=class_counts,
            result_url=result_url_json
        )

        print(f"✅ 检测结果已存储，结果ID: {result_id}")

        # 10. 统计所有文件信息（用于本地报告）
        def safe_relative_path(file_path):
            """安全地获取相对路径"""
            try:
                return str(file_path.relative_to(Path.cwd()))
            except ValueError:
                return str(file_path)

        all_result_files = {
            "model_files": [safe_relative_path(f) for f in results_path.glob("weights/*.pt")],
            "performance_charts": [safe_relative_path(f) for f in results_path.glob("*curve.png")],
            "confusion_matrix": [safe_relative_path(f) for f in results_path.glob("confusion_matrix*.png")],
            "training_data": [safe_relative_path(f) for f in results_path.glob("*.csv")],
            "config_files": [safe_relative_path(f) for f in results_path.glob("*.yaml")],
            "training_samples": [safe_relative_path(f) for f in results_path.glob("train_batch*.jpg")],
            "validation_samples": [safe_relative_path(f) for f in results_path.glob("val_batch*.jpg")],
            "labels_visualization": [safe_relative_path(f) for f in results_path.glob("labels.jpg")]
        }

        # 11. 保存完整报告到本地
        complete_report = {
            "database_storage_info": {
                "project_id": project_id,
                "task_id": task_id,
                "task_uuid": task_uuid,
                "result_id": result_id,
                "stored_tables": ["projects", "detection_tasks", "detection_results"],
                "skipped_tables": ["detections"],
                "skip_reason": "训练阶段无具体检测框数据，如需要可通过推理脚本补充",
                "storage_method": "minimal_path_storage"
            },
            "project_info": {
                "name": project_name,
                "scene_type": scene_type,
                "model_path": str(model_file),
                "classes": classes
            },
            "training_config": training_args,
            "training_metrics": training_metrics,
            "class_statistics": class_counts,
            "database_result_url": minimal_result_info,  # 数据库中存储的简化信息
            "complete_file_list": all_result_files,  # 完整的文件列表
            "training_history": {
                "epochs_data": df.to_dict('records')  # 保存完整的训练历史
            },
            "file_access_guide": {
                "model_file": str(model_file),
                "results_csv": str(csv_file),
                "config_file": str(args_file) if args_file.exists() else None,
                "results_directory": str(results_path)
            },
            "stored_at": datetime.now().isoformat()
        }

        # 保存报告
        report_dir = Path("runs/database_reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"training_report_{task_uuid}.json"

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(complete_report, f, indent=4, ensure_ascii=False)

        # 12. 打印详细的存储摘要
        print("\n" + "=" * 70)
        print("📊 数据库存储摘要")
        print("=" * 70)

        print(f"🗄️  存储状态:")
        print(f"   ✅ projects表: 已存储")
        print(f"   ✅ detection_tasks表: 已存储")
        print(f"   ✅ detection_results表: 已存储（最简化路径）")
        print(f"   ⏭️  detections表: 已跳过（训练阶段无具体检测框数据）")

        print(f"\n🏷️  项目信息:")
        print(f"   项目ID: {project_id}")
        print(f"   项目名称: {project_name}")
        print(f"   场景类型: {scene_type}")
        print(f"   模型路径: {model_file}")

        print(f"\n📋 任务信息:")
        print(f"   任务ID: {task_id}")
        print(f"   任务UUID: {task_uuid}")
        print(f"   结果ID: {result_id}")
        print(f"   处理时间: {processing_time:.2f}秒")

        print(f"\n🎯 训练配置:")
        print(f"   训练轮数: {training_metrics['total_epochs']}")
        print(f"   图片尺寸: {training_args.get('imgsz', 640)}")
        print(f"   批次大小: {training_args.get('batch', 16)}")
        print(f"   总训练时间: {training_metrics['total_training_time']:.2f}秒")

        print(f"\n📈 最终性能指标:")
        print(f"   精确率: {training_metrics['precision']:.4f}")
        print(f"   召回率: {training_metrics['recall']:.4f}")
        print(f"   F1分数: {training_metrics['f1_score']:.4f}")
        print(f"   mAP@0.5: {training_metrics['mAP@0.5']:.4f}")
        print(f"   mAP@0.5:0.95: {training_metrics['mAP@0.5:0.95']:.4f}")

        print(f"\n🏆 最佳性能指标:")
        print(f"   最佳精确率: {training_metrics['best_precision']:.4f}")
        print(f"   最佳召回率: {training_metrics['best_recall']:.4f}")
        print(f"   最佳mAP@0.5: {training_metrics['best_mAP50']:.4f}")
        print(f"   最佳mAP@0.5:0.95: {training_metrics['best_mAP50_95']:.4f}")

        print(f"\n📁 数据库存储的路径信息:")
        print(f"   基础目录: {minimal_result_info.get('base_dir', minimal_result_info.get('dir'))}")
        print(f"   模型文件: {minimal_result_info['model']}")
        print(f"   结果文件: {minimal_result_info.get('results', minimal_result_info.get('csv'))}")

        print(f"\n📁 完整文件统计:")
        total_files = 0
        for file_type, files in all_result_files.items():
            if files:
                print(f"   {file_type}: {len(files)} 个文件")
                total_files += len(files)
        print(f"   总计: {total_files} 个文件")

        print(f"\n📄 完整报告: {report_path}")

        print(f"\nℹ️  存储说明:")
        print(f"   - 数据库中存储简化路径信息，节省空间")
        print(f"   - 完整文件列表保存在本地JSON报告中")
        print(f"   - 可通过基础目录 + 相对路径访问所有文件")
        print(f"   - 如需具体检测框数据，请运行推理脚本")

        print("=" * 70)

        return complete_report

    except Exception as e:
        print(f"❌ 存储过程出错: {e}")
        if 'task_id' in locals():
            db.update_task_status(task_id, 'failed', error_msg=str(e))
        raise
    finally:
        db.close()


def get_full_path_from_db_info(db_result_info, relative_path):
    """
    根据数据库存储的信息和相对路径，获取完整路径

    Args:
        db_result_info: 从数据库result_url字段解析的JSON信息
        relative_path: 相对路径，如 "weights/best.pt"

    Returns:
        完整的文件路径
    """
    base_dir = db_result_info.get('base_dir') or db_result_info.get('dir')
    return str(Path(base_dir) / relative_path)


if __name__ == "__main__":
    print("🚀 开始存储训练结果到数据库（最简化路径存储）...")

    try:
        # 使用你的实际结果目录路径
        results_directory = "runs/detect/runs/train/warning_sign"

        # 检查目录是否存在
        if not os.path.exists(results_directory):
            print(f"❌ 结果目录不存在: {results_directory}")
            print("请确认路径是否正确")

            # 尝试查找可能的路径
            possible_paths = [
                "runs/train/warning_sign",
                "runs/detect/runs/train/warning_sign",
                "runs/train/exp/warning_sign"
            ]

            print("\n🔍 尝试查找可能的路径:")
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"   ✅ 找到: {path}")
                else:
                    print(f"   ❌ 不存在: {path}")

            exit(1)

        report = store_training_results_to_db(
            results_dir=results_directory,
            project_name="警示标志检测",
            scene_type="warning_sign"
        )

        print("\n🎉 存储完成！")
        print("✅ 训练结果已成功保存到数据库的3个表中")
        print("📋 数据库中存储简化路径，完整信息保存在本地报告中")
        print("🔧 可使用 get_full_path_from_db_info() 函数重构完整路径")

        # 演示如何使用存储的路径信息
        print("\n💡 路径重构示例:")
        db_info = report["database_result_url"]
        model_full_path = get_full_path_from_db_info(db_info, db_info["model"])
        print(f"   模型完整路径: {model_full_path}")

    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        print("请检查以下文件是否存在:")
        print("  - runs/detect/runs/train/warning_sign/weights/best.pt")
        print("  - runs/detect/runs/train/warning_sign/results.csv")
        print("  - runs/detect/runs/train/warning_sign/args.yaml")

    except Exception as e:
        print(f"❌ 存储失败: {e}")
        import traceback

        traceback.print_exc()