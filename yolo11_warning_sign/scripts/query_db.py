# scripts/query_db.py
import sys

sys.path.append('.')

import json
from pathlib import Path
from database.db_manager import DatabaseManager


def safe_get_stats(stats, key, default=0):
    """安全地获取统计数据"""
    if isinstance(stats, dict):
        return stats.get(key, default)
    elif isinstance(stats, (int, float)):
        return stats if key == 'count' else default
    else:
        return default


def query_stored_results():
    """查询存储的训练结果"""

    db = DatabaseManager()

    try:
        print("🔍 查询数据库中的训练结果...")
        print("=" * 60)

        # 1. 查询所有项目
        projects = db.execute_query("SELECT * FROM projects ORDER BY created_at DESC")

        if not projects:
            print("❌ 数据库中没有找到任何项目")
            return

        print(f"📊 找到 {len(projects)} 个项目:")

        for project in projects:
            project_id, name, scene_type, model_path, classes_json, status, created_at = project

            try:
                classes = json.loads(classes_json)
            except (json.JSONDecodeError, TypeError):
                classes = {"unknown": "unknown"}

            print(f"\n🏷️  项目 #{project_id}: {name}")
            print(f"   场景类型: {scene_type}")
            print(f"   模型路径: {model_path}")
            print(f"   类别: {classes}")
            print(f"   状态: {'启用' if status else '禁用'}")
            print(f"   创建时间: {created_at}")

            # 2. 查询该项目的任务
            tasks = db.execute_query(
                "SELECT * FROM detection_tasks WHERE project_id = %s ORDER BY created_at DESC",
                (project_id,)
            )

            print(f"   📋 任务数量: {len(tasks)}")

            for task in tasks:
                task_db_id, task_uuid, proj_id, file_name, file_type, file_path, confidence, status, error_msg, created_at, completed_at, processing_time = task

                print(f"      任务 #{task_db_id} ({task_uuid[:8]}...)")
                print(f"      文件: {file_name} ({file_type})")
                print(f"      状态: {status}")
                if processing_time:
                    print(f"      处理时间: {processing_time:.2f}秒")
                if error_msg:
                    print(f"      错误信息: {error_msg}")

                # 3. 查询该任务的结果
                results = db.execute_query(
                    "SELECT * FROM detection_results WHERE task_id = %s",
                    (task_db_id,)
                )

                for result in results:
                    result_id, task_id, width, height, total_frames, total_detections, class_counts_json, result_url, created_at = result

                    print(f"         📈 结果 #{result_id}:")
                    print(f"         尺寸: {width}x{height}")
                    print(f"         检测数: {total_detections}")

                    # 解析类别统计（处理不同的数据格式）
                    if class_counts_json:
                        try:
                            class_counts = json.loads(class_counts_json)

                            if isinstance(class_counts, dict):
                                for class_name, stats in class_counts.items():
                                    print(f"         {class_name} 性能:")

                                    # 处理不同格式的统计数据
                                    if isinstance(stats, dict):
                                        # 新格式：包含详细统计
                                        print(f"           精确率: {safe_get_stats(stats, 'precision'):.4f}")
                                        print(f"           召回率: {safe_get_stats(stats, 'recall'):.4f}")
                                        print(f"           mAP@0.5: {safe_get_stats(stats, 'mAP@0.5'):.4f}")
                                        print(f"           F1分数: {safe_get_stats(stats, 'f1_score'):.4f}")

                                        # 显示最佳指标（如果有）
                                        if 'best_precision' in stats:
                                            print(
                                                f"           最佳精确率: {safe_get_stats(stats, 'best_precision'):.4f}")
                                            print(f"           最佳召回率: {safe_get_stats(stats, 'best_recall'):.4f}")
                                            print(
                                                f"           最佳mAP@0.5: {safe_get_stats(stats, 'best_mAP@0.5'):.4f}")

                                        # 显示训练信息（如果有）
                                        if 'training_epochs' in stats:
                                            print(f"           训练轮数: {safe_get_stats(stats, 'training_epochs')}")
                                            print(
                                                f"           训练时间: {safe_get_stats(stats, 'training_time_seconds'):.2f}秒")

                                    elif isinstance(stats, (int, float)):
                                        # 旧格式：只有数量
                                        print(f"           检测数量: {stats}")

                                    else:
                                        print(f"           数据: {stats}")

                            else:
                                print(f"         统计数据: {class_counts}")

                        except (json.JSONDecodeError, TypeError) as e:
                            print(f"         统计数据解析错误: {e}")
                            print(f"         原始数据: {class_counts_json}")

                    # 解析结果URL信息
                    if result_url:
                        try:
                            url_info = json.loads(result_url)
                            print(f"         📁 文件信息:")

                            # 处理不同格式的URL信息
                            base_dir = url_info.get('base_dir') or url_info.get('dir')
                            if base_dir:
                                print(f"           基础目录: {base_dir}")

                            model_file = url_info.get('model')
                            if model_file:
                                print(f"           模型文件: {model_file}")

                                # 构建完整路径示例
                                if base_dir:
                                    full_model_path = str(Path(base_dir) / model_file)
                                    print(f"           模型完整路径: {full_model_path}")

                            results_file = url_info.get('results') or url_info.get('csv')
                            if results_file:
                                print(f"           结果文件: {results_file}")

                            config_file = url_info.get('config')
                            if config_file:
                                print(f"           配置文件: {config_file}")

                            # 显示其他信息
                            if 'note' in url_info:
                                print(f"           说明: {url_info['note']}")

                            # 如果是旧格式，显示文件列表
                            if 'model_files' in url_info:
                                model_files = url_info.get('model_files', [])
                                if model_files:
                                    print(f"           模型文件列表: {len(model_files)} 个")
                                    for mf in model_files[:3]:  # 只显示前3个
                                        print(f"             - {mf}")
                                    if len(model_files) > 3:
                                        print(f"             ... 还有 {len(model_files) - 3} 个文件")

                        except json.JSONDecodeError:
                            print(f"         📁 文件信息: {result_url}")
                        except Exception as e:
                            print(f"         📁 文件信息解析错误: {e}")

        print("\n" + "=" * 60)
        print("✅ 查询完成")

    except Exception as e:
        print(f"❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        db.close()


def get_project_files(project_id):
    """获取指定项目的所有文件路径"""

    db = DatabaseManager()

    try:
        # 查询项目的结果信息
        query = """
        SELECT dr.result_url 
        FROM detection_results dr
        JOIN detection_tasks dt ON dr.task_id = dt.id
        WHERE dt.project_id = %s
        """

        results = db.execute_query(query, (project_id,))

        if not results:
            print(f"❌ 项目 #{project_id} 没有找到结果")
            return None

        # 解析结果URL
        result_url = results[0][0]

        try:
            url_info = json.loads(result_url)
        except json.JSONDecodeError:
            print(f"❌ 无法解析项目 #{project_id} 的URL信息")
            return None

        base_dir = url_info.get('base_dir') or url_info.get('dir')

        if not base_dir:
            print(f"❌ 项目 #{project_id} 没有基础目录信息")
            return None

        # 构建完整路径
        file_paths = {
            'base_directory': base_dir,
            'model_file': str(Path(base_dir) / url_info['model']) if url_info.get('model') else None,
            'results_csv': str(Path(base_dir) / (url_info.get('results') or url_info.get('csv'))) if url_info.get(
                'results') or url_info.get('csv') else None,
            'config_file': str(Path(base_dir) / url_info['config']) if url_info.get('config') else None
        }

        return file_paths

    except Exception as e:
        print(f"❌ 获取项目文件失败: {e}")
        return None
    finally:
        db.close()


def clean_failed_tasks():
    """清理失败的任务（可选功能）"""

    db = DatabaseManager()

    try:
        # 查询失败的任务
        failed_tasks = db.execute_query("SELECT id, task_id FROM detection_tasks WHERE status = 'failed'")

        if not failed_tasks:
            print("✅ 没有找到失败的任务")
            return

        print(f"🔍 找到 {len(failed_tasks)} 个失败的任务")

        for task in failed_tasks:
            task_id, task_uuid = task
            print(f"   任务 #{task_id} ({task_uuid[:8]}...)")

        # 询问是否删除
        response = input("\n是否删除这些失败的任务？(y/N): ").strip().lower()

        if response == 'y':
            for task in failed_tasks:
                task_id = task[0]
                # 删除相关的检测结果
                db.execute_query("DELETE FROM detection_results WHERE task_id = %s", (task_id,))
                # 删除任务
                db.execute_query("DELETE FROM detection_tasks WHERE id = %s", (task_id,))

            print(f"✅ 已删除 {len(failed_tasks)} 个失败的任务")
        else:
            print("❌ 取消删除操作")

    except Exception as e:
        print(f"❌ 清理失败: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    query_stored_results()

    # 演示获取特定项目的文件路径
    print("\n" + "=" * 60)
    print("💡 获取项目文件路径示例:")

    db = DatabaseManager()
    try:
        # 获取最新的成功项目
        projects = db.execute_query("""
            SELECT p.id, p.name 
            FROM projects p
            JOIN detection_tasks dt ON p.id = dt.project_id
            WHERE dt.status = 'completed'
            ORDER BY p.created_at DESC
            LIMIT 1
        """)

        if projects:
            project_id, project_name = projects[0]
            print(f"最新成功项目: #{project_id} - {project_name}")

            file_paths = get_project_files(project_id)
            if file_paths:
                print(f"文件路径:")
                for key, path in file_paths.items():
                    if path:
                        print(f"  {key}: {path}")
        else:
            print("没有找到成功完成的项目")

    except Exception as e:
        print(f"❌ 获取示例失败: {e}")
    finally:
        db.close()

    # 提供清理选项
    print("\n" + "=" * 60)
    print("🧹 数据库维护选项:")
    response = input("是否要清理失败的任务？(y/N): ").strip().lower()
    if response == 'y':
        clean_failed_tasks()