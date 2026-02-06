from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import uuid
from pathlib import Path
import time
from werkzeug.utils import secure_filename
from detection.universal_detector import UniversalDetector
from database.db_manager import DatabaseManager
import json

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 配置路径
UPLOAD_FOLDER = Path('static/uploads')
RESULT_FOLDER = Path('static/results')

# 确保文件夹存在
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# 多模型配置（更新后）
MODELS_CONFIG = {
    'door': {
        'name': '门状态检测',
        'paths': [
            r"D:\PythonProject1\runs\detect\runs\train\door_detection\weights\best.pt"
        ],
        'description': '检测门的开关状态',
        'classes': ['door-close', 'door-open']
    },
    'vest': {
        'name': '反光背心检测',
        'paths': [
            r"D:\PythonProject2\reflective_vest_detection\runs\train\reflective_vest_v2_optimized\weights\best.pt"
        ],
        'description': '检测是否穿着反光背心',
        'classes': ['vest']
    },
    'safety_belt': {
        'name': '五点式安全带检测',
        'paths': [
            r"C:\Users\86138\Desktop\模型文件\五点式安全带\best.pt"
        ],
        'description': '检测五点式安全带佩戴情况',
        'classes': ['no', 'yes']
    },
    'face_mask': {
        'name': '防护面罩检测',
        'paths': [
            r"D:\集成\yolo11_warning_sign\runs\detect\runs\train\face_mask_detection\weights\best.pt"
        ],
        'description': '检测工作人员是否佩戴防护面罩',
        'classes': []
    },
    'goggles': {  # 🆕 新增护目镜检测模型
        'name': '护目镜检测',
        'paths': [
            r"D:\集成\yolo11_warning_sign\runs\detect\goggles_detection\yolo11n_goggles\weights\best.pt"
        ],
        'description': '检测工作人员是否佩戴护目镜',
        'classes': []  # 动态获取
    },
    'fire_extinguisher': {
        'name': '灭火器检测',
        'paths': [
            r"D:\集成\yolo11_warning_sign\runs\detect\runs\detect\fire_extinguisher_detection\weights\best.pt",
            "runs/detect/fire_extinguisher_detection/weights/best.pt"
        ],
        'description': '检测消防灭火器设备',
        'classes': []
    },
    'warning_sign': {
        'name': '警示标志检测',
        'paths': [
            "runs/detect/runs/train/warning_sign/weights/best.pt",
            "runs/train/warning_sign/weights/best.pt",
            "best.pt"
        ],
        'description': '检测各类警示标志',
        'classes': []
    },
    'tripod': {
        'name': '三脚架检测',
        'paths': [
            "runs/detect/runs/train/tripod_detector/weights/best.pt",
            "runs/train/tripod_detector/weights/best.pt"
        ],
        'description': '检测施工现场三脚架',
        'classes': []
    },
    'safety_barrier': {
        'name': '安全防护栏检测',
        'paths': [
            "runs/detect/runs/train/safety_barrier_detector/weights/best.pt",
            "runs/train/safety_barrier_detector/weights/best.pt"
        ],
        'description': '检测安全防护栏',
        'classes': []
    }
}

# 存储所有检测器
detectors = {}
current_detector_type = 'goggles'  # 🔄 默认使用护目镜检测


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def find_model_path(possible_paths):
    """从可能的路径列表中找到存在的模型文件"""
    for path in possible_paths:
        if Path(path).exists():
            return path
    return None


def init_detectors():
    """初始化所有可用的检测器"""
    global detectors, current_detector_type

    print("\n" + "=" * 60)
    print("🚀 初始化检测器...")
    print("=" * 60)

    for model_type, config in MODELS_CONFIG.items():
        try:
            model_path = find_model_path(config['paths'])

            if model_path is None:
                print(f"⚠️  {config['name']}: 未找到模型文件")
                print(f"   尝试的路径: {config['paths']}")
                continue

            detector = UniversalDetector(model_path)
            detectors[model_type] = {
                'detector': detector,
                'config': config,
                'model_path': model_path
            }

            print(f"✅ {config['name']}: 初始化成功")
            print(f"   模型路径: {model_path}")
            print(f"   支持类别: {detector.get_model_info()['class_names']}")

        except Exception as e:
            print(f"❌ {config['name']}: 初始化失败 - {e}")

    # 设置默认检测器
    if detectors:
        if current_detector_type not in detectors:
            current_detector_type = list(detectors.keys())[0]
        print(f"🎯 默认检测器: {detectors[current_detector_type]['config']['name']}")

    print("=" * 60)
    print(f"✅ 成功初始化 {len(detectors)} 个检测器")
    print("=" * 60 + "\n")

    return len(detectors) > 0


def get_current_detector():
    """获取当前选中的检测器"""
    if current_detector_type in detectors:
        return detectors[current_detector_type]['detector']

    # 如果当前检测器不可用，返回第一个可用的
    if detectors:
        return list(detectors.values())[0]['detector']

    return None


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/models')
def get_models():
    """获取所有可用的模型"""
    models_info = []

    for model_type, detector_info in detectors.items():
        config = detector_info['config']
        detector = detector_info['detector']
        model_info = detector.get_model_info()

        models_info.append({
            'type': model_type,
            'name': config['name'],
            'description': config['description'],
            'model_path': detector_info['model_path'],
            'class_names': model_info['class_names'],
            'num_classes': model_info['num_classes'],
            'is_current': model_type == current_detector_type
        })

    return jsonify({
        'success': True,
        'models': models_info,
        'current_model': current_detector_type
    })


@app.route('/api/switch_model', methods=['POST'])
def switch_model():
    """切换检测模型"""
    global current_detector_type

    try:
        data = request.get_json()
        model_type = data.get('model_type')

        if not model_type:
            return jsonify({
                'success': False,
                'error': '未指定模型类型'
            })

        if model_type not in detectors:
            return jsonify({
                'success': False,
                'error': f'模型 {model_type} 不可用'
            })

        current_detector_type = model_type
        detector_info = detectors[model_type]

        return jsonify({
            'success': True,
            'message': f'已切换到 {detector_info["config"]["name"]}',
            'model_info': {
                'type': model_type,
                'name': detector_info['config']['name'],
                'description': detector_info['config']['description']
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/detect', methods=['POST'])
def detect():
    """检测接口"""
    try:
        # 获取当前检测器
        detector = get_current_detector()

        if detector is None:
            return jsonify({
                'success': False,
                'error': '没有可用的检测器，请检查模型文件是否存在'
            })

        # 检查是否有文件上传
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': '没有上传文件'
            })

        file = request.files['image']

        # 检查文件名
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': '没有选择文件'
            })

        # 检查文件类型
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': '不支持的文件类型，请上传 PNG, JPG, JPEG, GIF, BMP 格式的图片'
            })

        # 生成唯一文件名
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()

        # 保存上传的文件
        upload_filename = f"{file_id}.{file_extension}"
        upload_path = UPLOAD_FOLDER / upload_filename
        file.save(upload_path)

        # 获取置信度阈值
        confidence = float(request.form.get('confidence', 0.7))
        confidence = max(0.1, min(1.0, confidence))  # 限制在0.1-1.0之间

        # 进行检测
        detection_result = detector.detect_image(upload_path, confidence)

        if not detection_result['success']:
            return jsonify(detection_result)

        # 绘制检测结果
        result_filename = f"result_{file_id}.{file_extension}"
        result_path = RESULT_FOLDER / result_filename

        draw_success = detector.draw_detections(
            upload_path,
            detection_result['detections'],
            result_path
        )

        if not draw_success:
            return jsonify({
                'success': False,
                'error': '绘制检测结果失败'
            })

        # 存储到数据库（可选）
        try:
            store_detection_to_db(detection_result, str(upload_path), str(result_path))
        except Exception as e:
            print(f"存储到数据库失败: {e}")
            # 不影响检测结果返回

        # 构建返回结果
        result = {
            'success': True,
            'model_type': current_detector_type,
            'model_name': detectors[current_detector_type]['config']['name'],
            'detection_count': detection_result['detection_count'],
            'detections': detection_result['detections'],
            'avg_confidence': detection_result['avg_confidence'],
            'process_time': detection_result['process_time'],
            'image_width': detection_result['image_width'],
            'image_height': detection_result['image_height'],
            'original_image_url': f'/static/uploads/{upload_filename}',
            'result_image_url': f'/static/results/{result_filename}',
            'model_info': detector.get_model_info()
        }

        return jsonify(result)

    except Exception as e:
        print(f"检测过程出错: {e}")
        return jsonify({
            'success': False,
            'error': f'检测过程出错: {str(e)}'
        })


def store_detection_to_db(detection_result, original_path, result_path):
    """将检测结果存储到数据库"""
    try:
        db = DatabaseManager()

        # 创建或获取项目
        project_id = get_or_create_project(db)

        # 创建检测任务
        task_id, task_uuid = db.create_detection_task(
            project_id=project_id,
            file_name=Path(original_path).name,
            file_type='image',
            file_path=original_path,
            confidence=0.5  # 默认置信度
        )

        # 更新任务状态
        db.update_task_status(task_id, 'completed', processing_time=detection_result['process_time'])

        # 准备类别统计
        class_counts = {}
        for detection in detection_result['detections']:
            class_name = detection['class_name']
            if class_name not in class_counts:
                class_counts[class_name] = []
            class_counts[class_name].append(detection['confidence'])

        # 计算每个类别的统计信息
        class_stats = {}
        for class_name, confidences in class_counts.items():
            class_stats[class_name] = {
                'count': len(confidences),
                'avg_confidence': sum(confidences) / len(confidences),
                'max_confidence': max(confidences),
                'min_confidence': min(confidences)
            }

        # 创建检测结果记录
        result_url_info = {
            'original_image': original_path,
            'result_image': result_path,
            'detection_time': time.time(),
            'model_type': current_detector_type
        }

        result_id = db.create_detection_result(
            task_id=task_id,
            width=detection_result['image_width'],
            height=detection_result['image_height'],
            total_frames=1,  # 单张图片
            total_detections=detection_result['detection_count'],
            class_counts=class_stats,
            result_url=json.dumps(result_url_info, ensure_ascii=False)
        )

        # 存储具体的检测框信息
        for detection in detection_result['detections']:
            db.add_detection(
                task_id=task_id,
                frame_number=1,  # 图片只有一帧
                class_name=detection['class_name'],
                confidence=detection['confidence'],
                bbox=detection['bbox']
            )

        print(f"✅ 检测结果已存储到数据库，任务ID: {task_id}")

    except Exception as e:
        print(f"❌ 存储到数据库失败: {e}")
        raise
    finally:
        if 'db' in locals():
            db.close()


def get_or_create_project(db):
    """获取或创建项目"""
    try:
        # 根据当前模型类型查找项目
        detector_info = detectors[current_detector_type]
        scene_type = current_detector_type

        # 使用新的方法获取项目
        project = db.get_project_by_scene_type(scene_type)

        if project:
            print(f"✅ 找到现有项目: {project['name']} (ID: {project['id']})")
            return project['id']

        # 创建新项目
        model_info = detector_info['detector'].get_model_info()
        project_id = db.create_project(
            name=detector_info['config']['name'],
            scene_type=scene_type,
            model_path=model_info['model_path'],
            classes=model_info['class_names']
        )

        return project_id

    except Exception as e:
        print(f"❌ 获取或创建项目失败: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.route('/api/history')
def get_history():
    """获取检测历史"""
    try:
        db = DatabaseManager()

        # 可选：按模型类型过滤
        model_type = request.args.get('model_type', None)

        # 查询最近的检测记录
        if model_type:
            query = """
            SELECT dt.id, dt.task_id, dt.file_name, dt.completed_at, dt.processing_time,
                   dr.total_detections, dr.class_counts, dr.width, dr.height, dr.result_url,
                   p.scene_type
            FROM detection_tasks dt
            JOIN detection_results dr ON dt.id = dr.task_id
            JOIN projects p ON dt.project_id = p.id
            WHERE dt.status = 'completed' AND dt.file_type = 'image' AND p.scene_type = %s
            ORDER BY dt.completed_at DESC
            LIMIT 20
            """
            results = db.execute_query(query, (model_type,))
        else:
            query = """
            SELECT dt.id, dt.task_id, dt.file_name, dt.completed_at, dt.processing_time,
                   dr.total_detections, dr.class_counts, dr.width, dr.height, dr.result_url,
                   p.scene_type
            FROM detection_tasks dt
            JOIN detection_results dr ON dt.id = dr.task_id
            JOIN projects p ON dt.project_id = p.id
            WHERE dt.status = 'completed' AND dt.file_type = 'image'
            ORDER BY dt.completed_at DESC
            LIMIT 20
            """
            results = db.execute_query(query)

        history = []
        for result in results:
            task_id, task_uuid, file_name, completed_at, processing_time, total_detections, class_counts_json, width, height, result_url, scene_type = result

            try:
                class_counts = json.loads(class_counts_json) if class_counts_json else {}
            except:
                class_counts = {}

            history.append({
                'task_id': task_id,
                'task_uuid': task_uuid,
                'file_name': file_name,
                'completed_at': str(completed_at),
                'processing_time': processing_time,
                'total_detections': total_detections,
                'class_counts': class_counts,
                'image_size': f"{width} × {height}",
                'model_type': scene_type,
                'model_name': MODELS_CONFIG.get(scene_type, {}).get('name', scene_type)
            })

        return jsonify({
            'success': True,
            'history': history
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })
    finally:
        if 'db' in locals():
            db.close()


@app.route('/api/stats')
def get_stats():
    """获取统计信息"""
    try:
        db = DatabaseManager()

        # 按模型类型统计
        stats_by_model = {}

        for model_type in detectors.keys():
            # 该模型的检测次数
            total_tasks = db.execute_query(
                """SELECT COUNT(*) FROM detection_tasks dtJOIN projects p ON dt.project_id = p.id
                   WHERE dt.status = 'completed' AND dt.file_type = 'image' AND p.scene_type = %s""",
                (model_type,)
            )[0][0]

            # 该模型检测到的物体数
            total_detections = db.execute_query(
                """SELECT SUM(dr.total_detections) FROM detection_results dr
                   JOIN detection_tasks dt ON dr.task_id = dt.id
                   JOIN projects p ON dt.project_id = p.id
                   WHERE dt.file_type = 'image' AND p.scene_type = %s""",
                (model_type,)
            )[0][0] or 0

            # 平均处理时间
            avg_time = db.execute_query(
                """SELECT AVG(dt.processing_time) FROM detection_tasks dt
                   JOIN projects p ON dt.project_id = p.id
                   WHERE dt.status = 'completed' AND dt.file_type = 'image' AND p.scene_type = %s""",
                (model_type,)
            )[0][0] or 0

            stats_by_model[model_type] = {
                'name': detectors[model_type]['config']['name'],
                'total_tasks': total_tasks,
                'total_detections': total_detections,
                'avg_processing_time': round(avg_time, 2)
            }

        # 总体统计
        total_tasks_all = db.execute_query(
            "SELECT COUNT(*) FROM detection_tasks WHERE status = 'completed' AND file_type = 'image'"
        )[0][0]

        total_detections_all = db.execute_query(
            "SELECT SUM(total_detections) FROM detection_results dr JOIN detection_tasks dt ON dr.task_id = dt.id WHERE dt.file_type = 'image'"
        )[0][0] or 0

        return jsonify({
            'success': True,
            'stats': {
                'total_tasks': total_tasks_all,
                'total_detections': total_detections_all,
                'by_model': stats_by_model
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })
    finally:
        if 'db' in locals():
            db.close()


@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': '文件太大，请上传小于16MB的图片'
    }), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'error': '请求的资源不存在'
    }), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({
        'success': False,
        'error': '服务器内部错误'
    }), 500


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 启动多模型目标检测系统")
    print("=" * 60)

    # 初始化所有检测器
    success = init_detectors()

    if not success:
        print("\n" + "=" * 60)
        print("❌ 没有可用的检测器")
        print("=" * 60)
        print("请确保以下至少一个模型文件存在:")
        for model_type, config in MODELS_CONFIG.items():
            print(f"\n{config['name']}:")
            for path in config['paths']:
                print(f"  - {path}")
        print("=" * 60)
    else:
        print(f"\n✅ 系统初始化完成，当前使用: {detectors[current_detector_type]['config']['name']}")
        print("🌐 访问 http://localhost:5000 开始使用")
        print("=" * 60 + "\n")

        # 启动Flask应用
        app.run(debug=True, host='0.0.0.0', port=5000)