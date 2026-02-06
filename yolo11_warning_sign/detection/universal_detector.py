from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import time


class UniversalDetector:
    def __init__(self, model_path):
        """初始化检测器"""
        self.model_path = model_path
        self.model = YOLO(model_path)
        print(f"✅ 模型加载成功: {model_path}")

    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_path': self.model_path,
            'class_names': list(self.model.names.values()),
            'num_classes': len(self.model.names)
        }

    def detect_image(self, image_path, confidence=0.5):
        """检测图片"""
        try:
            start_time = time.time()

            # 执行检测
            results = self.model.predict(
                source=str(image_path),
                conf=confidence,
                verbose=False
            )

            process_time = time.time() - start_time

            # 解析结果
            result = results[0]
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                return {
                    'success': True,
                    'detection_count': 0,
                    'detections': [],
                    'avg_confidence': 0,
                    'process_time': process_time,
                    'image_width': result.orig_shape[1],
                    'image_height': result.orig_shape[0]
                }

            detections = []
            confidences = []

            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # 获取类别和置信度
                class_id = int(box.cls[0].cpu().numpy())
                confidence_score = float(box.conf[0].cpu().numpy())
                class_name = self.model.names[class_id]

                confidences.append(confidence_score)

                detections.append({
                    'class_name': class_name,
                    'confidence': confidence_score,
                    'bbox': {
                        'x': int(x1),
                        'y': int(y1),
                        'w': int(x2 - x1),
                        'h': int(y2 - y1)
                    }
                })

            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return {
                'success': True,
                'detection_count': len(detections),
                'detections': detections,
                'avg_confidence': avg_confidence,
                'process_time': process_time,
                'image_width': result.orig_shape[1],
                'image_height': result.orig_shape[0]
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def draw_detections(self, image_path, detections, output_path):
        """绘制检测结果 - 处理中文类别名称"""
        try:
            # 读取图片
            image = cv2.imread(str(image_path))
            if image is None:
                return False

            # 定义颜色 (BGR格式)
            colors = [
                (0, 255, 0),  # 绿色
                (255, 0, 0),  # 蓝色
                (0, 0, 255),  # 红色
                (255, 255, 0),  # 青色
                (255, 0, 255),  # 品红色
                (0, 255, 255),  # 黄色
            ]

            # 中文到英文的映射表
            chinese_to_english = {
                '安全防护栏': 'Safety Barrier',
                '三脚架': 'Tripod',
                '反光背心': 'Reflective Vest',
                '门关闭': 'Door Closed',
                '门打开': 'Door Open',
                '未佩戴': 'Not Wearing',
                '已佩戴': 'Wearing',
                # 如果已经是英文，保持不变
                'safety_barrier': 'Safety Barrier',
                'tripod': 'Tripod',
                'vest': 'Reflective Vest',
                'door-close': 'Door Closed',
                'door-open': 'Door Open',
                'no': 'Not Wearing',
                'yes': 'Wearing',
            }

            # 绘制检测框
            for i, detection in enumerate(detections):
                bbox = detection['bbox']
                x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']

                # 选择颜色
                color = colors[i % len(colors)]

                # 绘制矩形框
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)

                # 处理类别名称
                class_name = detection['class_name']
                english_name = chinese_to_english.get(class_name, class_name)

                # 绘制标签
                label = f"{english_name}: {detection['confidence']:.2f}"

                # 计算文字尺寸
                font_scale = 0.7
                thickness = 2
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

                # 绘制标签背景
                cv2.rectangle(image, (x, y - label_size[1] - 10),
                              (x + label_size[0] + 10, y), color, -1)

                # 绘制标签文字
                cv2.putText(image, label, (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            # 保存结果图片
            cv2.imwrite(str(output_path), image)
            return True

        except Exception as e:
            print(f"绘制检测结果失败: {e}")
            return False