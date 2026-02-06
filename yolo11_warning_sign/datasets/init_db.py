# database/init_db.py
import pymysql
import os
from dotenv import load_dotenv

load_dotenv()


def create_database():
    """创建数据库和表"""

    # 先连接到MySQL服务器（不指定数据库）
    connection = pymysql.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', 3306)),
        user=os.getenv('DB_USER', 'root'),
        password=os.getenv('DB_PASSWORD', ''),
        charset='utf8mb4'
    )

    try:
        with connection.cursor() as cursor:
            # 创建数据库
            db_name = os.getenv('DB_NAME', 'yolo_detection')
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            cursor.execute(f"USE {db_name}")

            # 创建projects表
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INT PRIMARY KEY AUTO_INCREMENT,
                name VARCHAR(100) NOT NULL COMMENT '项目名称',
                scene_type VARCHAR(50) NOT NULL COMMENT '场景类型：door, person, vehicle等',
                model_path VARCHAR(255) NOT NULL COMMENT '模型文件路径',
                classes JSON NOT NULL COMMENT '类别列表',
                status TINYINT DEFAULT 1 COMMENT '状态：1启用 0禁用',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                INDEX idx_scene_type (scene_type)
            ) COMMENT='项目表'
            """)

            # 创建detection_tasks表
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS detection_tasks (
                id INT PRIMARY KEY AUTO_INCREMENT,
                task_id VARCHAR(36) NOT NULL UNIQUE COMMENT '任务UUID',
                project_id INT NOT NULL COMMENT '项目ID',

                file_name VARCHAR(255) NOT NULL COMMENT '文件名',
                file_type ENUM('image', 'video') NOT NULL COMMENT '文件类型',
                file_path VARCHAR(255) NOT NULL COMMENT '文件路径',

                confidence DECIMAL(3,2) DEFAULT 0.50 COMMENT '置信度阈值',

                status ENUM('pending', 'processing', 'completed', 'failed') DEFAULT 'pending',
                error_msg TEXT COMMENT '错误信息',

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP NULL COMMENT '完成时间',
                processing_time DECIMAL(10,2) COMMENT '处理时间（秒）',

                FOREIGN KEY (project_id) REFERENCES projects(id),
                INDEX idx_task_id (task_id),
                INDEX idx_status (status),
                INDEX idx_created_at (created_at)
            ) COMMENT='检测任务表'
            """)

            # 创建detection_results表
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS detection_results (
                id INT PRIMARY KEY AUTO_INCREMENT,
                task_id INT NOT NULL COMMENT '任务ID',

                width INT COMMENT '宽度',
                height INT COMMENT '高度',
                total_frames INT COMMENT '总帧数（视频）',

                total_detections INT DEFAULT 0 COMMENT '总检测数',
                class_counts JSON COMMENT '各类别统计',

                result_url VARCHAR(500) COMMENT '结果文件URL',

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (task_id) REFERENCES detection_tasks(id) ON DELETE CASCADE,
                INDEX idx_task_id (task_id)
            ) COMMENT='检测结果表'
            """)

            # 创建detections表
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                task_id INT NOT NULL COMMENT '任务ID',

                frame_number INT COMMENT '帧号（视频用）',
                class_name VARCHAR(50) NOT NULL COMMENT '类别名称',
                confidence DECIMAL(5,4) NOT NULL COMMENT '置信度',

                bbox JSON NOT NULL COMMENT '边界框 {"x1":100, "y1":200, "x2":300, "y2":400}',

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (task_id) REFERENCES detection_tasks(id) ON DELETE CASCADE,
                INDEX idx_task_id (task_id),
                INDEX idx_class_name (class_name)
            ) COMMENT='检测详情表'
            """)

            connection.commit()
            print("数据库和表创建成功！")

    finally:
        connection.close()


if __name__ == "__main__":
    create_database()