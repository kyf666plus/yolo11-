import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path


class DatabaseManager:
    def __init__(self, db_path='detection_system.db'):
        """初始化数据库管理器"""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.connect()
        self.init_database()

    def connect(self):
        """连接数据库"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            print("数据库连接成功")
        except Exception as e:
            print(f"数据库连接失败: {e}")
            raise

    def init_database(self):
        """初始化数据库表"""
        try:
            # 项目表
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name VARCHAR(100) NOT NULL,
                    scene_type VARCHAR(50) NOT NULL,
                    model_path VARCHAR(255) NOT NULL,
                    classes TEXT NOT NULL,
                    status INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 检测任务表
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS detection_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id VARCHAR(36) NOT NULL UNIQUE,
                    project_id INTEGER NOT NULL,
                    file_name VARCHAR(255) NOT NULL,
                    file_type VARCHAR(10) NOT NULL,
                    file_path VARCHAR(255) NOT NULL,
                    confidence REAL DEFAULT 0.50,
                    status VARCHAR(20) DEFAULT 'pending',
                    error_msg TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    processing_time REAL,
                    FOREIGN KEY (project_id) REFERENCES projects(id)
                )
            ''')

            # 检测结果表
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS detection_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER NOT NULL,
                    width INTEGER,
                    height INTEGER,
                    total_frames INTEGER,
                    total_detections INTEGER DEFAULT 0,
                    class_counts TEXT,
                    result_url VARCHAR(500),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (task_id) REFERENCES detection_tasks(id) ON DELETE CASCADE
                )
            ''')

            # 检测详情表
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER NOT NULL,
                    frame_number INTEGER,
                    class_name VARCHAR(50) NOT NULL,
                    confidence REAL NOT NULL,
                    bbox TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (task_id) REFERENCES detection_tasks(id) ON DELETE CASCADE
                )
            ''')

            # 创建索引
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_scene_type ON projects(scene_type)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_task_id ON detection_tasks(task_id)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON detection_tasks(status)')

            self.conn.commit()
            print("数据库表初始化成功")

        except Exception as e:
            print(f"数据库初始化失败: {e}")
            raise

    def execute_query(self, query, params=None):
        """执行查询"""
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            return self.cursor.fetchall()
        except Exception as e:
            print(f"查询执行失败: {e}")
            print(f"SQL: {query}")
            print(f"参数: {params}")
            raise

    def execute_update(self, query, params=None):
        """执行更新"""
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            self.conn.commit()
            return self.cursor.lastrowid
        except Exception as e:
            print(f"更新执行失败: {e}")
            print(f"SQL: {query}")
            print(f"参数: {params}")
            self.conn.rollback()
            raise

    def create_project(self, name, scene_type, model_path, classes):
        """创建项目"""
        try:
            # 将类别列表转换为 JSON 字符串
            classes_json = json.dumps(classes, ensure_ascii=False)

            query = '''
                INSERT INTO projects (name, scene_type, model_path, classes, status)
                VALUES (?, ?, ?, ?, 1)
            '''

            project_id = self.execute_update(
                query,
                (name, scene_type, str(model_path), classes_json)
            )

            print(f"✅ 项目创建成功，ID: {project_id}")
            return project_id

        except Exception as e:
            print(f"❌ 创建项目失败: {e}")
            raise

    def get_project_by_scene_type(self, scene_type):
        """根据场景类型获取项目"""
        try:
            query = "SELECT id, name, model_path, classes FROM projects WHERE scene_type = ? AND status = 1 LIMIT 1"
            result = self.execute_query(query, (scene_type,))

            if result:
                project_id, name, model_path, classes_json = result[0]
                return {
                    'id': project_id,
                    'name': name,
                    'model_path': model_path,
                    'classes': json.loads(classes_json)
                }
            return None

        except Exception as e:
            print(f"❌ 获取项目失败: {e}")
            raise

    def create_detection_task(self, project_id, file_name, file_type, file_path, confidence=0.5):
        """创建检测任务"""
        try:
            task_uuid = str(uuid.uuid4())

            query = '''
                INSERT INTO detection_tasks 
                (task_id, project_id, file_name, file_type, file_path, confidence, status)
                VALUES (?, ?, ?, ?, ?, ?, 'pending')
            '''

            task_id = self.execute_update(
                query,
                (task_uuid, project_id, file_name, file_type, str(file_path), confidence)
            )

            print(f"✅ 检测任务创建成功，ID: {task_id}, UUID: {task_uuid}")
            return task_id, task_uuid

        except Exception as e:
            print(f"❌ 创建检测任务失败: {e}")
            raise

    def update_task_status(self, task_id, status, error_msg=None, processing_time=None):
        """更新任务状态"""
        try:
            if status == 'completed':
                query = '''
                    UPDATE detection_tasks 
                    SET status = ?, completed_at = ?, processing_time = ?
                    WHERE id = ?
                '''
                self.execute_update(
                    query,
                    (status, datetime.now(), processing_time, task_id)
                )
            elif status == 'failed':
                query = '''
                    UPDATE detection_tasks 
                    SET status = ?, error_msg = ?, completed_at = ?
                    WHERE id = ?
                '''
                self.execute_update(
                    query,
                    (status, error_msg, datetime.now(), task_id)
                )
            else:
                query = "UPDATE detection_tasks SET status = ? WHERE id = ?"
                self.execute_update(query, (status, task_id))

            print(f"✅ 任务状态更新成功: {status}")

        except Exception as e:
            print(f"❌ 更新任务状态失败: {e}")
            raise

    def create_detection_result(self, task_id, width, height, total_frames,
                                total_detections, class_counts, result_url):
        """创建检测结果"""
        try:
            # 将类别统计转换为 JSON 字符串
            class_counts_json = json.dumps(class_counts, ensure_ascii=False)

            query = '''
                INSERT INTO detection_results 
                (task_id, width, height, total_frames, total_detections, class_counts, result_url)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            '''

            result_id = self.execute_update(
                query,
                (task_id, width, height, total_frames, total_detections,
                 class_counts_json, result_url)
            )

            print(f"✅ 检测结果创建成功，ID: {result_id}")
            return result_id

        except Exception as e:
            print(f"❌ 创建检测结果失败: {e}")
            raise

    def add_detection(self, task_id, frame_number, class_name, confidence, bbox):
        """添加检测详情"""
        try:
            # 将边界框转换为 JSON 字符串
            bbox_json = json.dumps(bbox, ensure_ascii=False)

            query = '''
                INSERT INTO detections 
                (task_id, frame_number, class_name, confidence, bbox)
                VALUES (?, ?, ?, ?, ?)
            '''

            detection_id = self.execute_update(
                query,
                (task_id, frame_number, class_name, confidence, bbox_json)
            )

            return detection_id

        except Exception as e:
            print(f"❌ 添加检测详情失败: {e}")
            raise

    def get_task_results(self, task_id):
        """获取任务结果"""
        try:
            query = '''
                SELECT dr.*, dt.file_name, dt.file_type, dt.processing_time
                FROM detection_results dr
                JOIN detection_tasks dt ON dr.task_id = dt.id
                WHERE dt.task_id = ?
            '''

            result = self.execute_query(query, (task_id,))

            if result:
                return {
                    'id': result[0][0],
                    'task_id': result[0][1],
                    'width': result[0][2],
                    'height': result[0][3],
                    'total_frames': result[0][4],
                    'total_detections': result[0][5],
                    'class_counts': json.loads(result[0][6]) if result[0][6] else {},
                    'result_url': result[0][7],
                    'file_name': result[0][9],
                    'file_type': result[0][10],
                    'processing_time': result[0][11]
                }
            return None

        except Exception as e:
            print(f"❌ 获取任务结果失败: {e}")
            raise

    def get_detections_by_task(self, task_id):
        """获取任务的所有检测详情"""
        try:
            query = '''
                SELECT d.frame_number, d.class_name, d.confidence, d.bbox
                FROM detections d
                JOIN detection_tasks dt ON d.task_id = dt.id
                WHERE dt.id = ?
                ORDER BY d.frame_number, d.id
            '''

            results = self.execute_query(query, (task_id,))

            detections = []
            for row in results:
                detections.append({
                    'frame_number': row[0],
                    'class_name': row[1],
                    'confidence': row[2],
                    'bbox': json.loads(row[3])
                })

            return detections

        except Exception as e:
            print(f"❌ 获取检测详情失败: {e}")
            raise

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            print("数据库连接已关闭")