"""
记忆工程项目配置文件
"""
import os

# 项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Demo配置
MAX_PHOTOS = 50  # Demo阶段最多处理50张照片
DEMO_MODE = True  # Demo模式

# API配置 - 从环境变量读取
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
AMAP_API_KEY = "263f3f40b5ac8921c2a98616ffa96201"  # 高德地图API（逆地理编码）
VLM_MODEL = "gemini-2.0-flash"
LLM_MODEL = "gemini-2.5-flash"  # 画像生成使用 Flash 2.5

# 人脸识别配置
FACE_THRESHOLD = 0.70  # 相似度阈值（降低到0.70，提高召回率）
FACE_MODEL = "Facenet512"  # DeepFace使用的模型（换用更准确的Facenet512）
FACE_DETECTOR = "opencv"  # 人脸检测器
FACE_ALIGN = True  # 是否对齐人脸
FACE_MIN_SIZE = 70  # 最小人脸尺寸（像素），过滤小脸

# 图片处理配置
MAX_IMAGE_SIZE = 1536  # 压缩后最大边长
JPEG_QUALITY = 85  # JPEG质量
DEDUP_TIME_WINDOW = 60  # 去重时间窗口（秒）

# 事件提取配置
EVENT_TIME_THRESHOLD = 2  # 时间阈值（小时）
EVENT_DISTANCE_THRESHOLD = 1  # 地点距离阈值（km）

# 关系推断配置
if DEMO_MODE:
    # Demo阶段：降低触发条件
    MIN_PHOTO_COUNT = 2  # 出现2次就行
    MIN_TIME_SPAN_DAYS = 1  # 认识1天就行
    MIN_SCENE_VARIETY = 1  # 1种场景就行
else:
    # 正式阶段：严格条件
    MIN_PHOTO_COUNT = 5
    MIN_TIME_SPAN_DAYS = 7
    MIN_SCENE_VARIETY = 3

# 存储路径
VLM_CACHE_PATH = os.path.join(CACHE_DIR, "Vyoyo.json")  # VLM分析结果
FACE_DB_PATH = os.path.join(CACHE_DIR, "face_db.json")
FEATURE_DB_PATH = os.path.join(CACHE_DIR, "vlm_feature_db.json")  # VLM特征库
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "事件yoyo.json")  # 事件提取结果
DETAILED_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "memory_detailed.md")
PROFILE_REPORT_PATH = os.path.join(OUTPUT_DIR, "user_profile_report.md")  # 用户画像报告（FBI级别）

# 错误处理
MAX_RETRIES = 3  # 最大重试次数
RETRY_DELAY = 1  # 重试延迟（秒）
CONTINUE_ON_ERROR = True  # 出错后继续处理

# 输出配置
SHOW_PROGRESS = True  # 显示进度条
SHOW_PHOTO_DETAILS = False  # 显示每张照片的处理详情

# 关系类型定义
RELATIONSHIP_TYPES = {
    "family": {"label": "家人", "description": "有血缘或婚姻关系"},
    "partner": {"label": "伴侣", "description": "亲密关系"},
    "close_friend": {"label": "密友", "description": "频繁出现，关系亲密"},
    "friend": {"label": "朋友", "description": "偶尔一起活动"},
    "colleague": {"label": "同事", "description": "工作关系"},
    "acquaintance": {"label": "熟人", "description": "很少出现，关系不深"},
}

# 事件类型定义
EVENT_TYPES = [
    "社交", "工作", "休闲", "用餐", "运动", "旅行", "购物", "学习", "其他"
]
