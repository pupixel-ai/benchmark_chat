"""
记忆工程项目配置文件
"""
import os
from urllib.parse import quote_plus

# 项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
RUNTIME_DIR = os.path.join(PROJECT_ROOT, "runtime")
TASKS_DIR = os.path.join(RUNTIME_DIR, "tasks")
BUNDLED_FACE_RECOGNITION_SRC_PATH = os.path.join(PROJECT_ROOT, "vendor", "face_recognition_src")

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Demo配置
MAX_PHOTOS = 50  # Demo阶段最多处理50张照片
DEMO_MODE = True  # Demo模式
MAX_UPLOAD_PHOTOS = 100

# Web 服务配置
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("PORT", os.getenv("BACKEND_PORT", "8000")))
BACKEND_RELOAD = os.getenv("BACKEND_RELOAD", "false").lower() == "true"
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
RUNS_URL_PREFIX = "/runs"
ASSET_URL_PREFIX = "/api/assets"
AUTH_SESSION_COOKIE_NAME = os.getenv("AUTH_SESSION_COOKIE_NAME", "memory_session")
AUTH_SESSION_DAYS = int(os.getenv("AUTH_SESSION_DAYS", "14"))
COOKIE_SECURE = os.getenv(
    "COOKIE_SECURE",
    "true" if FRONTEND_ORIGIN.startswith("https://") else "false",
).lower() == "true"


def _parse_origin_list(value: str) -> tuple[str, ...]:
    return tuple(
        item.rstrip("/")
        for item in (part.strip() for part in value.split(","))
        if item
    )


_extra_cors_origins = _parse_origin_list(os.getenv("CORS_ALLOW_ORIGINS", ""))
CORS_ALLOW_ORIGINS = tuple(
    dict.fromkeys(
        _extra_cors_origins
        or (
            FRONTEND_ORIGIN.rstrip("/"),
            "http://127.0.0.1:3000",
            "http://localhost:3000",
        )
    )
)


def _normalize_database_url(value: str) -> str:
    if value.startswith("mysql://"):
        return "mysql+pymysql://" + value[len("mysql://") :]
    return value


def _railway_mysql_url() -> str | None:
    direct_url = os.getenv("MYSQL_URL") or os.getenv("MYSQL_PUBLIC_URL")
    if direct_url:
        return _normalize_database_url(direct_url)

    host = os.getenv("MYSQLHOST")
    port = os.getenv("MYSQLPORT", "3306")
    user = os.getenv("MYSQLUSER")
    password = os.getenv("MYSQLPASSWORD")
    database = os.getenv("MYSQLDATABASE")

    if not all([host, user, password, database]):
        return None

    quoted_user = quote_plus(user)
    quoted_password = quote_plus(password)
    return f"mysql+pymysql://{quoted_user}:{quoted_password}@{host}:{port}/{database}"


DATABASE_URL = _normalize_database_url(
    os.getenv("DATABASE_URL")
    or _railway_mysql_url()
    or f"sqlite:///{os.path.join(RUNTIME_DIR, 'local_preview.db')}"
)
SQL_ECHO = os.getenv("SQL_ECHO", "false").lower() == "true"

# 对象存储配置（Railway Bucket / S3 兼容）
OBJECT_STORAGE_BUCKET = os.getenv("BUCKET", os.getenv("OBJECT_STORAGE_BUCKET", ""))
OBJECT_STORAGE_ENDPOINT = os.getenv("ENDPOINT", os.getenv("OBJECT_STORAGE_ENDPOINT", ""))
OBJECT_STORAGE_REGION = os.getenv("REGION", os.getenv("OBJECT_STORAGE_REGION", "auto"))
OBJECT_STORAGE_ACCESS_KEY_ID = os.getenv("ACCESS_KEY_ID", os.getenv("OBJECT_STORAGE_ACCESS_KEY_ID", ""))
OBJECT_STORAGE_SECRET_ACCESS_KEY = os.getenv("SECRET_ACCESS_KEY", os.getenv("OBJECT_STORAGE_SECRET_ACCESS_KEY", ""))
OBJECT_STORAGE_PREFIX = os.getenv("OBJECT_STORAGE_PREFIX", "tasks")
OBJECT_STORAGE_ADDRESSING_STYLE = os.getenv("OBJECT_STORAGE_ADDRESSING_STYLE", "auto")

# API配置 - 从环境变量读取
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
AMAP_API_KEY = os.getenv("AMAP_API_KEY", "")

# 代理服务配置（可选）
USE_API_PROXY = os.getenv("USE_API_PROXY", "false").lower() == "true"
API_PROXY_URL = os.getenv("API_PROXY_URL", "")  # 代理服务基础 URL
API_PROXY_KEY = os.getenv("API_PROXY_KEY", "")  # 代理服务 API Key
API_PROXY_MODEL = os.getenv("API_PROXY_MODEL", "gemini-2.0-flash")  # 代理支持的模型

VLM_MODEL = "gemini-2.0-flash"
LLM_MODEL = "gemini-2.5-flash"  # 画像生成使用 Flash 2.5

# 人脸识别配置（默认使用仓库内置的 vendored face-recognition 源码）
FACE_RECOGNITION_SRC_PATH = os.getenv(
    "FACE_RECOGNITION_SRC_PATH",
    BUNDLED_FACE_RECOGNITION_SRC_PATH,
)
FACE_MODEL_NAME = os.getenv("FACE_MODEL_NAME", "buffalo_l")
FACE_MAX_SIDE = int(os.getenv("FACE_MAX_SIDE", "1920"))
FACE_DET_THRESHOLD = float(os.getenv("FACE_DET_THRESHOLD", "0.60"))
FACE_SIM_THRESHOLD = float(os.getenv("FACE_SIM_THRESHOLD", "0.50"))
FACE_MIN_SIZE = int(os.getenv("FACE_MIN_SIZE", "48"))  # 最小人脸尺寸（像素）
FACE_PROVIDERS = tuple(
    provider.strip()
    for provider in os.getenv("FACE_PROVIDERS", "CPUExecutionProvider").split(",")
    if provider.strip()
)

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
FACE_INDEX_PATH = os.path.join(CACHE_DIR, "faces.index")
FACE_STATE_PATH = os.path.join(CACHE_DIR, "face_recognition_state.json")
FACE_OUTPUT_PATH = os.path.join(CACHE_DIR, "face_recognition_output.json")
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
