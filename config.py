"""
记忆工程 v2.0 项目配置文件
合并：新工程 FACE_* 配置 + 老工程 VLM/LLM/关系配置
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


def _fallback_load_dotenv(dotenv_path: str | os.PathLike[str]) -> None:
    file_path = Path(dotenv_path)
    if not file_path.exists():
        return
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


_fallback_load_dotenv(Path(__file__).resolve().parent / ".env")

# 项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
RUNTIME_DIR = os.path.join(PROJECT_ROOT, "runtime")
BUNDLED_FACE_RECOGNITION_SRC_PATH = os.path.join(PROJECT_ROOT, "vendor", "face_recognition_src")

# ─── 版本 ────────────────────────────────────────────────────────
TASK_VERSION_V0315 = "v0315"
TASK_VERSION_V0317 = "v0317"
AVAILABLE_TASK_VERSIONS = (TASK_VERSION_V0315, TASK_VERSION_V0317)
APP_VERSION = os.getenv("APP_VERSION", TASK_VERSION_V0317).strip() or TASK_VERSION_V0317
DEFAULT_TASK_VERSION = os.getenv("DEFAULT_TASK_VERSION", APP_VERSION).strip() or APP_VERSION
if DEFAULT_TASK_VERSION not in AVAILABLE_TASK_VERSIONS:
    DEFAULT_TASK_VERSION = TASK_VERSION_V0317

# ─── 上传限制 ────────────────────────────────────────────────────
MAX_UPLOAD_PHOTOS = int(os.getenv("MAX_UPLOAD_PHOTOS", "500"))
DEFAULT_NORMALIZE_LIVE_PHOTOS = os.getenv("DEFAULT_NORMALIZE_LIVE_PHOTOS", "true").lower() == "true"

# ─── VLM 并发与缓存刷新 ──────────────────────────────────────────
VLM_MAX_CONCURRENCY = int(os.getenv("VLM_MAX_CONCURRENCY", "4"))
VLM_ENABLE_PRIORITY_SCHEDULING = os.getenv("VLM_ENABLE_PRIORITY_SCHEDULING", "false").lower() == "true"
VLM_CACHE_FLUSH_EVERY_N = int(os.getenv("VLM_CACHE_FLUSH_EVERY_N", "20"))
VLM_CACHE_FLUSH_INTERVAL_SECONDS = float(os.getenv("VLM_CACHE_FLUSH_INTERVAL_SECONDS", "60.0"))

# ─── LLM 提供商 ──────────────────────────────────────────────────
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openrouter").strip().lower() or "openrouter"
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "benchmark_chat").strip()

# ─── Memory Module — 嵌入向量 ────────────────────────────────────
MEMORY_REAL_EMBEDDINGS_ENABLED = os.getenv("MEMORY_REAL_EMBEDDINGS_ENABLED", "false").lower() == "true"
MEMORY_EMBEDDING_PROVIDER = os.getenv("MEMORY_EMBEDDING_PROVIDER", "").strip()
MEMORY_EMBEDDING_MODEL = os.getenv("MEMORY_EMBEDDING_MODEL", "").strip()
MEMORY_EMBEDDING_VERSION = os.getenv("MEMORY_EMBEDDING_VERSION", "v1").strip() or "v1"
MEMORY_EMBEDDING_TIMEOUT_SECONDS = float(os.getenv("MEMORY_EMBEDDING_TIMEOUT_SECONDS", "30.0"))

# ─── Memory Module — 外部存储（Milvus / Neo4j / Redis） ──────────
MEMORY_EXTERNAL_SINKS_ENABLED = os.getenv("MEMORY_EXTERNAL_SINKS_ENABLED", "false").lower() == "true"
MEMORY_MILVUS_URI = os.getenv("MEMORY_MILVUS_URI", "").strip()
MEMORY_MILVUS_USER = os.getenv("MEMORY_MILVUS_USER", "").strip()
MEMORY_MILVUS_PASSWORD = os.getenv("MEMORY_MILVUS_PASSWORD", "").strip()
MEMORY_MILVUS_TOKEN = os.getenv("MEMORY_MILVUS_TOKEN", "").strip()
MEMORY_MILVUS_DB_NAME = os.getenv("MEMORY_MILVUS_DB_NAME", "default").strip() or "default"
MEMORY_MILVUS_COLLECTION = os.getenv("MEMORY_MILVUS_COLLECTION", "memory_units").strip() or "memory_units"
MEMORY_MILVUS_EVIDENCE_COLLECTION = os.getenv("MEMORY_MILVUS_EVIDENCE_COLLECTION", "memory_evidence").strip() or "memory_evidence"
MEMORY_MILVUS_UNITS_COLLECTION = os.getenv("MEMORY_MILVUS_UNITS_COLLECTION", "memory_units").strip() or "memory_units"
MEMORY_MILVUS_VECTOR_DIM = int(os.getenv("MEMORY_MILVUS_VECTOR_DIM", "1536"))
MEMORY_NEO4J_URI = os.getenv("MEMORY_NEO4J_URI", "").strip()
MEMORY_NEO4J_USERNAME = os.getenv("MEMORY_NEO4J_USERNAME", "").strip()
MEMORY_NEO4J_PASSWORD = os.getenv("MEMORY_NEO4J_PASSWORD", "").strip()
MEMORY_NEO4J_DATABASE = os.getenv("MEMORY_NEO4J_DATABASE", "neo4j").strip() or "neo4j"
MEMORY_REDIS_URL = os.getenv("MEMORY_REDIS_URL", "").strip()
MEMORY_REDIS_PREFIX = os.getenv("MEMORY_REDIS_PREFIX", "mem:").strip() or "mem:"

# ─── 通用 ────────────────────────────────────────────────────────
MAX_PHOTOS = int(os.getenv("MAX_PHOTOS", "999999"))
DEMO_MODE = True
SHOW_PROGRESS = os.getenv("SHOW_PROGRESS", "true").lower() == "true"
SHOW_PHOTO_DETAILS = False

# ─── API ─────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GOOGLE_GEMINI_BASE_URL = os.getenv("GOOGLE_GEMINI_BASE_URL", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip() or "https://openrouter.ai/api/v1"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
CRITIC_PROVIDER = os.getenv("CRITIC_PROVIDER", "anthropic").strip().lower()
CRITIC_MODEL = os.getenv("CRITIC_MODEL", "claude-sonnet-4-6").strip()
PROFILE_LLM_PROVIDER = os.getenv("PROFILE_LLM_PROVIDER", "").strip().lower()
PROFILE_LLM_MODEL = os.getenv("PROFILE_LLM_MODEL", "deepseek/deepseek-chat-v3-0324").strip() or "deepseek/deepseek-chat-v3-0324"
REFLECTION_AGENT_PROVIDER = os.getenv("REFLECTION_AGENT_PROVIDER", "openrouter").strip().lower() or "openrouter"
REFLECTION_AGENT_OPENROUTER_API_KEY = os.getenv("REFLECTION_AGENT_OPENROUTER_API_KEY", "").strip() or OPENROUTER_API_KEY
REFLECTION_AGENT_MODEL = os.getenv("REFLECTION_AGENT_MODEL", "").strip()
REFLECTION_AGENT_TEMPERATURE = float(os.getenv("REFLECTION_AGENT_TEMPERATURE", "0.1"))
AMAP_API_KEY = os.getenv("AMAP_API_KEY", "").strip()
PROFILE_AGENT_ROOT = os.getenv(
    "PROFILE_AGENT_ROOT",
    "/Users/vigar07/Downloads/profile_agent_reflect",
).strip() or "/Users/vigar07/Downloads/profile_agent_reflect"
FEISHU_WEBHOOK_URL = os.getenv("FEISHU_WEBHOOK_URL", "").strip()
FEISHU_APP_ID = os.getenv("FEISHU_APP_ID", "").strip()
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET", "").strip()
FEISHU_CALLBACK_VERIFICATION_TOKEN = os.getenv("FEISHU_CALLBACK_VERIFICATION_TOKEN", "").strip()
FEISHU_CALLBACK_ENCRYPT_KEY = os.getenv("FEISHU_CALLBACK_ENCRYPT_KEY", "").strip()
FEISHU_API_BASE_URL = os.getenv("FEISHU_API_BASE_URL", "https://open.feishu.cn").strip() or "https://open.feishu.cn"
FEISHU_LONG_CONNECTION_LOG_LEVEL = os.getenv("FEISHU_LONG_CONNECTION_LOG_LEVEL", "INFO").strip().upper() or "INFO"
FEISHU_DEFAULT_RECEIVE_ID = os.getenv("FEISHU_DEFAULT_RECEIVE_ID", "").strip()
FEISHU_DEFAULT_RECEIVE_ID_TYPE = os.getenv("FEISHU_DEFAULT_RECEIVE_ID_TYPE", "open_id").strip() or "open_id"
FEISHU_APPROVAL_RECEIVE_ID = os.getenv("FEISHU_APPROVAL_RECEIVE_ID", "").strip()
FEISHU_APPROVAL_RECEIVE_ID_TYPE = os.getenv("FEISHU_APPROVAL_RECEIVE_ID_TYPE", "chat_id").strip() or "chat_id"
FEISHU_DIFFICULT_CASE_RECEIVE_ID = os.getenv("FEISHU_DIFFICULT_CASE_RECEIVE_ID", "").strip()
FEISHU_DIFFICULT_CASE_RECEIVE_ID_TYPE = os.getenv("FEISHU_DIFFICULT_CASE_RECEIVE_ID_TYPE", "chat_id").strip() or "chat_id"
REVIEW_BASE_URL = os.getenv("REVIEW_BASE_URL", "http://localhost:8080").strip() or "http://localhost:8080"

VLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-chat-v3-0324")

# ─── 人脸识别（InsightFace + FAISS） ─────────────────────────────
FACE_RECOGNITION_SRC_PATH = os.getenv(
    "FACE_RECOGNITION_SRC_PATH",
    BUNDLED_FACE_RECOGNITION_SRC_PATH,
).strip() or BUNDLED_FACE_RECOGNITION_SRC_PATH
FACE_MODEL_NAME = os.getenv("FACE_MODEL_NAME", "buffalo_l").strip() or "buffalo_l"
FACE_MAX_SIDE = int(os.getenv("FACE_MAX_SIDE", "1920"))
FACE_DET_THRESHOLD = float(os.getenv("FACE_DET_THRESHOLD", "0.60"))
FACE_SIM_THRESHOLD = float(os.getenv("FACE_SIM_THRESHOLD", "0.50"))
FACE_MIN_SIZE = int(os.getenv("FACE_MIN_SIZE", "48"))
FACE_MATCH_TOP_K = int(os.getenv("FACE_MATCH_TOP_K", "5"))
FACE_MATCH_MARGIN_THRESHOLD = float(os.getenv("FACE_MATCH_MARGIN_THRESHOLD", "0.03"))
FACE_MATCH_WEAK_DELTA = float(os.getenv("FACE_MATCH_WEAK_DELTA", "0.055"))
FACE_MATCH_MIN_QUALITY_GRAY_ZONE = float(os.getenv("FACE_MATCH_MIN_QUALITY_GRAY_ZONE", "0.40"))
FACE_MATCH_HIGH_QUALITY_THRESHOLD = float(
    os.getenv("FACE_MATCH_HIGH_QUALITY_THRESHOLD", str(FACE_MATCH_MIN_QUALITY_GRAY_ZONE))
)
FACE_PROVIDERS = tuple(
    provider.strip()
    for provider in os.getenv("FACE_PROVIDERS", "CPUExecutionProvider").split(",")
    if provider.strip()
)
FACE_LANDMARKS_ENABLED = os.getenv("FACE_LANDMARKS_ENABLED", "true").lower() == "true"
FACE_LANDMARK_MODEL_PATH = os.getenv(
    "FACE_LANDMARK_MODEL_PATH",
    os.path.join(RUNTIME_DIR, "models", "face_landmarker.task"),
).strip() or os.path.join(RUNTIME_DIR, "models", "face_landmarker.task")
FACE_LANDMARK_MODEL_URL = os.getenv(
    "FACE_LANDMARK_MODEL_URL",
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
).strip() or "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
FACE_POSE_PROFILE_YAW_THRESHOLD = float(os.getenv("FACE_POSE_PROFILE_YAW_THRESHOLD", "18.0"))
FACE_PROFILE_RESCUE_DELTA = float(os.getenv("FACE_PROFILE_RESCUE_DELTA", "0.04"))
FACE_PROFILE_RESCUE_MARGIN = float(os.getenv("FACE_PROFILE_RESCUE_MARGIN", "0.02"))
FACE_PROFILE_RESCUE_MIN_QUALITY = float(
    os.getenv("FACE_PROFILE_RESCUE_MIN_QUALITY", str(FACE_MATCH_MIN_QUALITY_GRAY_ZONE))
)
FACE_SAME_PHOTO_MATCH_THRESHOLD = float(os.getenv("FACE_SAME_PHOTO_MATCH_THRESHOLD", "0.52"))

LFW_BENCHMARK_DIR = os.getenv(
    "LFW_BENCHMARK_DIR",
    os.path.join(RUNTIME_DIR, "benchmarks", "lfw"),
)
FACE_MATCH_THRESHOLD_PATH = os.getenv(
    "FACE_MATCH_THRESHOLD_PATH",
    os.path.join(LFW_BENCHMARK_DIR, "latest.json"),
)

# ─── 图片处理 ────────────────────────────────────────────────────
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1024"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))

# ─── 用户名与时间戳 ─────────────────────────────────────────────
USER_NAME = os.environ.get('MEMORY_USER_NAME', 'default')
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ─── 运行产物保留上限 ────────────────────────────────────────────
MAX_OUTPUT_RUNS_PER_USER = int(os.getenv("MAX_OUTPUT_RUNS_PER_USER", "20"))

# ─── 用户专属目录 ────────────────────────────────────────────────
USER_CACHE_DIR = os.path.join(CACHE_DIR, USER_NAME)
USER_OUTPUT_DIR = os.path.join(OUTPUT_DIR, f"{USER_NAME}_记忆测试_{RUN_TIMESTAMP}")

# 数据集目录（用户测试数据与项目代码隔离）
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
TASKS_DIR = OUTPUT_DIR  # 向后兼容 backend/app.py 的 TASKS_DIR 引用

os.makedirs(USER_CACHE_DIR, exist_ok=True)
# 注意：不再在 import 时自动创建 USER_OUTPUT_DIR
# 只在实际运行 pipeline 时才创建，避免产生大量空目录

# ─── 缓存路径（用户隔离） ────────────────────────────────────────
VLM_CACHE_PATH = os.path.join(USER_CACHE_DIR, f"{USER_NAME}.json")
FEATURE_DB_PATH = os.path.join(USER_CACHE_DIR, f"vlm_feature_db_{USER_NAME}.json")

# ─── 人脸识别缓存路径 ────────────────────────────────────────────
FACE_INDEX_PATH = os.path.join(USER_CACHE_DIR, "faces.index")
FACE_STATE_PATH = os.path.join(USER_CACHE_DIR, "face_recognition_state.json")
FACE_OUTPUT_PATH = os.path.join(USER_CACHE_DIR, "face_recognition_output.json")
DEDUP_REPORT_PATH = os.path.join(USER_CACHE_DIR, "dedupe_report.json")

# ─── 输出路径（用户+版本隔离） ───────────────────────────────────
OUTPUT_PATH = os.path.join(USER_OUTPUT_DIR, f"events_{USER_NAME}_{RUN_TIMESTAMP}.json")
DETAILED_OUTPUT_PATH = os.path.join(USER_OUTPUT_DIR, f"memory_detailed_{USER_NAME}_{RUN_TIMESTAMP}.md")
RELATIONSHIPS_OUTPUT_PATH = os.path.join(USER_OUTPUT_DIR, f"relationships_{USER_NAME}_{RUN_TIMESTAMP}.json")
PROFILE_STRUCTURED_PATH = os.path.join(USER_OUTPUT_DIR, f"profile_structured_{USER_NAME}_{RUN_TIMESTAMP}.json")
PROFILE_REPORT_PATH = os.path.join(USER_OUTPUT_DIR, f"user_profile_{USER_NAME}_{RUN_TIMESTAMP}.md")
PROFILE_DEBUG_PATH = os.path.join(USER_OUTPUT_DIR, f"profile_debug_{USER_NAME}_{RUN_TIMESTAMP}.json")
RELATIONSHIP_DOSSIERS_PATH = os.path.join(USER_OUTPUT_DIR, f"relationship_dossiers_{USER_NAME}_{RUN_TIMESTAMP}.json")
GROUP_ARTIFACTS_PATH = os.path.join(USER_OUTPUT_DIR, f"group_artifacts_{USER_NAME}_{RUN_TIMESTAMP}.json")
PROFILE_FACT_DECISIONS_PATH = os.path.join(USER_OUTPUT_DIR, f"profile_fact_decisions_{USER_NAME}_{RUN_TIMESTAMP}.json")
DOWNSTREAM_AUDIT_REPORT_PATH = os.path.join(USER_OUTPUT_DIR, f"downstream_audit_report_{USER_NAME}_{RUN_TIMESTAMP}.json")
FACE_PIPELINE_RESULT_PATH = os.path.join(USER_OUTPUT_DIR, f"face_pipeline_result_{RUN_TIMESTAMP}.json")
FACE_PIPELINE_REPORT_PATH = os.path.join(USER_OUTPUT_DIR, f"face_pipeline_report_{RUN_TIMESTAMP}.md")

# ─── 错误处理 ────────────────────────────────────────────────────
MAX_RETRIES = 3
RETRY_DELAY = 1
CONTINUE_ON_ERROR = True

# ─── 事件提取配置 ────────────────────────────────────────────────
EVENT_TIME_THRESHOLD = 2
EVENT_DISTANCE_THRESHOLD = 1

# ─── 关系推断配置 ────────────────────────────────────────────────
if DEMO_MODE:
    MIN_PHOTO_COUNT = 2
    MIN_TIME_SPAN_DAYS = 1
    MIN_SCENE_VARIETY = 1
else:
    MIN_PHOTO_COUNT = 5
    MIN_TIME_SPAN_DAYS = 7
    MIN_SCENE_VARIETY = 3

RELATIONSHIP_TYPES = {
    "family": {"label": "家人", "description": "血缘或法定家庭关系"},
    "romantic": {"label": "恋人", "description": "确认的浪漫伴侣关系"},
    "bestie": {"label": "密友", "description": "最亲密的非恋爱关系"},
    "close_friend": {"label": "好友", "description": "频繁互动的朋友"},
    "friend": {"label": "朋友", "description": "偶尔一起活动"},
    "classmate_colleague": {"label": "同学/同事", "description": "学校或工作场景中的固定关系"},
    "activity_buddy": {"label": "搭子", "description": "特定活动的伙伴（健身搭子、饭搭子等）"},
    "acquaintance": {"label": "点头之交", "description": "很少出现，关系不深"},
}

# 亲密度权重（LP2 v3.0）
INTIMACY_WEIGHT_FREQUENCY = 0.40
INTIMACY_WEIGHT_INTERACTION = 0.35
INTIMACY_WEIGHT_SCENE_DIVERSITY = 0.25

# 互动打分表（contact_type → score）
INTERACTION_SCORES = {
    "kiss": 1.0,
    "hug": 1.0,
    "holding_hands": 0.8,
    "arm_in_arm": 0.8,
    "selfie_together": 0.5,
    "shoulder_lean": 0.5,
    "sitting_close": 0.4,
    "standing_near": 0.3,
    "no_contact": 0.2,
}

# 私密场景类型（权重 1.5x）
PRIVATE_SCENE_TYPES = {
    "家", "卧室", "客厅", "公寓", "宿舍",
    "home", "bedroom", "living room", "apartment", "dorm",
}

# 事件类型定义
EVENT_TYPES = [
    "社交", "工作", "休闲", "用餐", "运动", "旅行", "购物", "学习", "其他"
]

# ─── 后端服务配置 ────────────────────────────────────────────
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1").strip() or "127.0.0.1"
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
BACKEND_RELOAD = os.getenv("BACKEND_RELOAD", "false").lower() == "true"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(RUNTIME_DIR, 'local_preview.db')}").strip()
SQL_ECHO = os.getenv("SQL_ECHO", "false").lower() == "true"
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5180").strip() or "http://localhost:5180"
CORS_ALLOW_ORIGINS = tuple(
    o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", FRONTEND_ORIGIN).split(",") if o.strip()
)
ALLOW_SELF_REGISTRATION = os.getenv("ALLOW_SELF_REGISTRATION", "true").lower() == "true"
HIGH_SECURITY_MODE = os.getenv("HIGH_SECURITY_MODE", "false").lower() == "true"
APP_ROLE = os.getenv("APP_ROLE", "control-plane").strip() or "control-plane"
ASSET_URL_PREFIX = os.getenv("ASSET_URL_PREFIX", "").strip()
AUTH_SESSION_COOKIE_NAME = os.getenv("AUTH_SESSION_COOKIE_NAME", "session_token").strip() or "session_token"
AUTH_SESSION_DAYS = int(os.getenv("AUTH_SESSION_DAYS", "30"))
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"
WORKER_BOOT_TIMEOUT_SECONDS = int(os.getenv("WORKER_BOOT_TIMEOUT_SECONDS", "300"))
WORKER_INTERNAL_PORT = int(os.getenv("WORKER_INTERNAL_PORT", "9000"))
WORKER_SHARED_TOKEN = os.getenv("WORKER_SHARED_TOKEN", "").strip()

# ─── 对象存储（S3 兼容） ──────────────────────────────────────
OBJECT_STORAGE_ACCESS_KEY_ID = os.getenv("OBJECT_STORAGE_ACCESS_KEY_ID", "").strip()
OBJECT_STORAGE_SECRET_ACCESS_KEY = os.getenv("OBJECT_STORAGE_SECRET_ACCESS_KEY", "").strip()
OBJECT_STORAGE_ENDPOINT = os.getenv("OBJECT_STORAGE_ENDPOINT", "").strip()
OBJECT_STORAGE_REGION = os.getenv("OBJECT_STORAGE_REGION", "").strip()
OBJECT_STORAGE_BUCKET = os.getenv("OBJECT_STORAGE_BUCKET", "").strip()
OBJECT_STORAGE_PREFIX = os.getenv("OBJECT_STORAGE_PREFIX", "").strip()
OBJECT_STORAGE_ADDRESSING_STYLE = os.getenv("OBJECT_STORAGE_ADDRESSING_STYLE", "auto").strip() or "auto"

# ─── AWS / Worker ────────────────────────────────────────────
AWS_REGION = os.getenv("AWS_REGION", "us-east-1").strip() or "us-east-1"
BEDROCK_REGION = os.getenv("BEDROCK_REGION", AWS_REGION).strip() or AWS_REGION
BEDROCK_REQUEST_TIMEOUT_SECONDS = int(os.getenv("BEDROCK_REQUEST_TIMEOUT_SECONDS", "60"))
RESULT_TTL_HOURS = int(os.getenv("RESULT_TTL_HOURS", "72"))
WORKER_ORCHESTRATION_ENABLED = os.getenv("WORKER_ORCHESTRATION_ENABLED", "false").lower() == "true"
WORKER_AMI_ID = os.getenv("WORKER_AMI_ID", "").strip()
WORKER_INSTANCE_TYPE = os.getenv("WORKER_INSTANCE_TYPE", "g4dn.xlarge").strip()
WORKER_IAM_INSTANCE_PROFILE = os.getenv("WORKER_IAM_INSTANCE_PROFILE", "").strip()
WORKER_SECURITY_GROUP_ID = os.getenv("WORKER_SECURITY_GROUP_ID", "").strip()
WORKER_SUBNET_IDS = os.getenv("WORKER_SUBNET_IDS", "").strip()
WORKER_INSTANCE_NAME_PREFIX = os.getenv("WORKER_INSTANCE_NAME_PREFIX", "me-worker").strip()
WORKER_LAUNCH_TEMPLATE_ID = os.getenv("WORKER_LAUNCH_TEMPLATE_ID", "").strip()
WORKER_LAUNCH_TEMPLATE_VERSION = os.getenv("WORKER_LAUNCH_TEMPLATE_VERSION", "").strip()

def normalize_task_version(version: str | None) -> str:
    v = (version or "").strip()
    return v if v in AVAILABLE_TASK_VERSIONS else DEFAULT_TASK_VERSION
