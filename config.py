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
    from vendor.face_recognition_src.face_recognition.config import PipelineConfig as PipelineConfig
except Exception:
    class PipelineConfig:  # pragma: no cover - optional legacy shim
        pass

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


def _fallback_load_dotenv(dotenv_path: str) -> None:
    if not os.path.exists(dotenv_path):
        return
    with open(dotenv_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            os.environ[key] = value


def _first_nonempty_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        normalized = value.strip()
        if normalized:
            return normalized
    return default


if load_dotenv is not None:
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
else:
    _fallback_load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Demo配置
MAX_PHOTOS = 50  # Demo阶段最多处理50张照片
DEMO_MODE = True  # Demo模式
MAX_UPLOAD_PHOTOS = int(os.getenv("MAX_UPLOAD_PHOTOS", "5000"))
TASK_VERSION_V0312 = "v0312"
TASK_VERSION_V0315 = "v0315"
TASK_VERSION_V0317 = "v0317"
TASK_VERSION_V0317_HEAVY = "v0317-Heavy"
TASK_VERSION_V0321_2 = "v0321.2"
TASK_VERSION_V0321_3 = "v0321.3"
TASK_VERSION_V0323 = "v0323"
TASK_VERSION_V0325 = "v0325"
TASK_VERSION_V0327_EXP = "v0327-exp"
TASK_VERSION_V0327_DB = "v0327-db"
TASK_VERSION_V0327_DB_QUERY = "v0327-db-query"
AVAILABLE_TASK_VERSIONS = (
    TASK_VERSION_V0312,
    TASK_VERSION_V0315,
    TASK_VERSION_V0317,
    TASK_VERSION_V0317_HEAVY,
    TASK_VERSION_V0321_2,
    TASK_VERSION_V0321_3,
    TASK_VERSION_V0323,
    TASK_VERSION_V0325,
    TASK_VERSION_V0327_EXP,
    TASK_VERSION_V0327_DB,
    TASK_VERSION_V0327_DB_QUERY,
)
APP_VERSION = os.getenv("APP_VERSION", TASK_VERSION_V0317).strip() or TASK_VERSION_V0317
DEFAULT_TASK_VERSION = os.getenv("DEFAULT_TASK_VERSION", TASK_VERSION_V0327_DB_QUERY).strip() or TASK_VERSION_V0327_DB_QUERY
if DEFAULT_TASK_VERSION not in AVAILABLE_TASK_VERSIONS:
    DEFAULT_TASK_VERSION = TASK_VERSION_V0327_DB_QUERY
DEFAULT_NORMALIZE_LIVE_PHOTOS = os.getenv("DEFAULT_NORMALIZE_LIVE_PHOTOS", "true").lower() == "true"

# Web 服务配置
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("PORT", os.getenv("BACKEND_PORT", "8000")))
BACKEND_RELOAD = os.getenv("BACKEND_RELOAD", "false").lower() == "true"
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
CONTROL_PLANE_INTERNAL_ORIGIN = (
    os.getenv("CONTROL_PLANE_INTERNAL_ORIGIN", f"http://127.0.0.1:{BACKEND_PORT}").strip()
    or f"http://127.0.0.1:{BACKEND_PORT}"
)
RUNS_URL_PREFIX = "/runs"
ASSET_URL_PREFIX = "/api/assets"
AUTH_SESSION_COOKIE_NAME = os.getenv("AUTH_SESSION_COOKIE_NAME", "memory_session")
AUTH_SESSION_DAYS = int(os.getenv("AUTH_SESSION_DAYS", "14"))
HIGH_SECURITY_MODE = os.getenv("HIGH_SECURITY_MODE", "false").lower() == "true"
ALLOW_SELF_REGISTRATION = (
    False
    if HIGH_SECURITY_MODE
    else os.getenv("ALLOW_SELF_REGISTRATION", "true").lower() == "true"
)
COOKIE_SECURE = os.getenv(
    "COOKIE_SECURE",
    "true" if FRONTEND_ORIGIN.startswith("https://") else "false",
).lower() == "true"
APP_ROLE = os.getenv("APP_ROLE", "control-plane").strip().lower() or "control-plane"
WORKER_ORCHESTRATION_ENABLED = os.getenv("WORKER_ORCHESTRATION_ENABLED", "false").lower() == "true"
AWS_REGION = os.getenv("AWS_REGION", "").strip()
WORKER_LAUNCH_TEMPLATE_ID = os.getenv("WORKER_LAUNCH_TEMPLATE_ID", "").strip()
WORKER_LAUNCH_TEMPLATE_VERSION = os.getenv("WORKER_LAUNCH_TEMPLATE_VERSION", "").strip() or "$Default"
WORKER_AMI_ID = os.getenv("WORKER_AMI_ID", "").strip()
WORKER_INSTANCE_TYPE = os.getenv("WORKER_INSTANCE_TYPE", "").strip()
WORKER_IAM_INSTANCE_PROFILE = os.getenv("WORKER_IAM_INSTANCE_PROFILE", "").strip()
WORKER_SECURITY_GROUP_ID = os.getenv("WORKER_SECURITY_GROUP_ID", "").strip()
WORKER_INSTANCE_NAME_PREFIX = os.getenv("WORKER_INSTANCE_NAME_PREFIX", "memory-worker").strip() or "memory-worker"
WORKER_INTERNAL_PORT = int(os.getenv("WORKER_INTERNAL_PORT", "9000"))
WORKER_SHARED_TOKEN = os.getenv("WORKER_SHARED_TOKEN", "").strip()
SERVICE_AUTH_TOKEN = os.getenv("SERVICE_AUTH_TOKEN", "").strip() or WORKER_SHARED_TOKEN
WORKER_TASK_ROOT = os.getenv("WORKER_TASK_ROOT", "/mnt/secure-tasks").strip() or "/mnt/secure-tasks"
RESULT_TTL_HOURS = int(os.getenv("RESULT_TTL_HOURS", "24"))
WORKER_POLL_SECONDS = int(os.getenv("WORKER_POLL_SECONDS", "3"))
WORKER_BOOT_TIMEOUT_SECONDS = int(os.getenv("WORKER_BOOT_TIMEOUT_SECONDS", "300"))
KAFKA_BOOTSTRAP_SERVERS = tuple(
    item
    for item in (part.strip() for part in os.getenv("KAFKA_BOOTSTRAP_SERVERS", "").split(","))
    if item
)
KAFKA_ENABLED = os.getenv("KAFKA_ENABLED", "true").lower() == "true" and bool(KAFKA_BOOTSTRAP_SERVERS)
KAFKA_TERMINAL_TOPIC = (
    os.getenv("KAFKA_TERMINAL_TOPIC", "memory.task.terminal.v1").strip()
    or "memory.task.terminal.v1"
)
KAFKA_SURVEY_IMPORT_TOPIC = (
    os.getenv("KAFKA_SURVEY_IMPORT_TOPIC", "memory.task.survey-import.v1").strip()
    or "memory.task.survey-import.v1"
)
KAFKA_CLIENT_ID = (
    os.getenv("KAFKA_CLIENT_ID", "memory-engineering-terminal-publisher").strip()
    or "memory-engineering-terminal-publisher"
)
KAFKA_SECURITY_PROTOCOL = os.getenv("KAFKA_SECURITY_PROTOCOL", "").strip()
KAFKA_SASL_MECHANISM = os.getenv("KAFKA_SASL_MECHANISM", "").strip()
KAFKA_SASL_USERNAME = _first_nonempty_env("KAFKA_SASL_USERNAME", "KAFKA_USERNAME")
KAFKA_SASL_PASSWORD = _first_nonempty_env("KAFKA_SASL_PASSWORD", "KAFKA_PASSWORD")
KAFKA_MESSAGE_MAX_BYTES = int(os.getenv("KAFKA_MESSAGE_MAX_BYTES", str(800 * 1024)))
KAFKA_PUBLISHER_BATCH_SIZE = int(os.getenv("KAFKA_PUBLISHER_BATCH_SIZE", "20"))
KAFKA_PUBLISHER_POLL_SECONDS = float(os.getenv("KAFKA_PUBLISHER_POLL_SECONDS", "2"))
KAFKA_PUBLISHER_LOCK_SECONDS = int(os.getenv("KAFKA_PUBLISHER_LOCK_SECONDS", "60"))
KAFKA_PUBLISHER_RETRY_SECONDS = int(os.getenv("KAFKA_PUBLISHER_RETRY_SECONDS", "10"))


def normalize_task_version(value: str | None, *, fallback: str = DEFAULT_TASK_VERSION) -> str:
    candidate = (value or "").strip() or fallback
    if candidate not in AVAILABLE_TASK_VERSIONS:
        raise ValueError(f"不支持的任务版本: {candidate}")
    return candidate


def _parse_csv_list(value: str) -> tuple[str, ...]:
    return tuple(
        item
        for item in (part.strip() for part in value.split(","))
        if item
    )


def _parse_origin_list(value: str) -> tuple[str, ...]:
    return tuple(item.rstrip("/") for item in _parse_csv_list(value))


_extra_cors_origins = _parse_origin_list(os.getenv("CORS_ALLOW_ORIGINS", ""))
WORKER_SUBNET_IDS = _parse_csv_list(os.getenv("WORKER_SUBNET_IDS", ""))
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
OBJECT_STORAGE_BUCKET = _first_nonempty_env("BUCKET", "OBJECT_STORAGE_BUCKET")
OBJECT_STORAGE_ENDPOINT = _first_nonempty_env("ENDPOINT", "OBJECT_STORAGE_ENDPOINT")
OBJECT_STORAGE_REGION = _first_nonempty_env("REGION", "OBJECT_STORAGE_REGION", default="auto")
OBJECT_STORAGE_ACCESS_KEY_ID = _first_nonempty_env("ACCESS_KEY_ID", "OBJECT_STORAGE_ACCESS_KEY_ID")
OBJECT_STORAGE_SECRET_ACCESS_KEY = _first_nonempty_env("SECRET_ACCESS_KEY", "OBJECT_STORAGE_SECRET_ACCESS_KEY")
OBJECT_STORAGE_PREFIX = os.getenv("OBJECT_STORAGE_PREFIX", "tasks")
OBJECT_STORAGE_ADDRESSING_STYLE = os.getenv("OBJECT_STORAGE_ADDRESSING_STYLE", "auto")

# Memory 外部存储适配器（可选）
MEMORY_EXTERNAL_SINKS_ENABLED = os.getenv("MEMORY_EXTERNAL_SINKS_ENABLED", "false").lower() == "true"
MEMORY_REDIS_URL = os.getenv("MEMORY_REDIS_URL", "").strip()
MEMORY_REDIS_PREFIX = os.getenv("MEMORY_REDIS_PREFIX", "memory").strip() or "memory"
MEMORY_NEO4J_URI = os.getenv("MEMORY_NEO4J_URI", "").strip()
MEMORY_NEO4J_USERNAME = os.getenv("MEMORY_NEO4J_USERNAME", "").strip()
MEMORY_NEO4J_PASSWORD = os.getenv("MEMORY_NEO4J_PASSWORD", "").strip()
MEMORY_NEO4J_DATABASE = os.getenv("MEMORY_NEO4J_DATABASE", "").strip()
MEMORY_MILVUS_URI = os.getenv("MEMORY_MILVUS_URI", "").strip()
MEMORY_MILVUS_USER = os.getenv("MEMORY_MILVUS_USER", "").strip()
MEMORY_MILVUS_PASSWORD = os.getenv("MEMORY_MILVUS_PASSWORD", "").strip()
MEMORY_MILVUS_TOKEN = os.getenv("MEMORY_MILVUS_TOKEN", "").strip()
MEMORY_MILVUS_DB_NAME = os.getenv("MEMORY_MILVUS_DB_NAME", "").strip()
MEMORY_MILVUS_COLLECTION = os.getenv("MEMORY_MILVUS_COLLECTION", "memory_segments").strip() or "memory_segments"
MEMORY_MILVUS_UNITS_COLLECTION = (
    os.getenv("MEMORY_MILVUS_UNITS_COLLECTION", "memory_units_v2").strip() or "memory_units_v2"
)
MEMORY_MILVUS_EVIDENCE_COLLECTION = (
    os.getenv("MEMORY_MILVUS_EVIDENCE_COLLECTION", "memory_evidence_v2").strip() or "memory_evidence_v2"
)
MEMORY_MILVUS_VECTOR_DIM = int(os.getenv("MEMORY_MILVUS_VECTOR_DIM", "512"))
MEMORY_QUERY_V1_ENABLED = os.getenv("MEMORY_QUERY_V1_ENABLED", "true").lower() == "true"
MEMORY_QUERY_V1_SHADOW_COMPARE = os.getenv("MEMORY_QUERY_V1_SHADOW_COMPARE", "false").lower() == "true"
MEMORY_QUERY_V1_EVENT_COLLECTION = (
    os.getenv("MEMORY_QUERY_V1_EVENT_COLLECTION", "event_views_v1").strip() or "event_views_v1"
)
MEMORY_QUERY_V1_EVIDENCE_COLLECTION = (
    os.getenv("MEMORY_QUERY_V1_EVIDENCE_COLLECTION", "evidence_docs_v1").strip() or "evidence_docs_v1"
)
MEMORY_REAL_EMBEDDINGS_ENABLED = os.getenv("MEMORY_REAL_EMBEDDINGS_ENABLED", "false").lower() == "true"
MEMORY_EMBEDDING_PROVIDER = os.getenv("MEMORY_EMBEDDING_PROVIDER", "auto").strip().lower() or "auto"
MEMORY_EMBEDDING_MODEL = os.getenv("MEMORY_EMBEDDING_MODEL", "").strip()
MEMORY_EMBEDDING_VERSION = os.getenv("MEMORY_EMBEDDING_VERSION", "v1").strip() or "v1"
MEMORY_EMBEDDING_TIMEOUT_SECONDS = float(os.getenv("MEMORY_EMBEDDING_TIMEOUT_SECONDS", "30"))

# API配置 - 从环境变量读取
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
AMAP_API_KEY = os.getenv("AMAP_API_KEY", "")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "").strip().lower()
VLM_PROVIDER = os.getenv("VLM_PROVIDER", "").strip().lower()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = (
    os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
    or "https://openrouter.ai/api/v1"
)
OPENROUTER_SITE_URL = (
    os.getenv("OPENROUTER_SITE_URL", FRONTEND_ORIGIN).strip()
    or FRONTEND_ORIGIN
)
OPENROUTER_APP_NAME = (
    os.getenv("OPENROUTER_APP_NAME", "Memory Engineering").strip()
    or "Memory Engineering"
)
OPENROUTER_VLM_MODEL = (
    os.getenv("OPENROUTER_VLM_MODEL", "google/gemini-3.1-pro-preview").strip()
    or "google/gemini-3.1-pro-preview"
)
OPENROUTER_LLM_MODEL = (
    os.getenv("OPENROUTER_LLM_MODEL", "google/gemini-3.1-pro-preview").strip()
    or "google/gemini-3.1-pro-preview"
)
OPENROUTER_AGENT_MODEL = (
    os.getenv("OPENROUTER_AGENT_MODEL", "qwen/qwen3.5-plus-02-15").strip()
    or "qwen/qwen3.5-plus-02-15"
)
PROFILE_LLM_PROVIDER = (
    os.getenv("PROFILE_LLM_PROVIDER", "bedrock").strip().lower()
    or "bedrock"
)
PROFILE_LLM_MODEL = (
    os.getenv("PROFILE_LLM_MODEL", "global.anthropic.claude-opus-4-6-v1").strip()
    or "global.anthropic.claude-opus-4-6-v1"
)
PROFILE_AGENT_ROOT = os.getenv("PROFILE_AGENT_ROOT", "").strip()
V0323_OPENROUTER_MODEL = (
    os.getenv("V0323_OPENROUTER_MODEL", "google/gemini-3.1-pro-preview").strip()
    or "google/gemini-3.1-pro-preview"
)
V0325_OPENROUTER_LLM_MODEL = (
    os.getenv("V0325_OPENROUTER_LLM_MODEL", "google/gemini-3.1-pro-preview").strip()
    or "google/gemini-3.1-pro-preview"
)
V0325_OPENROUTER_VLM_MODEL = (
    os.getenv("V0325_OPENROUTER_VLM_MODEL", "google/gemini-3.1-pro-preview").strip()
    or "google/gemini-3.1-pro-preview"
)
V0323_LP1_MAX_OUTPUT_TOKENS = int(os.getenv("V0323_LP1_MAX_OUTPUT_TOKENS", "24576"))
OPENROUTER_REASONING_EFFORT = os.getenv("OPENROUTER_REASONING_EFFORT", "").strip().lower()
BEDROCK_REGION = (
    os.getenv("BEDROCK_REGION", AWS_REGION or "ap-southeast-1").strip()
    or "ap-southeast-1"
)
BEDROCK_VLM_MODEL_PRIMARY = (
    os.getenv(
        "BEDROCK_VLM_MODEL_PRIMARY",
        "amazon.nova-2-pro-preview-20251202-v1:0",
    ).strip()
    or "amazon.nova-2-pro-preview-20251202-v1:0"
)
BEDROCK_VLM_MODEL_FALLBACK = (
    os.getenv("BEDROCK_VLM_MODEL_FALLBACK", "amazon.nova-pro-v1:0").strip()
    or "amazon.nova-pro-v1:0"
)
BEDROCK_VLM_MODEL_POLICY = (
    os.getenv("BEDROCK_VLM_MODEL_POLICY", "primary").strip().lower()
    or "primary"
)
BEDROCK_LLM_MODEL = (
    os.getenv("BEDROCK_LLM_MODEL", "global.anthropic.claude-opus-4-6-v1").strip()
    or "global.anthropic.claude-opus-4-6-v1"
)
BEDROCK_RELATIONSHIP_LLM_MODEL = (
    os.getenv("BEDROCK_RELATIONSHIP_LLM_MODEL", BEDROCK_LLM_MODEL).strip()
    or BEDROCK_LLM_MODEL
)
BEDROCK_RELATIONSHIP_LLM_FALLBACK_MODEL = (
    os.getenv("BEDROCK_RELATIONSHIP_LLM_FALLBACK_MODEL", BEDROCK_LLM_MODEL).strip()
    or BEDROCK_LLM_MODEL
)
BEDROCK_MAX_OUTPUT_TOKENS = int(os.getenv("BEDROCK_MAX_OUTPUT_TOKENS", "8192"))
BEDROCK_VLM_MAX_OUTPUT_TOKENS = int(os.getenv("BEDROCK_VLM_MAX_OUTPUT_TOKENS", "4096"))
BEDROCK_LLM_MAX_OUTPUT_TOKENS = int(os.getenv("BEDROCK_LLM_MAX_OUTPUT_TOKENS", str(BEDROCK_MAX_OUTPUT_TOKENS)))
BEDROCK_RELATIONSHIP_MAX_OUTPUT_TOKENS = int(
    os.getenv("BEDROCK_RELATIONSHIP_MAX_OUTPUT_TOKENS", "2048")
)
BEDROCK_REQUEST_TIMEOUT_SECONDS = int(os.getenv("BEDROCK_REQUEST_TIMEOUT_SECONDS", "120"))
RELATIONSHIP_REQUEST_TIMEOUT_SECONDS = int(os.getenv("RELATIONSHIP_REQUEST_TIMEOUT_SECONDS", "45"))
RELATIONSHIP_MAX_RETRIES = max(1, int(os.getenv("RELATIONSHIP_MAX_RETRIES", "2")))
V0323_LP1_MAX_ATTEMPTS = max(1, int(os.getenv("V0323_LP1_MAX_ATTEMPTS", "2")))
V0323_LP2_TIMEOUT_SCHEDULE_SECONDS = [
    max(1, int(item.strip()))
    for item in os.getenv("V0323_LP2_TIMEOUT_SCHEDULE_SECONDS", "60,120").split(",")
    if item.strip()
] or [60, 120]
RELATIONSHIP_MAX_CONCURRENCY = max(1, int(os.getenv("RELATIONSHIP_MAX_CONCURRENCY", "3")))
RELATIONSHIP_MIN_DISTINCT_DAYS = int(os.getenv("RELATIONSHIP_MIN_DISTINCT_DAYS", "2"))
RELATIONSHIP_MIN_CO_OCCURRENCE = int(os.getenv("RELATIONSHIP_MIN_CO_OCCURRENCE", "3"))
RELATIONSHIP_MIN_INTIMACY_SCORE = float(os.getenv("RELATIONSHIP_MIN_INTIMACY_SCORE", "0.35"))
VLM_MAX_CONCURRENCY = max(1, int(os.getenv("VLM_MAX_CONCURRENCY", "4")))
VLM_CACHE_FLUSH_EVERY_N = max(1, int(os.getenv("VLM_CACHE_FLUSH_EVERY_N", "10")))
VLM_CACHE_FLUSH_INTERVAL_SECONDS = max(
    1.0,
    float(os.getenv("VLM_CACHE_FLUSH_INTERVAL_SECONDS", "15")),
)
VLM_ENABLE_PRIORITY_SCHEDULING = os.getenv("VLM_ENABLE_PRIORITY_SCHEDULING", "true").lower() == "true"

# 代理服务配置（可选）
USE_API_PROXY = os.getenv("USE_API_PROXY", "false").lower() == "true"
API_PROXY_URL = os.getenv("API_PROXY_URL", "")  # 代理服务基础 URL
API_PROXY_KEY = os.getenv("API_PROXY_KEY", "")  # 代理服务 API Key
API_PROXY_MODEL = os.getenv("API_PROXY_MODEL", "gemini-2.0-flash")  # 代理支持的模型

def _resolve_model_provider(explicit: str = "", *, fallback: str = "") -> str:
    explicit = explicit.strip().lower()
    if explicit in {"gemini", "proxy", "openrouter", "bedrock"}:
        return explicit
    fallback = fallback.strip().lower()
    if fallback in {"gemini", "proxy", "openrouter", "bedrock"}:
        return fallback
    if USE_API_PROXY:
        return "proxy"
    if OPENROUTER_API_KEY:
        return "openrouter"
    if GEMINI_API_KEY.startswith("sk-"):
        return "openrouter"
    return "gemini"


MODEL_PROVIDER = _resolve_model_provider(MODEL_PROVIDER)
VLM_PROVIDER = _resolve_model_provider(VLM_PROVIDER, fallback=MODEL_PROVIDER)
LLM_PROVIDER = _resolve_model_provider(LLM_PROVIDER, fallback=MODEL_PROVIDER)
RELATIONSHIP_FOLLOWS_MAIN_LLM = os.getenv("RELATIONSHIP_FOLLOWS_MAIN_LLM", "true").lower() == "true"
RELATIONSHIP_PROVIDER = _resolve_model_provider(
    os.getenv("RELATIONSHIP_PROVIDER", ""),
    fallback=LLM_PROVIDER if RELATIONSHIP_FOLLOWS_MAIN_LLM else LLM_PROVIDER,
)

if VLM_PROVIDER == "openrouter":
    VLM_MODEL = OPENROUTER_VLM_MODEL
elif VLM_PROVIDER == "bedrock":
    VLM_MODEL = (
        BEDROCK_VLM_MODEL_FALLBACK
        if BEDROCK_VLM_MODEL_POLICY == "fallback"
        else BEDROCK_VLM_MODEL_PRIMARY
    )
else:
    VLM_MODEL = "gemini-2.0-flash"

if LLM_PROVIDER == "openrouter":
    LLM_MODEL = OPENROUTER_LLM_MODEL
elif LLM_PROVIDER == "bedrock":
    LLM_MODEL = BEDROCK_LLM_MODEL
else:
    LLM_MODEL = "gemini-2.5-flash"  # 画像生成使用 Flash 2.5

# 人脸识别配置（默认使用仓库内置的 vendored face-recognition 源码）
FACE_RECOGNITION_SRC_PATH = os.getenv(
    "FACE_RECOGNITION_SRC_PATH",
    BUNDLED_FACE_RECOGNITION_SRC_PATH,
).strip() or BUNDLED_FACE_RECOGNITION_SRC_PATH
FACE_MODEL_NAME = os.getenv("FACE_MODEL_NAME", "buffalo_l")
FACE_MAX_SIDE = int(os.getenv("FACE_MAX_SIDE", "1920"))
FACE_DET_THRESHOLD = float(os.getenv("FACE_DET_THRESHOLD", "0.60"))
FACE_SIM_THRESHOLD = float(os.getenv("FACE_SIM_THRESHOLD", "0.50"))
FACE_MIN_SIZE = int(os.getenv("FACE_MIN_SIZE", "48"))  # 最小人脸尺寸（像素）
FACE_MATCH_TOP_K = int(os.getenv("FACE_MATCH_TOP_K", "5"))
FACE_MATCH_MARGIN_THRESHOLD = float(os.getenv("FACE_MATCH_MARGIN_THRESHOLD", "0.03"))
FACE_MATCH_WEAK_DELTA = float(os.getenv("FACE_MATCH_WEAK_DELTA", "0.055"))
FACE_MATCH_MIN_QUALITY_GRAY_ZONE = float(os.getenv("FACE_MATCH_MIN_QUALITY_GRAY_ZONE", "0.40"))
FACE_MATCH_HIGH_QUALITY_THRESHOLD = float(
    os.getenv("FACE_MATCH_HIGH_QUALITY_THRESHOLD", str(FACE_MATCH_MIN_QUALITY_GRAY_ZONE))
)
LFW_BENCHMARK_DIR = os.getenv(
    "LFW_BENCHMARK_DIR",
    os.path.join(RUNTIME_DIR, "benchmarks", "lfw"),
)
MEMORY_BENCHMARK_DIR = os.getenv(
    "MEMORY_BENCHMARK_DIR",
    os.path.join(RUNTIME_DIR, "benchmarks", "memory_v0317"),
)
FACE_MATCH_THRESHOLD_PATH = os.getenv(
    "FACE_MATCH_THRESHOLD_PATH",
    os.path.join(LFW_BENCHMARK_DIR, "latest.json"),
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

# 图片处理配置
MAX_IMAGE_SIZE = 1536  # 压缩后最大边长
JPEG_QUALITY = 85  # JPEG质量

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
LLM_MEMORY_CONTRACT_PATH = os.path.join(OUTPUT_DIR, "memory_contract.json")
LLM_CHUNK_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "llm_chunks.json")
DEDUP_REPORT_PATH = os.path.join(CACHE_DIR, "dedupe_report.json")

# 错误处理
MAX_RETRIES = 3  # 最大重试次数
RETRY_DELAY = 1  # 重试延迟（秒）
CONTINUE_ON_ERROR = True  # 出错后继续处理

# 分层切分与大批量处理配置
LLM_BURST_GAP_SECONDS = int(os.getenv("LLM_BURST_GAP_SECONDS", "90"))
LLM_BURST_MAX_DURATION_SECONDS = int(os.getenv("LLM_BURST_MAX_DURATION_SECONDS", "180"))
LLM_BURST_MAX_PHOTOS = int(os.getenv("LLM_BURST_MAX_PHOTOS", "30"))
LLM_SESSION_HARD_GAP_SECONDS = int(os.getenv("LLM_SESSION_HARD_GAP_SECONDS", str(4 * 60 * 60)))
LLM_SESSION_STRONG_GAP_SECONDS = int(os.getenv("LLM_SESSION_STRONG_GAP_SECONDS", str(30 * 60)))
LLM_SESSION_SOFT_GAP_SECONDS = int(os.getenv("LLM_SESSION_SOFT_GAP_SECONDS", str(2 * 60 * 60)))
LLM_SESSION_NEAR_DISTANCE_KM = float(os.getenv("LLM_SESSION_NEAR_DISTANCE_KM", "1.5"))
LLM_SESSION_HARD_DISTANCE_KM = float(os.getenv("LLM_SESSION_HARD_DISTANCE_KM", "20"))
LLM_SLICE_MAX_PHOTOS = int(os.getenv("LLM_SLICE_MAX_PHOTOS", "18"))
LLM_SLICE_MAX_RARE_CLUES = int(os.getenv("LLM_SLICE_MAX_RARE_CLUES", "14"))
LLM_SLICE_MIN_PHOTOS = int(os.getenv("LLM_SLICE_MIN_PHOTOS", "4"))
LLM_SLICE_MAX_BURSTS = int(os.getenv("LLM_SLICE_MAX_BURSTS", "12"))
LLM_SLICE_HARD_MAX_BURSTS = int(os.getenv("LLM_SLICE_HARD_MAX_BURSTS", "20"))
LLM_SLICE_OVERLAP_BURSTS = int(os.getenv("LLM_SLICE_OVERLAP_BURSTS", "1"))
LLM_SLICE_MAX_INFO_SCORE = float(os.getenv("LLM_SLICE_MAX_INFO_SCORE", "36"))
LLM_SLICE_MAX_DENSITY_SCORE = float(os.getenv("LLM_SLICE_MAX_DENSITY_SCORE", "18"))

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
