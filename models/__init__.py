"""
数据模型定义（v2.0 — 合并新工程 Photo/Person + 老工程 Relationship/Event/UserProfile）
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class Photo:
    """照片对象"""
    photo_id: str
    filename: str
    path: str
    timestamp: datetime
    location: Dict[str, Any]  # {lat, lng, name}
    source_hash: Optional[str] = None  # 原始上传文件的稳定哈希
    original_path: Optional[str] = None  # 原始照片路径（HEIC等）
    compressed_path: Optional[str] = None  # 压缩后的JPEG路径（用于VLM）
    boxed_path: Optional[str] = None  # 带框图片路径（用于VLM提示）
    face_image_hash: Optional[str] = None  # face-recognition 侧的图像哈希
    primary_person_id: Optional[str] = None  # 当前相册的主用户 ID
    processing_errors: Dict[str, str] = field(default_factory=dict)

    # 人脸识别结果
    faces: List[Dict] = field(default_factory=list)
    # [
    #   {
    #     "face_id": str,
    #     "person_id": "Person_001",
    #     "score": 0.98,
    #     "similarity": 0.87,
    #     "faiss_id": 12,
    #     "bbox": [x1, y1, x2, y2],
    #     "bbox_xywh": {"x": 10, "y": 20, "w": 30, "h": 40},
    #     "kps": [...]
    #   }
    # ]

    # VLM分析结果
    vlm_analysis: Optional[Dict] = None


@dataclass
class Person:
    """人物对象"""
    person_id: str
    name: str
    features: List[Any] = field(default_factory=list)  # 人脸特征向量
    photo_count: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    avg_confidence: float = 0.0
    avg_quality: float = 0.0
    high_quality_face_count: int = 0


@dataclass
class Event:
    """事件对象（v2.1 - 增强版，支持meta_info和objective_fact）"""
    event_id: str
    date: str
    time_range: str
    duration: str
    title: str
    type: str  # 社交/工作/休闲/用餐/运动/旅行/节日/医疗/生活
    participants: List[str]
    location: str
    description: str
    photo_count: int
    confidence: float
    reason: str

    # 核心叙事字段
    narrative: str = ""
    narrative_synthesis: str = ""

    # 新增 v2.1 字段
    meta_info: Dict[str, Any] = field(default_factory=dict)
    objective_fact: Dict[str, Any] = field(default_factory=dict)

    # 社交分析
    social_interaction: Dict[str, Any] = field(default_factory=dict)
    social_dynamics: List[Dict[str, Any]] = field(default_factory=list)

    # 标签
    lifestyle_tags: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    # 画像证据
    social_slices: List[Dict[str, Any]] = field(default_factory=list)
    persona_evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relationship:
    """关系对象（v3.1 — 从"判标签"到"建档案"）"""
    person_id: str
    relationship_type: str  # family/romantic/bestie/close_friend/friend/classmate_colleague/activity_buddy/acquaintance
    intimacy_score: float   # 0-1, code-computed
    status: str             # new/growing/stable/fading/gone
    confidence: float
    reasoning: str
    shared_events: List[Dict] = field(default_factory=list)  # [{event_id, date, narrative}]
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserProfile:
    """用户画像对象"""
    basic_info: Dict[str, Any]
    lifestyle: Dict[str, Any]
    personality: Dict[str, Any]
    interests: Dict[str, Any]
    values: Dict[str, Any]
    place: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    one_more_thing: Optional[str] = None
