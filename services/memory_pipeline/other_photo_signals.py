"""提取他拍信号的独立 Tool。

核心职责：
- 遍历所有 VLM 结果，识别"他拍"部分（非自拍、非身份照）
- 为每个人物构建"他拍轮廓"（OtherPhotoProfile）
- 预先计算所有 veto 规则及其证据
- 生成 confidence_score，表示该人作为"他拍主角"的合理性
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from .types import PersonScreening


# ===== 规则常量 =====
PORTRAIT_SHOOTING_KEYWORDS = (
    "人像拍摄",
    "portrait",
    "展示照",
    "主角作为拍摄者记录",
    "as photographer",
    "拍摄者记录",
    "记录",
)
SELFIE_KEYWORDS = ("自拍", "selfie", "mirror selfie", "镜中自拍", "front camera")
IDENTITY_KEYWORDS = ("证件", "工牌", "student id", "学生证", "id card", "badge", "证件照")
MIRROR_KEYWORDS = ("镜", "mirror")
USER_VIEW_KEYWORDS = ("第一视角", "as photographer", "拍摄者", "主角作为拍摄者", "user view", "recording")

# ===== 阈值常量 =====
PRIMARY_OTHER_PHOTO_MIN_COUNT = 4
PRIMARY_OTHER_PHOTO_LEAD = 2
PRIMARY_OTHER_PHOTO_MIN_SCENES = 2
PRIMARY_PORTRAIT_TARGET_RATIO_MAX = 0.65
PRIMARY_PHOTOGRAPHED_SUBJECT_RATIO_MAX = 0.6
PRIMARY_PHOTOGRAPHER_NOFACE_RATIO = 0.6


@dataclass
class OtherPhotoProfile:
    """他拍候选的轮廓。"""

    person_id: str
    non_selfie_photo_count: int  # 他拍总数
    cross_scene_presence_count: int  # 出现的场景数

    # ===== 比例指标 =====
    portrait_target_ratio: float  # 他拍中人像比例（被拍主体）
    photographed_subject_ratio: float  # 他拍中被拍主体比例（更细致）

    # ===== 证据 =====
    photographed_subject_photo_ids: List[str] = field(default_factory=list)
    other_photo_ids: List[str] = field(default_factory=list)

    # ===== 他拍候选的评估 =====
    veto_reasons: List[Dict[str, Any]] = field(default_factory=list)
    """所有否决理由及证据。每条包含：
    {
        "type": "veto_type_name",
        "reason": "文字说明",
        "evidence": {...具体数据...}
    }
    """

    confidence_score: float = 0.0  # 作为"他拍主角"的合理性 [0.0-1.0]
    confidence_reasoning: str = ""

    # ===== 完整证据链 =====
    evidence_refs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_other_photo_signals(
    face_db: Dict[str, Any],
    vlm_results: List[Dict[str, Any]],
    screening: Dict[str, PersonScreening],
    album_trace: Dict[str, Any],
    portrait_threshold: float = PRIMARY_PORTRAIT_TARGET_RATIO_MAX,
    subject_threshold: float = PRIMARY_PHOTOGRAPHED_SUBJECT_RATIO_MAX,
) -> List[OtherPhotoProfile]:
    """提取每个人物的"他拍轮廓"并预先计算 veto。

    Args:
        face_db: 人脸数据库
        vlm_results: VLM 分析结果列表
        screening: 人物筛查结果
        album_trace: 相册级别的追踪信息（包含 no_face_or_user_view_ratio 等）
        portrait_threshold: 人像比例阈值（默认 0.65）
        subject_threshold: 被拍主体比例阈值（默认 0.6）

    Returns:
        OtherPhotoProfile 列表，每个人物一条记录
    """

    # [1] 初始化结构
    profiles: Dict[str, OtherPhotoProfile] = {}

    for person_id in face_db.keys():
        # 跳过被筛查排除的人物
        if screening.get(person_id) and screening[person_id].memory_value == "block":
            continue

        profiles[person_id] = OtherPhotoProfile(
            person_id=person_id,
            non_selfie_photo_count=0,
            cross_scene_presence_count=0,
            portrait_target_ratio=0.0,
            photographed_subject_ratio=0.0,
        )

    # [2] 遍历照片，识别他拍
    scene_keys_per_person: Dict[str, set] = {pid: set() for pid in profiles.keys()}
    portrait_target_hits: Dict[str, int] = {pid: 0 for pid in profiles.keys()}
    photographed_subject_hits: Dict[str, int] = {pid: 0 for pid in profiles.keys()}

    for item in vlm_results or []:
        photo_id = str(item.get("photo_id") or "")
        analysis = item.get("vlm_analysis", {}) or {}
        summary = str(analysis.get("summary", "") or "")
        event = analysis.get("event", {}) or {}
        scene = analysis.get("scene", {}) or {}
        activity = event.get("activity", "") if isinstance(event, dict) else str(event or "")
        location = scene.get("location_detected", "") if isinstance(scene, dict) else str(scene or "")
        haystack = " ".join([summary, str(activity), str(location)]).lower()

        people = [
            str(person.get("person_id"))
            for person in analysis.get("people", []) or []
            if isinstance(person, dict) and person.get("person_id")
        ]

        # 先判断是否是自拍或身份照
        is_selfie = any(keyword in haystack for keyword in SELFIE_KEYWORDS)
        is_identity = any(keyword in haystack for keyword in IDENTITY_KEYWORDS)

        if is_selfie or is_identity:
            continue

        # ✓ 进入"他拍"处理
        for person_id in people:
            if person_id not in profiles:
                continue

            profile = profiles[person_id]
            profile.non_selfie_photo_count += 1
            profile.other_photo_ids.append(photo_id)

            # 跟踪场景
            scene_key = str(location or "").strip().lower()
            if scene_key:
                scene_keys_per_person[person_id].add(scene_key)

            # 子判断：是不是人像拍摄（portrait）
            if any(keyword in haystack for keyword in PORTRAIT_SHOOTING_KEYWORDS):
                portrait_target_hits[person_id] += 1
                photographed_subject_hits[person_id] += 1
                profile.photographed_subject_photo_ids.append(photo_id)

    # [3] 计算比例和跨场景
    for person_id, profile in profiles.items():
        profile.cross_scene_presence_count = len(scene_keys_per_person[person_id])
        non_selfie_count = max(profile.non_selfie_photo_count, 1)
        profile.portrait_target_ratio = round(portrait_target_hits[person_id] / non_selfie_count, 3)
        profile.photographed_subject_ratio = round(photographed_subject_hits[person_id] / non_selfie_count, 3)

    # [4] 预先计算 veto 和 confidence
    for profile in profiles.values():
        _compute_veto_reasons_and_score(
            profile,
            screening,
            album_trace,
            portrait_threshold=portrait_threshold,
            subject_threshold=subject_threshold,
        )

    return list(profiles.values())


def _compute_veto_reasons_and_score(
    profile: OtherPhotoProfile,
    screening: Dict[str, PersonScreening],
    album_trace: Dict[str, Any],
    portrait_threshold: float = PRIMARY_PORTRAIT_TARGET_RATIO_MAX,
    subject_threshold: float = PRIMARY_PHOTOGRAPHED_SUBJECT_RATIO_MAX,
) -> None:
    """逐条计算 veto，并评估 confidence_score。

    在 profile 对象上原地修改：
    - veto_reasons: 所有否决理由
    - confidence_score: 作为他拍主角的合理性评分
    - evidence_refs: 完整证据链
    """

    vetoes: List[Dict[str, Any]] = []
    evidence_list: List[Dict[str, Any]] = []

    # ===== Veto 1: 他拍数不足 =====
    if profile.non_selfie_photo_count < PRIMARY_OTHER_PHOTO_MIN_COUNT:
        veto = {
            "type": "non_selfie_photo_count_too_low",
            "reason": f"他拍数仅 {profile.non_selfie_photo_count} 张，低于最小阈值 {PRIMARY_OTHER_PHOTO_MIN_COUNT}",
            "evidence": {
                "count": profile.non_selfie_photo_count,
                "threshold": PRIMARY_OTHER_PHOTO_MIN_COUNT,
            },
        }
        vetoes.append(veto)
        evidence_list.append({
            "source": "other_photo_count",
            "veto_type": veto["type"],
            "photo_count": profile.non_selfie_photo_count,
        })

    # ===== Veto 2: 人像比例过高 =====
    if profile.portrait_target_ratio >= portrait_threshold:
        veto = {
            "type": "portrait_target_ratio_too_high",
            "reason": (
                f"他拍中人像比例 {profile.portrait_target_ratio:.1%}，"
                f"超过阈值 {portrait_threshold:.1%}，表示该候选主要作为被拍对象"
            ),
            "evidence": {
                "ratio": profile.portrait_target_ratio,
                "threshold": portrait_threshold,
                "portrait_hits": len(profile.photographed_subject_photo_ids),
                "total_non_selfie": profile.non_selfie_photo_count,
            },
        }
        vetoes.append(veto)
        evidence_list.append({
            "source": "portrait_ratio",
            "veto_type": veto["type"],
            "ratio": profile.portrait_target_ratio,
            "photo_ids": profile.photographed_subject_photo_ids,
        })

    # ===== Veto 3: 被拍主体比例过高 =====
    if profile.photographed_subject_ratio >= subject_threshold:
        veto = {
            "type": "photographed_subject_ratio_too_high",
            "reason": (
                f"他拍中被拍主体比例 {profile.photographed_subject_ratio:.1%}，"
                f"超过阈值 {subject_threshold:.1%}，该候选很可能主要被用户拍摄"
            ),
            "evidence": {
                "ratio": profile.photographed_subject_ratio,
                "threshold": subject_threshold,
                "subject_hits": len(profile.photographed_subject_photo_ids),
                "total_non_selfie": profile.non_selfie_photo_count,
            },
        }
        vetoes.append(veto)
        evidence_list.append({
            "source": "photographed_subject_ratio",
            "veto_type": veto["type"],
            "ratio": profile.photographed_subject_ratio,
            "photo_ids": profile.photographed_subject_photo_ids,
        })

    # ===== Veto 4: 场景多样性不足 =====
    if profile.cross_scene_presence_count < PRIMARY_OTHER_PHOTO_MIN_SCENES:
        veto = {
            "type": "scene_diversity_too_low",
            "reason": (
                f"跨场景次数 {profile.cross_scene_presence_count}，"
                f"低于最小阈值 {PRIMARY_OTHER_PHOTO_MIN_SCENES}，"
                f"表示该候选仅在特定场景出现"
            ),
            "evidence": {
                "count": profile.cross_scene_presence_count,
                "threshold": PRIMARY_OTHER_PHOTO_MIN_SCENES,
            },
        }
        vetoes.append(veto)
        evidence_list.append({
            "source": "scene_diversity",
            "veto_type": veto["type"],
            "scene_count": profile.cross_scene_presence_count,
        })

    # ===== Veto 5: 人物低价值或不可信 =====
    screening_result = screening.get(profile.person_id)
    if screening_result and screening_result.memory_value == "low_value":
        veto = {
            "type": "person_memory_value_low",
            "reason": f"{profile.person_id} 被筛查为 low_value",
            "evidence": {"memory_value": "low_value"},
        }
        vetoes.append(veto)
        evidence_list.append({
            "source": "screening",
            "veto_type": veto["type"],
            "memory_value": "low_value",
        })

    if screening_result and screening_result.person_kind != "real_person":
        veto = {
            "type": f"person_kind_not_real",
            "reason": f"{profile.person_id} 被筛查为 {screening_result.person_kind}",
            "evidence": {"person_kind": screening_result.person_kind},
        }
        vetoes.append(veto)
        evidence_list.append({
            "source": "screening",
            "veto_type": veto["type"],
            "person_kind": screening_result.person_kind,
        })

    # ===== Veto 6: 相册模式判定（用户主要在拍别人） =====
    if (
        album_trace.get("no_face_or_user_view_ratio", 0.0) >= PRIMARY_PHOTOGRAPHER_NOFACE_RATIO
        and profile.photographed_subject_ratio >= 0.4
    ):
        veto = {
            "type": "album_shooting_mode_detected",
            "reason": (
                f"相册中 {album_trace.get('no_face_or_user_view_ratio', 0.0):.1%} 是无脸/用户视角，"
                f"且该候选被拍比例达 {profile.photographed_subject_ratio:.1%}，"
                f"表示用户主要在用摄像头拍别人"
            ),
            "evidence": {
                "no_face_or_user_view_ratio": album_trace.get("no_face_or_user_view_ratio", 0.0),
                "person_photographed_subject_ratio": profile.photographed_subject_ratio,
            },
        }
        vetoes.append(veto)
        evidence_list.append({
            "source": "album_mode",
            "veto_type": veto["type"],
            "no_face_ratio": album_trace.get("no_face_or_user_view_ratio", 0.0),
        })

    # [5] 计算 confidence score
    score = 1.0
    reasoning_parts = []

    if not vetoes:
        # 没有否决理由
        score = 0.85
        reasoning_parts.append("没有检测到否决条件")
    else:
        # 有否决理由，按严重程度降分
        score = max(0.0, 0.85 - 0.15 * len(vetoes))
        reasoning_parts.append(f"检测到 {len(vetoes)} 条否决理由：{', '.join(v['type'] for v in vetoes)}")

    # 额外奖励：高频且跨场景
    if profile.non_selfie_photo_count >= 6:
        score = min(1.0, score + 0.05)
        reasoning_parts.append("他拍高频（≥6张），加分")

    if profile.cross_scene_presence_count >= 3:
        score = min(1.0, score + 0.05)
        reasoning_parts.append("跨场景多（≥3个），加分")

    profile.veto_reasons = vetoes
    profile.confidence_score = round(score, 2)
    profile.confidence_reasoning = " | ".join(reasoning_parts)
    profile.evidence_refs = evidence_list
