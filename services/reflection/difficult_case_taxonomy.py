"""Difficult case taxonomy: named "diseases" that affect protagonist/relationship/profile fields.

Each disease has:
- A name (Chinese)
- A description of the condition
- Detection rules (how to identify it from cross-user patterns)
- Affected dimensions
- Suggested remediation direction (not a fix, but what kind of new capability is needed)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class DifficultDisease:
    """A named disease affecting a field dimension."""
    disease_id: str
    lane: str  # protagonist / relationship / profile
    name: str  # Chinese disease name
    description: str  # What this disease is
    condition: str  # When does this disease occur
    remediation_direction: str  # What kind of new capability is needed
    affected_dimensions: List[str] = field(default_factory=list)
    affected_users: List[str] = field(default_factory=list)
    case_count: int = 0
    evidence_examples: List[str] = field(default_factory=list)
    severity: str = "medium"  # high / medium / low

    def to_dict(self) -> Dict[str, Any]:
        return {
            "disease_id": self.disease_id,
            "lane": self.lane,
            "name": self.name,
            "description": self.description,
            "condition": self.condition,
            "remediation_direction": self.remediation_direction,
            "affected_dimensions": self.affected_dimensions,
            "affected_users": self.affected_users,
            "case_count": self.case_count,
            "evidence_examples": self.evidence_examples[:5],
            "severity": self.severity,
        }


# ── Disease detection rules ──

def detect_diseases(
    patterns: List[Dict[str, Any]],
    missing_capabilities: List[Dict[str, Any]],
    all_case_facts: List[Dict[str, Any]],
    total_users: int,
) -> List[DifficultDisease]:
    """Detect named diseases from cross-user patterns and missing capabilities."""
    diseases: List[DifficultDisease] = []

    # ═══ PROTAGONIST diseases ═══
    diseases.extend(_detect_protagonist_diseases(patterns, all_case_facts, total_users))

    # ═══ RELATIONSHIP diseases ═══
    diseases.extend(_detect_relationship_diseases(patterns, all_case_facts, total_users))

    # ═══ PROFILE diseases ═══
    diseases.extend(_detect_profile_diseases(patterns, missing_capabilities, all_case_facts, total_users))

    return diseases


def _detect_protagonist_diseases(
    patterns: List[Dict], cases: List[Dict], total_users: int,
) -> List[DifficultDisease]:
    diseases: List[DifficultDisease] = []
    proto_patterns = [p for p in patterns if p.get("lane") == "protagonist"]

    # Disease: 脸部缺失症
    # When: primary_decision fails because no face detected in album
    proto_cases = [c for c in cases if _is_lane(c, "protagonist")]
    no_face_users = set()
    for c in proto_cases:
        trace = c.get("decision_trace") or {}
        upstream = c.get("upstream_output") or {}
        reasoning = str(trace.get("retention_reason", "")) + str(upstream.get("reasoning", ""))
        if any(kw in reasoning.lower() for kw in ("no face", "无人脸", "face_count", "零人脸", "photographer_mode")):
            no_face_users.add(c.get("user_name", ""))
    if no_face_users:
        diseases.append(DifficultDisease(
            disease_id="protagonist_no_face",
            lane="protagonist",
            name="脸部缺失症",
            description="用户相册中缺少可识别的人脸，导致系统无法确定主角身份",
            condition="当相册中完全没有正面人脸，或人脸质量过低无法匹配时触发",
            remediation_direction="需要非人脸的身份识别方式（如手部特征、穿着风格、拍摄视角模式）或社媒头像关联",
            affected_dimensions=["primary_decision"],
            affected_users=sorted(no_face_users),
            case_count=len([c for c in proto_cases if c.get("user_name") in no_face_users]),
            severity="high",
        ))

    # Disease: 主角模糊症
    # When: multiple candidates with similar frequency, system can't decide
    ambiguous_users = set()
    for c in proto_cases:
        upstream = c.get("upstream_output") or {}
        conf = float(upstream.get("confidence") or 1.0)
        if conf < 0.6:
            ambiguous_users.add(c.get("user_name", ""))
    if ambiguous_users:
        diseases.append(DifficultDisease(
            disease_id="protagonist_ambiguous",
            lane="protagonist",
            name="主角模糊症",
            description="存在多个高频出现的人物，系统无法确定哪个是主角",
            condition="当 2+ 人物出现频率接近，且缺少自拍等明确的第一人称视角线索时触发",
            remediation_direction="需要自拍检测、拍摄视角分析、或用户手动确认主角的交互流程",
            affected_dimensions=["primary_decision"],
            affected_users=sorted(ambiguous_users),
            case_count=len([c for c in proto_cases if c.get("user_name") in ambiguous_users]),
            severity="high",
        ))

    # Disease: 主角判定全失败 (catch-all for remaining protagonist failures)
    failed_users = set()
    for p in proto_patterns:
        if p.get("failure_mode") in ("unknown", "missing_signal"):
            for u in p.get("affected_users", []):
                if u not in no_face_users and u not in ambiguous_users:
                    failed_users.add(u)
    if failed_users:
        diseases.append(DifficultDisease(
            disease_id="protagonist_undiagnosed",
            lane="protagonist",
            name="主角判定失败（未归因）",
            description="主角识别失败，但原因未能自动归类",
            condition="系统输出与 GT 不匹配，且不属于已知的脸部缺失或模糊症",
            remediation_direction="需要人工查看具体 case 后补充诊断规则",
            affected_dimensions=["primary_decision"],
            affected_users=sorted(failed_users),
            case_count=sum(p.get("total_case_count", 0) for p in proto_patterns),
            severity="medium",
        ))

    return diseases


def _detect_relationship_diseases(
    patterns: List[Dict], cases: List[Dict], total_users: int,
) -> List[DifficultDisease]:
    diseases: List[DifficultDisease] = []
    rel_patterns = [p for p in patterns if p.get("lane") == "relationship"]
    rel_cases = [c for c in cases if _is_lane(c, "relationship")]

    # Disease: 场景缺失症
    # When: a relationship type consistently has no supporting scene evidence
    for p in rel_patterns:
        rel_type = p.get("dimension", "").replace("relationship_type:", "")
        if p.get("failure_mode") in ("missing_signal", "unknown") and p.get("total_case_count", 0) >= 3:
            diseases.append(DifficultDisease(
                disease_id=f"rel_scene_missing_{rel_type}",
                lane="relationship",
                name=f"「{rel_type}」场景缺失症",
                description=f"系统在识别 {rel_type} 关系时缺少足够的场景证据",
                condition=f"当用户相册中缺少与 {rel_type} 关系相关的互动场景（如家庭聚会、亲密合影等）时触发",
                remediation_direction="需要更丰富的场景理解能力，或接入社媒互动数据作为补充证据",
                affected_dimensions=[p.get("dimension", "")],
                affected_users=p.get("affected_users", []),
                case_count=p.get("total_case_count", 0),
                severity="high" if p.get("total_case_count", 0) >= 10 else "medium",
            ))

    # Disease: 自我误识症
    # When: the user themselves is misidentified as another person
    self_misid_users = set()
    for c in rel_cases:
        gt = c.get("gt_payload") or {}
        notes = str(gt.get("notes", "")) + str(gt.get("gt_value", ""))
        if any(kw in notes for kw in ("自己", "本人", "is_self", "用户自己")):
            self_misid_users.add(c.get("user_name", ""))
    if self_misid_users:
        diseases.append(DifficultDisease(
            disease_id="rel_self_misidentification",
            lane="relationship",
            name="自我误识症",
            description="系统将用户本人的不同照片误识别为其他人物，建立了虚假关系",
            condition="当用户的外貌变化（发型、妆容、光线）导致人脸识别将同一人判为不同 person_id 时触发",
            remediation_direction="需要更强的跨外貌变化人脸匹配能力，或主角聚类后置校验机制",
            affected_dimensions=["relationship:self_misid"],
            affected_users=sorted(self_misid_users),
            case_count=len([c for c in rel_cases if c.get("user_name") in self_misid_users]),
            severity="high",
        ))

    # Disease: 虚拟人物症
    # When: AI-generated images or screenshots are treated as real relationships
    virtual_users = set()
    for c in rel_cases:
        gt = c.get("gt_payload") or {}
        notes = str(gt.get("notes", "")) + str(gt.get("gt_value", ""))
        if any(kw in notes for kw in ("AI生图", "截图", "虚拟", "数字媒体", "screenshot")):
            virtual_users.add(c.get("user_name", ""))
    if virtual_users:
        diseases.append(DifficultDisease(
            disease_id="rel_virtual_person",
            lane="relationship",
            name="虚拟人物症",
            description="系统将 AI 生成图片、截图中的人物或数字媒体人物误判为真实社交关系",
            condition="当相册中包含 AI 生图、社交媒体截图、明星照片等非真实互动场景时触发",
            remediation_direction="需要 AI 生图检测能力 + 截图/媒体内容识别，将非真实场景过滤",
            affected_dimensions=["relationship:virtual_person"],
            affected_users=sorted(virtual_users),
            case_count=len([c for c in rel_cases if c.get("user_name") in virtual_users]),
            severity="medium",
        ))

    return diseases


def _detect_profile_diseases(
    patterns: List[Dict],
    missing_caps: List[Dict],
    cases: List[Dict],
    total_users: int,
) -> List[DifficultDisease]:
    diseases: List[DifficultDisease] = []
    profile_patterns = [p for p in patterns if p.get("lane") == "profile"]

    # Disease: 域级信息荒漠
    # When: an entire field domain is empty across many users
    for mc in missing_caps:
        if mc.get("capability_type") == "new_data_source":
            fields = mc.get("affected_fields", [])
            users = mc.get("affected_users", [])
            domain = _extract_domain(fields)
            if len(users) >= 1:
                diseases.append(DifficultDisease(
                    disease_id=f"profile_desert_{domain}",
                    lane="profile",
                    name=f"「{domain}」信息荒漠",
                    description=f"字段域 {domain} 在多个用户中持续为空，照片信息不足以推断",
                    condition=f"当用户相册中缺少与 {domain} 相关的场景/物品/文字线索时触发",
                    remediation_direction="需要接入社媒数据、用户问卷等外部信息源，或降低该域字段的期望",
                    affected_dimensions=fields,
                    affected_users=users,
                    case_count=sum(p.get("total_case_count", 0) for p in profile_patterns if any(f in p.get("dimension", "") for f in fields)),
                    severity="high" if len(users) > 1 else "medium",
                ))

    # Disease: COT 无效循环
    # When: field_cot is repeatedly recommended but never succeeds
    for mc in missing_caps:
        if mc.get("capability_type") == "rule_redesign":
            diseases.append(DifficultDisease(
                disease_id="profile_cot_loop",
                lane="profile",
                name="COT 无效循环",
                description="修改 COT 规则被反复推荐但实际无法改善字段质量",
                condition="当同一修复面（field_cot）在多个 pattern 中出现但历史上未产出有效改善时触发",
                remediation_direction="可能需要更根本的架构变更：新的证据检索方式、新的字段推理链路、或该字段本身不适合从照片推断",
                affected_dimensions=mc.get("affected_patterns", []),
                affected_users=[],
                case_count=0,
                severity="medium",
            ))

    # Disease: 姓名不可得症
    # When: name field consistently empty (photos rarely contain name info)
    name_patterns = [p for p in profile_patterns if "identity.name" in p.get("dimension", "")]
    if name_patterns:
        all_users = set()
        for p in name_patterns:
            all_users.update(p.get("affected_users", []))
        if all_users:
            diseases.append(DifficultDisease(
                disease_id="profile_name_unreachable",
                lane="profile",
                name="姓名不可得症",
                description="照片中几乎不包含姓名信息，系统无法推断用户姓名",
                condition="当相册中无证件照、无署名文档、无社交媒体个人页截图时触发",
                remediation_direction="需要社媒账号关联、证件 OCR、或用户主动提供的补充渠道",
                affected_dimensions=["long_term_facts.identity.name"],
                affected_users=sorted(all_users),
                case_count=sum(p.get("total_case_count", 0) for p in name_patterns),
                severity="low",
            ))

    return diseases


# ── Helpers ──

def _is_lane(case: Dict, lane: str) -> bool:
    entity_type = str(case.get("entity_type") or "").strip()
    dimension = str(case.get("dimension") or "").strip()
    signal_source = str(case.get("signal_source") or "").strip()
    if lane == "protagonist":
        return entity_type == "primary_person" or "primary" in dimension.lower() or signal_source == "mainline_primary"
    if lane == "relationship":
        return entity_type == "relationship_candidate" or dimension.startswith("relationship:") or signal_source == "mainline_relationship"
    if lane == "profile":
        return entity_type == "profile_field" or dimension.startswith(("long_term_", "short_term_")) or signal_source == "mainline_profile"
    return False


def _extract_domain(fields: List[str]) -> str:
    """Extract common domain from field list."""
    if not fields:
        return "unknown"
    # Take the first field's second-level domain
    parts = fields[0].split(".")
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return fields[0]
