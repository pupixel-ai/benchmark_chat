from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Tuple

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, PROFILE_LLM_MODEL, PROFILE_LLM_PROVIDER
from services.consistency_checker import build_consistency_report

from .evidence_utils import build_evidence_payload, extract_ids_from_refs, flatten_ref_buckets
from .profile_agent import ProfileAgent
from .rule_asset_loader import (
    apply_runtime_field_spec_updates,
    clear_runtime_rule_overlays,
    get_effective_field_specs,
    temporary_rule_overlay,
)
from .types import FactFieldDecision, FieldBundle, FieldSpec, MemoryState


def _default_cot_steps(field_key: str, strong_evidence: List[str]) -> List[str]:
    primary_requirement = strong_evidence[0] if strong_evidence else "满足字段最小证据门槛"
    return [
        f"先看该字段允许证据源里的主线信号，优先确认 {primary_requirement}",
        "再确认这些信号稳定绑定主角本人，而不是同框人物、环境信息、截图内容或外部上下文",
        "当证据支持更细粒度的子类时，优先输出子类标签（如 college_student 而非 student，rock_music 而非 music，digital_photography 而非 photography）；泛类标签只在无法区分子类时使用",
        "只有在无法区分子类或证据不足以支撑任何标签时才输出泛化标签或 null",
    ]


def _default_owner_resolution_steps(field_key: str, owner_checks: List[str]) -> List[str]:
    if owner_checks:
        return list(owner_checks)
    return [
        "先检查证据是否明确指向主角本人",
        "如果证据只指向他人物品、他人关系、环境背景或截图文本，则不计入主角字段",
    ]


def _default_time_reasoning_steps(field_key: str, time_layer_rule: str) -> List[str]:
    if field_key.startswith("short_term_facts."):
        return [
            "先对比最近窗口与长期基线，确认这是近期变化而不是长期稳定属性",
            "如果只是一两次偶发事件或短期噪声，不输出近期标签",
        ]
    if time_layer_rule == "long_term_only":
        return [
            "只接受跨事件、跨时间窗口的稳定模式",
            "如果证据只来自单次事件或短期窗口，退回 null",
        ]
    return [
        "优先确认该字段对应的是长期模式还是短期变化",
        "若无法区分长期/短期层级，则按更保守的层级处理或退回 null",
    ]


def _default_counter_evidence_checks(field_key: str, hard_blocks: List[str], weak_evidence_caution: List[str]) -> List[str]:
    checks = [
        "检查是否存在与草案相反的证据没有被引用",
        "检查是否把单次事件、短期噪声或同框他人线索过度外推成主角标签",
    ]
    checks.extend(hard_blocks[:2])
    checks.extend(weak_evidence_caution[:2])
    return checks


FIELD_COT_OVERRIDES: Dict[str, Dict[str, List[str]]] = {
    "long_term_facts.identity.name": {
        "cot_steps": [
            "先看实名、稳定昵称或称呼是否直接绑定主角本人",
            "再排除聊天截图、联系人名称、他人物品或外部上下文带来的误绑定",
            "只有当名字能稳定回到主角本人时才输出非 null 标签",
        ],
    },
    "long_term_facts.identity.gender": {
        "cot_steps": [
            "先看主角本人是否在多张照片中稳定出现同类外观线索",
            "再排除单张图、滤镜、妆造和同框他人带来的偏差",
            "若主体不明或证据不稳定，保守输出 null",
        ],
    },
    "long_term_facts.social_identity.education": {
        "cot_steps": [
            "先看跨事件校园/课堂主线，确认不是一次路过学校或临时活动",
            "再看校名、课程、课堂、作业等证据是否能稳定绑定到主角本人",
            "当证据支持时，尽量输出具体的教育信息：学校名称、专业方向、学历阶段（如'某大学计算机专业本科'而非仅'大学'）",
            "最后只有在校园主线连续成立时才输出教育标签",
        ],
        "owner_resolution_steps": [
            "主体归属检查：主角本人必须明确在校园/课堂证据里出现，不能只看到环境",
            "主体归属检查：如果学校信息主要属于同框他人，则该证据无效",
        ],
        "time_reasoning_steps": [
            "时间层判断：教育属于长期字段，至少需要跨事件重复成立",
            "时间层判断：若只是一场讲座、一次访问、一次校园路过，则不能写成长期教育背景",
        ],
        "counter_evidence_checks": [
            "反证检查：有没有“路过”“walk by”“visit”这类临时校园证据在反驳就读主线",
            "反证检查：是不是只有课件、海报、环境，没有主角本人和教育行为闭环",
        ],
    },
    "long_term_facts.social_identity.career": {
        "cot_steps": [
            "career 输出具体的职业方向或专业领域（如 design / software_engineering / marketing / logistics），不输出 student——student 属于 role 字段",
            "如果主角是学生且没有明确的职业/兼职方向，career 应输出 null 而非 student",
            "先看是否存在多事件重复的工作/职业场景主线",
            "再看设备、时间投入、成果或重复参与是否能形成职业闭环",
            "没有连续职业主线时不输出 career",
        ],
    },
    "long_term_facts.material.brand_preference": {
        "cot_steps": [
            "先看同一品牌是否跨事件重复出现",
            "再核验该品牌是否明确属于主角长期拥有或长期使用",
            "广告、商品截图、他人物品不能直接升格为品牌偏好",
        ],
    },
    "long_term_facts.geography.cross_border": {
        "cot_steps": [
            "cross_border 输出跨境活动模式（如 domestic_only / occasional_international / frequent_traveler），不要输出字符串 'none'——没有证据时输出 JSON null",
            "有跨境证据（护照、国外地名、外币、国际航班）才输出具体模式",
            "所有证据都指向国内活动时输出 domestic_only，不要输出 null（这本身就是有意义的判断）",
        ],
    },
    "long_term_facts.geography.location_anchors": {
        "cot_steps": [
            "先读地点锚点特征，再回看这些地点是否真的承载主角生活主线",
            "再排除学校地、旅行地、过路地和他人地点被误写成常驻锚点",
            "只有跨事件生活主线成立时才输出 location_anchors",
        ],
    },
    "long_term_facts.relationships.intimate_partner": {
        "cot_steps": [
            "先只读取关系层最终裁决后的 romantic 结果",
            "再确认画像层没有绕过关系层自行补伴侣",
            "没有稳定 romantic 关系时，必须输出 null",
        ],
    },
    "long_term_facts.relationships.social_groups": {
        "cot_steps": [
            "先读取圈层识别器产出的 GroupArtifact",
            "再确认群组不是由一次多人同框或单次活动合照误推出来的",
            "没有稳定群组强证据时，不输出 social_groups",
        ],
    },
    "long_term_facts.relationships.pets": {
        "cot_steps": [
            "先看动物是否跨事件重复出现",
            "再看居家场景、照护行为、用品和喂养线索，确认是否属于主角本人",
            "朋友宠物、路边动物和随手拍动物不能直接写成主角养宠",
        ],
    },
    "long_term_facts.relationships.parenting": {
        "cot_steps": [
            "先看是否存在连续的儿童照护行为",
            "再看家庭场景与 family 关系线是否同时成立",
            "活动现场儿童、朋友孩子和游客小孩不能直接写成 parenting",
        ],
    },
    "long_term_facts.relationships.living_situation": {
        "cot_steps": [
            "先看重复居家事件是否形成稳定生活场景",
            "再看稳定同住人、伴侣或家人线索是否清晰",
            "酒店、民宿、宿舍和短暂停留默认高风险，需要保守 null",
        ],
    },
    "long_term_facts.hobbies.frequent_activities": {
        "cot_steps": [
            "先读长期事件统计里的 top 活动",
            "再过滤最近窗口的噪声活动，避免把偶然高频写成长期高频",
            "只有稳定 top 活动成立时才输出 frequent_activities",
        ],
    },
    "short_term_facts.current_displacement": {
        "cot_steps": [
            "current_displacement 只在主角近期离开了常驻地时才输出（如出差、旅行、搬迁），格式为'从 A 到 B'或目的地描述",
            "如果主角一直在常驻城市活动（如 location_anchors 中的城市），不算位移，输出 null",
            "普通家校通勤、城市内移动不算 displacement",
            "不要输出当前所在城市名——那属于 location_anchors 字段",
        ],
    },
    "short_term_facts.recent_interests": {
        "cot_steps": [
            "recent_interests 输出近期关注的话题/领域（如 skincare_brands / cruise_travel / indie_music），是注意力层面的关注方向",
            "与 recent_habits 的边界：interests 是主题（在关注什么），habits 是行为（在反复做什么）。例：对护肤品牌的研究是 interest，每天护肤是 habit",
            "先看最近新增主题是否在多个近期事件里重复出现",
            "再排除单张截图、一次跟风和短期热点带来的噪声",
        ],
    },
    "short_term_facts.recent_habits": {
        "cot_steps": [
            "recent_habits 输出近期反复出现的具体行为模式（如 weekly_date_nights / daily_skincare_routine / frequent_cafe_visits），强调'反复做的事'",
            "与 recent_interests 的边界：habits 是行为层面的重复动作，interests 是主题层面的关注方向",
            "与 phase_change 的边界：habits 是持续的重复行为，phase_change 是一次性的阶段转折",
            "不要输出泛标签（social_dating 太泛），要输出具体行为模式",
        ],
    },
    "short_term_facts.phase_change": {
        "cot_steps": [
            "phase_change 输出近期生活阶段的转折点（如 entering_relationship / started_new_job / moved_to_new_city），是低频的大事件",
            "不要输出当前状态描述（skincare_focus 不是阶段变化——那属于 recent_interests 或 recent_habits）",
            "必须体现'变化'：从什么状态转到了什么状态",
            "如果近期没有明显的阶段转折，输出 null",
        ],
    },
    "short_term_expression.motivation_shift": {
        "cot_steps": [
            "motivation_shift 输出近期内驱力的方向转变（如 from_solo_exploration_to_relationship_building / from_academic_to_career_preparation），必须体现'从 A 到 B'的变化",
            "不要输出当前状态标签（relationship_focused 没有体现变化——那属于 current_mood 或 mental_state）",
            "与 phase_change 的边界：phase_change 是外部生活阶段变化，motivation_shift 是内部驱动力方向变化",
            "如果没有明显的动机转变，输出 null",
        ],
    },
    "long_term_expression.personality_mbti": {
        "cot_steps": [
            "先看跨事件行为模式、社交频率和独处比例是否形成稳定人格线索",
            "再排除单次情绪、单次活动或节庆周造成的短期噪声",
            "只有稳定行为模式成立时才输出 personality_mbti，否则保守 null",
        ],
    },
    "long_term_expression.attitude_style": {
        "cot_steps": [
            "先看穿搭、场景选择和物品搭配是否跨事件重复",
            "再排除单次 dress code、主题活动和借用物带来的噪声",
            "只有稳定风格成立时才输出 attitude_style",
        ],
    },
    "long_term_expression.aesthetic_tendency": {
        "cot_steps": [
            "先看构图、色调、物品选择和装修风格是否跨事件重复",
            "再排除单次场地风格、滤镜和主题布景误导",
            "只有稳定审美模式成立时才输出 aesthetic_tendency",
        ],
    },
    "long_term_expression.visual_creation_style": {
        "cot_steps": [
            "先看自拍、风景、合照和构图习惯是否跨事件重复",
            "再排除单次拍摄任务或单次旅行相册带来的偏差",
            "只有稳定拍照模式成立时才输出 visual_creation_style",
        ],
    },
    "short_term_expression.current_mood": {
        "cot_steps": [
            "current_mood 输出当前的情绪基调（如 happy / anxious / relaxed / romantic / stressed），是短期的情感状态",
            "与 mental_state 的边界：mood 是'感觉怎么样'（情绪），mental_state 是'精神状态如何'（认知/能量）",
            "不要输出行为描述（socially_active 不是 mood，是行为模式——那属于 social_energy）",
            "只有近期多条信号一致时才输出",
        ],
    },
    "short_term_expression.mental_state": {
        "cot_steps": [
            "mental_state 输出当前的精神/认知状态（如 focused / fatigued / energized / overwhelmed / content），是精神能量和认知负荷的描述",
            "与 current_mood 的边界：mental_state 是'脑子在什么状态'（专注/疲惫/充沛），mood 是'心情怎么样'（开心/焦虑）",
            "不要输出社交行为描述（socially_active 不是 mental_state——那属于 social_energy）",
            "如果没有连续证据，输出 null",
        ],
    },
    "short_term_expression.stress_signal": {
        "cot_steps": [
            "先看近期是否出现连续高压信号，如深夜活动、密集事务或明显关系压力",
            "再排除考试周、项目周或节庆活动造成的局部峰值",
            "只有近期压力信号连续出现时才输出 stress_signal",
        ],
    },
    "short_term_expression.social_energy": {
        "cot_steps": [
            "先看近期社交频率、活动规模和互动强度是否出现一致变化",
            "再排除单周高频聚会或单次社交低谷造成的偏差",
            "只有近期社交能量模式稳定时才输出 social_energy",
        ],
    },
    "long_term_facts.identity.age_range": {
        "cot_steps": [
            "age_range 只输出年龄范围（如 18-22 / 25-30 / early_twenties / late_thirties），绝对不输出身份标签（student / worker 等——那些属于 role 字段）",
            "优先使用硬证据：身份证出生年份、证件年龄、OCR 中的年份信息",
            "硬证据缺失时，可从 education（大学→18-24）、career 经验年限、外观线索等间接推断年龄范围",
            "只输出年龄数字范围或年龄段描述，不输出身份类别",
        ],
    },
    "long_term_facts.identity.role": {
        "cot_steps": [
            "先确认主角是否有明确的校园/工作主线证据（至少 2 条跨事件）",
            "role 只输出身份类别（student / employee / freelancer 等），不输出专业、学历或具体职业名称——那些属于 education 和 career 字段",
            "若校园和工作证据并存，选跨事件证据更强的那一面作为主标签",
            "证据不足以区分具体身份类别时才输出 null",
        ],
        "owner_resolution_steps": [
            "校园/工作线索是否明确指向主角本人而非同框他人",
            "他人大学录取通知书、他人工牌不能算主角的 role 证据",
        ],
    },
    "long_term_facts.hobbies.interests": {
        "cot_steps": [
            "先读 top_candidates 列表和 detail_snippet 中的具体兴趣主题，确认跨事件重复性",
            "如果证据中有具体的活动名称、器材、品牌、场地类型等细节，输出到能区分的最细粒度（如证据显示吉他+摇滚T恤→输出具体音乐类型，而非'音乐'）",
            "用器材、服装、场景细节和 OCR 文字作为细化依据，但不要猜测证据中没有的子类",
            "只有当细节证据完全缺失时才输出泛化兴趣标签",
        ],
    },
    "long_term_facts.hobbies.frequent_activities": {
        "cot_steps": [
            "先读长期事件统计里的 top 活动及具体活动场景",
            "如果证据中有具体的活动场景和频次细节，输出到能区分的最细粒度，而非宽泛分类",
            "再过滤最近窗口的噪声活动，避免把偶然高频写成长期高频",
            "只有稳定 top 活动成立时才输出",
        ],
    },
    "long_term_facts.time.event_cycles": {
        "cot_steps": [
            "event_cycles 输出周期性重复事件的具体模式和频率（如 biweekly_social_outings / monthly_travel / weekly_gym），不输出泛化活动类别（social_activities——那属于 interests/frequent_activities）",
            "必须体现周期性：多久一次、什么活动、是否稳定重复",
            "如果没有发现明确的周期性模式，输出 null 而非泛化活动标签",
        ],
    },
    "long_term_facts.time.sleep_pattern": {
        "cot_steps": [
            "sleep_pattern 输出具体的睡眠时间规律（如 late_sleeper_past_midnight / early_riser / irregular_between_11pm_2am），不输出单个词 irregular——这没有信息量",
            "从深夜/清晨活动时间戳推断入睡和起床时间范围",
            "与 life_rhythm 的边界：sleep_pattern 只关注睡眠相关时间点，不描述整体生活节奏",
            "证据不足时输出 null，不要猜",
        ],
    },
    "long_term_facts.social_identity.career_phase": {
        "cot_steps": [
            "career_phase 输出职业/学业的发展阶段（如 undergraduate / postgraduate / entry_level / mid_career / senior / retired），不输出身份类别（student/employee——那属于 role）",
            "对学生：区分 undergraduate / postgraduate / phd_candidate",
            "对职场人：区分 intern / entry_level / mid_career / senior / management",
            "如果无法判断具体阶段，输出 null",
        ],
    },
    "long_term_facts.social_identity.professional_dedication": {
        "cot_steps": [
            "professional_dedication 输出主角在主业上的投入模式和专注方向，不输出身份类别（student/employee——那属于 role）",
            "示例格式：full_time_focused / part_time_with_side_projects / academic_with_social_balance / career_driven",
            "从活动时间分配、工作/学习场景频率、副业/兼职线索中推断",
            "如果只能确定身份但无法判断投入模式，输出 null 而非重复 role 的值",
        ],
    },
    "long_term_facts.material.asset_level": {
        "cot_steps": [
            "先看跨事件的消费场景、品牌层次和生活条件证据",
            "输出的标签应体现这个人的实际资产特征，而非套用通用社会阶层标签",
            "只有跨事件社经证据稳定时才输出，一次性体验不算",
        ],
    },
    "long_term_facts.material.spending_style": {
        "cot_steps": [
            "先看跨事件的消费场景等级和品牌层次",
            "用 detail_snippet 中的品牌名、价格线索和场景等级来判断具体的消费偏好方向",
            "只有跨事件消费模式稳定时才输出，礼物和借用物不算",
        ],
    },
    "long_term_facts.time.life_rhythm": {
        "cot_steps": [
            "life_rhythm 输出整体生活节奏模式（如 nine_to_five / night_owl_social / weekend_active_weekday_quiet / flexible_schedule），不输出单个词 irregular——这没有信息量",
            "从工作日 vs 周末、日间 vs 夜间的活动分布差异中提取具体模式",
            "与 sleep_pattern 的边界：life_rhythm 描述整体活动节奏，sleep_pattern 只描述睡眠时间规律",
            "只有时间模式在多个事件中稳定重复时才输出",
        ],
    },
}


def _spec(
    field_key: str,
    risk_level: str,
    allowed_sources: List[str],
    strong_evidence: List[str],
    cot_steps: List[str] | None = None,
    owner_resolution_steps: List[str] | None = None,
    time_reasoning_steps: List[str] | None = None,
    counter_evidence_checks: List[str] | None = None,
    weak_evidence: List[str] | None = None,
    hard_blocks: List[str] | None = None,
    owner_checks: List[str] | None = None,
    time_layer_rule: str = "flexible",
    weak_evidence_caution: List[str] | None = None,
    reflection_questions: List[str] | None = None,
    reflection_rounds: int = 1,
    requires_social_media: bool = False,
    requires_protagonist_face: bool = False,
    field_boundary: str = "",
    cross_field_caution: str = "",
) -> FieldSpec:
    overrides = FIELD_COT_OVERRIDES.get(field_key, {})
    resolved_owner_checks = owner_checks or []
    resolved_hard_blocks = hard_blocks or []
    resolved_null_preferred = weak_evidence_caution or []
    return FieldSpec(
        field_key=field_key,
        risk_level=risk_level,
        allowed_sources=allowed_sources,
        strong_evidence=strong_evidence,
        cot_steps=cot_steps or overrides.get("cot_steps") or _default_cot_steps(field_key, strong_evidence),
        owner_resolution_steps=owner_resolution_steps or overrides.get("owner_resolution_steps") or _default_owner_resolution_steps(field_key, resolved_owner_checks),
        time_reasoning_steps=time_reasoning_steps or overrides.get("time_reasoning_steps") or _default_time_reasoning_steps(field_key, time_layer_rule),
        counter_evidence_checks=counter_evidence_checks or overrides.get("counter_evidence_checks") or _default_counter_evidence_checks(field_key, resolved_hard_blocks, resolved_null_preferred),
        weak_evidence=weak_evidence or [],
        hard_blocks=resolved_hard_blocks,
        owner_checks=resolved_owner_checks,
        time_layer_rule=time_layer_rule,
        weak_evidence_caution=resolved_null_preferred,
        reflection_questions=reflection_questions or [],
        reflection_rounds=reflection_rounds,
        requires_social_media=requires_social_media,
        requires_protagonist_face=requires_protagonist_face,
        field_boundary=field_boundary,
        cross_field_caution=cross_field_caution,
    )


FIELD_SPECS: Dict[str, FieldSpec] = {
    "long_term_facts.identity.name": _spec("long_term_facts.identity.name", "P1", ["vlm", "feature"], ["实名或稳定昵称"]),
    "long_term_facts.identity.gender": _spec("long_term_facts.identity.gender", "P1", ["vlm"], ["多次稳定主角外观"], requires_protagonist_face=True),
    "long_term_facts.identity.age_range": _spec("long_term_facts.identity.age_range", "P1", ["vlm", "event", "feature"], ["跨事件年龄阶段线索"]),
    "long_term_facts.identity.role": _spec("long_term_facts.identity.role", "P0", ["event", "vlm", "feature"], ["至少2条校园/工作主线证据"], hard_blocks=["主体归属不清"]),
    "long_term_facts.identity.race": _spec("long_term_facts.identity.race", "P0", ["vlm", "event"], ["多事件稳定外观"], hard_blocks=["单张外貌"], requires_protagonist_face=True),
    "long_term_facts.identity.nationality": _spec("long_term_facts.identity.nationality", "P0", ["vlm", "event", "feature"], ["长期地点与文化线索"], hard_blocks=["只靠城市或外貌"]),
    "long_term_facts.social_identity.education": _spec("long_term_facts.social_identity.education", "P0", ["event", "vlm", "feature"], ["至少2条跨事件校园/课堂证据"], hard_blocks=["路过学校"]),
    "long_term_facts.social_identity.career": _spec("long_term_facts.social_identity.career", "P0", ["event", "feature"], ["多事件工作场景闭环"], hard_blocks=["一次性项目"]),
    "long_term_facts.social_identity.career_phase": _spec("long_term_facts.social_identity.career_phase", "P1", ["event", "feature"], ["稳定身份主线 + 最近变化"]),
    "long_term_facts.social_identity.professional_dedication": _spec("long_term_facts.social_identity.professional_dedication", "P1", ["event", "feature"], ["连续工作/学习事件"]),
    "long_term_facts.social_identity.language_culture": _spec("long_term_facts.social_identity.language_culture", "P1", ["vlm", "event", "feature"], ["稳定文本/语言环境"]),
    "long_term_facts.social_identity.political_preference": _spec("long_term_facts.social_identity.political_preference", "P0", ["event", "vlm", "feature"], ["明确政治符号/标语/活动参与"], hard_blocks=["从国籍或文化推断政治立场"]),
    "long_term_facts.material.asset_level": _spec("long_term_facts.material.asset_level", "P0", ["event", "vlm", "feature"], ["多条社经证据 + 明确归属"], hard_blocks=["一次性体验"]),
    "long_term_facts.material.spending_style": _spec("long_term_facts.material.spending_style", "P0", ["event", "vlm", "feature"], ["跨事件重复消费模式"], hard_blocks=["礼物/借用物"]),
    "long_term_facts.material.brand_preference": _spec("long_term_facts.material.brand_preference", "P0", ["event", "vlm", "feature"], ["同一品牌跨事件重复出现"], hard_blocks=["广告/商品截图"]),
    "long_term_facts.material.income_model": _spec("long_term_facts.material.income_model", "P1", ["event", "feature"], ["role+career+生活方式闭环"]),
    "long_term_facts.material.signature_items": _spec("long_term_facts.material.signature_items", "P1", ["vlm", "event", "feature"], ["物品跨事件重复且归属明确"]),
    "long_term_facts.geography.location_anchors": _spec("long_term_facts.geography.location_anchors", "P0", ["event", "vlm", "feature"], ["跨事件重复地点锚点"], hard_blocks=["学校/旅行/过路地混写"]),
    "long_term_facts.geography.mobility_pattern": _spec("long_term_facts.geography.mobility_pattern", "P0", ["event", "feature"], ["持续位移模式"], hard_blocks=["单次旅行"]),
    "long_term_facts.geography.cross_border": _spec("long_term_facts.geography.cross_border", "P1", ["event", "feature"], ["明确跨境线索"]),
    "long_term_facts.time.life_rhythm": _spec("long_term_facts.time.life_rhythm", "P1", ["event", "feature"], ["稳定作息模式"]),
    "long_term_facts.time.event_cycles": _spec("long_term_facts.time.event_cycles", "P1", ["event", "feature"], ["重复事件模式"]),
    "long_term_facts.time.sleep_pattern": _spec("long_term_facts.time.sleep_pattern", "P1", ["event", "feature"], ["持续深夜/清晨重复"]),
    "long_term_facts.relationships.intimate_partner": _spec("long_term_facts.relationships.intimate_partner", "P0", ["relationship", "feature"], ["LP2稳定 romantic"], hard_blocks=["画像自行补伴侣"]),
    "long_term_facts.relationships.close_circle_size": _spec("long_term_facts.relationships.close_circle_size", "P0", ["relationship", "feature"], ["LP2核心关系统计"], hard_blocks=["画像自由改写"]),
    "long_term_facts.relationships.social_groups": _spec("long_term_facts.relationships.social_groups", "P1", ["group", "relationship", "event", "feature"], ["GroupArtifact稳定群组"], hard_blocks=["只凭多人同框"]),
    "long_term_facts.relationships.pets": _spec("long_term_facts.relationships.pets", "P0", ["vlm", "event", "feature"], ["动物跨事件 + 居家/照护线索"], hard_blocks=["朋友宠物"]),
    "long_term_facts.relationships.parenting": _spec("long_term_facts.relationships.parenting", "P0", ["event", "vlm", "relationship", "feature"], ["连续照护行为 + family线索"], hard_blocks=["朋友孩子"]),
    "long_term_facts.relationships.living_situation": _spec("long_term_facts.relationships.living_situation", "P0", ["event", "relationship", "feature"], ["重复居家事件 + 稳定同住线索"], hard_blocks=["酒店/宿舍/临住"]),
    "long_term_facts.hobbies.interests": _spec("long_term_facts.hobbies.interests", "P1", ["event", "vlm", "feature"], ["同类活动跨事件重复"]),
    "long_term_facts.hobbies.frequent_activities": _spec("long_term_facts.hobbies.frequent_activities", "P1", ["event", "feature"], ["长期 top 活动"]),
    "long_term_facts.hobbies.solo_vs_social": _spec("long_term_facts.hobbies.solo_vs_social", "P1", ["event", "relationship", "feature"], ["事件规模 + 关系结果"], requires_protagonist_face=True),
    "long_term_facts.physiology.fitness_level": _spec("long_term_facts.physiology.fitness_level", "P1", ["event", "vlm", "feature"], ["持续运动事件"]),
    "long_term_facts.physiology.diet_mode": _spec("long_term_facts.physiology.diet_mode", "P1", ["event", "vlm", "feature"], ["稳定饮食模式"]),
    "long_term_facts.physiology.health_conditions": _spec("long_term_facts.physiology.health_conditions", "P1", ["event", "vlm", "feature"], ["跨事件健康相关线索"], hard_blocks=["从单次就医或单张药品推断长期健康状况"]),
    "short_term_facts.life_events": _spec("short_term_facts.life_events", "P1", ["event", "relationship", "feature"], ["近期窗口重大事件"]),
    "short_term_facts.phase_change": _spec("short_term_facts.phase_change", "P1", ["event", "relationship", "feature"], ["近期与长期基线系统偏移"]),
    "short_term_facts.spending_shift": _spec("short_term_facts.spending_shift", "P1", ["event", "vlm", "feature"], ["近期消费模式持续偏移"]),
    "short_term_facts.current_displacement": _spec("short_term_facts.current_displacement", "P0", ["event", "feature"], ["近期连续临时位移状态"], hard_blocks=["家校通勤"]),
    "short_term_facts.recent_habits": _spec("short_term_facts.recent_habits", "P1", ["event", "feature"], ["近期连续重复行为"]),
    "short_term_facts.recent_interests": _spec("short_term_facts.recent_interests", "P1", ["event", "vlm", "feature"], ["近期主题多次重复"]),
    "short_term_facts.physiological_state": _spec("short_term_facts.physiological_state", "P1", ["event", "vlm", "feature"], ["近期持续生理状态变化"], hard_blocks=["单次疲劳或单次运动后状态"]),
    "long_term_expression.personality_mbti": _spec("long_term_expression.personality_mbti", "P0", ["event", "relationship", "feature"], ["跨事件行为模式 + 至少3条行为证据"], hard_blocks=["单次情绪", "单次活动"], requires_social_media=True),
    "long_term_expression.morality": _spec("long_term_expression.morality", "P1", ["event", "relationship", "feature"], ["稳定价值取向二阶特征"], hard_blocks=["从单个场景外推道德"], requires_social_media=True),
    "long_term_expression.philosophy": _spec("long_term_expression.philosophy", "P1", ["event", "relationship", "feature"], ["稳定生活取向二阶特征"], hard_blocks=["从单个活动外推生活哲学"], requires_social_media=True),
    "long_term_expression.attitude_style": _spec("long_term_expression.attitude_style", "P1", ["vlm", "event", "feature"], ["穿搭/场景选择跨事件重复"], hard_blocks=["单次 dress code"]),
    "long_term_expression.aesthetic_tendency": _spec("long_term_expression.aesthetic_tendency", "P1", ["vlm", "event", "feature"], ["构图/色调/物品选择跨事件重复"], hard_blocks=["滤镜或场地主题"]),
    "long_term_expression.visual_creation_style": _spec("long_term_expression.visual_creation_style", "P1", ["vlm", "event", "feature"], ["拍照模式跨事件重复"], hard_blocks=["样本太少"]),
    "short_term_expression.current_mood": _spec("short_term_expression.current_mood", "P1", ["event", "vlm", "relationship", "feature"], ["近期多条情绪线索"], hard_blocks=["单张表情"]),
    "short_term_expression.mental_state": _spec("short_term_expression.mental_state", "P1", ["event", "vlm", "relationship", "feature"], ["近期连续心理状态线索"], hard_blocks=["只看表情"], requires_social_media=True),
    "short_term_expression.motivation_shift": _spec("short_term_expression.motivation_shift", "P1", ["event", "relationship", "feature"], ["近期投入方向持续变化"], hard_blocks=["单次冲动"], requires_social_media=True),
    "short_term_expression.stress_signal": _spec("short_term_expression.stress_signal", "P1", ["event", "relationship", "feature"], ["近期压力信号重复出现"], hard_blocks=["单次忙碌"], requires_social_media=True),
    "short_term_expression.social_energy": _spec("short_term_expression.social_energy", "P1", ["event", "relationship", "feature"], ["近期社交模式持续变化"], hard_blocks=["单周聚会"]),
}

LONG_TERM_FACT_FIELD_SPECS: Dict[str, FieldSpec] = {
    key: value for key, value in FIELD_SPECS.items() if key.startswith("long_term_facts.")
}

SHORT_TERM_FACT_FIELD_SPECS: Dict[str, FieldSpec] = {
    key: value for key, value in FIELD_SPECS.items() if key.startswith("short_term_facts.")
}

LONG_TERM_EXPRESSION_FIELD_SPECS: Dict[str, FieldSpec] = {
    key: value for key, value in FIELD_SPECS.items() if key.startswith("long_term_expression.")
}

SHORT_TERM_EXPRESSION_FIELD_SPECS: Dict[str, FieldSpec] = {
    key: value for key, value in FIELD_SPECS.items() if key.startswith("short_term_expression.")
}

LONG_TERM_FACT_NON_DAILY_SENSITIVE_FIELDS = {
    "long_term_facts.material.asset_level",
    "long_term_facts.material.spending_style",
    "long_term_facts.material.brand_preference",
    "long_term_facts.geography.location_anchors",
    "long_term_facts.geography.mobility_pattern",
    "long_term_facts.hobbies.interests",
    "long_term_facts.hobbies.frequent_activities",
}

NON_DAILY_EVENT_KEYWORDS = (
    "旅行",
    "travel",
    "trip",
    "车展",
    "expo",
    "exhibition",
    "festival",
    "展会",
    "主题活动",
    "打卡",
    "concert",
    "parade",
    "fair",
)

SOLITARY_SIGNAL_KEYWORDS = (
    "独处",
    "alone",
    "solo",
    "reading alone",
    "安静",
    "quiet",
    "看书",
)


def _normalize_string_list(values: Any) -> List[str]:
    normalized: List[str] = []
    for value in values or []:
        candidate = ""
        if isinstance(value, dict):
            for key in ("canonical_name", "raw_name", "brand_name", "name", "text", "value"):
                if value.get(key):
                    candidate = str(value[key]).strip()
                    break
        elif value is not None:
            candidate = str(value).strip()
        if candidate:
            normalized.append(candidate)
    return list(dict.fromkeys(normalized))


def _determine_subject_role(
    primary_person_id: str | None,
    people: List[str],
    face_person_ids: List[str],
    summary: str,
    activity: str,
) -> str:
    primary = str(primary_person_id or "").strip()
    people_set = {person for person in people if person}
    face_person_set = {person for person in face_person_ids if person}
    if primary and primary != "主角" and (primary in people_set or primary in face_person_set):
        return "protagonist_present"

    signal = " ".join(filter(None, [summary, activity]))
    protagonist_view_markers = ("主角", "自拍", "selfie", "first-person", "first person", "POV", "pov")
    if (not people_set and not face_person_set) and any(marker.lower() in signal.lower() for marker in protagonist_view_markers):
        return "protagonist_view"
    if primary == "主角" and "主角" in signal:
        return "protagonist_view"
    return "other_people_only"


def build_profile_context(state: MemoryState) -> Dict[str, Any]:
    vlm_observations = []
    primary_person_id = (state.primary_decision or {}).get("primary_person_id")
    for item in state.vlm_results or []:
        analysis = item.get("vlm_analysis", {}) or {}
        scene = analysis.get("scene", {}) or {}
        event = analysis.get("event", {}) or {}
        people = [p.get("person_id") for p in analysis.get("people", []) if isinstance(p, dict) and p.get("person_id")]
        face_person_ids = _normalize_string_list(item.get("face_person_ids", []))
        summary = analysis.get("summary", "")
        activity = event.get("activity", "") if isinstance(event, dict) else ""
        vlm_observations.append(
            {
                "photo_id": item.get("photo_id"),
                "timestamp": item.get("timestamp"),
                "summary": summary,
                "location": scene.get("location_detected", "") if isinstance(scene, dict) else "",
                "activity": activity,
                "people": people,
                "details": list(analysis.get("details", []) or []),
                "ocr_hits": _normalize_string_list(analysis.get("ocr_hits", [])),
                "brands": _normalize_string_list(analysis.get("brands", [])),
                "place_candidates": _normalize_string_list(analysis.get("place_candidates", [])),
                "face_person_ids": face_person_ids,
                "media_kind": item.get("media_kind"),
                "is_reference_like": bool(item.get("is_reference_like")),
                "subject_role": _determine_subject_role(primary_person_id, people, face_person_ids, summary, activity),
            }
        )
    return {
        "primary_person_id": primary_person_id,
        "events": list(state.events or []),
        "relationships": list(state.relationships or []),
        "groups": list(state.groups or []),
        "vlm_observations": vlm_observations,
        "feature_refs": _build_context_feature_refs(
            events=list(state.events or []),
            relationships=list(state.relationships or []),
            groups=list(state.groups or []),
        ),
        "social_media_available": False,
        "resolved_facts": {},
    }


def build_empty_structured_profile() -> Dict[str, Any]:
    return {
        "long_term_facts": {
            "identity": _section(["name", "gender", "age_range", "role", "race", "nationality"]),
            "social_identity": _section(["education", "career", "career_phase", "professional_dedication", "language_culture", "political_preference"]),
            "material": _section(["asset_level", "spending_style", "brand_preference", "income_model", "signature_items"]),
            "geography": _section(["location_anchors", "mobility_pattern", "cross_border"]),
            "time": _section(["life_rhythm", "event_cycles", "sleep_pattern"]),
            "relationships": _section(["intimate_partner", "close_circle_size", "social_groups", "pets", "parenting", "living_situation"]),
            "hobbies": _section(["interests", "frequent_activities", "solo_vs_social"]),
            "physiology": _section(["fitness_level", "health_conditions", "diet_mode"]),
        },
        "short_term_facts": _section(["life_events", "phase_change", "spending_shift", "current_displacement", "recent_habits", "recent_interests", "physiological_state"]),
        "long_term_expression": _section(["personality_mbti", "morality", "philosophy", "attitude_style", "aesthetic_tendency", "visual_creation_style"]),
        "short_term_expression": _section(["current_mood", "mental_state", "motivation_shift", "stress_signal", "social_energy"]),
    }


def generate_structured_profile(
    state: MemoryState,
    llm_processor: Any | None = None,
    rule_overlay: Dict[str, Any] | None = None,
    target_field_keys: set | None = None,
) -> Dict[str, Any]:
    context = state.profile_context or build_profile_context(state)
    structured = build_empty_structured_profile()
    effective_llm_processor = llm_processor or _resolve_profile_llm_processor(context)
    effective_field_specs = get_active_field_specs(rule_overlay=rule_overlay)
    profile_agent = ProfileAgent(effective_field_specs)
    with temporary_rule_overlay(rule_overlay):
        profile_result = profile_agent.run(context, structured, llm_processor=effective_llm_processor, target_field_keys=target_field_keys)
    structured = profile_result["structured"]
    consistency = build_consistency_report(state.events or [], state.relationships or [], structured)
    return {
        "structured": structured,
        "field_decisions": profile_result["field_decisions"],
        "llm_batch_debug": profile_result.get("llm_batch_debug", []),
        "consistency": consistency,
    }


def get_active_field_specs(*, rule_overlay: Dict[str, Any] | None = None) -> Dict[str, FieldSpec]:
    return get_effective_field_specs(base_field_specs=FIELD_SPECS, overlay_bundle=rule_overlay)


def clear_runtime_field_spec_updates() -> None:
    clear_runtime_rule_overlays()


def _resolve_profile_llm_processor(context: Dict[str, Any]) -> Any | None:
    primary_person_id = context.get("primary_person_id")
    if PROFILE_LLM_PROVIDER == "openrouter":
        try:
            from .profile_llm import OpenRouterProfileLLMProcessor

            return OpenRouterProfileLLMProcessor(
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL,
                model=PROFILE_LLM_MODEL,
                primary_person_id=primary_person_id,
            )
        except Exception:
            pass
    try:
        from services.llm_processor import LLMProcessor

        return LLMProcessor(primary_person_id=primary_person_id)
    except Exception:
        return None


def _deterministic_field_value(field_key: str, context: Dict[str, Any]) -> Tuple[Any, float]:
    relationships = context.get("relationships", [])
    groups = context.get("groups", [])
    events = context.get("events", [])

    if field_key == "long_term_facts.relationships.intimate_partner":
        romantic = next((rel.person_id for rel in relationships if rel.relationship_type == "romantic"), None)
        return romantic, 0.88 if romantic else 0.0
    if field_key == "long_term_facts.relationships.close_circle_size":
        return _compute_close_circle_size(relationships), 0.82
    if field_key == "long_term_facts.relationships.social_groups":
        group_names = [group.group_type_candidate for group in groups]
        return group_names or None, 0.76 if group_names else 0.0
    if field_key == "long_term_facts.time.sleep_pattern":
        ratio = _late_night_event_ratio(events)
        if ratio >= 0.5:
            return "night_owl", 0.65
        if ratio > 0:
            return "irregular", 0.55
        return None, 0.0
    if field_key == "long_term_facts.relationships.living_situation":
        inferred = _infer_living_situation_from_events_and_relationships(events, relationships)
        if inferred is not None:
            return inferred, 0.78
        return None, 0.0
    return None, 0.0


def _llm_field_value(bundle: FieldBundle, context: Dict[str, Any], llm_processor: Any) -> Tuple[Any, float]:
    resolved_facts_clause = ""
    if bundle.field_key.startswith(("long_term_expression.", "short_term_expression.")) and context.get("resolved_facts"):
        resolved_facts_clause = f"\n已定稿facts:\n{context.get('resolved_facts')}\n"
    prompt = f"""你是客观画像字段判定 agent。
字段: {bundle.field_key}
风险等级: {bundle.field_spec.risk_level}
强证据: {bundle.field_spec.strong_evidence}
字段级COT:
主判顺序:
{_format_list_for_prompt(bundle.field_spec.cot_steps)}
主体归属检查:
{_format_list_for_prompt(bundle.field_spec.owner_resolution_steps)}
时间层判断:
{_format_list_for_prompt(bundle.field_spec.time_reasoning_steps)}
反证检查:
{_format_list_for_prompt(bundle.field_spec.counter_evidence_checks)}
以下情况证据较弱，需要更谨慎地判断（不是直接输出 null，而是需要更充分的证据支撑）:
{_format_list_for_prompt(bundle.field_spec.weak_evidence_caution or ["证据不足时需要更多证据才能下结论"])}
反思问题: {bundle.field_spec.reflection_questions}
允许证据:
{bundle.allowed_refs}
{resolved_facts_clause}

请严格按上面的字段级COT在内部逐步判断，但不要输出推理过程。
若主体归属不清、时间层级不清、或反证更强，需要更充分的证据支撑才能输出结论。当确实没有任何证据时才输出 null。

请只输出 JSON:
{{
  "value": null,
  "confidence": 0.0
}}"""
    try:
        result = llm_processor._call_llm_via_official_api(prompt, response_mime_type="application/json")
    except Exception:
        return None, 0.0
    if not isinstance(result, dict):
        return None, 0.0
    return result.get("value"), float(result.get("confidence", 0.0) or 0.0)


def _build_tag_evidence(
    allowed_refs: Dict[str, List[Dict[str, Any]]],
    supporting_refs: Dict[str, List[Dict[str, Any]]],
    contradicting_refs: Dict[str, List[Dict[str, Any]]],
    gate_result: Dict[str, Any],
    field_key: str,
) -> Dict[str, Any]:
    normalized_supporting = flatten_ref_buckets(supporting_refs)
    normalized_contradicting = flatten_ref_buckets(contradicting_refs)
    ids = extract_ids_from_refs(normalized_supporting)
    evidence = build_evidence_payload(
        photo_ids=ids["photo_ids"],
        event_ids=ids["event_ids"],
        person_ids=ids["person_ids"],
        group_ids=ids["group_ids"],
        feature_names=ids["feature_names"],
        supporting_refs=normalized_supporting,
        contradicting_refs=normalized_contradicting,
    )
    evidence["events"] = supporting_refs.get("events", [])
    evidence["relationships"] = supporting_refs.get("relationships", [])
    evidence["vlm_observations"] = supporting_refs.get("vlm_observations", []) or allowed_refs.get("vlm_observations", [])
    evidence["group_artifacts"] = supporting_refs.get("group_artifacts", [])
    evidence["feature_refs"] = supporting_refs.get("feature_refs", [])
    constraint_notes = [] if not gate_result["must_null"] else [f"cleared_by_field_gate:{field_key}"]
    if gate_result.get("silent_by_missing_social_media"):
        constraint_notes.append("silent_by_missing_social_media")
    evidence["constraint_notes"] = constraint_notes
    evidence["summary"] = f"field_judge:{field_key}"
    return evidence


def _build_field_reasoning(
    *,
    field_key: str,
    value: Any,
    evidence: Dict[str, Any],
    gate_result: Dict[str, Any],
    reflection_1: Dict[str, Any],
    reflection_2: Dict[str, Any],
) -> str:
    evidence_ids = (
        evidence.get("event_ids", [])[:2]
        + evidence.get("photo_ids", [])[:2]
        + evidence.get("person_ids", [])[:1]
        + evidence.get("feature_names", [])[:2]
    )
    evidence_clause = "、".join(evidence_ids) if evidence_ids else "当前证据池"
    if value is None:
        if gate_result.get("silent_by_missing_social_media"):
            return f"{field_key} 当前缺少社媒模态，未进入推断；审查 {evidence_clause} 后静默输出 null。"
        if reflection_2.get("null_reason") == "null_due_to_expression_conflict_reflection":
            return f"{field_key} 读取已定稿 facts 后，又在事件反证阶段发现冲突；审查 {evidence_clause} 后未通过表达层反思，回退为 null。"
        if reflection_2.get("null_reason") == "null_due_to_non_daily_event_reflection":
            return f"{field_key} 的支持信号主要来自非日常高密度事件；审查 {evidence_clause} 后为防止非日常事件干扰回退为 null。"
        if gate_result.get("must_null"):
            return f"{field_key} 因强证据不足被字段闸门清空；已检查 {evidence_clause} 后仍不满足非 null 条件。"
        if reflection_2.get("decision") == "null":
            return f"{field_key} 在反思阶段发现证据冲突或风险过高；审查 {evidence_clause} 后回退为 null。"
        return f"{field_key} 在审查 {evidence_clause} 后没有形成稳定结论，因此保守输出 null。"
    if reflection_1.get("issues_found"):
        return f"{field_key} 主要依据 {evidence_clause} 得到当前结论，并在反思后排除了 {', '.join(reflection_1['issues_found'])}。"
    return f"{field_key} 主要依据 {evidence_clause} 得到当前结论，且没有发现足以推翻该字段的反证。"


def _format_list_for_prompt(items: List[str]) -> str:
    if not items:
        return "- 无"
    return "\n".join(f"- {item}" for item in items)


def _section(keys: Iterable[str]) -> Dict[str, Any]:
    return {key: _empty_tag_object() if not key.endswith("_preference") else _empty_tag_object() for key in keys}


def _empty_tag_object() -> Dict[str, Any]:
    evidence = build_evidence_payload()
    evidence["events"] = []
    evidence["relationships"] = []
    evidence["vlm_observations"] = []
    evidence["group_artifacts"] = []
    evidence["feature_refs"] = []
    evidence["constraint_notes"] = []
    evidence["summary"] = ""
    return {
        "value": None,
        "confidence": 0.0,
        "evidence": evidence,
        "reasoning": "",
    }


def _assign_tag_object(payload: Dict[str, Any], field_key: str, value: Dict[str, Any]) -> None:
    path = field_key.split(".")
    current = payload
    for part in path[:-1]:
        current = current.setdefault(part, {})
    current[path[-1]] = value


def _contains_any_keyword(text: str, keywords: Tuple[str, ...]) -> bool:
    normalized = str(text or "").lower()
    return any(keyword.lower() in normalized for keyword in keywords)


def _is_non_daily_event_ref(ref: Dict[str, Any]) -> bool:
    return _contains_any_keyword(
        " ".join(
            filter(
                None,
                [
                    ref.get("signal", ""),
                    ref.get("type", ""),
                    ref.get("description", ""),
                    ref.get("narrative_synthesis", ""),
                ],
            )
        ),
        NON_DAILY_EVENT_KEYWORDS,
    )


def _late_night_event_ratio(events: List[Any]) -> float:
    if not events:
        return 0.0
    late_night_events = 0
    for event in events:
        time_range = getattr(event, "time_range", "") or ""
        start_time = time_range.split(" - ")[0].strip()
        if start_time[:2].isdigit() and int(start_time[:2]) >= 22:
            late_night_events += 1
    return round(late_night_events / max(len(events), 1), 2)


def _expression_conflicts_with_events(field_key: str, context: Dict[str, Any]) -> bool:
    resolved_facts = context.get("resolved_facts", {}) or {}
    long_term_facts = resolved_facts.get("long_term_facts", {}) or {}
    hobby_facts = long_term_facts.get("hobbies", {}) or {}
    solo_vs_social = ((hobby_facts.get("solo_vs_social") or {}).get("value"))

    event_signals: List[str] = []
    for event in context.get("events", []):
        event_signals.append(
            " ".join(
                filter(
                    None,
                    [
                        getattr(event, "title", ""),
                        getattr(event, "type", ""),
                        getattr(event, "location", ""),
                        getattr(event, "description", ""),
                        getattr(event, "narrative_synthesis", ""),
                    ],
                )
            )
        )
    for observation in context.get("vlm_observations", []):
        event_signals.append(
            " ".join(
                filter(
                    None,
                    [
                        observation.get("summary", ""),
                        observation.get("location", ""),
                        observation.get("activity", ""),
                    ],
                )
            )
        )

    if field_key == "long_term_expression.personality_mbti" and solo_vs_social == "social":
        return any(_contains_any_keyword(signal, SOLITARY_SIGNAL_KEYWORDS) for signal in event_signals)
    return False


def _build_context_feature_refs(
    *,
    events: List[Any],
    relationships: List[Any],
    groups: List[Any],
) -> List[Dict[str, Any]]:
    close_circle_size = _compute_close_circle_size(relationships)
    return [
        {"feature_name": "event_count", "value": len(events)},
        {"feature_name": "relationship_count", "value": len(relationships)},
        {"feature_name": "group_count", "value": len(groups)},
        {"feature_name": "close_circle_size", "value": close_circle_size},
    ]


def _compute_close_circle_size(relationships: List[Any]) -> int:
    close_types = {"romantic", "family", "bestie", "close_friend"}
    person_ids: List[str] = []
    for rel in relationships:
        rel_type = str(getattr(rel, "relationship_type", "") or "")
        confidence = float(getattr(rel, "confidence", 0.0) or 0.0)
        person_id = str(getattr(rel, "person_id", "") or "")
        if not person_id:
            continue
        if rel_type in close_types and confidence >= 0.6:
            person_ids.append(person_id)
    return len(set(person_ids))


def _infer_living_situation_from_events_and_relationships(events: List[Any], relationships: List[Any]) -> str | None:
    shared_home_rel_ids = {
        str(rel.person_id)
        for rel in relationships
        if str(getattr(rel, "relationship_type", "")) in {"romantic", "family"}
        and float(getattr(rel, "confidence", 0.0) or 0.0) >= 0.7
    }
    if not shared_home_rel_ids:
        return None

    home_event_hits = 0
    for event in events:
        location_text = " ".join(
            [
                str(getattr(event, "location", "") or ""),
                str(getattr(event, "title", "") or ""),
                str(getattr(event, "description", "") or ""),
            ]
        ).lower()
        participants = set(getattr(event, "participants", []) or [])
        if not participants.intersection(shared_home_rel_ids):
            continue
        if any(keyword in location_text for keyword in ("home", "room", "宿舍", "家", "apartment", "卧室")):
            home_event_hits += 1
    if home_event_hits >= 2:
        return "shared"
    return None
