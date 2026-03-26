"""
LLM处理模块：事件提取、关系推断、画像生成
"""
import json
import re
import time
from datetime import datetime
from typing import List, Dict, Optional
from models import Event, Relationship, UserProfile
from config import (
    GEMINI_API_KEY, LLM_MODEL, MIN_PHOTO_COUNT, MIN_TIME_SPAN_DAYS, MIN_SCENE_VARIETY,
    GOOGLE_GEMINI_BASE_URL, INTIMACY_WEIGHT_FREQUENCY, INTIMACY_WEIGHT_INTERACTION,
    INTIMACY_WEIGHT_SCENE_DIVERSITY, INTERACTION_SCORES, PRIVATE_SCENE_TYPES,
)
from utils import calculate_distance, time_overlap, is_weekend

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - test env may not install requests
    requests = None
from services.relationship_rules import (
    apply_relationship_type_veto,
    apply_status_redlines,
    blend_relationship_confidence,
    determine_relationship_status,
    infer_relationship_candidates,
    score_relationship_confidence,
)
from services.consistency_checker import build_consistency_report


class LLMProcessor:
    """LLM处理器 - 使用 HTTP 请求调用代理服务"""

    def __init__(self, primary_person_id: str = None):
        self.model = LLM_MODEL
        self.primary_person_id = primary_person_id

        if not GEMINI_API_KEY:
            raise ValueError("未配置 GEMINI_API_KEY")
        if not GOOGLE_GEMINI_BASE_URL:
            raise ValueError("未配置 GOOGLE_GEMINI_BASE_URL")

        self.api_key = GEMINI_API_KEY
        self.base_url = GOOGLE_GEMINI_BASE_URL
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        print(f"[LLM] 使用 HTTP 请求方式")
        print(f"[LLM] BaseURL: {self.base_url}")
        print(f"[LLM] 模型: {self.model}")

    def _extract_json_from_text(self, text: str, target_key: str = "events") -> dict:
        """
        从混合文本中鲁棒提取包含 target_key 的 JSON 对象。

        策略链：
        1. 直接 json.loads
        2. ```json 代码块提取
        3. 花括号平衡匹配，找包含 target_key 的最大 JSON 块
        """
        # 策略1：直接解析
        try:
            result = json.loads(text)
            if isinstance(result, dict) and target_key in result:
                return result
        except json.JSONDecodeError:
            pass

        # 策略2：```json 代码块（贪婪匹配最大块）
        json_blocks = re.findall(r'```json\s*\n(.*?)\n\s*```', text, re.DOTALL)
        for block in json_blocks:
            try:
                result = json.loads(block.strip())
                if isinstance(result, dict) and target_key in result:
                    print(f"调试：从```json代码块提取到JSON，包含{len(result.get(target_key, []))}个{target_key}")
                    return result
            except json.JSONDecodeError:
                continue

        # 策略3：花括号平衡匹配 - 找包含 target_key 的 JSON 对象
        search_pattern = f'"{target_key}"'
        idx = text.find(search_pattern)
        while idx != -1:
            # 向前找到包含此 key 的最近 '{'
            brace_start = text.rfind('{', 0, idx)
            if brace_start == -1:
                idx = text.find(search_pattern, idx + 1)
                continue

            # 从 brace_start 开始，用平衡计数找到匹配的 '}'
            depth = 0
            in_string = False
            escape_next = False
            for i in range(brace_start, len(text)):
                ch = text[i]
                if escape_next:
                    escape_next = False
                    continue
                if ch == '\\' and in_string:
                    escape_next = True
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[brace_start:i + 1]
                        try:
                            result = json.loads(candidate)
                            if isinstance(result, dict) and target_key in result:
                                print(f"调试：花括号匹配提取到JSON，包含{len(result.get(target_key, []))}个{target_key}")
                                return result
                        except json.JSONDecodeError:
                            break
                        break

            idx = text.find(search_pattern, idx + 1)

        return None

    def _require_requests(self) -> None:
        if requests is None:
            raise RuntimeError("requests 未安装，无法调用 HTTP LLM 接口")

    def extract_events(self, vlm_results: List[Dict]) -> List[Event]:
        """
        从VLM结果中提取事件（直接让LLM分析全部数据）

        Args:
            vlm_results: VLM分析结果列表

        Returns:
            事件列表
        """
        # 直接让LLM分析全部VLM数据，不进行预分段
        all_events = self._extract_events_from_all_photos(vlm_results)

        # 物理约束检查
        all_events = self._check_constraints(all_events)

        return all_events

    def _extract_events_from_all_photos(self, vlm_results: List[Dict]) -> List[Event]:
        """
        用LLM从全部照片中提取事件（两步法）
        Step 1: 2.5-flash 生成文本分析
        Step 2: 2.5-flash + responseMimeType 将文本转为 JSON
        """
        prompt = self._create_event_extraction_prompt(vlm_results)

        try:
            # Step 1: 生成文本分析
            print("  [Step 1/2] 生成事件分析...", flush=True)
            result = self._call_llm_via_official_api(prompt)

            if result is None:
                print(f"警告：LLM返回None")
                return []

            # 理想情况：直接返回了 JSON
            if isinstance(result, dict) and "events" in result:
                print(f"  [Step 1/2] 直接获得JSON，{len(result['events'])}个事件")
                return [self._build_event(ed, i) for i, ed in enumerate(result["events"])]

            # 尝试从文本提取 JSON
            if isinstance(result, dict) and "text" in result:
                text = result["text"]
                extracted = self._extract_json_from_text(text, "events")
                if extracted:
                    print(f"  [Step 1/2] 从文本提取到{len(extracted['events'])}个事件")
                    return [self._build_event(ed, i) for i, ed in enumerate(extracted["events"])]

                # Step 2: 文本转 JSON
                print(f"  [Step 2/2] 文本转JSON（{len(text)}字符）...", flush=True)
                if len(text) > 30000:
                    text = text[:30000]

                convert_prompt = f"""将以下事件分析文本转换为JSON格式。只输出JSON，不要其他内容。

要求：
- 每个事件都必须补齐 `confidence` 和 `reason`
- `confidence` 是事件级置信度（0-1）
- 判断标准：
  - 多张照片、时间地点一致、参与者明确、叙事闭环完整 → 高置信度（0.75-0.95）
  - 证据较完整但仍有少量模糊 → 中高置信度（0.6-0.74）
  - 单张照片、地点模糊、参与者不清或叙事跳跃 → 较低置信度（0.35-0.59）
  - 不允许省略 `confidence`

格式：
{{"events": [{{"event_id": "EVT_001", "meta_info": {{"title": "标题", "timestamp": "时间", "location_context": "地点", "photo_count": 0}}, "objective_fact": {{"scene_description": "描述", "participants": []}}, "narrative_synthesis": "叙事", "social_dynamics": [], "persona_evidence": {{"behavioral": [], "aesthetic": [], "socioeconomic": []}}, "tags": [], "confidence": 0.8, "reason": "依据"}}]}}

分析文本：
{text}"""

                json_result = self._call_llm_via_official_api(convert_prompt, response_mime_type="application/json")

                if json_result and isinstance(json_result, dict):
                    if "events" in json_result:
                        print(f"  [Step 2/2] 转换成功，{len(json_result['events'])}个事件")
                        return [self._build_event(ed, i) for i, ed in enumerate(json_result["events"])]
                    if "text" in json_result:
                        extracted = self._extract_json_from_text(json_result["text"], "events")
                        if extracted:
                            print(f"  [Step 2/2] 二次提取成功，{len(extracted['events'])}个事件")
                            return [self._build_event(ed, i) for i, ed in enumerate(extracted["events"])]

                print(f"  事件提取：两步法均失败")

            return []

        except Exception as e:
            print(f"警告：事件提取失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _build_event(self, event_data: Dict, index: int) -> Event:
        """辅助方法：从事件数据构建Event对象"""
        # 兼容新旧格式：从meta_info提取或直接使用
        meta_info = dict(event_data.get("meta_info", {}) or {})
        objective_fact = dict(event_data.get("objective_fact", {}) or {})
        participants = self._normalize_event_participants(
            objective_fact.get("participants") or event_data.get("participants", [])
        )
        social_dynamics = self._normalize_social_dynamics(event_data.get("social_dynamics", []))
        objective_fact["participants"] = participants
        event_confidence_baseline = self._compute_event_confidence_baseline(
            event_data=event_data,
            meta_info=meta_info,
            objective_fact=objective_fact,
            social_dynamics=social_dynamics,
        )
        event_confidence = self._resolve_event_confidence(
            raw_confidence=event_data.get("confidence"),
            baseline=event_confidence_baseline,
        )

        # 解析timestamp从meta_info或直接字段
        timestamp = meta_info.get("timestamp") or event_data.get("time_range", "")
        if timestamp:
            # 从timestamp中提取日期和时间范围
            parts = timestamp.split(" - ")
            if len(parts) == 2:
                date_part = parts[0].split()[0]
                time_range = f"{parts[0].split()[-1]} - {parts[1].split()[-1]}" if " " in parts[1] else f"{parts[0].split()[-1]} - {parts[1]}"
            else:
                date_part = timestamp.split()[0] if " " in timestamp else ""
                time_range = timestamp
        else:
            date_part = event_data.get("date", "")
            time_range = event_data.get("time_range", "")

        event = Event(
            event_id=event_data.get("event_id", f"EVT_{index + 1:03d}"),
            date=date_part,
            time_range=time_range,
            duration=event_data.get("duration", ""),
            title=meta_info.get("title") or event_data.get("title", ""),
            type=event_data.get("type", "其他"),
            participants=participants,
            location=meta_info.get("location_context") or event_data.get("location", ""),
            description=objective_fact.get("scene_description") or event_data.get("description", ""),
            photo_count=int(meta_info.get("photo_count") or event_data.get("photo_count", 0)),
            confidence=event_confidence,
            reason=event_data.get("reason", ""),
            narrative="",
            narrative_synthesis=event_data.get("narrative_synthesis", ""),
            meta_info=meta_info,
            objective_fact=objective_fact,
            social_interaction={},
            social_dynamics=social_dynamics,

            lifestyle_tags=[],
            tags=event_data.get("tags", []),
            social_slices=[],
            persona_evidence=event_data.get("persona_evidence", {})
        )
        return event

    def _compute_event_confidence_baseline(
        self,
        event_data: Dict,
        meta_info: Dict,
        objective_fact: Dict,
        social_dynamics: List[Dict],
    ) -> float:
        """根据事件结构完整度计算事件级 baseline confidence。"""
        score = 0.28

        photo_count = int(meta_info.get("photo_count") or event_data.get("photo_count", 0) or 0)
        if photo_count >= 4:
            score += 0.18
        elif photo_count >= 2:
            score += 0.12
        elif photo_count >= 1:
            score += 0.06

        if meta_info.get("title") or event_data.get("title"):
            score += 0.07
        if meta_info.get("timestamp") or event_data.get("date") or event_data.get("time_range"):
            score += 0.10
        if meta_info.get("location_context") or event_data.get("location"):
            score += 0.08
        if objective_fact.get("scene_description") or event_data.get("description"):
            score += 0.08
        if objective_fact.get("participants"):
            score += 0.08
        if event_data.get("narrative_synthesis"):
            score += 0.08
        if event_data.get("type") and event_data.get("type") != "其他":
            score += 0.05
        if social_dynamics:
            score += min(0.08, 0.04 + 0.02 * len(social_dynamics))
        persona_evidence = event_data.get("persona_evidence", {}) or {}
        if any(persona_evidence.get(key) for key in ("behavioral", "aesthetic", "socioeconomic")):
            score += 0.05
        if event_data.get("tags"):
            score += 0.03

        return round(max(0.25, min(score, 0.95)), 3)

    def _resolve_event_confidence(self, raw_confidence, baseline: float) -> float:
        """融合 LLM confidence 和代码 baseline，避免整批回落到 0.5。"""
        if raw_confidence in (None, ""):
            return baseline

        llm_confidence = self._normalize_confidence(raw_confidence, default=baseline)
        blended = 0.6 * llm_confidence + 0.4 * baseline
        return round(max(0.25, min(blended, 0.95)), 3)

    def _normalize_primary_person_reference(self, raw_value):
        """把 LP1 中对主角的历史别名归一到真实 primary_person_id。"""
        if not isinstance(raw_value, str):
            return raw_value

        value = raw_value.strip()
        if not value or not self.primary_person_id:
            return value

        compact = re.sub(r"[\s\-]+", "", value).replace("【", "").replace("】", "").lower()
        primary_aliases = {
            "主角",
            "主角本人",
            "主角person_id",
            "主角personid",
            "用户",
            "用户本人",
            "用户person_id",
            "用户personid",
            "primary_person_id",
            "primarypersonid",
        }

        if compact in primary_aliases or re.fullmatch(r"person_?0+", compact):
            return self.primary_person_id

        return value

    def _normalize_event_participants(self, participants) -> List[str]:
        """清理 LP1 participants，避免主角占位符污染下游召回。"""
        normalized = []
        seen = set()

        for participant in participants or []:
            if not isinstance(participant, str):
                continue
            person_ref = self._normalize_primary_person_reference(participant)
            if not person_ref or person_ref in seen:
                continue
            normalized.append(person_ref)
            seen.add(person_ref)

        return normalized

    def _normalize_social_dynamics(self, social_dynamics) -> List[Dict]:
        """只对 person_id 类字段做兜底纠偏，其他字段保持原样。"""
        normalized = []

        for item in social_dynamics or []:
            if not isinstance(item, dict):
                continue
            dynamic = dict(item)
            dynamic["target_id"] = self._normalize_primary_person_reference(dynamic.get("target_id"))
            normalized.append(dynamic)

        return normalized

    def _create_event_extraction_prompt(self, vlm_results: List[Dict]) -> str:
        """创建事件提取prompt（适配新的简化VLM格式）"""
        photos_info = []

        for item in vlm_results:
            # 获取地点：优先使用VLM识别的location_detected，其次EXIF GPS
            exif_location = item['location'].get('name', '未知') if item.get('location') else '未知'
            scene_info = item['vlm_analysis'].get('scene', {})
            if isinstance(scene_info, dict):
                vlm_location = scene_info.get('location_detected', '')
                location_type = scene_info.get('location_type', '')
                scene_details = ', '.join(scene_info.get('environment_details', []))
            else:
                vlm_location = ''
                location_type = ''
                scene_details = str(scene_info)
            location = vlm_location if vlm_location else exif_location

            # 获取人物详情
            people_details = item['vlm_analysis'].get('people', [])
            people_str = ""
            for p in people_details:
                pid = p.get('person_id', '未知')
                appearance = p.get('appearance', '')
                clothing = p.get('clothing', '')
                activity = p.get('activity', '')
                interaction = p.get('interaction', '')
                expression = p.get('expression', '')
                people_str += f"""
    - {pid}:
      外貌: {appearance}
      穿着: {clothing}
      动作: {activity}
      互动: {interaction}
      表情: {expression}"""

            # 获取实体关系三元组
            relations = item['vlm_analysis'].get('relations', [])
            relations_str = ""
            for r in relations:
                if isinstance(r, dict):
                    relations_str += f"\n    - {r.get('subject', '')} → {r.get('relation', '')} → {r.get('object', '')}"

            # 获取事件信息
            event_info = item['vlm_analysis'].get('event', {})
            if isinstance(event_info, str):
                event_activity = event_info
                event_social = ''
                event_mood = ''
                story_hints = ''
            else:
                event_activity = event_info.get('activity', '')
                event_social = event_info.get('social_context', '')
                event_mood = event_info.get('mood', '')
                hints = event_info.get('story_hints', [])
                story_hints = ', '.join(hints) if isinstance(hints, list) else str(hints)

            # 获取细节
            details = item['vlm_analysis'].get('details', '')
            if isinstance(details, list):
                details = ', '.join(details)
            elif isinstance(details, dict):
                details = str(details)

            # 获取人物ID列表
            vlm_people = item['vlm_analysis'].get('people', [])
            person_ids = [p.get('person_id', '') for p in vlm_people]

            info = f"""
【照片 {item['photo_id']}】
时间: {item['timestamp']}
地点: {location}{f" ({location_type})" if location_type else ""}
人物ID: {', '.join(person_ids) if person_ids else '无'}
人物详情:{people_str if people_str else " 无"}
实体关系:{relations_str if relations_str else " 无"}
VLM描述: {item['vlm_analysis'].get('summary', '')}
场景细节: {scene_details}
活动: {event_activity}
社交背景: {event_social}
氛围: {event_mood}
{f"故事线索: {story_hints}" if story_hints else ""}
细节: {details}
---
"""
            photos_info.append(info)

        # 提取日期用于背景推理
        dates = [item['timestamp'][:10] for item in vlm_results if item.get('timestamp')]
        if not dates:
            date_info = "照片拍摄时间范围：未知"
        else:
            date_info = f"照片拍摄时间范围：{dates[0]} 至 {dates[-1]}，共{len(set(dates))}天"

        primary_person_output_id = self.primary_person_id or "主角对应的 Person_###"
        if self.primary_person_id:
            primary_person_contract = f"""
Primary Person Contract (人物主键约束):
- 当前数据集中，【主角】唯一对应的 person_id 是 {self.primary_person_id}
- 任何涉及用户/主角本人的 participants 或 social_dynamics.target_id，都必须直接写 {self.primary_person_id}
- 不允许输出 `Person_0` / `person_0` / `主角person_id` / `用户person_id` 这类历史别名或占位符
- `participants` 只能填写真实出现的 Person_###；若推断存在未识别的拍摄者，只能写进 `scene_description` 或 `social_dynamics.social_clue`，不要把说明文字写进 `participants`
"""
        else:
            primary_person_contract = """
Primary Person Contract (人物主键约束):
- 若涉及用户/主角本人，必须使用输入中真实出现的 Person_###，不要使用 `Person_0` / `person_0` / `主角person_id`
- `participants` 只能填写真实出现的 Person_###
"""

        prompt = f"""Role: 你是一位资深的人类学专家与社会学行为分析师，擅长从破碎的相册VLM元数据中，通过"时空-行为-关系"三维建模，将零散照片还原为逻辑连贯的"原子事件"，并为后续的用户人格画像（Who/Whom/What）提供高价值的推导证据。

Task:
1. 原子事件聚类：分析用户相册的VLM（视觉语言模型）原始数据（包含时间戳、地点、人物ID、场景描述），将零散的照片聚类为逻辑连贯的"客观事件"，并识别核心社交关系。
2. 微观线索提取：从物象、构图、互动中提取能反映用户身份、社交与性格的"切片"。
3. 特征向量标记：为每个事件打上维度标签，便于全局画像合成。
4. 为每个事件给出事件级 confidence（0-1）和 reason。confidence 必须综合考虑：照片数、时间地点一致性、参与者清晰度、叙事闭环程度。不要省略该字段。

{primary_person_contract}

Core Principles (核心原则):
- 视角法则（关键）：
  - 判定自拍：若画面出现手臂延伸、持手机姿势或镜面反射 → 用户主动记录，表明记录欲、自我展示
  - 判定他拍：若【主角】出现在中远景、双手未持拍摄设备且姿态自然 → 必须判定为"他拍"，现场存在"隐形成员（Invisible Photographer）"，这是识别亲密同伴的关键线索
  - 判定监控视角：画面为俯视、高角度、固定机位 → 公共/工作场所
  - 身份确认：截图或拍摄证件中的人物信息通常是用户本人
- 证据溯源：所有推断必须基于事实（如：通过Chanel包装推测经济能力，通过凌晨时段推测生活节奏）。严禁无具体证据的主观推测。

Analysis Logic (分析逻辑):

1. 时空聚类（Event Grouping）：
   - 若相邻照片时间差在4小时内，且地点（location）相同或场景（scene）逻辑连贯（如：从商场到商场餐厅），应归为一个"事件"
   - 若时间跨度大或地点发生显著位移，应开启新事件

2. 地点的逻辑推断：
   - 即使坐标缺失，也要通过环境要素判定场景属性：
   - 睡衣/厨具/床品 → 私密居家（家庭生活、个人空间）
   - 显示器/工牌/会议室 → 职业办公（工作身份、职业地位）
   - 摩托车/街道/商场 → 都市户外（出行习惯、生活节奏）
   - 餐具/菜单/餐桌 → 餐饮场所（饮食习惯、社交活动）
   - 奢侈品牌、包装 → 消费场景（经济能力、品味偏好）

3. 社交建模（Social Mapping）：
   - 识别和统计每个事件中出现的person_id
   - 核心同伴（Core Companion）：行为亲密（经常合影、身体接近）、有明显互动（对视、共同动作）、高频多事件出现 → 候选：家人、伴侣、密友
   - 频繁伙伴（Frequent Companion）：出现在3-8个事件中，多个不同场景（咖啡馆、餐厅、户外） → 候选：朋友、同事
   - 环境人物（Incidental Person）：仅在特定公共场所出现一次且无互动 → 不推断关系

4. 微观线索（Micro-Clues）：
   - 身份线索：工牌、员工证件、专业设备、消费小票、屏幕内容、APP界面 → 推导：职业身份、学历背景
   - 社交线索：拍摄者与被拍者的距离（亲密度）、身体语言（朝向/肢体接触）、合影频率与场景、出现地点（家/公司） → 推导：关系亲密度、互动模式
   - 性格线索：构图的秩序感（整齐/随意）、拍摄内容的重复性（如反复拍猫或拍路标）、截图的类型（工作/娱乐/学习）、修图风格 → 推导：性格倾向、专注力、表达欲
   - 审美线索：服装品牌（奢侈品 vs 快时尚）、配色偏好（素色 vs 鲜艳）、构图风格（对称 vs 自由）、物品选择（极简 vs 繁复） → 推导：审美水平、品味、生活品质
   - 社会经济线索：品牌档次（Chanel vs 淘宝）、消费场景（米其林餐厅 vs 街边小店）、生活设施（别墅 vs 出租房）、出行方式（私家车 vs 公交） → 推导：经济能力、消费观、阶层位置

Output Format (严格执行以下JSON格式):

{{
  "events": [
    {{
      "event_id": "唯一编号",
      "narrative_synthesis": "一句话深度还原：[身份/状态]在[环境]中[出于什么动机/情感]进行了[核心行为]，体现了[某种生活特质]。",
      "meta_info": {{
        "title": "事件核心动作短语",
        "timestamp": "YYYY-MM-DD HH:mm - HH:mm",
        "location_context": "推测的地点性质（如：私密居家/职业办公/公共街区）",
        "photo_count": "涉及的照片总数"
      }},
      "objective_fact": {{
        "scene_description": "客观事实：环境细节、品牌、构图视角（强调自拍/他拍/监控视角）、人物动作。",
        "participants": ["{primary_person_output_id}", "Person_002"]
      }},
      "social_dynamics": [
        {{
          "target_id": "Person_002",
          "interaction_type": "互动类型（如：他拍摄影师/共同进餐/合影）",
          "social_clue": "判定关系的证据（如：物理距离、眼神指向、视角高低）",
          "relation_hypothesis": "初步关系推断",
          "confidence": 0.9
        }}
      ],
      "persona_evidence": {{
        "behavioral": ["性格/习惯特征"],
        "aesthetic": ["视觉偏好/秩序感特征"],
        "socioeconomic": ["阶层/价值取向特征"]
      }},
      "tags": ["#标签1", "#标签2"],
      "confidence": 0.85,
      "reason": "事件成立依据"
    }}
  ]
}}

事件级 confidence 判断标准：
- 0.75-0.95：多张照片支撑，时间地点连贯，参与者明确，叙事闭环完整
- 0.60-0.74：证据较完整，但仍有少量模糊或推断跳跃
- 0.35-0.59：单张照片、地点模糊、参与者不清或叙事闭环弱
- 不允许省略 `confidence`

请开始分析：

{date_info}

{''.join(photos_info)}
"""

        return prompt

    def _extract_date(self, timestamp_str: str) -> str:
        """从时间戳中提取日期"""
        try:
            dt = datetime.fromisoformat(timestamp_str)
            return dt.strftime("%Y-%m-%d")
        except:
            return ""

    def _check_constraints(self, events: List[Event]) -> List[Event]:
        """
        物理约束检查

        Args:
            events: 事件列表

        Returns:
            检查后的事件列表
        """
        # 检查时间冲突
        for i, event1 in enumerate(events):
            for event2 in events[i+1:]:
                if time_overlap(event1.time_range, event2.time_range):
                    print(f"警告：事件{event1.event_id}和{event2.event_id}时间重叠")

        return events

    def infer_relationships(self, vlm_results: List[Dict], face_db: Dict, events=None) -> List[Relationship]:
        """
        推断人物关系（LP2 v3.0）

        Args:
            vlm_results: VLM结果列表
            face_db: 人脸库（可能是 dict 或 Person 对象）
            events: LP1 提取的事件列表（用于事件召回）

        Returns:
            关系列表
        """
        relationships = []

        # 遍历所有人脸库中的人物（除了主角）
        for person_id, person_info in face_db.items():
            # 跳过主角
            if self.primary_person_id and person_id == self.primary_person_id:
                continue

            # 检查是否满足触发条件
            should_infer, reason = self._should_infer_relationship(person_id, person_info, vlm_results)

            if not should_infer:
                print(f"跳过 {person_id}: {reason}")
                continue

            # 收集证据（含事件召回和 contact_types）
            evidence = self._collect_relationship_evidence(person_id, vlm_results, events)

            # 推断关系（含亲密度计算）
            relationship = self._infer_relationship(person_id, evidence, vlm_results, face_db)
            relationships.append(relationship)

        return relationships

    def _should_infer_relationship(self, person_id: str, person_info: Dict, vlm_results: List[Dict]) -> tuple:
        """
        判断是否应该推断关系

        Returns:
            (是否应该推断, 原因)
        """
        from datetime import datetime

        # 处理 person_info 可能是 dict 的情况
        if isinstance(person_info, dict):
            photo_count = person_info.get("photo_count", 0)
            first_seen = person_info.get("first_seen")
            last_seen = person_info.get("last_seen")
        else:
            photo_count = getattr(person_info, 'photo_count', 0)
            first_seen = getattr(person_info, 'first_seen', None)
            last_seen = getattr(person_info, 'last_seen', None)

        # 条件1：出现次数
        if photo_count < MIN_PHOTO_COUNT:
            return False, f"出现次数太少（{photo_count}次，需要≥{MIN_PHOTO_COUNT}次）"

        # 条件2：时间跨度
        if first_seen and last_seen:
            # 如果是字符串，需要转换为 datetime
            if isinstance(first_seen, str):
                try:
                    first_seen = datetime.fromisoformat(first_seen)
                except:
                    pass

            if isinstance(last_seen, str):
                try:
                    last_seen = datetime.fromisoformat(last_seen)
                except:
                    pass

            # 确保都是 datetime 对象后才计算
            if isinstance(first_seen, datetime) and isinstance(last_seen, datetime):
                time_span = (last_seen - first_seen).days
                if time_span < MIN_TIME_SPAN_DAYS:
                    return False, f"认识时间太短（{time_span}天，需要≥{MIN_TIME_SPAN_DAYS}天）"

        # 条件3：场景多样性
        scenes = self._get_person_scenes(person_id, vlm_results)
        if len(scenes) < MIN_SCENE_VARIETY:
            return False, f"场景数据不足（{len(scenes)}种场景，需要≥{MIN_SCENE_VARIETY}种）"

        return True, "可以推断"

    def _get_person_scenes(self, person_id: str, vlm_results: List[Dict]) -> List[str]:
        """获取人物出现的场景"""
        scenes = set()

        for result in vlm_results:
            # 从VLM分析结果中获取人物列表
            vlm_people = result["vlm_analysis"].get("people", [])
            for person in vlm_people:
                if person.get("person_id") == person_id:
                    scene = result["vlm_analysis"].get("scene", {}).get("location_detected", "")
                    if scene:
                        scenes.add(scene)
                    break

        return list(scenes)

    def _parse_interaction_text(self, text: str) -> float:
        """解析 people[].interaction 自由文本中的互动关键词，回退打分"""
        if not text:
            return 0.2
        text_lower = text.lower()
        keyword_scores = [
            (["亲吻", "kiss", "接吻"], 1.0),
            (["拥抱", "hug", "抱"], 1.0),
            (["牵手", "holding_hands", "hand"], 0.8),
            (["搂", "arm_in_arm", "挽"], 0.8),
            (["自拍", "selfie"], 0.5),
            (["肩", "shoulder", "lean"], 0.5),
            (["亲密", "intimate", "close"], 0.4),
            (["旁边", "near", "beside", "next"], 0.3),
        ]
        best = 0.2
        for keywords, score in keyword_scores:
            if any(kw in text_lower for kw in keywords):
                best = max(best, score)
        return best

    def _compute_intimacy_score(self, person_id: str, evidence: Dict, vlm_results: List[Dict], face_db: Dict) -> float:
        """纯代码计算亲密度分数（0-1），不调 LLM"""
        # --- 频率维度 ---
        person_count = evidence["photo_count"]
        total_non_protagonist = 0
        for pid, pinfo in face_db.items():
            if self.primary_person_id and pid == self.primary_person_id:
                continue
            if isinstance(pinfo, dict):
                total_non_protagonist += pinfo.get("photo_count", 0)
            else:
                total_non_protagonist += getattr(pinfo, "photo_count", 0)
        if total_non_protagonist > 0:
            relative_freq = person_count / total_non_protagonist
        else:
            relative_freq = 0
        freq_score = min(relative_freq * 3, 1.0)

        # --- 互动维度 ---
        interaction_scores_list = []
        for result in vlm_results:
            vlm_people = result.get("vlm_analysis", {}).get("people", [])
            for person in vlm_people:
                if person.get("person_id") == person_id:
                    contact_type = person.get("contact_type", "")
                    if contact_type and contact_type in INTERACTION_SCORES:
                        interaction_scores_list.append(INTERACTION_SCORES[contact_type])
                    else:
                        # 回退到 interaction 文本解析
                        interaction_scores_list.append(
                            self._parse_interaction_text(person.get("interaction", ""))
                        )
                    break
        if interaction_scores_list:
            interaction_score = sum(interaction_scores_list) / len(interaction_scores_list)
        else:
            interaction_score = 0.2

        # --- 场景多样性维度 ---
        person_scene_types = set()
        all_scene_types = set()
        private_count = 0
        total_person_scenes = 0
        for result in vlm_results:
            scene_data = result.get("vlm_analysis", {}).get("scene", {})
            if isinstance(scene_data, dict):
                scene_type = scene_data.get("location_detected", "")
            else:
                scene_type = ""
            if scene_type:
                all_scene_types.add(scene_type)
            vlm_people = result.get("vlm_analysis", {}).get("people", [])
            for person in vlm_people:
                if person.get("person_id") == person_id:
                    if scene_type:
                        person_scene_types.add(scene_type)
                        total_person_scenes += 1
                        if any(pt in scene_type.lower() for pt in PRIVATE_SCENE_TYPES):
                            private_count += 1
                    break
        if all_scene_types:
            diversity = len(person_scene_types) / len(all_scene_types)
        else:
            diversity = 0
        # 私密场景加权
        if total_person_scenes > 0:
            private_ratio = private_count / total_person_scenes
            diversity = min(diversity * (1 + 0.5 * private_ratio), 1.0)
        scene_score = min(diversity, 1.0)

        # --- 加权求和 ---
        intimacy = (
            freq_score * INTIMACY_WEIGHT_FREQUENCY
            + interaction_score * INTIMACY_WEIGHT_INTERACTION
            + scene_score * INTIMACY_WEIGHT_SCENE_DIVERSITY
        )
        return round(min(intimacy, 1.0), 3)

    def _recall_shared_events(self, person_id: str, events: List) -> List[Dict]:
        """从 LP1 事件中召回与 person_id 的共同事件（≤10 个）"""
        if not events:
            return []
        # 筛选包含 person_id 的事件
        shared = [e for e in events if person_id in (e.participants if hasattr(e, 'participants') else [])]
        if not shared:
            return []

        def event_to_dict(e):
            return {
                "event_id": e.event_id,
                "date": e.date,
                "title": e.title,
                "location": e.location,
                "photo_count": e.photo_count,
                "description": e.description,
                "participants": e.participants,
                "narrative_synthesis": getattr(e, 'narrative_synthesis', ''),
                "social_dynamics": getattr(e, 'social_dynamics', []),
            }

        if len(shared) <= 2:
            return [event_to_dict(e) for e in shared]

        # 必选：第一个 + 最后一个
        result = [event_to_dict(shared[0]), event_to_dict(shared[-1])]
        middle = shared[1:-1]

        # 中间按优先级取 top 8
        def priority_key(e):
            return -(e.photo_count or 0)
        middle_sorted = sorted(middle, key=priority_key)
        for e in middle_sorted[:8]:
            result.append(event_to_dict(e))

        return result

    def _collect_relationship_evidence(self, person_id: str, vlm_results: List[Dict], events=None) -> Dict:
        """收集关系推断的证据（优先消费 LP1 结果，VLM 仅用于亲密度打分）"""
        import statistics
        from collections import defaultdict

        evidence = {
            "photo_count": 0,
            "time_span": "",
            "time_span_days": 0,
            "recent_gap_days": 0,
            "scenes": [],
            "private_scene_ratio": 0.0,
            "dominant_scene_ratio": 0.0,
            "interaction_behavior": [],
            "weekend_frequency": "",
            "with_user_only": True,
            "sample_scenes": [],
            "contact_types": [],
            "rela_events": [],
            "monthly_frequency": 0.0,
            "trend_detail": {},
            "co_appearing_persons": [],
            "anomalies": [],
        }

        # 找出包含该人物的所有照片
        co_photos = []
        for result in vlm_results:
            vlm_people = result["vlm_analysis"].get("people", [])
            for person in vlm_people:
                if person.get("person_id") == person_id:
                    co_photos.append(result)
                    break

        evidence["photo_count"] = len(co_photos)
        if not co_photos:
            return evidence

        # --- 时间跨度 ---
        timestamps = sorted([datetime.fromisoformat(r["timestamp"]) for r in co_photos])
        first = timestamps[0]
        last = timestamps[-1]
        span_days = (last - first).days + 1
        evidence["time_span"] = f"{span_days}天"
        evidence["time_span_days"] = span_days

        # --- 月均频率 ---
        evidence["monthly_frequency"] = round(len(co_photos) / max(span_days / 30, 1), 1)

        # --- 场景提取（VLM，亲密度打分需要） ---
        scene_counts = {}
        private_scene_hits = 0
        for photo in co_photos:
            scene_data = photo["vlm_analysis"].get("scene", {})
            scene = scene_data.get("location_detected", "") if isinstance(scene_data, dict) else ""
            if scene and scene not in evidence["scenes"]:
                evidence["scenes"].append(scene)
            if scene:
                scene_counts[scene] = scene_counts.get(scene, 0) + 1
                if any(pt in scene.lower() for pt in PRIVATE_SCENE_TYPES):
                    private_scene_hits += 1

        if co_photos:
            evidence["private_scene_ratio"] = round(private_scene_hits / len(co_photos), 2)
        if scene_counts:
            evidence["dominant_scene_ratio"] = round(max(scene_counts.values()) / len(co_photos), 2)

        # --- 互动行为（优先从 LP1 social_dynamics 聚合） ---
        rela_events = self._recall_shared_events(person_id, events)
        evidence["rela_events"] = rela_events

        if rela_events:
            for ev in rela_events:
                for dyn in ev.get("social_dynamics", []):
                    if dyn.get("target_id") == person_id:
                        it = dyn.get("interaction_type", "")
                        if it and it not in evidence["interaction_behavior"]:
                            evidence["interaction_behavior"].append(it)
        else:
            # Fallback: 无 LP1 事件时，从 VLM 原始数据提取
            for photo in co_photos:
                for rel in photo["vlm_analysis"].get("relations", []):
                    if isinstance(rel, dict):
                        subj, obj, relation = rel.get("subject", ""), rel.get("object", ""), rel.get("relation", "")
                        if (subj == person_id or obj == person_id) and relation:
                            interaction = f"{subj} {relation} {obj}"
                            if interaction not in evidence["interaction_behavior"]:
                                evidence["interaction_behavior"].append(interaction)

        # --- contact_types（VLM 直接读，LP1 没有这个粒度） ---
        for photo in co_photos:
            for person in photo.get("vlm_analysis", {}).get("people", []):
                if person.get("person_id") == person_id:
                    ct = person.get("contact_type", "")
                    if ct:
                        evidence["contact_types"].append(ct)
                    break

        # --- 周末频率 ---
        weekend_count = sum(1 for t in timestamps if is_weekend(t))
        evidence["weekend_frequency"] = "高" if weekend_count / len(co_photos) > 0.5 else "低"
        dataset_last = max(datetime.fromisoformat(r["timestamp"]) for r in vlm_results if r.get("timestamp"))
        evidence["recent_gap_days"] = max((dataset_last - last).days, 0)

        # --- 是否只和主角二人 ---
        for photo in co_photos:
            vlm_people = photo.get("vlm_analysis", {}).get("people", [])
            other_people = [p.get("person_id") for p in vlm_people
                           if p.get("person_id") and p.get("person_id") not in (person_id, self.primary_person_id)]
            if other_people:
                evidence["with_user_only"] = False
                break

        # --- sample_scenes（保留少量样本给 prompt 展示） ---
        for photo in co_photos[:5]:
            scene_data = photo["vlm_analysis"].get("scene", {})
            scene = scene_data.get("location_detected", "") if isinstance(scene_data, dict) else ""
            event_data = photo["vlm_analysis"].get("event", {})
            evidence["sample_scenes"].append({
                "timestamp": photo["timestamp"],
                "scene": scene,
                "summary": photo["vlm_analysis"].get("summary", ""),
                "activity": event_data.get("activity", "") if isinstance(event_data, dict) else str(event_data),
                "social_context": event_data.get("social_context", "") if isinstance(event_data, dict) else ""
            })

        # --- 趋势细节（前后半段频率对比） ---
        if len(timestamps) >= 4:
            mid = first + (last - first) / 2
            first_half = [t for t in timestamps if t <= mid]
            second_half = [t for t in timestamps if t > mid]
            first_half_days = max((mid - first).days, 1)
            second_half_days = max((last - mid).days, 1)
            first_freq = round(len(first_half) / (first_half_days / 30), 1)
            second_freq = round(len(second_half) / (second_half_days / 30), 1)
            if second_freq > first_freq * 1.5:
                direction = "up"
            elif second_freq < first_freq * 0.5:
                direction = "down"
            else:
                direction = "flat"
            evidence["trend_detail"] = {
                "first_half_freq": first_freq,
                "second_half_freq": second_freq,
                "direction": direction,
                "change_ratio": round(second_freq / max(first_freq, 0.1), 1),
            }

        # --- 第三方共现人物 ---
        third_party = defaultdict(int)
        for photo in co_photos:
            for p in photo.get("vlm_analysis", {}).get("people", []):
                pid = p.get("person_id", "")
                if pid and pid not in (person_id, self.primary_person_id):
                    third_party[pid] += 1
        evidence["co_appearing_persons"] = [
            {"person_id": pid, "co_count": cnt, "co_ratio": round(cnt / len(co_photos), 2)}
            for pid, cnt in sorted(third_party.items(), key=lambda x: -x[1])
            if cnt >= 2
        ]

        # --- 异常检测（统计突变） ---
        anomalies = []

        # 1) 频率突变：相邻照片间隔的 z-score
        if len(timestamps) >= 4:
            intervals = [(timestamps[i+1] - timestamps[i]).days for i in range(len(timestamps)-1)]
            mean_iv = statistics.mean(intervals)
            std_iv = statistics.stdev(intervals) if len(intervals) > 1 else 0
            if std_iv > 0:
                for i, gap in enumerate(intervals):
                    z = (gap - mean_iv) / std_iv
                    if z < -2:
                        anomalies.append({
                            "type": "frequency_surge",
                            "description": f"Interval {gap}d vs avg {mean_iv:.0f}d",
                            "date": timestamps[i+1].strftime("%Y-%m-%d"),
                        })
                    elif z > 2:
                        anomalies.append({
                            "type": "frequency_drop",
                            "description": f"Gap {gap}d vs avg {mean_iv:.0f}d",
                            "date": timestamps[i+1].strftime("%Y-%m-%d"),
                        })

        # 2) 共现模式突变：高频第三方最近消失 / 新第三方首次出现
        if len(co_photos) >= 6:
            cutoff = len(co_photos) * 2 // 3
            early_photos = co_photos[:cutoff]
            recent_photos = co_photos[cutoff:]

            def get_pids(photo):
                return {p.get("person_id") for p in photo.get("vlm_analysis", {}).get("people", []) if p.get("person_id")}

            early_pids = set()
            for p in early_photos:
                early_pids |= get_pids(p)

            for cp in evidence["co_appearing_persons"]:
                if cp["co_ratio"] >= 0.5:
                    recent_count = sum(1 for p in recent_photos if cp["person_id"] in get_pids(p))
                    if recent_count == 0:
                        anomalies.append({
                            "type": "companion_disappeared",
                            "description": f"{cp['person_id']} (co_ratio {cp['co_ratio']}) absent recently",
                            "date": recent_photos[0]["timestamp"][:10],
                        })

            for p in recent_photos:
                for pid in get_pids(p):
                    if pid not in (self.primary_person_id, person_id) and pid not in early_pids:
                        anomalies.append({
                            "type": "new_companion",
                            "description": f"{pid} first appeared in shared photos",
                            "date": p["timestamp"][:10],
                        })
                        early_pids.add(pid)

        # 3) 场景突变：最近出现的首次新场景
        if len(co_photos) >= 6:
            early_scenes = set()
            for p in early_photos:
                s = p.get("vlm_analysis", {}).get("scene", {})
                loc = s.get("location_detected", "") if isinstance(s, dict) else ""
                if loc:
                    early_scenes.add(loc)
            for p in recent_photos:
                s = p.get("vlm_analysis", {}).get("scene", {})
                loc = s.get("location_detected", "") if isinstance(s, dict) else ""
                if loc and loc not in early_scenes:
                    anomalies.append({
                        "type": "new_scene",
                        "description": f"First time at '{loc}' together",
                        "date": p["timestamp"][:10],
                    })
                    early_scenes.add(loc)

        evidence["anomalies"] = anomalies

        return evidence

    def _infer_relationship(self, person_id: str, evidence: Dict, vlm_results: List[Dict], face_db: Dict) -> Relationship:
        """Soft-constraint LP2：代码给建议和红线，LLM 主导最终判断。"""
        intimacy_score = self._compute_intimacy_score(person_id, evidence, vlm_results, face_db)
        candidate_types = infer_relationship_candidates(evidence, intimacy_score)
        code_status_suggestion = determine_relationship_status(evidence)
        code_type_suggestion = candidate_types[0] if candidate_types else "acquaintance"
        code_confidence_baseline = score_relationship_confidence(
            evidence=evidence,
            intimacy_score=intimacy_score,
            relationship_type=code_type_suggestion,
            candidate_types=candidate_types or [code_type_suggestion],
        )

        shared_events = [
            {"event_id": ev.get("event_id", ""), "date": ev.get("date", ""), "narrative": ev.get("narrative_synthesis", "") or ev.get("title", "")}
            for ev in evidence.get("rela_events", [])
        ]

        prompt = self._create_relationship_prompt(
            person_id,
            evidence,
            intimacy_score,
            candidate_types=candidate_types,
            code_status_suggestion=code_status_suggestion,
        )

        try:
            result = self._call_llm_via_official_api(prompt)
            llm_relationship_type = result.get("relationship_type") or code_type_suggestion
            llm_status = result.get("status") or code_status_suggestion
            llm_confidence = self._normalize_confidence(
                result.get("confidence"),
                default=code_confidence_baseline,
            )
            llm_reason = result.get("reason", "")
        except Exception as e:
            print(f"警告：关系推断失败 ({person_id}): {e}")
            llm_relationship_type = code_type_suggestion
            llm_status = code_status_suggestion
            llm_confidence = code_confidence_baseline
            llm_reason = f"LLM调用失败，回退代码建议: {e}"

        final_type, type_vetoes = apply_relationship_type_veto(llm_relationship_type, candidate_types)
        final_status, status_vetoes = apply_status_redlines(llm_status, evidence)
        applied_vetoes = type_vetoes + status_vetoes
        final_confidence = blend_relationship_confidence(
            llm_confidence=llm_confidence,
            code_confidence_baseline=code_confidence_baseline,
            applied_vetoes=applied_vetoes,
        )

        reason = (
            f"候选关系建议: {', '.join(candidate_types)} | "
            f"代码状态建议: {code_status_suggestion} | "
            f"代码置信度基线: {code_confidence_baseline:.2f} | "
            f"LLM关系: {llm_relationship_type} | "
            f"LLM状态: {llm_status} | "
            f"LLM置信度: {llm_confidence:.2f}"
        )
        if applied_vetoes:
            reason += f" | 护栏触发: {', '.join(applied_vetoes)}"
        if llm_reason:
            reason += f" | LLM依据: {llm_reason}"

        evidence["decision_trace"] = {
            "candidate_types": candidate_types,
            "code_status_suggestion": code_status_suggestion,
            "code_confidence_baseline": code_confidence_baseline,
            "llm_relationship_type": llm_relationship_type,
            "llm_status": llm_status,
            "llm_confidence": llm_confidence,
            "final_relationship_type": final_type,
            "final_status": final_status,
            "final_confidence": final_confidence,
            "applied_vetoes": applied_vetoes,
        }

        return Relationship(
            person_id=person_id,
            relationship_type=final_type,
            intimacy_score=intimacy_score,
            status=final_status,
            confidence=final_confidence,
            reasoning=reason,
            shared_events=shared_events,
            evidence=evidence,
        )

    def _create_relationship_prompt(
        self,
        person_id: str,
        evidence: Dict,
        intimacy_score: float,
        candidate_types: List[str] | None = None,
        code_status_suggestion: str = "",
    ) -> str:
        """创建关系推断 prompt（LP2 v3.3 — 中文 prompt，保留拍摄者自然语言 fallback）"""
        candidate_types = candidate_types or []
        contact_types_str = ', '.join(evidence.get("contact_types", [])) if evidence.get("contact_types") else "none detected"
        interaction_str = ', '.join(evidence.get("interaction_behavior", [])) if evidence.get("interaction_behavior") else "none observed"

        # 趋势细节
        trend = evidence.get("trend_detail", {})
        trend_str = "insufficient data"
        if trend:
            trend_str = f"{trend['direction']} (first half: {trend['first_half_freq']}/mo → second half: {trend['second_half_freq']}/mo, ×{trend['change_ratio']})"

        # 第三方共现
        co_persons = evidence.get("co_appearing_persons", [])
        co_str = ', '.join([f"{c['person_id']}({c['co_ratio']:.0%})" for c in co_persons[:5]]) if co_persons else "none"

        # 异常
        anomalies = evidence.get("anomalies", [])
        anomaly_str = ""
        if anomalies:
            for a in anomalies[:5]:
                anomaly_str += f"\n- [{a['date']}] {a['type']}: {a['description']}"
        else:
            anomaly_str = "\nNone detected."

        # 事件召回（含 LP1 的 social_dynamics）
        rela_events = evidence.get("rela_events", [])
        events_str = ""
        if rela_events:
            for ev in rela_events[:10]:
                line = f"\n- [{ev.get('date', '')}] {ev.get('title', '')} @ {ev.get('location', '')} ({ev.get('photo_count', 0)} photos)"
                narr = ev.get('narrative_synthesis', '')
                if narr:
                    line += f"\n  Narrative: {narr}"
                for dyn in ev.get('social_dynamics', []):
                    if dyn.get('target_id') == person_id:
                        line += f"\n  Interaction: {dyn.get('interaction_type', '')} | Clue: {dyn.get('social_clue', '')} | Hypothesis: {dyn.get('relation_hypothesis', '')}"
                events_str += line
        else:
            events_str = "\nNo shared events extracted."

        prompt = f"""# 角色
你是一位像真人秀制片人分析素材一样阅读相册的社交关系分析师。你的任务是判断 {person_id} 在 {self.primary_person_id or "the user (photographer)"} 的生活里扮演什么角色。

# 证据
- 共现情况：{evidence['photo_count']} 张照片，跨度 {evidence['time_span']}（月均 {evidence.get('monthly_frequency', 0)}/月）
- 场景：{', '.join(evidence.get('scenes', []))}
- 接触类型：{contact_types_str}
- 互动行为（来自事件层）：{interaction_str}
- 周末频率：{evidence.get('weekend_frequency', 'unknown')}
- 是否主要是一对一：{'是' if evidence.get('with_user_only', True) else '否'}
- 第三方共现人物：{co_str}
- 亲密度分数：{intimacy_score:.3f}
- 趋势：{trend_str}

**共同事件**：{events_str}

**异常信号**：{anomaly_str}

# 软提示（不是硬规则）
- 代码给出的候选关系提示：{', '.join(candidate_types) if candidate_types else 'acquaintance'}
- 代码给出的关系状态建议：{code_status_suggestion or 'stable'}
- 以上只是建议，不是硬性规则；如果证据强烈支持另一种低风险解释，你可以覆盖代码提示。

# 关系类型（必须且只能选一个）
- **family** — 家人或法定家庭成员。重点看：年龄差 >15 岁、家庭/节日场景。
- **romantic** — 明确的恋爱对象。重点看：亲吻/拥抱、高频一对一、私密场景、intimacy_score >0.6。
- **bestie** — 死党/最铁好友。重点看：高频、多场景、自拍、周末和夜间同框多。
- **close_friend** — 稳定且较亲近的朋友。重点看：中等频率、多个轻松休闲场景。
- **friend** — 普通朋友，偶尔一起玩。重点看：频率较低、场景有限。
- **classmate_colleague** — 同学或同事。重点看：校园/办公室/教室、工作日白天。
- **activity_buddy** — 某类固定活动搭子（如健身、早午餐、徒步搭子）。重点看：单一主导的非工作/非学校场景、重复模式明显。
- **acquaintance** — 泛泛之交。重点看：频率很低、基本只出现在群体合照里。

# 关系状态（必须且只能选一个）
- **new** — 认识时间 <30 天
- **growing** — 最近频率/亲密度明显上升
- **stable** — 长期稳定存在
- **fading** — 频率/亲密度在下降
- **gone** — 超过 60 天没有再出现

# 任务
1. 选择 relationship_type（8 选 1）。
2. 选择 status（5 选 1），并结合趋势数据判断。
3. 给出 confidence 和 reason。

对 romantic / family 必须非常保守。如果你选择了候选提示之外的关系类型，必须在 reason 里说明为什么覆盖代码建议。

输出 JSON：
{{
  "relationship_type": "bestie",
  "status": "stable",
  "confidence": 0.85,
  "reason": "高频共现（15 张，3 次/月）、场景多样、主要集中在周末、经常一对一出现、亲密度 0.72，趋势平稳"
}}"""

        return prompt

    def _normalize_confidence(self, raw_value, default: float = 0.5) -> float:
        """把 LLM confidence 归一到 [0, 1]。"""
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = float(default)
        return max(0.0, min(value, 1.0))

    def generate_profile(
        self,
        events: List[Event],
        relationships: List[Relationship],
        vlm_results: List[Dict] | None = None,
        face_db: Dict | None = None,
    ) -> dict:
        """
        生成用户画像（两步：结构化 JSON + Markdown 报告）

        Returns:
            {"structured": dict, "report": str}
        """
        # 构建事件和关系的上下文（两步共用）
        events_str = self._build_events_context(events)
        relationships_str = self._build_relationships_context(relationships)
        face_context = self._build_profile_face_context(face_db or {})
        raw_vlm_context = self._build_profile_vlm_context(vlm_results or [])

        # Step 1: 结构化画像 JSON
        print("    [Step 1/2] 生成结构化画像标签...", flush=True)
        structured_profile = self._generate_structured_profile(
            events_str,
            relationships_str,
            face_context=face_context,
            raw_vlm_context=raw_vlm_context,
        )

        # Step 2: Markdown 画像报告（引用 Step 1 的结构化结果）
        print("    [Step 2/2] 生成画像分析报告...", flush=True)
        report_payload = self._generate_profile_report(events_str, relationships_str, structured_profile)
        consistency_report = build_consistency_report(events, relationships, structured_profile)

        return {
            "structured": structured_profile,
            "report": report_payload.get("public_report", ""),
            "debug": {
                "report_reasoning": report_payload.get("reasoning_trace", {}),
            },
            "consistency": consistency_report,
        }

    def _generate_structured_profile(
        self,
        events_str: str,
        relationships_str: str,
        face_context: str = "",
        raw_vlm_context: str = "",
    ) -> dict:
        """Call 1: 生成结构化画像 JSON"""
        prompt = self._create_structured_profile_prompt(
            events_str,
            relationships_str,
            face_context=face_context,
            raw_vlm_context=raw_vlm_context,
        )

        try:
            result = self._call_llm_via_official_api(prompt, response_mime_type="application/json")
            if isinstance(result, dict) and "text" in result:
                extracted = self._extract_json_from_text(result["text"], "long_term_facts")
                if extracted:
                    return extracted
            if isinstance(result, dict):
                return result
            print("警告：结构化画像 JSON 解析失败，返回空结构")
            return {}
        except Exception as e:
            print(f"警告：结构化画像生成失败: {e}")
            return {}

    def _generate_profile_report(self, events_str: str, relationships_str: str, structured_profile: dict) -> dict:
        """Call 2: 生成 public_report + reasoning_trace"""
        prompt = self._create_profile_report_prompt(events_str, relationships_str, structured_profile)

        try:
            result = self._call_llm_via_official_api(prompt, response_mime_type="application/json")
            if isinstance(result, dict) and "text" in result:
                extracted = self._extract_json_from_text(result["text"], "public_report")
                if extracted:
                    return {
                        "public_report": extracted.get("public_report", ""),
                        "reasoning_trace": extracted.get("reasoning_trace", {}),
                    }
            if isinstance(result, dict):
                return {
                    "public_report": result.get("public_report", ""),
                    "reasoning_trace": result.get("reasoning_trace", {}),
                }
            return {"public_report": result or "", "reasoning_trace": {}}
        except Exception as e:
            print(f"警告：画像报告生成失败: {e}")
            return {
                "public_report": f"# 用户全维画像分析报告\n\n## 生成失败\n\n{e}",
                "reasoning_trace": {"error": str(e)},
            }

    def _build_events_context(self, events: List[Event]) -> str:
        """构建事件上下文（Call 1 和 Call 2 共用）"""
        events_str = ""
        for event in events:
            events_str += f"""
### {event.event_id}: {event.title}
- 时间: {event.date} {event.time_range}（{event.duration}）| 类型: {event.type} | 地点: {event.location}
- 参与者: {', '.join(event.participants) if event.participants else '无'} | 照片数: {event.photo_count}
- 描述: {event.description}
- 叙事: {event.narrative_synthesis if event.narrative_synthesis else '无'}
- 标签: {', '.join(event.tags) if event.tags else '无'}
- 画像证据: 行为[{', '.join(event.persona_evidence.get('behavioral', []))}] 审美[{', '.join(event.persona_evidence.get('aesthetic', []))}] 社经[{', '.join(event.persona_evidence.get('socioeconomic', []))}]
"""
            for dyn in event.social_dynamics:
                events_str += f"  - {dyn.get('target_id', '?')}: {dyn.get('interaction_type', '')} | {dyn.get('relation_hypothesis', '')} ({dyn.get('confidence', 0):.0%})\n"
        return events_str

    def _build_relationships_context(self, relationships: List[Relationship]) -> str:
        """构建关系上下文（Call 1 和 Call 2 共用）"""
        relationships_str = ""
        for rel in relationships:
            ev = rel.evidence
            co_persons = ev.get('co_appearing_persons', [])
            co_str = ', '.join([f"{c['person_id']}({c['co_ratio']:.0%})" for c in co_persons[:3]]) if co_persons else "无"
            trend = ev.get('trend_detail', {})
            trend_str = f"{trend.get('direction', 'N/A')}(×{trend.get('change_ratio', 'N/A')})" if trend else "数据不足"
            anomalies = ev.get('anomalies', [])
            anomaly_str = '; '.join([f"{a['type']}@{a['date']}" for a in anomalies[:3]]) if anomalies else "无"
            shared_ev_str = '; '.join([f"{se.get('date','')}: {se.get('narrative','')}" for se in rel.shared_events[:5]]) if rel.shared_events else "无"
            relationships_str += f"""
### {rel.person_id} — {rel.relationship_type} (intimacy: {rel.intimacy_score:.2f}, status: {rel.status})
- 月均: {ev.get('monthly_frequency', 0)}/月 | 趋势: {trend_str} | 置信度: {rel.confidence:.0%}
- 共现: {ev.get('photo_count', 0)}次, 跨度{ev.get('time_span', '?')} | 场景: {', '.join(ev.get('scenes', []))}
- 互动: {', '.join(ev.get('interaction_behavior', []))} | 第三方: {co_str} | 异常: {anomaly_str}
- 共同事件: {shared_ev_str}
- 推理: {rel.reasoning}
"""
        return relationships_str

    def _build_profile_face_context(self, face_db: Dict | None) -> str:
        """为 LP3 构建紧凑的人脸识别摘要。"""
        subject = getattr(self, "primary_person_id", None) or "the user (photographer)"
        if not face_db:
            return f"""## 人脸识别结果摘要
- 当前分析对象（主角）: {subject}
- 人脸库为空，暂无可用的人脸统计
"""

        def _photo_count(item: Dict | object) -> int:
            if isinstance(item, dict):
                return int(item.get("photo_count", 0) or 0)
            return int(getattr(item, "photo_count", 0) or 0)

        def _avg_confidence(item: Dict | object) -> float:
            if isinstance(item, dict):
                return float(item.get("avg_confidence", 0.0) or 0.0)
            return float(getattr(item, "avg_confidence", 0.0) or 0.0)

        ranked_people = sorted(face_db.items(), key=lambda kv: _photo_count(kv[1]), reverse=True)
        lines = [
            "## 人脸识别结果摘要",
            f"- 当前分析对象（主角）: {subject}",
            f"- 人脸库总人数: {len(face_db)}",
        ]

        if self.primary_person_id and self.primary_person_id in face_db:
            primary = face_db[self.primary_person_id]
            lines.append(
                f"- 主角出现: {_photo_count(primary)} 张 | 平均置信度: {_avg_confidence(primary):.0%}"
            )
        else:
            lines.append("- 当前未稳定识别出主角 person_id，统一按拍摄者视角解释")

        lines.append("- 主要人物出场统计（Top 8）:")
        for person_id, person_info in ranked_people[:8]:
            lines.append(
                f"  - {person_id}: {_photo_count(person_info)} 张 | 平均置信度 {_avg_confidence(person_info):.0%}"
            )
        return "\n".join(lines)

    def _build_profile_vlm_context(self, vlm_results: List[Dict], limit: int = 12) -> str:
        """为 LP3 构建原始 VLM 证据摘录，避免整包 prompt 膨胀。"""
        if not vlm_results:
            return "## 原始VLM数据摘录\n- 无可用 VLM 数据"

        lines = ["## 原始VLM数据摘录"]
        sorted_results = sorted(vlm_results, key=lambda item: item.get("timestamp", ""))
        for item in sorted_results[:limit]:
            analysis = item.get("vlm_analysis", {}) or {}
            scene = analysis.get("scene", {}) if isinstance(analysis.get("scene"), dict) else {}
            event = analysis.get("event", {}) if isinstance(analysis.get("event"), dict) else {}
            people = analysis.get("people", []) if isinstance(analysis.get("people"), list) else []
            people_ids = ", ".join(
                p.get("person_id", "?") for p in people[:4] if isinstance(p, dict)
            ) or "无"
            lines.append(
                f"- {item.get('photo_id', 'UNKNOWN')} | {item.get('timestamp', '未知时间')} | "
                f"地点: {scene.get('location_detected', '未知')} | 活动: {event.get('activity', '未知')} | "
                f"社交: {event.get('social_context', '未知')} | 人物: {people_ids} | "
                f"摘要: {analysis.get('summary', '无')}"
            )
        if len(sorted_results) > limit:
            lines.append(f"- 仅保留前 {limit} 条原始 VLM 摘录以控制 prompt 长度")
        return "\n".join(lines)

    def _create_structured_profile_prompt(
        self,
        events_str: str,
        relationships_str: str,
        face_context: str = "",
        raw_vlm_context: str = "",
    ) -> str:
        """Call 1: 结构化画像 JSON prompt"""
        subject = getattr(self, "primary_person_id", None) or "the user (photographer)"
        return f"""# Role
你是一位世界级的行为分析专家和 FBI 级别的人格画像师，擅长通过"行为残迹"还原人类灵魂，注重每个画像标签推理的逻辑。

# Task
基于人脸识别结果、原始VLM数据和原始事件数据，产出结构化的画像，并结合关系推断结果与代码预计算特征完成每个标签的判断。
- 当前分析对象（主角）: {subject}
- 对每个标签都输出 `value + confidence(0-1)`。
- 无法从数据稳定推断的标签输出 `null`。
- 低于 0.4 的 confidence 代表高度不确定，但仍需给出你最接近证据的猜测。
- 标记为 [SOCIAL_MEDIA_REQUIRED] 的标签必须始终输出 null。
- 必须尊重代码约束信号与硬约束；如果代码约束信号已明确证据不足，优先输出 null。
- 每个字段都先看人脸/VLM原始证据，再看事件与关系抽象结果，避免只复述结构化标签名而忽视底层证据。
- 只推断主角，不要把照片里其他人的外貌特征误投到主角身上。特别是主角缺失时，不能用画面里别人去推 `gender / age_range / race / nationality`。
- 长期标签必须依赖跨事件重复证据；单张照片线索最多影响短期标签，或作为低置信度猜测。
- 当 events / relationships / raw VLM 冲突时，长期事实优先信“跨事件模式”和 LP2 最终关系，不要被单张图异常带偏。

# 输出 JSON Schema

所有**非 null 的可推断标签**都必须使用同一个 Tag Object 结构：

{{
  "value": "标签值或 null",
  "confidence": 0.0,
  "evidence": {{
    "events": [
      {{
        "event_id": "EVT_001",
        "signal": "这条事件里最关键的支持线索",
        "why": "它为什么支持当前标签"
      }}
    ],
    "relationships": [
      {{
        "person_id": "Person_002",
        "relationship_type": "close_friend",
        "signal": "这段关系提供的支持线索",
        "why": "它为什么支持当前标签"
      }}
    ],
    "vlm_observations": [
      {{
        "photo_id": "PHOTO_001",
        "signal": "原始 VLM 里看到的关键视觉线索",
        "why": "它为什么支持当前标签"
      }}
    ],
        "feature_refs": [
      {{
        "feature": "career_evidence_count",
        "value": 2,
        "why": "这条代码约束信号如何影响该标签"
      }}
    ],
    "constraint_notes": [
      "如字段被代码清空或覆盖，在这里写明原因"
    ],
    "summary": "用一句话说明这条标签主要基于哪些证据得出"
  }}
}}

规则：
- `events[].event_id` 只能引用上面的 `EVT_###`
- `relationships[].person_id` 只能引用 LP2 中实际存在的 `Person_###`
- `vlm_observations[].photo_id` 只能引用原始 VLM 摘录里出现过的 `photo_id`
- `feature_refs` 用来记录代码约束信号如何影响该标签
- `constraint_notes` 用来记录字段被代码清空、覆盖或降级的原因
- 如果没有直接证据，对应数组留空，并在 `summary` 里说明为什么仍给出该判断或为什么保持空值

{{
  "long_term_facts": {{
    "identity": {{
      "name": {{"value": "string or null", "confidence": 0.0}},
      "gender": {{"value": "male/female/non_binary or null", "confidence": 0.0}},
      "age_range": {{"value": "e.g. 20-24", "confidence": 0.0}},
      "role": {{"value": "student/worker/freelancer or null", "confidence": 0.0}},
      "race": {{"value": "string or null", "confidence": 0.0}},
      "nationality": {{"value": "string or null", "confidence": 0.0}}
    }},
    "social_identity": {{
      "education": {{"value": "string or null", "confidence": 0.0}},
      "career": {{"value": "string or null", "confidence": 0.0}},
      "career_phase": {{"value": "ascending/stable/declining or null", "confidence": 0.0}},
      "professional_dedication": {{"value": "high/medium/low or null", "confidence": 0.0}},
      "language_culture": {{"value": "string or null", "confidence": 0.0}},
      "political_preference": null
    }},
    "material": {{
      "asset_level": {{"value": "high/middle/struggling or null", "confidence": 0.0}},
      "spending_style": {{"value": "value/experience/luxury or null", "confidence": 0.0}},
      "brand_preference": {{"value": ["brand1"], "confidence": 0.0}},
      "income_model": {{"value": "salary/freelance/business/family or null", "confidence": 0.0}},
      "signature_items": {{"value": ["item1"], "confidence": 0.0}}
    }},
    "geography": {{
      "location_anchors": {{"value": ["place1", "place2"], "confidence": 0.0}},
      "mobility_pattern": {{"value": "string or null", "confidence": 0.0}},
      "cross_border": {{"value": true/false, "confidence": 0.0}}
    }},
    "time": {{
      "life_rhythm": {{"value": "string or null", "confidence": 0.0}},
      "event_cycles": {{"value": ["pattern1"], "confidence": 0.0}},
      "sleep_pattern": {{"value": "early_bird/night_owl/irregular or null", "confidence": 0.0}}
    }},
    "relationships": {{
      "intimate_partner": {{"value": "person_id or null", "confidence": 0.0}},
      "close_circle_size": {{"value": 0, "confidence": 0.0}},
      "social_groups": {{"value": ["group1"], "confidence": 0.0}},
      "pets": {{"value": "string or null", "confidence": 0.0}},
      "parenting": {{"value": "no_children/has_children or null", "confidence": 0.0}},
      "living_situation": {{"value": "alone/shared/with_family or null", "confidence": 0.0}}
    }},
    "hobbies": {{
      "interests": {{"value": [{{"name": "hobby", "dedication": "high/medium/low"}}], "confidence": 0.0}},
      "frequent_activities": {{"value": ["activity1"], "confidence": 0.0}},
      "solo_vs_social": {{"value": "mostly_solo/mostly_social/balanced or null", "confidence": 0.0}}
    }},
    "physiology": {{
      "fitness_level": {{"value": "active/moderate/sedentary or null", "confidence": 0.0}},
      "health_conditions": null,
      "diet_mode": {{"value": "string or null", "confidence": 0.0}}
    }}
  }},
  "short_term_facts": {{
    "life_events": {{"value": "string or null", "confidence": 0.0}},
    "phase_change": {{"value": "string or null", "confidence": 0.0}},
    "spending_shift": {{"value": "string or null", "confidence": 0.0}},
    "current_displacement": {{"value": "at_home/traveling/relocated or null", "confidence": 0.0}},
    "recent_habits": {{"value": ["habit1"], "confidence": 0.0}},
    "recent_interests": {{"value": "string or null", "confidence": 0.0}},
    "physiological_state": null
  }},
  "long_term_expression": {{
    "personality_mbti": {{"value": "XXXX or null", "confidence": 0.0}},
    "morality": {{"value": "string or null", "confidence": 0.0}},
    "philosophy": {{"value": "string or null", "confidence": 0.0}},
    "attitude_style": {{"value": "refined/relaxed or null", "confidence": 0.0}},
    "aesthetic_tendency": {{"value": "minimalist/retro/modern/classical or null", "confidence": 0.0}},
    "visual_creation_style": {{"value": "string or null", "confidence": 0.0}},
    "intimacy_display": {{"value": "low_key/public or null", "confidence": 0.0}},
    "boundary_sense": {{"value": "strong/open or null", "confidence": 0.0}},
    "solo_vs_group_photos": {{"value": "solo_dominant/group_dominant/balanced or null", "confidence": 0.0}},
    "topic_blocks": null,
    "persona_blocks": null,
    "humor_type": null,
    "vocabulary_style": null,
    "symbol_habits": null,
    "narrative_length": null
  }},
  "short_term_expression": {{
    "current_mood": {{"value": "down/fired_up/neutral or null", "confidence": 0.0}},
    "mood_trend": {{"value": "rising/stable/falling or null", "confidence": 0.0}},
    "social_fatigue": {{"value": "high_energy/low_energy or null", "confidence": 0.0}},
    "temporary_persona": null,
    "expression_threshold": null,
    "recent_recurring_topics": null
  }}
}}

# 数据输入

{face_context or "## 人脸识别结果摘要\n- 无"}

{raw_vlm_context or "## 原始VLM数据摘录\n- 无"}

## 原始事件数据
{events_str}

## 关系推断结果
{relationships_str}

只输出 JSON，不要附加解释。"""

    def _create_profile_report_prompt(self, events_str: str, relationships_str: str, structured_profile: dict) -> str:
        """Call 2: public_report + reasoning_trace JSON prompt"""
        import json
        profile_json_str = json.dumps(structured_profile, ensure_ascii=False, indent=2) if structured_profile else "{}"
        subject = getattr(self, "primary_person_id", None) or "the user (photographer)"

        return f"""# Role
你是一位世界级的行为分析专家和 FBI 级别的人格画像师，擅长通过"行为残迹"还原人类灵魂。

# Task
基于已生成的**结构化画像标签**、原始事件数据和关系数据，生成两个层次的输出：
1. `reasoning_trace`：内部调试用，存储推理依据和过程。
2. `public_report`：最终对外展示的《用户全维画像分析报告》。

- 当前分析对象（主角）: {subject}
- 推理时必须引用结构化标签作为证据（如：根据 `identity.role: student (0.7)` + 事件 EVT_003 推断...）。
- 把推理依据和过程写入 reasoning_trace，明确记录标签证据、事件证据、关系证据和章节结论。
- 最终对外输出的 public_report 只能保留逻辑连贯、用词凝练的文字结论，不要出现结构化字段名、置信度、event_id、证据列表、表格或推理草稿箱。
- public_report 可以使用自然语言小标题，但每一段都应是人类可直接阅读的结论，而不是证据罗列。

# Reasoning Process (Chain of Thought)

## Step 1: 行为基频分析
- 确定用户的"正常节律"：作息时间、主要活动地点、工作/学习模式
- 偏离常规的行为是偶然的还是结构性的？

## Step 2: 线索交叉印证
- 空间+消费：消费场景与经济水平是否一致？
- 时间+社交：社交活动的时间分布说明什么？
- 地点+人物：特定人物是否只在特定场景出现？

## Step 3: 异常值探测
- 频率突变、关系切换、场景变化等"剧变时刻"
- 对应人生事件推测

## Step 4: 矛盾仲裁
- 证据冲突的深层解读（选择性奢侈、场景化消费等）

---

# Output JSON Schema

{{
  "public_report": "# 用户全维画像分析报告\\n\\n## 事实身份与社会角色\\n用自然语言输出结论 ...",
  "reasoning_trace": {{
    "tag_evidence": [
      "long_term_facts.identity.role: student (0.7)",
      "long_term_facts.relationships.intimate_partner: Person_002 (0.82)"
    ],
    "event_evidence": [
      "EVT_003: 连续多周末与同一人物在校园外休闲场景共现"
    ],
    "relationship_evidence": [
      "Person_002: romantic / stable / intimacy 0.84"
    ],
    "reasoning_steps": [
      "先根据结构化标签建立身份、节奏和社交主轴，再回到事件中验证时间演进。"
    ],
    "section_support": [
      {{
        "section": "事实身份与社会角色",
        "basis": "identity.role + 校园事件",
        "conclusion": "自然语言总结"
      }}
    ]
  }}
}}

# Constraint & Rules
1. **证据闭环**：reasoning_trace 中的每条结论都必须能回指结构化标签或具体事件/关系。
2. **动态演进**：如数据跨度较长，指出时间维度上的变化，并把时间变化写入 reasoning_trace。
3. **留白原则**：低置信度标签在 public_report 中用"高概率、疑似、待观察"等自然语言处理，不直接输出分数。
4. **对外洁净**：public_report 不得暴露结构化字段名、confidence、event_id、证据条目或推理过程。

---

# 结构化画像标签（来自 Step 1）

```json
{profile_json_str}
```

# 用户事件数据

{events_str}

# 用户社交关系数据

{relationships_str}

请按上述 JSON Schema 输出结果，不要输出额外说明。"""

    def _call_llm_via_official_api(self, prompt: str, response_mime_type: str = None, model_override: str = None) -> dict:
        """通过 HTTP 请求调用代理服务的 Gemini API

        Args:
            prompt: 提示词
            response_mime_type: 强制返回格式（如 "application/json"）
            model_override: 覆盖默认模型（如事件提取用 gemini-2.0-flash）
        """
        self._require_requests()
        model = model_override or self.model

        # 构建请求体
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }

        # 强制 JSON 输出
        if response_mime_type:
            payload["generationConfig"] = {
                "responseMimeType": response_mime_type
            }

        # 构建请求 URL
        url = f"{self.base_url}/{model}:generateContent"
        max_retries = 3
        timeout = 300  # LLM处理也增加到300秒

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=self.headers,
                    timeout=timeout
                )

                if response.status_code == 200:
                    response_data = response.json()

                    # 提取文本内容
                    if "candidates" in response_data and len(response_data["candidates"]) > 0:
                        candidate = response_data["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            for part in candidate["content"]["parts"]:
                                if "text" in part:
                                    try:
                                        # 尝试解析为 JSON
                                        result = json.loads(part["text"])
                                        return result
                                    except json.JSONDecodeError:
                                        # 如果是 Markdown JSON，尝试提取
                                        text = part["text"]
                                        import re
                                        json_match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
                                        if json_match:
                                            json_str = json_match.group(1)
                                            try:
                                                result = json.loads(json_str)
                                                return result
                                            except:
                                                pass
                                        # 返回原始文本
                                        return {"text": text}

                    return None
                else:
                    error_msg = f"代理 API 返回状态码 {response.status_code}"
                    if response.text:
                        try:
                            error_data = response.json()
                            if isinstance(error_data, dict) and "error" in error_data:
                                error_msg += f": {error_data['error']}"
                        except:
                            error_msg += f": {response.text[:200]}"
                    raise Exception(error_msg)

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                # 超时或连接错误，进行重试
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避：2s, 4s
                    print(f"      [LLM重试 {attempt + 1}/{max_retries}] 等待 {wait_time}s 后重试...", flush=True)
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"LLM调用重试{max_retries}次仍失败: {e}")

    def _call_profile_via_official_api(self, prompt: str) -> str:
        """通过 HTTP 请求调用代理服务生成用户画像"""
        self._require_requests()
        # 构建请求体
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }

        # 构建请求 URL
        url = f"{self.base_url}/{self.model}:generateContent"
        timeout = 300  # 增加到300秒

        response = requests.post(
            url,
            json=payload,
            headers=self.headers,
            timeout=timeout
        )

        if response.status_code == 200:
            response_data = response.json()

            # 提取文本内容
            if "candidates" in response_data and len(response_data["candidates"]) > 0:
                candidate = response_data["candidates"][0]

                # 检查是否有content字段
                if "content" in candidate:
                    content = candidate["content"]

                    # 处理标准格式（parts中有text）
                    if "parts" in content:
                        for part in content["parts"]:
                            if "text" in part:
                                return part["text"]

                    # 处理思维链格式（只有role，no parts）
                    # 如果没有text，尝试从thinking字段提取
                    if "thinking" in content:
                        # 思维链模式，返回思维链内容作为报告
                        return f"# 用户全维画像分析报告\n\n## 思维过程\n\n{content.get('thinking', '分析中...')}"

            # 如果都没有找到，返回错误消息
            return ""
        else:
            error_msg = f"代理 API 返回状态码 {response.status_code}"
            if response.text:
                try:
                    error_data = response.json()
                    if isinstance(error_data, dict) and "error" in error_data:
                        error_msg += f": {error_data['error']}"
                except:
                    error_msg += f": {response.text[:200]}"
            raise Exception(error_msg)
