"""
LLM处理模块：事件提取、关系推断、画像生成
"""
import json
from datetime import datetime
from typing import List, Dict, Optional
from google import genai
from models import Event, Relationship, UserProfile
from config import GEMINI_API_KEY, LLM_MODEL, MIN_PHOTO_COUNT, MIN_TIME_SPAN_DAYS, MIN_SCENE_VARIETY
from utils import calculate_distance, time_overlap, is_weekend


class LLMProcessor:
    """LLM处理器"""

    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = LLM_MODEL

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
        用LLM从全部照片中提取事件

        Args:
            vlm_results: 全部VLM分析结果

        Returns:
            事件列表
        """
        prompt = self._create_event_extraction_prompt(vlm_results)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )

            result = json.loads(response.text)
            events = []

            for i, event_data in enumerate(result.get("events", [])):
                # 兼容新旧格式：从meta_info提取或直接使用
                meta_info = event_data.get("meta_info", {})
                objective_fact = event_data.get("objective_fact", {})

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
                    event_id=event_data.get("event_id", f"EVT_{i + 1:03d}"),
                    date=date_part,
                    time_range=time_range,
                    duration=event_data.get("duration", ""),  # 新格式可能没有
                    title=meta_info.get("title") or event_data.get("title", ""),
                    type=event_data.get("type", "其他"),  # 新格式可能没有
                    participants=objective_fact.get("participants") or event_data.get("participants", []),
                    location=meta_info.get("location_context") or event_data.get("location", ""),
                    description=objective_fact.get("scene_description") or event_data.get("description", ""),
                    photo_count=int(meta_info.get("photo_count") or event_data.get("photo_count", 0)),
                    confidence=event_data.get("confidence", 0.5),  # 新格式可能没有
                    reason=event_data.get("reason", ""),  # 新格式可能没有
                    # 核心叙事
                    narrative="",  # 新格式没有narrative字段
                    narrative_synthesis=event_data.get("narrative_synthesis", ""),
                    # 新增 v2.1 字段
                    meta_info=meta_info,
                    objective_fact=objective_fact,
                    # 社交分析
                    social_interaction={},  # 新格式没有social_interaction
                    social_dynamics=event_data.get("social_dynamics", []),
                    # 证据（新格式没有evidence_photos）
                    evidence_photos=[],
                    # 标签
                    lifestyle_tags=[],  # 新格式没有lifestyle_tags
                    tags=event_data.get("tags", []),
                    # 画像证据（新格式没有social_slices）
                    social_slices=[],
                    persona_evidence=event_data.get("persona_evidence", {})
                )
                events.append(event)

            return events

        except Exception as e:
            print(f"警告：事件提取失败: {e}")
            return []

    def _create_event_extraction_prompt(self, vlm_results: List[Dict]) -> str:
        """创建事件提取prompt（适配新的简化VLM格式）"""
        photos_info = []

        for item in vlm_results:
            # 获取地点：使用EXIF GPS，VLM现在不提供location_detected
            exif_location = item['location'].get('name', '未知') if item.get('location') else '未知'
            location = exif_location

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

            # 获取场景信息（VLM新格式：scene是字符串）
            scene_str = item['vlm_analysis'].get('scene', '')
            event_info = item['vlm_analysis'].get('event', {})

            # 兼容event可能是字符串的情况
            if isinstance(event_info, str):
                event_activity = event_info
                event_social = ''
                event_mood = ''
            else:
                event_activity = event_info.get('activity', '')
                event_social = event_info.get('social_context', '')
                event_mood = event_info.get('mood', '')

            # 获取时间信息
            time_info = item['vlm_analysis'].get('Time', {})
            if isinstance(time_info, dict):
                date_note = time_info.get('date', '')
                time_note = time_info.get('time', '')
            else:
                date_note = ''
                time_note = ''

            # 获取细节
            details = item['vlm_analysis'].get('details', '')
            if isinstance(details, list):
                details = ', '.join(details)
            elif isinstance(details, dict):
                details = str(details)

            # 获取人物ID列表（从VLM分析结果中）
            vlm_people = item['vlm_analysis'].get('people', [])
            person_ids = [p.get('person_id', '') for p in vlm_people]

            info = f"""
【照片 {item['photo_id']}】
时间: {item['timestamp']}
{f"时间背景: {date_note}, {time_note}" if date_note or time_note else ""}
地点: {location}
人物ID: {', '.join(person_ids) if person_ids else '无'}
人物详情:{people_str if people_str else " 无"}
VLM描述: {item['vlm_analysis'].get('summary', '')}
场景描述: {scene_str}
活动: {event_activity}
社交背景: {event_social}
氛围: {event_mood}
细节: {details}
---
"""
            photos_info.append(info)

        # 提取日期用于背景推理
        dates = [item['timestamp'][:10] for item in vlm_results]
        date_info = f"照片拍摄时间范围：{dates[0]} 至 {dates[-1]}，共{len(set(dates))}天"

        prompt = f"""Role: 你是一位资深的人类学专家与社会学行为分析师，擅长从破碎的相册VLM元数据中，通过"时空-行为-关系"三维建模，将零散照片还原为逻辑连贯的"原子事件"，并为后续的用户人格画像（Who/Whom/What）提供高价值的推导证据。

Task:
1. 原子事件聚类：分析用户相册的VLM（视觉语言模型）原始数据（包含时间戳、地点、人物ID、场景描述），将零散的照片聚类为逻辑连贯的"客观事件"，并识别核心社交关系。
2. 微观线索提取：从物象、构图、互动中提取能反映用户身份、社交与性格的"切片"。
3. 特征向量标记：为每个事件打上维度标签，便于全局画像合成。

Core Principles (核心原则):
- 视角法则（关键）：
  - 判定他拍：若用户（Person_0）出现在中远景、双手未持拍摄设备且姿态自然，必须判定为"他拍"。这意味着现场存在"隐形成员（Invisible Photographer）"，需将其纳入社交动态分析。
  - 判定自拍：若画面出现手臂延伸、持手机姿势或镜面反射。
  - 身份确认：截图或拍摄证件中的人物信息通常是用户本人。
- 证据溯源：所有推断必须基于事实（如：通过Chanel包装推测经济能力，通过凌晨时段推测生活节奏）。

Analysis Logic (分析逻辑):

1. 时空聚类（Event Grouping）：
   - 若相邻照片时间差在4小时内，且地点（location）相同或场景（scene）逻辑连贯（如：从商场到商场餐厅），应归为一个"事件"
   - 若时间跨度大或地点发生显著位移，应开启新事件

2. 地点的逻辑推断：
   - 即使坐标缺失，也要通过环境要素判定场景属性：
   - 睡衣/厨具/床品 → 私密居家
   - 显示器/工牌/会议室 → 职业办公
   - 摩托车/街道/商场 → 都市户外
   - 餐具/菜单/餐桌 → 餐饮场所

3. 社交建模（Social Mapping）：
   - 识别和统计每个事件中出现的person_id
   - 识别"核心同伴"：行为亲密、有互动，或在多个连续事件中高频出现的person_id（person_0是主角）
   - 识别"环境人物"：仅在特定公共场所出现一次且无互动的person_id

4. 微观线索（Micro-Clues）：
   - 身份线索：工牌、专业设备、特定的消费小票、屏幕内容、APP界面
   - 社交线索：拍摄者与被拍者的距离（亲密度）、身体语言（朝向/互动）、合影频率、出现场所（家/公司）
   - 性格线索：构图的秩序感、拍摄内容的重复性（如反复拍猫或拍路标）、截图的类型（工作/生活/审美）
   - 审美线索：服装品牌、配色偏好、构图风格、物品选择（如Chanel vs 快时尚）
   - 社会经济线索：品牌档次（奢侈品牌/轻奢/大众）、消费场景（高档餐厅/平价小店）、生活设施

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
        "participants": ["person_0", "包含识别出的隐形拍摄者或背景人物"]
      }},
      "social_dynamics": [
        {{
          "target_id": "人物ID",
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
      "tags": ["#标签1", "#标签2"]
    }}
  ]
}}

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

    def infer_relationships(self, vlm_results: List[Dict], face_db: Dict) -> List[Relationship]:
        """
        推断人物关系

        Args:
            vlm_results: VLM结果列表
            face_db: 人脸库

        Returns:
            关系列表
        """
        relationships = []

        # 遍历所有人脸库中的人物（除了主角）
        for person_id, person_info in face_db.items():
            if person_id == "person_0" or person_info.name == "主角":
                continue

            # 检查是否满足触发条件
            should_infer, reason = self._should_infer_relationship(person_id, person_info, vlm_results)

            if not should_infer:
                print(f"跳过 {person_id}: {reason}")
                continue

            # 收集证据
            evidence = self._collect_relationship_evidence(person_id, vlm_results)

            # 推断关系
            relationship = self._infer_relationship(person_id, evidence)
            relationships.append(relationship)

        return relationships

    def _should_infer_relationship(self, person_id: str, person_info: Dict, vlm_results: List[Dict]) -> tuple:
        """
        判断是否应该推断关系

        Returns:
            (是否应该推断, 原因)
        """
        # 条件1：出现次数
        if person_info.photo_count < MIN_PHOTO_COUNT:
            return False, f"出现次数太少（{person_info.photo_count}次，需要≥{MIN_PHOTO_COUNT}次）"

        # 条件2：时间跨度
        if person_info.first_seen and person_info.last_seen:
            time_span = (person_info.last_seen - person_info.first_seen).days
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

    def _collect_relationship_evidence(self, person_id: str, vlm_results: List[Dict]) -> Dict:
        """收集关系推断的证据"""
        evidence = {
            "photo_count": 0,
            "time_span": "",
            "scenes": [],
            "interaction_behavior": [],  # 互动行为列表
            "weekend_frequency": "",
            "with_user_only": True,
            "sample_scenes": []
        }

        # 找出包含该人物的所有照片
        co_photos = []
        for result in vlm_results:
            # 从VLM分析结果中获取人物列表
            vlm_people = result["vlm_analysis"].get("people", [])
            for person in vlm_people:
                if person.get("person_id") == person_id:
                    co_photos.append(result)
                    break

        evidence["photo_count"] = len(co_photos)

        if not co_photos:
            return evidence

        # 计算时间跨度
        timestamps = [datetime.fromisoformat(r["timestamp"]) for r in co_photos]
        first = min(timestamps)
        last = max(timestamps)
        evidence["time_span"] = f"{(last - first).days + 1}天"

        # 提取场景和互动行为
        for photo in co_photos:
            scene = photo["vlm_analysis"].get("scene", {}).get("location_detected", "")
            if scene and scene not in evidence["scenes"]:
                evidence["scenes"].append(scene)

            # 提取互动行为
            event_data = photo["vlm_analysis"].get("event", {})
            if isinstance(event_data, dict):
                interaction = event_data.get("interaction", "")
            else:
                interaction = str(event_data)
            if interaction and interaction not in evidence["interaction_behavior"]:
                evidence["interaction_behavior"].append(interaction)

            # 保留样本
            if len(evidence["sample_scenes"]) < 5:
                evidence["sample_scenes"].append({
                    "timestamp": photo["timestamp"],
                    "scene": scene,
                    "summary": photo["vlm_analysis"]["summary"],
                    "activity": photo["vlm_analysis"]["event"]["activity"]
                })

        # 计算周末频率
        weekend_count = sum([1 for r in co_photos if is_weekend(datetime.fromisoformat(r["timestamp"]))])
        if weekend_count / len(co_photos) > 0.5:
            evidence["weekend_frequency"] = "高"
        else:
            evidence["weekend_frequency"] = "低"

        # 判断是否只和用户在一起
        for photo in co_photos:
            # 从 VLM 分析结果中获取人物列表
            vlm_people = photo.get("vlm_analysis", {}).get("people", [])
            other_people = [p.get("person_id") for p in vlm_people
                           if p.get("person_id") and p.get("person_id") != person_id and p.get("person_id") != "person_0"]
            if other_people:
                evidence["with_user_only"] = False
                break

        return evidence

    def _infer_relationship(self, person_id: str, evidence: Dict) -> Relationship:
        """用LLM推断关系"""
        prompt = self._create_relationship_prompt(person_id, evidence)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )

            result = json.loads(response.text)

            return Relationship(
                person_id=person_id,
                relationship_type=result.get("relationship_type", "acquaintance"),
                label=result.get("label", "熟人"),
                confidence=result.get("confidence", 0.5),
                evidence=evidence,
                reason=result.get("reason", "")
            )

        except Exception as e:
            print(f"警告：关系推断失败 ({person_id}): {e}")
            return Relationship(
                person_id=person_id,
                relationship_type="acquaintance",
                label="熟人",
                confidence=0.3,
                evidence=evidence,
                reason=f"推断失败: {e}"
            )

    def _create_relationship_prompt(self, person_id: str, evidence: Dict) -> str:
        """创建关系推断prompt"""
        sample_scenes_str = ""
        for scene in evidence["sample_scenes"][:3]:
            sample_scenes_str += f"\n- {scene['timestamp']}: {scene['summary']}\n"

        # 互动行为描述
        interaction_str = ', '.join(evidence.get("interaction_behavior", [])) if evidence.get("interaction_behavior") else "无"

        prompt = f"""请根据以下证据，判断{person_id}和用户（person_0）的关系类型。

**证据**：
- 共同出现次数：{evidence['photo_count']}次
- 时间跨度：{evidence['time_span']}
- 常见场景：{', '.join(evidence['scenes'])}
- 互动行为：{interaction_str}
- 周末出现频率：{evidence['weekend_frequency']}
- 是否只和用户一起：{'是' if evidence['with_user_only'] else '否'}

**部分场景描述**：{sample_scenes_str}

**关系类型定义**：
- **family（家人）**：有血缘关系
- **partner（伴侣）**：亲密关系，经常单独在一起，有亲密互动
- **close_friend（密友）**：频繁出现，场景多样（休闲+日常），关系亲密
- **friend（朋友）**：偶尔一起活动，场景较单一
- **colleague（同事）**：主要在工作场景出现
- **acquaintance（熟人）**：很少出现，关系不深
- **date（约会对象）**：不是1V1的浪漫关系，有明确的约会行为

**任务**：
请判断关系类型，并说明理由。
要保证亲密关系的判断准确，不要误判。

输出JSON：
{{
  "relationship_type": "close_friend",
  "label": "密友",
  "confidence": 0.85,
  "reason": "频繁出现（15次），场景多样（咖啡馆、餐厅、户外），主要在周末，关系亲密"
}}"""

        return prompt

    def generate_profile(self, events: List[Event], relationships: List[Relationship]) -> str:
        """
        生成用户画像（Markdown格式报告）

        Args:
            events: 事件列表
            relationships: 关系列表

        Returns:
            Markdown格式的用户画像报告
        """
        prompt = self._create_profile_prompt(events, relationships)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
                # 不指定 response_mime_type，让LLM自由输出Markdown
            )

            return response.text

        except Exception as e:
            print(f"警告：画像生成失败: {e}")
            # 返回默认报告
            return f"""# 用户全维画像分析报告

## 生成失败

画像生成过程中发生错误：{e}

请检查输入数据或重试。
"""

    def _create_profile_prompt(self, events: List[Event], relationships: List[Relationship]) -> str:
        """创建画像生成prompt - FBI级别人格画像师版本"""
        # 构建事件详情（包含更多上下文）
        events_str = ""
        for event in events:
            # 提取evidence_photos作为证据链接
            evidence_str = ", ".join(event.evidence_photos) if event.evidence_photos else "无"

            events_str += f"""
### Event: {event.title}
- **ID**: {event.event_id}
- **时间**: {event.date} {event.time_range}（{event.duration}）
- **类型**: {event.type}
- **地点**: {event.location}
- **参与者**: {', '.join(event.participants) if event.participants else '无'}
- **描述**: {event.description}
- **叙事**: {event.narrative_synthesis if event.narrative_synthesis else '无'}
- **照片数**: {event.photo_count}张
- **证据照片**: {evidence_str}
- **标签**: {', '.join(event.tags) if event.tags else '无'}
- **画像证据**:
  - 行为特征: {', '.join(event.persona_evidence.get('behavioral', []))}
  - 审美特征: {', '.join(event.persona_evidence.get('aesthetic', []))}
  - 社会经济: {', '.join(event.persona_evidence.get('socioeconomic', []))}
- **社交动态**:
"""

            # 添加社交动态详情
            for dyn in event.social_dynamics:
                events_str += f"  - {dyn.get('target_id', '未知')}: {dyn.get('interaction_type', '')} | {dyn.get('relation_hypothesis', '')} (置信度: {dyn.get('confidence', 0):.0%})\n"

        # 构建关系详情
        relationships_str = ""
        for rel in relationships:
            evidence = rel.evidence
            relationships_str += f"""
### {rel.person_id}
- **推断关系**: {rel.label}
- **关系类型**: {rel.relationship_type}
- **置信度**: {rel.confidence:.0%}
- **共同出现**: {evidence.get('photo_count', 0)}次
- **时间跨度**: {evidence.get('time_span', '未知')}
- **常见场景**: {', '.join(evidence.get('scenes', []))}
- **互动行为**: {', '.join(evidence.get('interaction_behavior', []))}
- **推理依据**: {rel.reason}
"""

        prompt = f"""# Role
你是一位世界级的行为分析专家和 FBI 级别的人格画像师，擅长通过"行为残迹"还原人类灵魂。

# Task
分析提供的围绕主角的用户事件，产出《用户全维画像分析报告》。

# Reasoning Process (Chain of Thought - 必须在输出前按此逻辑分步思考)

## Step 1: 行为基频分析 (Baseline Establishing)
- 扫描日志全貌，确定该用户的"正常节律"：几点起床？几点活跃？主要坐标点 A（家）和 B（公司）在哪？
- [思考]：如果此人的行为偏离了 9-5 规律，这种偏离是偶然的还是结构性的（如：长期 11-23 点在软件园，暗示高薪程序员身份）？

## Step 2: 线索交叉印证 (Cross-Referencing)
- **空间+消费**：如果用户在 A 地（高档写字楼）停留，但购买的是"瑞幸 9.9 优惠券"，说明什么？（是实用主义的中产，还是入不敷出的奋斗者？）
- **时间+社交**：深夜 22:00 后的社交互动是发生在"办公室"还是"酒吧"？（判定是职场协作还是情感依赖。）

## Step 3: 异常值探测 (Anomaly Detection)
- 寻找这一年中的"剧变时刻"：某个月消费突然激增？某周突然换了常驻地？
- [思考]：这些异常点通常对应人生重大事件（跳槽、分手、搬家、暴富）。

## Step 4: 矛盾仲裁 (Conflict Resolution)
- 如果发现证据冲突（例如：既去奢饰品店又买临期食品），不要忽略。
- [思考]：这种矛盾点正是刻画人物"伪装"或"复杂价值观"的关键。

---

# Output Format (严格按此模版输出)

## [推理草稿箱 (Reasoning Scratchpad)]
*在给出报告前，请先简述你的核心推理发现：*
1. **时空锚点确认**：[地点 A 是 XX，地点 B 是 XX，职业初步判定为 XX]
2. **社交实体挖掘**：[发现 Person_X 出现了 N 次，关系特征为 XX]
3. **职业逻辑修正**：[针对工作时间非 9-18 的情况，我的解释逻辑是 XX]

---

## 1. 基础画像 (Base Persona)
- **姓名/称呼推测**：[结论] —— **证据链接**：(证据见 event_xxx)
- **估算年龄/生命周期**：[例如：28-32岁，正处于职业上升期] —— **证据链接**：(证据见 event_xxx)
- **核心身份/阶层**：[描述] —— **逻辑推导**：[基于消费水平与职业地点的综合推论]
- **常驻地/通勤模式**：[描述] —— **证据链接**：(证据见 event_xxx)
- **兴趣爱好**：[描述] —— **证据链接**：(证据见 event_xxx)
- **是否单身/伴侣是谁**：[描述] —— **证据链接**：(证据见 event_xxx)

## 2. 社交关系图谱 (Social Graph)
- **重要关系识别表**：
  | 实体ID/姓名 | 推测关系 | 亲密度 (1-10) | 关键证据 (event_id) | 判定理由 |
  | :--- | :--- | :--- | :--- | :--- |
  | | | | | |
- **社交性格总结**：[是社交能量输出者，还是被动接受者？]

## 3. 深度人格分析 (Psychological Profiling)
- **性格特征与 MBTI**：[结论] —— **心理学逻辑**：[为何从行为 X 推导出性格 Y]
- **价值观与底层驱动力**：[他/她真正在乎什么？]
- **审美偏好和对外展示**：[他/她认为什么是美的？他/她希望自己展示给外界的形象]
- **底层人格侧写**：[用一段感性的、带有文学洞察力的文字描述这个人的本质。]

---

# Constraint & Rules
1. **证据闭环**：严禁出现没有 event_id 支持的形容词。
2. **动态演进**：如果数据跨度为一年，必须指出用户在年初和年末是否有明显的性格或状态变化。
3. **留白原则**：对于模糊信息，使用"高概率、疑似、待进一步观察"等词汇，体现 FBI 专家的严谨。

---

# 用户事件数据

{events_str}

# 用户社交关系数据

{relationships_str}

请开始分析并生成报告："""

        return prompt
