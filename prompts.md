# 记忆工程 - 关键Prompt文档

## 1. 事件提取Prompt

```python
def _create_event_extraction_prompt(self, segment: List[Dict]) -> str:
    """创建事件提取prompt"""
    photos_info = []

    for item in segment:
        # 获取地点：优先使用VLM识别的地点，其次使用EXIF GPS
        vlm_location = item['vlm_analysis']['scene']['location_detected']
        exif_location = item['location'].get('name', '未知') if item.get('location') else '未知'
        location = vlm_location if vlm_location and vlm_location != '未知' else exif_location

        info = f"""
照片 {item['photo_id']}（{item['timestamp']}）：
- 地点：{location}
- VLM描述：{item['vlm_analysis']['summary']}
- 人物：{', '.join([p['person_id'] for p in item['faces']])}
- 活动：{item['vlm_analysis']['macro_event']['activity']}
- 氛围：{', '.join(item['vlm_analysis']['mood'])}
"""
        photos_info.append(info)

    # 提取日期用于背景推理
    dates = [item['timestamp'][:10] for item in segment]
    date_info = f"照片拍摄时间：{', '.join(set(dates))}"

    prompt = f"""以下是用户在某个时间段内拍摄的{len(segment)}张照片及其分析结果：

{date_info}

{''.join(photos_info)}

**重要 - 时间背景推理**：
请特别注意照片的拍摄时间，结合以下背景知识进行推理：
- **1月下旬-2月中旬**：春节期间（春运）
- **2月**：春节返程高峰
- **春节期间+火车** → 极可能是春节返程/春运
- **春节后+外地工作** → 可能是外地来工作（如北漂）

**任务**：
请分析这些照片，判断用户经历了几个**独立的事件**。

**判断标准**：
1. 时间连续性：时间间隔短（<2小时）→ 可能是同一事件
2. 地点一致性：地点相同或相近 → 可能是同一事件
3. 场景一致性：活动类型相同 → 可能是同一事件
4. 场景变化：活动类型明显不同（工作→午餐）→ 可能是不同事件
5. **时间背景**：结合拍摄时间推理事件含义（如2月21日+火车=春节返程）

**对每个事件**，请说明：
1. title：事件标题（**要求：结合时间背景，如"春节返程：火车旅途"而非简单的"火车旅行"**）
2. type：事件类型（社交/工作/休闲/用餐/运动/旅行/其他）
3. time_range：时间范围（如 14:00-15:30）
4. duration：持续时间（如 1.5小时）
5. location：地点
6. participants：参与人物（person_id列表）
7. description：事件描述（**要求：包含时间背景推理，如"春节返程途中"**）
8. photo_count：涉及照片数量
9. confidence：置信度（0-1）
10. reason：为什么认为这是一个独立事件（**要求：说明时间背景推理**）

输出JSON格式：
{{
  "events": [
    {{
      "title": "春节返程：火车旅途",
      "type": "旅行",
      "time_range": "14:00-15:30",
      "duration": "1.5小时",
      "location": "火车上",
      "participants": [],
      "description": "2月21日（春节期间）乘坐火车返程，从老家返回工作城市",
      "photo_count": 3,
      "confidence": 0.85,
      "reason": "时间连续（14:00-15:30），地点一致（火车上），2月21日为春节返程高峰期"
    }}
  ]
}}"""

    return prompt
```

---

## 2. 关系推断Prompt

```python
def _create_relationship_prompt(self, person_id: str, evidence: Dict) -> str:
    """创建关系推断prompt"""
    sample_scenes_str = ""
    for scene in evidence["sample_scenes"][:3]:
        sample_scenes_str += f"\n- {scene['timestamp']}: {scene['summary']}\n"

    # 互动行为描述
    interaction_str = ', '.join(evidence.get("interaction_behavior", [])) if evidence.get("interaction_behavior") else "无"

    prompt = f"""请根据以下证据，判断{person_id}和用户（person_1）的关系类型。

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
```

---

## 3. 用户画像Prompt

```python
def _create_profile_prompt(self, events: List[Event], relationships: List[Relationship]) -> str:
    """创建画像生成prompt"""
    events_str = ""
    for event in events[:15]:  # 最多15个事件
        events_str += f"\n- {event.date} {event.time_range}: {event.title}（{event.type}）\n"

    relationships_str = ""
    for rel in relationships:
        relationships_str += f"\n- {rel.person_id}: {rel.label}（置信度{rel.confidence:.0%}）\n"

    prompt = f"""基于以下记忆事件和人物关系，生成用户画像。

**用户的事件**：
{events_str}

**用户的社交关系**：
{relationships_str}

**重要 - 生活状态推断**：
根据事件的时间特征，推断用户的生活状态：
- **1-2月火车/飞机旅行** → 可能是春节返程/春运
- **春节后立即工作** → 可能是外地来工作（如北漂、沪漂、深漂）
- **春节期间独自在工作城市** → 可能是留守工作
- **独自拍照/工作** → 独立生活或独自打拼

请特别关注：
- 用户是否是"漂"一族（北漂、沪漂、深漂、杭漂等）
- 春节期间的活动模式反映的生活状态
- 节后立即工作反映的工作态度
- 独自活动反映的独立性

**任务**：
请生成用户画像，包含：

1. **basic_info**：
   - age_range：年龄段（从照片推断）
   - gender：性别（从照片推断）
   - location：常驻地（可能的工作城市）
   - hometown：家乡（如果能推断）
   - company：公司（如果能推断）
   - job：职业（如果能推断）
   - single：是否单身
   - wealth_status：财富状况（如果能推断）
   - life_stage：生活阶段（如"独自在大城市打拼的年轻人"）

2. **lifestyle**：
   - daily_routine：日常作息描述
   - social_frequency：社交频率
   - living_situation：生活状况（如"独自租房"）

3. **place**：
   - home_vicinity：家庭地址推断列表（基于常在家里的GPS定位，如"上海安福路30号"或"北京某小区"）
   - work_hub：典型办公地点推断列表（如"核心CBD A座写字楼"或"北辰大厦"）
   - third_space：常去的其他地点列表（如"某家Jazz Bar"、"xx复古黑胶店"、"小美普拉提馆"）

4. **personality**：
   - traits：性格特质（3-5个）
   - social_style：社交风格
   - work_style：工作风格
   - mbti：MBTI 4位码推断及理由（如["INTJ", "理由xxx"]）

5. **interests**：
   - categories：兴趣类别列表
   - weights：权重字典（0-1，表示兴趣强度）

6. **values**：
   - priorities：重视的事物列表
   - attitude：生活态度

7. **preferences**（审美与偏好）：
   - aesthetic_style：审美风格列表（如：精致爱美、复古怀旧、Clean Fit、工装风等）
   - brand_affinity：可能偏好的品牌调性列表（如Apple, Aesop, Leica等）

8. **one_more_thing**：
   - 一些其他有趣的观察或洞察（这个是加分项）

输出JSON格式，尽可能多地推断信息，无法推断的字段设为null或空列表。"""

    return prompt
```

---

## 4. VLM分析Prompt（单张照片分析）

```python
def _create_prompt(self, photo: Photo, face_db: Dict) -> str:
    """
    创建VLM prompt

    Args:
        photo: 照片对象
        face_db: 人脸库

    Returns:
        prompt字符串
    """
    # 构建人物描述
    people_desc = []
    for face in photo.faces:
        person_id = face["person_id"]
        person_info = face_db.get(person_id)

        desc = f"- {person_id}"

        if person_info:
            if person_id == "person_1" or person_info.name == "主角":
                desc += ": 主角（用户本人）"
            elif person_info.photo_count > 10:
                desc += ": 密友（经常出现）"
            elif person_info.photo_count > 3:
                desc += ": 朋友（出现过多次）"
            else:
                desc += ": 陌生人（很少出现）"
        else:
            desc += ": 未知"

        people_desc.append(desc)

    people_str = "\n".join(people_desc) if people_desc else "- 无"

    # 格式化时间
    time_str = photo.timestamp.strftime("%Y-%m-%d %H:%M")

    # 格式化地点
    location_str = photo.location.get("name", "未知") if photo.location else "未知"

    prompt = f"""请分析这张照片。

**照片中的人物**：
{people_str}

**EXIF信息**：
- 时间：{time_str}
- 地点：{location_str}

**任务**：
请分析并输出JSON，包含以下字段：

1. **summary**（必须）：
   - 一句话描述，必须包含人物
   - 例如："用户和person_2在咖啡馆聊天" ✓
   - 不要说："一张咖啡馆的照片" ✗

2. **mood**：整体氛围（如：轻松、愉快、温馨）

3. **vibes**：氛围标签（如：放松、日常、正式）

4. **macro_event**：
   - activity：活动类型（社交/工作/休闲/用餐/运动...）
   - social_context：社交背景（和朋友/独自/和家人/和同事）
   - interaction：人物互动（聊天/合作/聚会...）
   - confidence：置信度（0-1）

5. **people_details**（每个人）：
   - person_id
   - role：角色（主角/朋友/同事/家人...）
   - activity：在做什么
   - expression：表情（微笑/严肃/专注...）

6. **scene**：
   - location_detected：具体位置（如：室内咖啡馆）
   - location_type：室内/室外
   - visual_clues：关键视觉元素（列表）
   - environment_details：环境细节（列表）

7. **details**：关键物品/元素（列表）

8. **story_hints**：能推断的故事线索（列表）

**注意**：
- summary必须包含人物，不能只描述场景
- 如果能推断人物关系，请在people_details中说明
- story_hints可以从人物表情、动作、环境推断

输出JSON格式，不要有任何其他文字。"""

    return prompt
```

---

## 使用说明

这四个prompt分别位于：
1. **事件提取** - `services/llm_processor.py` 的 `_create_event_extraction_prompt()` 方法
2. **关系推断** - `services/llm_processor.py` 的 `_create_relationship_prompt()` 方法
3. **用户画像** - `services/llm_processor.py` 的 `_create_profile_prompt()` 方法
4. **VLM分析** - `services/vlm_analyzer.py` 的 `_create_prompt()` 方法

如需修改，请直接编辑对应方法中的prompt字符串，然后重新运行pipeline即可。
