"""
VLM分析模块
"""
import base64
import json
import mimetypes
import os
import time
from typing import List, Dict, Any
from models import Photo
from config import (
    API_PROXY_KEY,
    API_PROXY_MODEL,
    API_PROXY_URL,
    GEMINI_API_KEY,
    MAX_RETRIES,
    MODEL_PROVIDER,
    OPENROUTER_API_KEY,
    OPENROUTER_APP_NAME,
    OPENROUTER_BASE_URL,
    OPENROUTER_SITE_URL,
    OPENROUTER_VLM_MODEL,
    RETRY_DELAY,
    VLM_CACHE_PATH,
    VLM_MODEL,
)
from utils import save_json, load_json


class VLMAnalyzer:
    """VLM分析器 - 支持官方 Gemini、Gemini 代理和 OpenRouter"""

    def __init__(self, cache_path: str = VLM_CACHE_PATH):
        self.provider = MODEL_PROVIDER
        self.use_proxy = self.provider == "proxy"
        self.use_openrouter = self.provider == "openrouter"
        self.model = OPENROUTER_VLM_MODEL if self.use_openrouter else VLM_MODEL
        self.cache_path = cache_path
        self.results = []  # 缓存分析结果
        self._result_index: Dict[str, int] = {}
        self.requests = None
        self.genai = None
        self.types = None

        if self.use_proxy:
            # 使用代理服务
            if not API_PROXY_URL or not API_PROXY_KEY:
                raise ValueError("使用代理服务需要配置 API_PROXY_URL 和 API_PROXY_KEY")
            import requests

            self.requests = requests
            self.proxy_url = API_PROXY_URL
            self.proxy_key = API_PROXY_KEY
            self.proxy_model = API_PROXY_MODEL
            print(f"[VLM] 使用代理服务: {self.proxy_url}")
        elif self.use_openrouter:
            import requests

            self.requests = requests
            self.openrouter_api_key = OPENROUTER_API_KEY or GEMINI_API_KEY
            if not self.openrouter_api_key:
                raise ValueError("使用 OpenRouter 需要配置 OPENROUTER_API_KEY 或 GEMINI_API_KEY")
            self.openrouter_base_url = OPENROUTER_BASE_URL.rstrip("/")
            self.openrouter_site_url = OPENROUTER_SITE_URL
            self.openrouter_app_name = OPENROUTER_APP_NAME
            print(f"[VLM] 使用 OpenRouter: {self.model}")
        else:
            # 使用官方 Gemini API
            from google import genai
            from google.genai import types

            self.genai = genai
            self.types = types
            self.client = genai.Client(api_key=GEMINI_API_KEY)
            print(f"[VLM] 使用官方 Gemini API")

    def _coerce_text_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part for part in parts if part).strip()
        if isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str):
                return text.strip()
        raise ValueError(f"无法从内容中提取文本: {type(content).__name__}")

    def _extract_json_payload(self, raw_text: str) -> Dict[str, Any]:
        """从模型返回的文本中提取 JSON。"""
        text = raw_text.strip()
        text = text.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "").strip()

        if text.startswith("```"):
            lines = text.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = min((idx for idx in (text.find("{"), text.find("[")) if idx != -1), default=-1)
            end_object = text.rfind("}")
            end_array = text.rfind("]")
            end = max(end_object, end_array)
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise

    def _normalize_result(self, result: Any) -> Dict:
        """统一代理/官方 API 的返回结构。"""
        if result is None:
            return {}

        if isinstance(result, list):
            if not result:
                return {}
            return self._normalize_result(result[0])

        if isinstance(result, str):
            return self._extract_json_payload(result)

        if isinstance(result, dict):
            if "text" in result and isinstance(result["text"], str):
                return self._extract_json_payload(result["text"])
            return result

        raise ValueError(f"不支持的 VLM 返回类型: {type(result).__name__}")

    def _infer_reliable_primary_person_id(self, face_db: Dict) -> str | None:
        if not face_db:
            return None

        stats_by_person = []
        for person_id, person in face_db.items():
            photo_count = int(getattr(person, "photo_count", 0) or 0)
            first_seen = getattr(person, "first_seen", None)
            first_seen_value = first_seen.isoformat() if first_seen else ""
            stats_by_person.append((person_id, photo_count, first_seen_value))

        stats_by_person.sort(key=lambda item: (item[1], item[2]), reverse=True)
        top_person_id, top_photo_count, _ = stats_by_person[0]
        if top_photo_count < 2:
            return None

        tied_top_people = [person_id for person_id, photo_count, _ in stats_by_person if photo_count == top_photo_count]
        if len(tied_top_people) > 1:
            return None

        return top_person_id

    def analyze_photo(self, photo: Photo, face_db: Dict, primary_person_id: str = None) -> Dict:
        """
        分析单张照片

        Args:
            photo: 照片对象
            face_db: 人脸库
            primary_person_id: 主用户 person_id

        Returns:
            VLM分析结果
        """
        # 优先使用标注后的图片（带红框和person_id），其次使用压缩图
        image_path = (
            getattr(photo, 'boxed_path', None)
            or getattr(photo, 'annotated_path', None)
            or photo.compressed_path
        )

        if not image_path:
            photo.processing_errors["vlm"] = "未找到可供 VLM 分析的图片路径"
            return None

        photo.processing_errors.pop("vlm", None)

        # 如果没有指定主用户，只在存在稳定锚点时才推断，避免把偶发人物误当成主用户。
        if primary_person_id is None:
            primary_person_id = self._infer_reliable_primary_person_id(face_db)

        # 构建prompt
        prompt = self._create_prompt(photo, face_db, primary_person_id)

        try:
            # 读取图片
            with open(image_path, 'rb') as f:
                image_data = f.read()
            mime_type = self._guess_mime_type(image_path)

            result = None
            for attempt in range(MAX_RETRIES):
                try:
                    if self.use_proxy:
                        # 使用代理服务
                        result = self._analyze_via_proxy(prompt, image_data, mime_type)
                    elif self.use_openrouter:
                        result = self._analyze_via_openrouter(prompt, image_data, mime_type)
                    else:
                        # 使用官方 Gemini API
                        result = self._analyze_via_official_api(prompt, image_data, mime_type)
                    break
                except Exception as exc:
                    if attempt == MAX_RETRIES - 1 or not self._is_retryable_error(exc):
                        raise
                    delay_seconds = RETRY_DELAY * (2 ** attempt)
                    print(f"[VLM] 可重试错误，{delay_seconds}s 后重试 ({attempt + 1}/{MAX_RETRIES}): {exc}")
                    time.sleep(delay_seconds)

            result = self._normalize_result(result)

            if result:
                # 保存到照片对象
                photo.vlm_analysis = result
                return result
            else:
                return None

        except Exception as e:
            photo.processing_errors["vlm"] = str(e)
            print(f"警告：VLM分析失败 ({photo.filename}): {e}")
            return None

    def _guess_mime_type(self, image_path: str) -> str:
        guessed, _ = mimetypes.guess_type(image_path)
        return guessed or "image/jpeg"

    def _is_retryable_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        retry_keywords = ["429", "rate limit", "connection", "timeout", "temporarily unavailable", "reset by peer"]
        return any(keyword in message for keyword in retry_keywords)

    def _analyze_via_official_api(self, prompt: str, image_data: bytes, mime_type: str) -> Dict:
        """通过官方 Gemini API 分析"""
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                prompt,
                self.types.Part.from_bytes(data=image_data, mime_type=mime_type)
            ],
            config=self.genai.types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        result = json.loads(response.text)
        return result

    def _analyze_via_proxy(self, prompt: str, image_data: bytes, mime_type: str) -> Dict:
        """通过代理服务分析"""
        # 将图片转换为 base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # 构建请求
        headers = {
            "x-api-key": self.proxy_key,
            "Content-Type": "application/json"
        }

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": prompt
                        },
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_base64
                            }
                        }
                    ]
                }
            ]
        }

        # 调用代理 API
        url = f"{self.proxy_url}/api/gemini/v1beta/models/{self.proxy_model}:generateContent"

        response = self.requests.post(url, json=payload, headers=headers, timeout=60)

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
                                # 如果不是有效 JSON，返回原始文本
                                return {"text": part["text"]}

            return None
        else:
            error_msg = f"代理 API 返回状态码 {response.status_code}"
            if response.text:
                try:
                    error_data = response.json()
                    error_msg += f": {error_data.get('error', {}).get('message', response.text)}"
                except:
                    error_msg += f": {response.text[:200]}"
            raise Exception(error_msg)

    def _openrouter_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.openrouter_site_url,
            "X-Title": self.openrouter_app_name,
        }

    def _extract_openrouter_content(self, response_data: Dict[str, Any]) -> str:
        choices = response_data.get("choices", [])
        if not choices:
            raise ValueError("OpenRouter 未返回 choices")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not content and message.get("reasoning"):
            content = message["reasoning"]
        return self._coerce_text_content(content)

    def _analyze_via_openrouter(self, prompt: str, image_data: bytes, mime_type: str) -> Dict:
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.1,
        }

        response = self.requests.post(
            f"{self.openrouter_base_url}/chat/completions",
            json=payload,
            headers=self._openrouter_headers(),
            timeout=60,
        )

        if response.status_code == 200:
            response_data = response.json()
            text = self._extract_openrouter_content(response_data)
            return self._extract_json_payload(text)

        error_msg = f"OpenRouter 返回状态码 {response.status_code}"
        if response.text:
            try:
                error_data = response.json()
                error_msg += f": {error_data.get('error', {}).get('message', response.text)}"
            except Exception:
                error_msg += f": {response.text[:200]}"
        raise Exception(error_msg)

    def _create_prompt(self, photo: Photo, face_db: Dict, primary_person_id: str) -> str:
        """
        创建VLM prompt

        Args:
            photo: 照片对象
            face_db: 人脸库（仅用于显示人物ID）
            primary_person_id: 主用户 person_id

        Returns:
            prompt字符串
        """
        # 格式化时间
        time_str = photo.timestamp.strftime("%Y-%m-%d %H:%M")

        # 格式化地点：优先使用语义化地址，其次显示GPS坐标
        if photo.location and photo.location.get("name"):
            location_str = photo.location["name"]
        elif photo.location and photo.location.get("lat"):
            location_str = f"GPS: {photo.location['lat']:.4f}, {photo.location['lng']:.4f}"
        else:
            location_str = "未知"

        analysis_anchor = "【主用户】" if primary_person_id else "【拍摄者】"

        # 构建人物说明：只在有人脸时才说明谁是主用户
        people_section = ""
        if photo.faces and primary_person_id:
            # 获取主用户出现次数
            primary_person = face_db.get(primary_person_id) if primary_person_id else None
            primary_count = primary_person.photo_count if primary_person else 0

            # 有人脸，列出所有人脸并标注主用户
            people_desc = []
            for face in photo.faces:
                person_id = face["person_id"]
                if primary_person_id and person_id == primary_person_id:
                    people_desc.append(f"- {person_id}：【主用户】（高频人物）")
                else:
                    people_desc.append(f"- {person_id}")
            people_str = "\n".join(people_desc)

            # 检查主用户是否在照片中
            primary_in_photo = any(
                f["person_id"] == primary_person_id for f in photo.faces
            ) if primary_person_id else False

            if primary_in_photo:
                people_section = f"""
**人物说明**（照片中每个人脸用彩色框标注，标签位于人物上方）：
- {primary_person_id}（红色框）是【主用户】，是当前相册里出现频率最高的人物，出现在{primary_count}张照片中
- 其他人物保持原始 person_id / Person_### 编号
{people_str}

**分析原则**：围绕【主用户】这个身份锚点分析，但必须先判断其是否为现场实体人物
- 即便 {primary_person_id} 的人脸框出现在画面中，也要先判断该形象是否只存在于屏幕、海报、广告牌、包装印刷、镜面反射或手持照片等载体里
- 如果 {primary_person_id} 只出现在上述载体中，不得写成“【主用户】在现场出镜/站在现场/正在现场表演”
- 这类情况应明确写成“载体中的【主用户】”或“屏幕中显示【主用户】”，并优先考虑现场拍摄视角仍来自【主用户】本人
- 只有确认 {primary_person_id} 是现场实体人物时，summary 才能直接用“【主用户】正在……”
- people 数组中使用具体的 person_id
"""
            else:
                people_section = f"""
**人物说明**：
- 照片中的人物：{', '.join([f["person_id"] for f in photo.faces])}
- 【主用户】不在照片中，通常是拍摄者
- {primary_person_id or '未知'} 是当前相册里出现频率最高的人物，出现在{primary_count}张照片中

**分析原则**：
- summary 中使用"【主用户】"指代拍摄者
- 描述从【主用户】的拍摄视角观察到的场景
- people 数组中使用照片中具体的 person_id
"""
        elif photo.faces:
            people_section = f"""
**人物说明**：
- 当前批次没有可靠的主用户身份锚点，以下 person_id 只代表画面中可见的人物，不等于拍摄者本人：
{chr(10).join([f"- {f['person_id']}" for f in photo.faces])}

**分析原则**：
- 如果表达拍摄视角，summary 中使用"【拍摄者】"，不要把任何 person_id 写成【主用户】或拍摄者本人
- people 数组中只使用具体的 person_id
- 若人物出现在屏幕、海报、广告牌、包装印刷、镜面反射或手持照片中，必须明确写明“载体中的人物”，不要当成现场实体人物
- 若无法确认人物是否来自上述载体，不要做【主用户】绑定，只描述为“画面中的人物”或“疑似载体中的人物”
"""
        else:
            people_section = """
**人物说明**：照片中未检测到人脸，这是【主用户】的拍摄视角。

**分析原则**：从【主用户】的拍摄视角描述场景和事件，summary 中直接使用"【主用户】"。
"""
            if not primary_person_id:
                people_section = """
**人物说明**：照片中未检测到人脸，默认按【拍摄者】的拍摄视角理解。

**分析原则**：从【拍摄者】的拍摄视角描述场景和事件，summary 中直接使用"【拍摄者】"。
"""

        prompt = f"""## Role
你是一位精通视觉人类学与社会空间重构的专家。你的任务是针对相册内容，建立一套"高度复原、可供画像分析"的客观视觉档案。

### Context & Priorities
- **核心任务**: 以{analysis_anchor}的视角与画面中的可见人物为线索，还原每一张照片的物理现场、人物身份与社会学线索。
- **描述准则**: 拒绝模糊词汇（如：肤色白皙），追求物理参数（如：冷白皮、重磅棉、30度斜射光、具体手指动作）。
{people_section}
### EXIF信息
- 时间：{time_str}
- 地点：{location_str}

---

### Task: 结构化识别要求 (Output Schema)

请针对每张照片输出以下严格的 JSON 格式（严禁包含主观推测）：

1. **summary** (String, 必须):
   - 完整叙事句，包含：[精确时刻/天气] + [具体地理/室内场景] + [视角主体或主要人物的行为状态] + [核心事件进度]
   - 只有当 prompt 明确给出可靠主用户锚点时，才能使用【主用户】；否则用【拍摄者】或具体 person_id
   - 若人物位于屏幕、海报、广告牌、包装印刷、镜面反射或手持照片中，必须明确标注“载体中的人物”
   - 若无法确认人物是否来自上述载体，不要绑定为【主用户】，只描述为“画面中的人物”或“疑似载体中的人物”
   - 不要把手持照片默认写成“拍立得”；只有在品牌、相纸边框或器材特征明确可见时才能这么写，否则统一写“手持照片”或“即时成像照片”

2. **people** (List，无人脸时可为空数组):
   - **person_id** (String): 如 Person_001、Person_002
     也可能是 Person_001、Person_002 这类 face-recognition 原始编号
   - **appearance** (String): 性别、年龄段、发型细节、脸型特征、体型、修饰痕迹
   - **clothing** (String): 衣物材质（重磅棉/尼龙/真丝等）、版型、品牌Logo、配饰
   - **interaction** (String): 物理距离（亲密/社交/公共）、具体动作、与主角的关系
     这里的“主角”仅在存在可靠主用户锚点时才可理解为【主用户】；否则描述其与【拍摄者】或其他 person_id 的关系

3. **scene** (Object):
   - **environment_description** (String): 宏观场景/景色描述
   - **environment_details** (List): 环境细节列表（木质桌子、绿色植物、暖色调灯光等）
   - **location_detected** (String): 具体位置识别（如望京漫咖啡）
   - **visual_clues** (List): 关键视觉元素列表
   - **weather** (String, 可选): 天气（如果可见）
   - **layout** (Object, 可选): {{ "foreground": "近景", "midground": "主体", "background": "远景" }}
   - **lighting** (String, 可选): 光线性质与角度（左侧斜射自然光、室内冷色调荧光等）

4. **event** (Object):
   - **activity** (String): 具体活动类型（喝咖啡/吃饭/工作/运动/旅行/其他）
   - **social_context** (String): 社交背景（和朋友/独自/和家人/和同事）
   - **interaction** (String): 人物互动（聊天/合作/聚会...）
   - **mood** (String): 整体氛围（轻松、愉快、温馨、专注...）

5. **details** (List): 画面中值得关注的硬核线索（品牌Logo、App界面、证件、账单、书籍标题等）

---

### ⚠️ 输出要求 (Constraints)
- **拒绝模板**: 严禁多人描述雷同，必须捕捉细微差别
- **硬核线索**: 必须扫描并记录所有可见的品牌、文字、屏幕内容
- **严禁误绑身份**: 不要因为人脸框或 person_id 的存在，就把屏幕/海报中的人物认定为【主用户】
- **严禁误判在场**: 若【主用户】只出现在屏幕、海报、广告牌、包装印刷、镜面反射或手持照片中，不得写成【主用户】本人就在现场实体出镜
- **严禁滥用“拍立得”**: 看见手持照片时，除非视觉证据非常明确，不要直接写“拍立得”
- **JSON Only**: 仅输出结构化 JSON，不要任何开头语或解释

输出JSON格式（精简版，仅保留事件提取所需字段）：
{{
  "summary": "String",
  "people": [
    {{
      "person_id": "Person_001",
      "appearance": "String",
      "clothing": "String",
      "interaction": "String"
    }}
  ],
  "scene": {{
    "environment_description": "String",
    "environment_details": ["String"],
    "location_detected": "String",
    "visual_clues": ["String"],
    "weather": "String"
  }},
  "event": {{
    "activity": "String",
    "social_context": "String",
    "interaction": "String",
    "mood": "String"
  }},
  "details": ["String"],
  "key_objects": ["String"]
}}

说明：
- summary: 完整叙事句，包含时间、地点、视角主体或主要人物状态、核心事件
- 如果没有可靠主用户锚点，summary 应改为描述【拍摄者】或具体 person_id 的状态，不能硬写【主用户】
- 如果可靠主用户锚点只出现在载体中，summary 也不能写成【主用户】本人在现场实体出镜，而应写成“载体中的【主用户】”
- people: 若检测到人脸，必须逐个输出 person_id，并描述其外貌/穿着/互动
- scene.environment_details: 关键物品列表（桌椅、蛋糕、花门等，用于识别事件类型）
- scene.location_detected: 地点识别（餐厅、户外、海边等）
- event.activity: 活动类型（用餐、婚礼、运动等）
- event.mood: 整体氛围（轻松、愉快、温馨等）
- details: 额外硬核细节（屏幕、品牌、票据、装饰、道具）
- key_objects: 关键物品线索（蛋糕、礼物、文件等，用于识别具体事件）"""

        return prompt

    def save_cache(self):
        """保存VLM结果缓存"""
        data = {
            "metadata": {
                "total_photos": len(self.results),
                "model": self.model,
                "schema_version": 2,
                "face_id_scheme": "Person_###",
            },
            "photos": self.results
        }

        save_json(data, self.cache_path)

    def _rebuild_result_index(self):
        self._result_index = {}
        for index, item in enumerate(self.results):
            photo_id = item.get("photo_id")
            if photo_id:
                self._result_index[photo_id] = index

    def load_cache(self) -> bool:
        """
        加载VLM结果缓存

        Returns:
            是否成功加载
        """
        data = load_json(self.cache_path)

        metadata = data.get("metadata", {}) if data else {}
        if metadata.get("schema_version") != 2 or metadata.get("face_id_scheme") != "Person_###":
            if data:
                print("VLM缓存版本与当前人脸协议不兼容，忽略旧缓存")
            return False

        if data and data.get("photos"):
            normalized_results = []
            for item in data["photos"]:
                vlm_analysis = self._normalize_result(item.get("vlm_analysis"))
                normalized_item = dict(item)
                normalized_item["vlm_analysis"] = vlm_analysis
                normalized_item.setdefault("face_person_ids", [])
                normalized_results.append(normalized_item)

            self.results = normalized_results
            self._rebuild_result_index()
            print(f"加载VLM缓存：{len(self.results)} 张照片")
            return True

        return False

    def has_result(self, photo_id: str) -> bool:
        return photo_id in self._result_index

    def get_result(self, photo_id: str) -> Dict | None:
        index = self._result_index.get(photo_id)
        if index is None:
            return None
        return self.results[index]

    def add_result(self, photo: Photo, vlm_result: Dict, persist: bool = False):
        """添加一个VLM结果到缓存"""
        vlm_result = self._normalize_result(vlm_result)

        result = {
            "photo_id": photo.photo_id,
            "filename": photo.filename,
            "timestamp": photo.timestamp.isoformat(),
            "location": photo.location,
            "face_person_ids": [face["person_id"] for face in photo.faces if face.get("person_id")],
            "vlm_analysis": vlm_result
        }

        existing_index = self._result_index.get(photo.photo_id)
        if existing_index is not None:
            self.results[existing_index] = result
        else:
            self.results.append(result)
        self._rebuild_result_index()
        if persist:
            self.save_cache()
