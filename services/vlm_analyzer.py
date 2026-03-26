"""
VLM分析模块
"""
import json
import os
import base64
from typing import List, Dict
from models import Photo
from config import GEMINI_API_KEY, VLM_MODEL, VLM_CACHE_PATH, GOOGLE_GEMINI_BASE_URL
from utils import save_json, load_json

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - test env may not install requests
    requests = None
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


class VLMAnalyzer:
    """VLM分析器 - 使用 HTTP 请求调用代理服务"""

    def __init__(self):
        self.model = VLM_MODEL
        self.results = []  # 缓存分析结果

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

        print(f"[VLM] 使用 HTTP 请求方式")
        print(f"[VLM] BaseURL: {self.base_url}")
        print(f"[VLM] 模型: {self.model}")

    def _require_requests(self) -> None:
        if requests is None:
            raise RuntimeError("requests 未安装，无法调用 HTTP VLM 接口")

    def analyze_photo(self, photo: Photo, face_db: Dict, protagonist: str = None) -> Dict:
        """
        分析单张照片

        Args:
            photo: 照片对象
            face_db: 人脸库
            protagonist: 主角person_id（如果为None，则自动推断）

        Returns:
            VLM分析结果
        """
        # 优先使用带人脸框的图片（boxed_path），其次使用压缩图
        image_path = photo.boxed_path or photo.compressed_path

        if not image_path:
            return None

        # 构建prompt
        prompt = self._create_prompt(photo, face_db, protagonist)

        try:
            # 读取图片
            with open(image_path, 'rb') as f:
                image_data = f.read()

            # 使用官方 Gemini API
            result = self._analyze_via_official_api(prompt, image_data)

            if result:
                # 保存到照片对象
                photo.vlm_analysis = result
                return result
            else:
                return None

        except Exception as e:
            print(f"警告：VLM分析失败 ({photo.filename}): {e}")
            return None

    def _analyze_via_official_api(self, prompt: str, image_data: bytes) -> Dict:
        self._require_requests()
        """通过 HTTP 请求调用代理服务的 Gemini API（支持重试，超时时间：60秒）"""
        # 将图片转换为 base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # 构建请求体
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_base64
                            }
                        }
                    ]
                }
            ]
        }

        # 构建请求 URL
        url = f"{self.base_url}/{self.model}:generateContent"
        max_retries = 3
        timeout = 60  # 降低到60秒，避免长时间卡死

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
                                        # 如果不是有效 JSON，尝试从markdown代码块中提取
                                        import re
                                        text = part["text"]
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
                elif response.status_code == 429:
                    # 代理服务限流，等待后重试
                    if attempt < max_retries - 1:
                        wait_time = 3 + attempt * 2  # 3s, 5s, 7s
                        print(f"      [限流重试 {attempt + 1}/{max_retries}] 等待 {wait_time}s 后重试...", end='', flush=True)
                        time.sleep(wait_time)
                        print("\r", end='')  # 清除进度行
                        continue
                    else:
                        raise Exception(f"代理 API 限流，重试{max_retries}次仍失败")
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
                    print(f"      [重试 {attempt + 1}/{max_retries}] 等待 {wait_time}s 后重试...", end='', flush=True)
                    time.sleep(wait_time)
                    print("\r", end='')  # 清除进度行
                    continue
                else:
                    raise Exception(f"VLM分析重试{max_retries}次仍失败: {e}")

    def _create_prompt(self, photo: Photo, face_db: Dict, protagonist: str) -> str:
        """
        创建VLM prompt

        Args:
            photo: 照片对象
            face_db: 人脸库（仅用于显示人物ID）
            protagonist: 主角person_id

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

        # 构建人物说明：只在有人脸时才说明谁是主角
        people_section = ""
        if photo.faces:
            if protagonist is None:
                people_section = f"""
**人物说明**：
- 照片中的人物：{', '.join([f["person_id"] for f in photo.faces])}
- 当前无法从人脸层稳定识别出【主角】，统一按【主角】是拍摄者处理

**分析原则**：
- summary 中使用"【主角】"指代拍摄者（用户本人）
- 描述从【主角】的拍摄视角观察到的场景
- people 数组中使用照片中具体的 person_id
"""
            else:
                # 获取主角出现次数（兼容dict和Person对象）
                protagonist_info = face_db.get(protagonist)
                if protagonist_info:
                    protagonist_count = protagonist_info.get('photo_count') if isinstance(protagonist_info, dict) else getattr(protagonist_info, 'photo_count', 0)
                else:
                    protagonist_count = 0

                # 有人脸，列出所有人脸并标注主角
                people_desc = []
                for face in photo.faces:
                    person_id = face["person_id"]
                    if person_id == protagonist:
                        people_desc.append(f"- {person_id}：【主角】（用户本人）")
                    else:
                        people_desc.append(f"- {person_id}")
                people_str = "\n".join(people_desc)

                # 检查主角是否在照片中
                protagonist_in_photo = any(f["person_id"] == protagonist for f in photo.faces)

                if protagonist_in_photo:
                    # 主角在照片中
                    people_section = f"""
**人物说明**（照片中每个人脸用彩色框标注，标签位于人物上方）：
- {protagonist}（红色框）是【主角】，出现在{protagonist_count}张照片中
- 其他人物（蓝色框）用 Person_002、Person_003... 表示
{people_str}

**分析原则**：围绕【主角】分析所有内容
- summary 中使用"【主角】"指代主角（{protagonist}）
- people 数组中使用具体的 person_id
"""
                else:
                    # 主角不在照片中（拍摄者视角）
                    people_section = f"""
**人物说明**：
- 照片中的人物：{', '.join([f["person_id"] for f in photo.faces])}
- 【主角】不在照片中，是拍摄者
- {protagonist} 通常是【主角】，出现在{protagonist_count}张照片中

**分析原则**：
- summary 中使用"【主角】"指代拍摄者（主角）
- 描述从【主角】的拍摄视角观察到的场景
- people 数组中使用照片中具体的 person_id
"""
        else:
            # 无人脸，说明是主角拍摄视角
            people_section = """
**人物说明**：照片中未检测到人脸，这是【主角】的拍摄视角。

**分析原则**：从【主角】的拍摄视角描述场景和事件，summary 中直接使用"【主角】"。
"""

        prompt = f"""## Role
你是一位精通视觉人类学与社会空间重构的专家。你的任务是针对【主角】的相册，建立一套"高度复原、可供画像分析"的客观视觉档案。

### Context & Priorities
- **核心任务**: 以【主角】为圆心，还原每一张照片的物理现场、人物身份与社会学线索。
- **描述准则**: 拒绝模糊词汇（如：肤色白皙），追求物理参数（如：冷白皮、重磅棉、30度斜射光、具体手指动作）。
{people_section}
### EXIF信息
- 时间：{time_str}
- 地点：{location_str}

---

### Task: 结构化识别要求 (Output Schema)

请针对每张照片输出以下严格的 JSON 格式（严禁包含主观推测）：

1. **summary** (String, 必须):
   - 完整叙事句，包含：[精确时刻/天气] + [具体地理/室内场景] + [【主角】的行为状态] + [核心事件进度]

2. **people** (List，无人脸时可为空数组):
   - **person_id** (String): 必须与人脸标注一致（Person_001、Person_002...）
   - **appearance** (String): 性别、年龄段、发型细节、脸型特征、体型、修饰痕迹
   - **clothing** (String): 衣物材质（重磅棉/尼龙/真丝等）、版型、品牌Logo、配饰
   - **activity** (String): 人物当前动作/姿态
   - **interaction** (String): 与【主角】的物理距离（亲密/社交/公共）、具体互动动作
   - **contact_type** (String): 身体接触类型，从以下枚举中选择：kiss/hug/holding_hands/arm_in_arm/selfie_together/shoulder_lean/sitting_close/standing_near/no_contact
   - **expression** (String): 面部表情和情绪状态

3. **relations** (List): 画面中可观测到的实体关系三元组：
   - **subject** (String): 主体（person_id 或物品名）
   - **relation** (String): 关系动作（如 sitting_at, holding, interacting_with, placed_on, looking_at, standing_near）
   - **object** (String): 客体（person_id、物品名或场景元素）

4. **scene** (Object):
   - **environment_details** (List): 环境细节列表（木质桌子、绿色植物、暖色调灯光等），如有可见天气信息也写入此处
   - **location_detected** (String): 具体位置识别（如望京漫咖啡、星巴克XX店）
   - **location_type** (String): "室内" 或 "室外"

5. **event** (Object):
   - **activity** (String): 具体活动类型（喝咖啡/吃饭/工作/运动/旅行/购物/学习/其他）
   - **social_context** (String): 社交背景（和朋友/独自/和家人/和同事/和伴侣）
   - **mood** (String): 整体氛围（轻松、愉快、温馨、专注、忙碌...）
   - **story_hints** (List): 1-2条基于视觉证据的社交故事推断（如"可能是生日聚会"、"工作日加班"）

6. **details** (List): 画面中值得关注的硬核线索（品牌Logo、App界面、证件、账单、书籍标题、屏幕内容等）

---

### ⚠️ 输出要求 (Constraints)
- **拒绝模板**: 严禁多人描述雷同，必须捕捉细微差别
- **硬核线索**: 必须扫描并记录所有可见的品牌、文字、屏幕内容
- **JSON Only**: 仅输出结构化 JSON，不要任何开头语或解释

输出JSON格式：
{{
  "summary": "String",
  "people": [
    {{
      "person_id": "String",
      "appearance": "String",
      "clothing": "String",
      "activity": "String",
      "interaction": "String",
      "contact_type": "String",
      "expression": "String"
    }}
  ],
  "relations": [
    {{"subject": "Person_001", "relation": "String", "object": "String"}}
  ],
  "scene": {{
    "environment_details": ["String"],
    "location_detected": "String",
    "location_type": "String"
  }},
  "event": {{
    "activity": "String",
    "social_context": "String",
    "mood": "String",
    "story_hints": ["String"]
  }},
  "details": ["String"]
}}

说明：
- summary: 完整叙事句，包含时间、地点、【主角】行为、核心事件
- people: 照片中所有可检测到的人物（若无人脸则为空数组）
  - person_id: 人物ID（必须使用 Person_001/Person_002 等，与人脸标注一致）
  - appearance: 年龄段、性别、发型、脸型特征
  - clothing: 衣物描述（材质、版型、品牌等）
  - activity: 人物动作/姿态
  - interaction: 与【主角】的互动方式或距离关系
  - contact_type: 身体接触类型枚举（kiss/hug/holding_hands/arm_in_arm/selfie_together/shoulder_lean/sitting_close/standing_near/no_contact）
  - expression: 表情和情绪状态
- relations: 人物与人物、人物与物品、物品与场景之间的可观测空间/互动关系
- scene.environment_details: 环境细节列表（桌椅、蛋糕、花门等），含天气信息（如可见）
- scene.location_detected: 地点识别（餐厅、户外、海边等）
- scene.location_type: 室内或室外
- event.activity: 活动类型（用餐、婚礼、运动等）
- event.social_context: 社交背景（独自、和朋友、和家人、和同事等）
- event.mood: 整体氛围（轻松、愉快、温馨等）
- event.story_hints: 基于视觉证据的1-2条故事推断
- details: 硬核物品线索（品牌Logo、证件、文件、App界面等）"""

        return prompt

    def save_cache(self):
        """保存VLM结果缓存"""
        data = {
            "metadata": {
                "total_photos": len(self.results),
                "model": self.model
            },
            "photos": self.results
        }

        save_json(data, VLM_CACHE_PATH)

    def load_cache(self) -> bool:
        """
        加载VLM结果缓存

        Returns:
            是否成功加载
        """
        data = load_json(VLM_CACHE_PATH)

        if data and data.get("photos"):
            self.results = data["photos"]
            print(f"加载VLM缓存：{len(self.results)} 张照片")
            return True

        return False

    def analyze_photos_concurrent(self, photos: List[Photo], face_db: Dict, protagonist: str = None, max_workers: int = 2) -> None:
        """
        并发分析多张照片（使用线程池）

        Args:
            photos: 照片列表
            face_db: 人脸库
            protagonist: 主角person_id（由 main.py 传入）
            max_workers: 最大并发线程数（降低到2避免API限流）
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 按顺序提交任务，避免突发大量并发请求
            future_to_photo = {}
            for i, photo in enumerate(photos):
                future = executor.submit(self.analyze_photo, photo, face_db, protagonist)
                future_to_photo[future] = photo
                # 在提交任务间添加小延迟，避免突发并发
                if (i + 1) % max_workers == 0 and i < len(photos) - 1:
                    time.sleep(0.5)

            # 处理完成的任务
            completed = 0
            for future in as_completed(future_to_photo):
                photo = future_to_photo[future]
                completed += 1
                try:
                    result = future.result()
                    if result:
                        self.add_result(photo, result)
                except Exception as e:
                    print(f"警告：VLM分析失败 ({photo.filename}): {e}")

                # 简单进度显示
                print(f"\r处理中... [{completed}/{len(photos)}]", end='', flush=True)

            print()  # 换行

    def add_result(self, photo: Photo, vlm_result: Dict):
        """添加一个VLM结果到缓存"""
        # 处理vlm_result可能是list的情况（Gemini有时返回list包裹）
        if isinstance(vlm_result, list) and len(vlm_result) > 0:
            vlm_result = vlm_result[0]  # 取第一个元素

        result = {
            "photo_id": photo.photo_id,
            "filename": photo.filename,
            "timestamp": photo.timestamp.isoformat(),
            "location": photo.location,
            "vlm_analysis": vlm_result
        }

        self.results.append(result)
