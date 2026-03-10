"""
VLM分析模块
"""
import json
import os
from typing import List, Dict
from google import genai
from google.genai import types
from models import Photo
from config import GEMINI_API_KEY, VLM_MODEL, VLM_CACHE_PATH
from utils import save_json, load_json


class VLMAnalyzer:
    """VLM分析器"""

    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = VLM_MODEL
        self.results = []  # 缓存分析结果

    def analyze_photo(self, photo: Photo, face_db: Dict, protagonist: str = None) -> Dict:
        """
        分析单张照片

        Args:
            photo: 照片对象
            face_db: 人脸库
            protagonist: 主角person_id（如果为None，则使用person_0）

        Returns:
            VLM分析结果
        """
        # 优先使用标注后的图片（带红框和person_id），其次使用压缩图
        image_path = getattr(photo, 'annotated_path', None) or photo.compressed_path

        if not image_path:
            return None

        # 如果没有指定主角，尝试从face_db推断（出现次数最多的）
        if protagonist is None:
            protagonist = max(face_db.items(), key=lambda x: x[1].photo_count)[0] if face_db else "person_0"

        # 构建prompt
        prompt = self._create_prompt(photo, face_db, protagonist)

        try:
            # 读取图片
            with open(image_path, 'rb') as f:
                image_data = f.read()

            # 调用Gemini
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=image_data, mime_type="image/jpeg")
                ],
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )

            # 解析结果
            result = json.loads(response.text)

            # 保存到照片对象
            photo.vlm_analysis = result

            return result

        except Exception as e:
            print(f"警告：VLM分析失败 ({photo.filename}): {e}")
            return None

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
            # 获取主角出现次数
            protagonist_info = face_db.get(protagonist)
            protagonist_count = protagonist_info.photo_count if protagonist_info else 0

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
- 其他人物（蓝色框）用 person_1、person_2... 表示
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
   - **person_id** (String): 如 person_0、person_1
   - **appearance** (String): 性别、年龄段、发型细节、脸型特征、体型、修饰痕迹
   - **clothing** (String): 衣物材质（重磅棉/尼龙/真丝等）、版型、品牌Logo、配饰
   - **interaction** (String): 物理距离（亲密/社交/公共）、具体动作、与主角的关系

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
- **JSON Only**: 仅输出结构化 JSON，不要任何开头语或解释

输出JSON格式（精简版，仅保留事件提取所需字段）：
{{
  "summary": "String",
  "scene": {{
    "environment_details": ["String"],
    "location_detected": "String"
  }},
  "event": {{
    "activity": "String",
    "mood": "String"
  }},
  "key_objects": ["String"]
}}

说明：
- summary: 完整叙事句，包含时间、地点、【主角】行为、核心事件
- scene.environment_details: 关键物品列表（桌椅、蛋糕、花门等，用于识别事件类型）
- scene.location_detected: 地点识别（餐厅、户外、海边等）
- event.activity: 活动类型（用餐、婚礼、运动等）
- event.mood: 整体氛围（轻松、愉快、温馨等）
- key_objects: 关键物品线索（蛋糕、礼物、文件等，用于识别具体事件）"""

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
