# 记忆工程项目 - 开发文档

## 项目概述

基于多模态AI的个人相册分析系统，自动从照片中提取：
- **人脸识别** - 识别人物并建立人物库
- **VLM分析** - Gemini 2.0 Flash 理解照片内容
- **GPS定位** - 读取EXIF GPS，高德地图逆地理编码
- **事件提取** - LLM自动聚类生成生活事件
- **关系推断** - 基于共现频率推断人物关系
- **用户画像** - 生成多维度用户画像

---

## 目录结构

```
memory_engineering/
├── main.py                 # 主入口
├── config.py               # 配置文件（使用环境变量）
├── requirements.txt        # 依赖列表
├── .env.example            # API密钥配置模板
├── models/
│   └── __init__.py         # 数据模型（Photo, Person, Event, Relationship, UserProfile）
├── services/
│   ├── __init__.py
│   ├── image_processor.py  # 图片处理（HEIC转JPEG、GPS解析）
│   ├── face_recognition.py # 人脸识别（DeepFace + Facenet512）
│   ├── vlm_analyzer.py     # VLM分析（Gemini 2.0 Flash）
│   └── llm_processor.py    # LLM处理（事件提取、关系推断、画像生成）
├── utils/
│   └── __init__.py         # 工具函数
├── cache/                  # 运行时缓存
│   ├── vlm_results.json    # VLM分析结果缓存
│   ├── face_db.json       # 人脸数据库
│   ├── jpeg_images/       # 全尺寸JPEG（人脸识别用）
│   └── compressed_images/ # 压缩图片（VLM用）
└── output/                 # 输出结果
    ├── memory_output.json # 结构化数据
    └── memory_detailed.md  # Markdown报告
```

---

## 核心数据模型

### Photo（照片对象）
```python
@dataclass
class Photo:
    photo_id: str              # photo_001
    filename: str              # 原文件名
    path: str                  # 原文件路径
    timestamp: datetime         # EXIF拍摄时间
    location: Dict             # {lat, lng, name} GPS+地址
    original_path: Optional[str]  # HEIC原始路径
    compressed_path: Optional[str] # 压缩图路径
    faces: List[Dict]          # 人脸识别结果
    vlm_analysis: Optional[Dict] # VLM分析结果
```

### Event（事件对象）
```python
@dataclass
class Event:
    event_id: str
    date: str
    time_range: str
    duration: str
    title: str
    type: str                  # 社交/工作/休闲/用餐/运动/旅行/其他
    participants: List[str]
    location: str
    description: str
    photo_count: int
    confidence: float
    reason: str
    # 新增字段
    narrative: str             # 客观叙事（50-100字）
    social_interaction: Dict   # {core_person_ids, interaction_details}
    evidence_photos: List[str] # 照片ID列表
    lifestyle_tags: List[str]  # 生活标签
```

---

## 处理流程（8步）

```
[1/8] 加载照片 → ImageProcessor.load_photos()
       ↓ 读取EXIF（时间、GPS），按时间排序

[2/8] 转换HEIC → ImageProcessor.convert_to_jpeg()
       ↓ 保留EXIF，转换为全尺寸JPEG（人脸识别用）

[3/8] 人脸识别 → FaceRecognition.process_photo()
       ↓ DeepFace (Facenet512)，匹配或创建person_id

[4/8] 识别主角 → 找出现次数最多的人

[5/8] 压缩照片 → ImageProcessor.preprocess()
       ↓ 压缩到1536px（VLM用）

[6/8] VLM分析 → VLMAnalyzer.analyze_photo()
       ↓ Gemini 2.0 Flash理解照片，返回结构化数据

[7/8] LLM处理 → LLMProcessor
       ├─ extract_events()     # 提取事件（直接分析全部VLM数据）
       ├─ infer_relationships() # 推断关系
       └─ generate_profile()    # 生成画像

[8/8] 保存结果 → JSON + Markdown报告
```

---

## VLM数据结构（Gemini返回）

```json
{
  "summary": "一句话描述",
  "people": [
    {
      "person_id": "person_0",
      "appearance": "外貌描述",
      "clothing": "穿着描述",
      "activity": "在做什么",
      "interaction": "与主角互动",
      "expression": "表情"
    }
  ],
  "scene": {
    "environment_description": "环境描述",
    "environment_details": ["细节列表"],
    "location_detected": "VLM识别地点",
    "visual_clues": ["视觉元素"],
    "weather": "天气"
  },
  "event": {
    "activity": "活动描述",
    "social_context": "社交背景",
    "interaction": "互动",
    "mood": "氛围"
  },
  "Time": {
    "date": "日期特征",
    "time": "时间特征"
  },
  "details": ["其他细节"]
}
```

---

## LLM事件提取Prompt结构

**Role**: 人类学专家与社会学行为分析师

**输入**: 全部VLM结果，包含：
- 时间、GPS、人物ID
- 人物详情（外貌、穿着、动作、互动、表情）
- 场景详情（环境、细节、视觉线索、天气）
- 事件详情（活动、社交背景、互动、氛围）
- 时间背景（日期/时间特征）
- 其他细节

**输出格式**:
```json
{
  "events": [{
    "event_id": "EVT_001",
    "date": "2025-12-04",
    "title": "事件标题",
    "type": "社交/工作/休闲...",
    "time_range": "19:52 - 20:21",
    "duration": "约30分钟",
    "location": "具体地点",
    "participants": ["person_0"],
    "description": "简要描述",
    "narrative": "50-100字客观叙事",
    "social_interaction": {
      "core_person_ids": ["person_1"],
      "interaction_details": "互动详情"
    },
    "evidence_photos": ["photo_001"],
    "photo_count": 3,
    "confidence": 0.85,
    "reason": "判断理由",
    "lifestyle_tags": ["文化探索", "夜间社交"]
  }]
}
```

---

## 配置说明

### 环境变量 (.env)
```bash
GEMINI_API_KEY=your_gemini_api_key_here    # 必需
AMAP_API_KEY=your_amap_api_key_here      # 可选，用于地址解析
```

### config.py 关键参数
```python
# VLM/LLM
VLM_MODEL = "gemini-2.0-flash"
LLM_MODEL = "gemini-2.0-flash"

# 人脸识别
FACE_THRESHOLD = 0.70    # 相似度阈值
FACE_MIN_SIZE = 70       # 最小人脸尺寸

# 图片处理
MAX_IMAGE_SIZE = 1536    # 压缩后最大边长

# 关系推断
MIN_PHOTO_COUNT = 2      # 至少出现几次
MIN_TIME_SPAN_DAYS = 1   # 至少认识几天
```

---

## 依赖库

```
deepface>=0.0.79         # 人脸识别
google-genai>=1.0.0      # Gemini API
pillow>=10.0.0           # 图像处理
pillow-heif>=0.13.0      # HEIC支持
ExifRead>=3.0.0          # EXIF读取
python-dotenv>=1.0.0    # 环境变量
geopy>=2.3.0             # （备用，未使用）
```

---

## 已知问题

### 1. 人脸识别失败
**原因**: PNG文件路径包含中文字符
```
Input image must not have non-english characters
```
**影响**: 约11张PNG无法识别人脸
**待修复**: 复制文件到临时路径处理

### 2. VLM分析偶尔失败
**原因**: Gemini返回的JSON格式错误
**影响**: 个别照片分析失败
**处理**: 跳过失败照片继续处理

### 3. 压缩失败
**原因**: RGBA模式无法转换为JPEG
**影响**: 1张照片

### 4. 人脸识别不准
**原因**:
- 人脸太小（<70px）
- 侧脸、遮挡
- 光线不佳

---

## 待开发事项

1. **修复PNG中文路径问题** - 复制到临时路径处理
2. **支持视频分析** - 目前只支持照片
3. **增量处理** - 只处理新增照片
4. **Web界面** - 目前只有CLI
5. **批量处理优化** - 并发VLM调用
6. **人物命名** - 允许用户手动命名人物
7. **关系可视化** - 生成人物关系图
8. **时间轴可视化** - 生成交互式时间线
9. **导出格式** - 支持PDF、HTML等格式

---

## 运行命令

```bash
# 基本用法
python3 main.py --photos /path/to/photos --max-photos 50

# 使用缓存（跳过VLM分析）
python3 main.py --photos /path/to/photos --use-cache
```

---

## 输出文件

| 文件 | 说明 |
|------|------|
| `output/memory_output.json` | 结构化数据（事件、关系、画像、人脸库） |
| `output/memory_detailed.md` | Markdown格式详细报告 |
| `cache/vlm_results.json` | VLM缓存（可加速） |
| `cache/face_db.json` | 人脸数据库 |

---

## 开发注意事项

1. **API成本**: VLM和LLM调用Google Gemini API会产生费用
2. **隐私**: 照片会上传到Google服务器
3. **GPS解析**: 高德地图API需单独申请key
4. **人脸识别**: 首次运行会自动下载模型（较慢）
5. **HEIC处理**: 需要pillow-heif库支持
6. **中文路径**: 某些库（如DeepFace）不支持中文路径

---

## 最后更新

**时间**: 2026-03-04
**版本**: v1.0.0
**维护者**: Claude Code
