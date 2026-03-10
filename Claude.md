# 记忆工程 - Claude Code 工作指南

## 工程信息

**项目名称**: Memory Engineering v1.0.0
**路径**: `/Users/vigar07/Desktop/vigar/memory_engineering_v1.0.0_20260309`
**更新时间**: 2026-03-09

### 核心目标
利用多模态AI（人脸识别+VLM+LLM）自动从个人照相库智能提取生活事件、人物关系和用户画像

### 技术栈
- **人脸识别**: DeepFace + Facenet512
- **VLM分析**: Gemini 2.0 Flash
- **LLM处理**: Gemini 2.0 Flash
- **图像处理**: Pillow + pillow-heif（HEIC支持）
- **地址解析**: 高德地图API

### 核心流程（8步）
```
[1/8] 加载照片 → 读取EXIF（时间、GPS），按时间排序
[2/8] 转换HEIC → 转换为全尺寸JPEG（人脸识别用）
[3/8] 人脸识别 → DeepFace (Facenet512)，匹配或创建person_id
[4/8] 重排主角 → 找出现次数最多的人作为主角
[5/8] 压缩照片 → 压缩到1536px（VLM用）
[6/8] VLM分析 → Gemini 2.0 Flash理解照片内容
[7/8] LLM处理 → 提取事件、推断关系、生成画像
[8/8] 保存结果 → JSON + Markdown报告
```

### 关键文件
| 文件 | 说明 |
|------|------|
| `main.py` | 主入口，8步处理流程 |
| `config.py` | 全局配置（API、阈值、路径） |
| `models/__init__.py` | 数据模型（Photo, Person, Event, Relationship, UserProfile） |
| `services/image_processor.py` | 图片处理（HEIC转JPEG、GPS解析） |
| `services/face_recognition.py` | 人脸识别（DeepFace + Facenet512） |
| `services/vlm_analyzer.py` | VLM分析（Gemini 2.0 Flash） |
| `services/llm_processor.py` | LLM处理（事件提取、关系推断、画像生成） |
| `output/memory_output.json` | 结构化数据输出 |
| `output/memory_detailed.md` | Markdown详细报告 |

### 已知问题
- 人脸识别相似度过低（0.33 < 0.70阈值）
- PNG文件中文路径问题（DeepFace不支持）
- 人脸框偶尔绘制位置错误
- 多个人物相同出现次数时都被标记为主角

---

## 工作规则（必须遵守）

### 规则1：称呼约定
**每次回复都称呼用户为 Vigar**
示例: "Vigar，我已经完成了..."

### 规则2：决策确认
**遇到不确定的代码设计问题，必须先询问 Vigar，不得直接行动**
- 在可能有多种设计方案的情况下，询问 Vigar 的偏好
- 涉及架构改动、接口变更等决策性问题，需要事先确认
- 不确定时，使用 AskUserQuestion 工具咨询

### 规则3：兼容性代码
**不能写兼容性代码，除非 Vigar 主动要求**
- 删除未使用的代码而不是保留为备用
- 修改接口时直接更新所有调用，不做向后兼容处理
- 重构时不需要保留旧的实现方式

---

## 工程改造记录

### 修改日期: 2026-03-09
**操作**: 初始化 Claude.md 工作指南
**内容**: 记录工程信息和三条工作规则

