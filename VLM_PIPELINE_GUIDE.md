# VLM 分析完整管道 - 使用指南

## 📌 快速开始

完整的 VLM 分析管道：从随机采样照片 → 人脸识别 → 压缩 → VLM 视觉分析

## ⚠️ 前置条件

### 1️⃣ 获取 Gemini API Key

**步骤 1：访问 Google AI Studio**
- 打开浏览器访问：https://makersuite.google.com/app/apikey

**步骤 2：创建或复制 API Key**
- 点击 **"Create API Key"** 按钮
- 或者复制已有的 API Key

**步骤 3：配置到项目**
- 在项目根目录创建 `.env` 文件
- 填入 API Key（见下面的配置步骤）

### 2️⃣ 验证依赖

确保已安装必要的 Python 包：
```bash
pip install google-generativeai python-dotenv pillow exifread
```

## 🔧 配置步骤

### 方法 1：使用 .env 文件（推荐）

**第一步：创建 .env 文件**
```bash
# 在项目根目录运行
cat > .env << 'EOF'
GEMINI_API_KEY=paste_your_api_key_here
EOF
```

**第二步：编辑 .env 文件**
```bash
# 用你的编辑器打开 .env
# 替换 paste_your_api_key_here 为你的实际 API Key
nano .env
```

**第三步：验证配置**
```bash
# 脚本会自动检测 .env 文件
python3 test_pipeline_vlm.py --samples 1
```

### 方法 2：使用环境变量

```bash
# 在终端设置环境变量
export GEMINI_API_KEY=your_api_key_here

# 然后运行脚本
python3 test_pipeline_vlm.py --samples 10
```

### 方法 3：快速配置模板

```bash
# 复制模板文件
cp .env.template .env

# 编辑 .env，替换 your_gemini_api_key_here 为实际 Key
# nano .env

# 验证
python3 test_pipeline_vlm.py --help
```

## 🚀 使用脚本

### 基本用法

```bash
# 运行默认 10 张照片
python3 test_pipeline_vlm.py

# 运行 N 张照片
python3 test_pipeline_vlm.py --samples 20

# 使用固定随机种子（可重现结果）
python3 test_pipeline_vlm.py --samples 10 --seed 42

# 指定自定义数据目录
python3 test_pipeline_vlm.py --samples 10 --data-dir /path/to/photos

# 不保存 VLM 缓存（仅查看结果）
python3 test_pipeline_vlm.py --samples 5 --no-save
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--samples` | 10 | 采样照片数 |
| `--data-dir` | `data/raw` | 照片目录路径 |
| `--seed` | 无 | 随机种子（用于可重现结果） |
| `--no-save` | 否 | 不保存 VLM 缓存 |

## 📊 完整流程说明

脚本会按顺序执行以下 7 个步骤：

### [1/7] 加载照片
- 扫描 `data/raw` 目录
- 读取所有支持格式（JPG/JPEG/PNG/HEIC）
- 解析 EXIF 信息（时间戳、GPS）
- 按时间戳排序

### [2/7] 随机采样
- 从所有照片中随机选择 N 张
- 显示每张照片的：
  - 文件名
  - 拍摄时间
  - GPS 位置（逆地理编码）

### [3/7] HEIC 转 JPEG
- 将 HEIC/HEIF 格式转换为全尺寸 JPEG
- 保留原始的 EXIF 信息
- 用于人脸识别的输入

### [4/7] 人脸识别
- 使用 DeepFace (Facenet512) 检测人脸
- 提取人脸特征向量（128 维）
- 与已有人物库匹配或创建新人物
- 重排主角（出现最多的人为 person_0）
- 绘制带标签的人脸框

### [5/7] 压缩照片
- 压缩到 1536px 最大边长
- JPEG 质量 85
- 用于 VLM 分析的输入

### [6/7] VLM 分析（需要 API Key）
- 使用 **Google Gemini 2.0 Flash** 模型
- 对每张照片进行视觉分析
- 识别：
  - 场景和环境描述
  - 人物外貌和穿着
  - 活动和社交背景
  - 关键物品和线索
  - 整体氛围

### [7/7] 保存结果
- VLM 缓存：`cache/Vyoyo.json`
- 测试结果：`cache/test_pipeline_vlm_result.json`
- 人脸库：`cache/face_db.json`

## 📈 预期输出

运行脚本后，你会看到类似的输出：

```
============================================================
  [1/7] 加载照片
============================================================

从 /path/to/data/raw 加载所有照片...
✓ 加载完成：共 63 张照片

============================================================
  [2/7] 随机采样 (10 张)
============================================================

✓ 已随机选择 10 张照片:

   1. IMG_0746.jpeg               | 2024-03-30 16:52:36 | 广东广州白云区
  ...

============================================================
  [6/7] VLM分析
============================================================

使用模型：Gemini 2.0 Flash
处理照片数：10

  [1/10] IMG_0746.jpeg             ✓ 分析完成
  [2/10] IMG_8122.jpeg             ✓ 分析完成
  ...

✓ VLM分析完成
  成功：9 张
  失败：1 张

============================================================
  📊 完整流程统计
============================================================

✅ 照片处理
  • 加载: 63 张
  • 采样: 10 张
  • 有人脸: 3 张
  • 无人脸: 7 张

👤 人脸识别
  • 检测人脸: 8 张
  • 识别人物: 4 个
  • 主角: person_0（3 次）

🤖 VLM分析
  • 成功: 9 张
  • 失败: 1 张
  • 成功率: 90.0%

📋 识别出的人物
  • person_0 【主角】
      出现: 3 次 | 置信度: 0.82
  • person_1
      出现: 2 次 | 置信度: 0.71
  ...
```

## 💾 缓存和结果位置

所有生成的文件都保存在 `cache/` 目录：

| 文件 | 说明 |
|------|------|
| `face_db.json` | 人脸库（累积） |
| `Vyoyo.json` | VLM 分析结果缓存 |
| `test_pipeline_vlm_result.json` | 本次测试的摘要 |
| `jpeg_images/` | 转换后的全尺寸 JPEG |
| `compressed_images/` | 压缩后用于 VLM 的图片 |
| `boxed_images/` | 带人脸框的图片 |

## ⏱️ 性能预期

| 操作 | 时间 | 说明 |
|------|------|------|
| 加载 10 张照片 | <1 秒 | 快速 |
| 人脸识别 | 5-10 秒 | 取决于图片大小和人脸数量 |
| 压缩照片 | <5 秒 | 快速 |
| **VLM 分析** | **2-5 分钟** | **主要耗时步骤** |
| **总耗时** | **2-6 分钟** | 10 张照片 |

## 🐛 故障排除

### 问题：API Key 错误
```
❌ 错误：未检测到 GEMINI_API_KEY
```

**解决方案：**
1. 确保 `.env` 文件在项目根目录
2. 检查格式：`GEMINI_API_KEY=your_key_here`（没有空格）
3. 使用 `echo $GEMINI_API_KEY` 验证环境变量
4. 重启终端让环境变量生效

### 问题：API 限流错误
```
Error: Resource has been exhausted (status_code=429)
```

**解决方案：**
- 这表示超过了免费额度的限制
- 等待 24 小时后重试
- 或者在 https://cloud.google.com 升级账户

### 问题：没有检测到人脸
```
检测人脸: 0 张
```

**解决方案：**
- 这很正常，取决于照片质量
- 确保有足够清晰的人脸照片
- 检查 `cache/boxed_images/` 中绘制的框

### 问题：VLM 返回格式错误
```
警告：VLM分析失败 (photo.jpg): JSON decode error
```

**解决方案：**
- Gemini 可能返回非 JSON 格式
- 这通常是因为照片内容超出范围（如极端内容）
- 脚本会自动跳过失败的照片

## 📝 VLM 分析结果示例

成功的 VLM 分析会返回结构化 JSON：

```json
{
  "summary": "2024-03-30 下午4点30分，【主角】在广州白云区的咖啡馆内坐着，面前摆着一杯咖啡，表情专注地看着手机屏幕，身旁有朋友陪伴",
  "scene": {
    "environment_details": [
      "木质圆形桌子",
      "白色陶瓷咖啡杯",
      "暖色调吊灯",
      "植物装饰"
    ],
    "location_detected": "咖啡馆/咖啡店",
    "lighting": "室内暖色调照明"
  },
  "event": {
    "activity": "喝咖啡/社交",
    "mood": "轻松、愉快"
  },
  "key_objects": [
    "咖啡杯",
    "手机",
    "植物",
    "灯具"
  ]
}
```

## 🔗 与主流程的关联

✅ **兼容性**：VLM 缓存与 `main.py` 完全兼容

运行完此脚本后，可以继续执行主流程：

```bash
# 使用缓存的 VLM 结果，跳过 VLM 分析，直接进行事件提取和关系推断
python3 main.py --photos data/raw --max-photos 10 --use-cache
```

## 📚 更多信息

- **项目根目录**：`/Users/vigar07/Desktop/vigar/memory_engineering_v1.0.0_20260309/`
- **配置文件**：`config.py`
- **服务代码**：`services/vlm_analyzer.py`
- **测试脚本**：`test_pipeline_vlm.py`

## 💡 最佳实践

1. **首次使用**：先用 `--samples 1` 测试 API Key 是否正常
2. **开发调试**：使用 `--seed 42` 固定随机种子保证可重现性
3. **批量处理**：可以在脚本基础上修改逻辑进行批处理
4. **缓存管理**：如要重新分析，删除 `cache/Vyoyo.json` 即可

## 🆘 获取帮助

如遇问题，请检查：
1. ✅ Gemini API Key 是否正确配置
2. ✅ 网络连接是否正常
3. ✅ Python 版本 >= 3.9
4. ✅ 所需包是否安装（google-generativeai）
