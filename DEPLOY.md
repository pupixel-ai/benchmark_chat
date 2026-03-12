# 记忆工程项目 - 部署指南

## 系统要求

- Python 3.9+
- macOS / Linux
- 至少 4GB 内存

## Railway 部署

### 后端服务

1. 在 Railway 新建一个 Python service，根目录指向仓库根目录。
2. 再附加一个 MySQL service。
3. 在后端 service 的 Variables 里至少设置：

```bash
DATABASE_URL=${{MySQL.MYSQL_URL}}
FRONTEND_ORIGIN=https://你的前端域名
GEMINI_API_KEY=...
AMAP_API_KEY=...
```

如果不显式设置 `DATABASE_URL`，后端也会自动尝试读取 Railway 注入的 `MYSQL_URL`、`MYSQLHOST`、`MYSQLPORT`、`MYSQLUSER`、`MYSQLPASSWORD`、`MYSQLDATABASE`。

4. 再附加一个 Railway Bucket（或其他 S3 兼容对象存储），后端会优先读取：

```bash
BUCKET
ENDPOINT
REGION
ACCESS_KEY_ID
SECRET_ACCESS_KEY
```

也可以改用自定义变量：

```bash
OBJECT_STORAGE_BUCKET
OBJECT_STORAGE_ENDPOINT
OBJECT_STORAGE_REGION
OBJECT_STORAGE_ACCESS_KEY_ID
OBJECT_STORAGE_SECRET_ACCESS_KEY
OBJECT_STORAGE_PREFIX
OBJECT_STORAGE_ADDRESSING_STYLE
```

### 前端服务

前端建议作为单独的 Railway Node service 部署，根目录指向 `frontend/`，并设置：

```bash
NEXT_PUBLIC_API_BASE_URL=https://你的后端域名
```

### 关键说明

- 不要把数据库地址写成 `127.0.0.1`，Railway 容器内没有本机 MySQL。
- 仓库已内置 `vendor/face_recognition_src/face_recognition`，不再依赖本机 `/Users/...` 路径。
- 原始上传图、预览图、boxed 图、face crops、缓存与结果文件会同步到对象存储，并通过 `/api/assets/{task_id}/...` 由后端代理访问。
- `runtime/tasks/` 仍会作为任务执行时的临时工作目录存在，但持久化依赖对象存储而不是容器本地磁盘。

## 快速开始

### 1. 解压项目

```bash
unzip memory_engineering.zip
cd memory_engineering
```

### 2. 安装依赖

```bash
pip3 install -r requirements.txt
```

如果安装失败，尝试：

```bash
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

### 3. 配置API密钥

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件，填入你的密钥
nano .env  # 或使用其他编辑器
```

必需的API密钥：
- **GEMINI_API_KEY**: Gemini API密钥（[获取地址](https://makersuite.google.com/app/apikey)）

可选的API密钥：
- **AMAP_API_KEY**: 高德地图API密钥，用于地址解析（[获取地址](https://console.amap.com/dev/key/app)）

### 4. 运行测试

```bash
# 用几张照片测试
python3 main.py --photos /path/to/test/photos --max-photos 5
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--photos` | 照片目录路径（必需） | - |
| `--max-photos` | 最多处理多少张照片 | 50 |
| `--use-cache` | 使用VLM缓存（跳过VLM分析） | - |

## 运行示例

```bash
# 基本用法
python3 main.py --photos ~/Pictures/2026 --max-photos 10

# 使用缓存（快速调试，第二次运行时）
python3 main.py --photos ~/Pictures/2026 --use-cache
```

## 输出说明

运行完成后，查看 `output/` 目录：
- `memory_output.json` - 结构化数据（JSON格式）
- `memory_detailed.md` - 详细报告（Markdown格式）

## 项目结构

```
memory_engineering/
├── main.py              # 主程序入口
├── config.py            # 配置文件
├── requirements.txt     # Python依赖列表
├── .env.example         # API密钥配置模板
├── utils.py             # 工具函数
├── models/              # 数据模型定义
├── services/            # 核心服务
│   ├── image_processor.py    # 图片处理（HEIC转JPEG、压缩）
│   ├── face_recognition.py   # 人脸识别
│   ├── vlm_analyzer.py       # VLM分析（Gemini）
│   └── llm_processor.py      # LLM处理
├── cache/               # 运行时生成（缓存）
└── output/              # 输出结果
```

## 常见问题

### Q: 提示找不到模块
**A**: 确保在项目根目录运行，且已安装所有依赖

### Q: 提示API密钥无效
**A**: 检查 `.env` 文件是否正确配置，确认API密钥有效

### Q: HEIC照片无法处理
**A**: `pillow-heif` 已包含在 requirements.txt 中，确保已安装

### Q: 人脸识别不准
**A**: 编辑 `config.py`，调整 `FACE_THRESHOLD`（当前0.70）

### Q: 处理速度慢
**A**:
- 减少 `--max-photos` 数量
- 使用 `--use-cache` 参数跳过VLM分析

### Q: 人脸识别模型下载慢
**A**: 首次运行时会自动下载模型，可能需要几分钟

## 技术架构

| 模块 | 技术 |
|------|------|
| 人脸识别 | DeepFace (Facenet512 + OpenCV) |
| 视觉理解 | Gemini 2.0 Flash |
| 文本推理 | Gemini 2.0 Flash |
| 图像处理 | Pillow + pillow-heif |
| 地址解析 | 高德地图API |

## 注意事项

1. **API成本**: VLM和LLM调用Google Gemini API，会产生费用
2. **隐私**: 照片会上传到Google服务器进行分析
3. **性能**: 处理大量照片需要较长时间（VLM分析最耗时）

祝使用顺利！
