# 记忆工程项目 - 从相册提取记忆和画像

基于多模态AI的个人相册分析系统，自动提取事件、识别人物关系、生成用户画像。

现在仓库同时包含：
- Python 后端服务：包装现有 pipeline，支持上传任务、任务轮询、静态结果访问
- `frontend/`：Next.js + Tailwind CSS 前端路径，用于上传图片并展示人脸识别结果

## 功能特性

- **人脸识别**：基于本地 `face-recognition` 项目（InsightFace + 相似度索引），输出原生 `Person_###`
- **VLM分析**：使用Gemini 2.0 Flash理解照片内容（场景、事件、细节）
- **GPS定位**：自动读取照片GPS信息，支持地址解析
- **事件提取**：LLM自动识别和归类生活中的重要事件
- **关系推断**：基于共现频率推断人物关系
- **用户画像**：生成性格、兴趣、生活方式等多维度画像

## 系统要求

- Python 3.9+
- macOS / Linux
- 至少 4GB 内存

## 快速开始

### 1. 安装依赖

```bash
pip3 install -r requirements.txt
```

### 2. 配置API密钥

```bash
cp .env.example .env
```

必需的API密钥：
- **GEMINI_API_KEY**: [获取地址](https://makersuite.google.com/app/apikey)

可选的API密钥：
- **AMAP_API_KEY**: 高德地图（地址解析，[获取地址](https://console.amap.com/dev/key/app)）
- **API_PROXY_URL / API_PROXY_KEY**: 代理模式需要

### 3. 运行

```bash
python3 main.py --photos /path/to/photos --max-photos 10
```

如需显式覆盖仓库内置的人脸识别源码路径：

```bash
export FACE_RECOGNITION_SRC_PATH=/absolute/path/to/face-recognition/src
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--photos` | 照片目录路径（必需） | - |
| `--max-photos` | 最多处理多少张照片 | 50 |
| `--use-cache` | 使用VLM缓存（跳过VLM分析） | - |

## 输出结果

```
output/
├── 事件yoyo.json                 # 最终结构化结果
└── memory_detailed.md           # 详细报告

cache/
├── face_recognition_output.json # 原生人脸识别输出（前端可直接展示）
├── face_recognition_state.json  # 人脸状态缓存
└── faces.index                  # 相似度索引
```

## Web 模式

### 后端

本地开发默认会回落到 `runtime/local_preview.db`。如果你希望本地也使用 MySQL，可以先启动数据库：

```bash
docker compose up -d mysql
```

```bash
./.venv/bin/python backend/app.py
```

默认监听 `http://localhost:8000`，提供：
- `POST /api/tasks`：上传最多 100 张图片并创建任务
- `GET /api/tasks/{task_id}`：轮询任务状态与结果
- `GET /api/assets/{task_id}/...`：通过后端代理访问对象存储中的图片与任务产物

### 前端

前端代码位于 `frontend/`，使用 Next.js + Tailwind CSS。常用环境变量：

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

当前仓库只提供前端代码路径；如果本机未安装 Node.js，需要先安装后再执行 `npm install` / `npm run dev`。

生产部署时，前端建议通过 Nginx 与后端同域部署，此时可以把 `NEXT_PUBLIC_API_BASE_URL` 留空，让浏览器直接访问当前域名下的 `/api`。

## 配置说明

编辑 `config.py` 调整参数：

```python
# 人脸识别
FACE_SIM_THRESHOLD = 0.50  # 相似度阈值
FACE_DET_THRESHOLD = 0.60  # 检测阈值
FACE_MIN_SIZE = 48         # 最小人脸尺寸（像素）

# 图片处理
MAX_IMAGE_SIZE = 1536  # VLM用图的最大尺寸

# 关系推断
MIN_PHOTO_COUNT = 2    # 至少出现几次才推断关系
```

## 常见问题

**Q: 人脸识别不准？**
A: 优先调整 `FACE_SIM_THRESHOLD` 和 `FACE_DET_THRESHOLD`

**Q: 处理速度慢？**
A: 使用 `--use-cache` 参数，或减少 `--max-photos` 数量

**Q: HEIC照片无法处理？**
A: `pillow-heif` 已包含在依赖中，确保已安装

## 技术架构

| 模块 | 技术 |
|------|------|
| 人脸识别 | InsightFace + 本地 `face-recognition` |
| 视觉理解 | Gemini 2.0 Flash |
| 文本推理 | Gemini 2.0 Flash |
| 图像处理 | Pillow + pillow-heif |
| 地址解析 | 高德地图API |

## 注意事项

1. **API成本**: VLM和LLM调用会产生费用
2. **隐私**: 照片会上传到Google服务器进行分析
3. **性能**: 处理大量照片需要较长时间

## Railway 部署提示

- 后端优先读取 `DATABASE_URL`，如果没有设置，会自动尝试 Railway MySQL 注入的 `MYSQL_URL` / `MYSQLHOST` 等变量。
- 如果两者都没有，后端会回落到 `sqlite:///runtime/local_preview.db`，只适合本地预览，不适合正式部署。
- 仓库已经内置 `vendor/face_recognition_src/face_recognition`，部署时不再依赖你本机的 `/Users/...` 路径。
- Railway 后端建议设置 `FRONTEND_ORIGIN` 为前端实际域名，并将 `DATABASE_URL` 直接指向 `${{MySQL.MYSQL_URL}}`。
- 原始上传图、预览图、boxed 图、face crops、缓存与结果文件都会同步到 Railway Bucket / S3 兼容对象存储，并通过 `/api/assets/...` 稳定访问。

## 部署指南

详细部署说明请查看 [DEPLOY.md](DEPLOY.md)
