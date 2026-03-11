# 记忆工程项目 - 从相册提取记忆和画像

基于多模态AI的个人相册分析系统，自动提取事件、识别人物关系、生成用户画像。

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

如需显式指定本地人脸项目源码路径：

```bash
export FACE_RECOGNITION_SRC_PATH=/Users/ziyan/Documents/face-recognition/src
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

## 配置说明

编辑 `config.py` 调整参数：

```python
# 人脸识别
FACE_SIM_THRESHOLD = 0.50  # 相似度阈值
FACE_DET_THRESHOLD = 0.60  # 检测阈值
FACE_MIN_SIZE = 70         # 最小人脸尺寸（像素）

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

## 部署指南

详细部署说明请查看 [DEPLOY.md](DEPLOY.md)
