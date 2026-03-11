# VLM 分析功能设置指南

## ✅ 已为你创建的文件

| 文件 | 说明 |
|------|------|
| `test_pipeline_vlm.py` | 完整的 VLM 分析管道脚本（7 步流程） |
| `VLM_PIPELINE_GUIDE.md` | 详细使用文档 |
| `.env.template` | API Key 配置模板 |
| `SETUP_VLM.md` | 本文件（快速设置指南） |

## 🚀 快速设置（3 步）

### 第 1 步：获取 API Key

1. 打开浏览器访问：**https://makersuite.google.com/app/apikey**
2. 点击 **"Create API Key"** 或复制现有的 API Key
3. 复制 API Key（一长串文本）

> 💡 **免费额度**：每月 1500 次请求，足够测试使用

### 第 2 步：配置 API Key

在项目根目录创建 `.env` 文件：

```bash
# 一条命令创建 .env 文件
cat > .env << 'EOF'
GEMINI_API_KEY=your_api_key_here
EOF
```

然后编辑 `.env`，替换 `your_api_key_here` 为你的实际 API Key：

```bash
# 使用 nano 编辑器
nano .env

# 或使用其他编辑器
# 编辑器中找到这一行：
# GEMINI_API_KEY=your_api_key_here
# 替换为：
# GEMINI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 第 3 步：验证配置

```bash
# 测试脚本是否能找到 API Key
python3 test_pipeline_vlm.py --samples 1
```

如果看到进度条而不是错误，说明配置成功！✅

## 📖 使用方法

### 最简单的用法

```bash
# 运行默认 10 张照片的完整 VLM 分析
python3 test_pipeline_vlm.py
```

### 常用命令

```bash
# 运行 20 张照片
python3 test_pipeline_vlm.py --samples 20

# 使用固定随机种子（可重现结果）
python3 test_pipeline_vlm.py --samples 10 --seed 42

# 仅分析 5 张照片，不保存缓存
python3 test_pipeline_vlm.py --samples 5 --no-save
```

## ⏱️ 大概需要多长时间？

| 步骤 | 时间 |
|------|------|
| 加载 + 采样 + 人脸识别 | 5-10 秒 |
| VLM 分析 10 张照片 | 2-5 分钟 |
| **总计** | **2-6 分钟** |

VLM 分析是通过网络调用 Google Gemini API，速度取决于网络连接。

## 📊 会输出什么结果？

脚本完成后，你会看到：

1. **进度信息** - 每个步骤的完成状态
2. **人脸识别结果** - 识别出的人物及出现次数
3. **VLM 分析统计** - 成功/失败的照片数
4. **详细摘要** - 每张照片的 VLM 分析内容预览

示例输出：
```
✅ 照片处理
  • 加载: 63 张
  • 采样: 10 张
  • 有人脸: 3 张

👤 人脸识别
  • 检测人脸: 8 张
  • 识别人物: 4 个
  • 主角: person_0（3 次）

🤖 VLM分析
  • 成功: 9 张
  • 失败: 1 张
  • 成功率: 90.0%
```

## 💾 结果保存位置

所有结果都保存在 `cache/` 目录：

```
cache/
├── face_db.json                    # 人脸库
├── Vyoyo.json                      # VLM 分析缓存
├── test_pipeline_vlm_result.json   # 本次测试摘要
├── jpeg_images/                    # JPEG 图片
├── compressed_images/              # 压缩图片
└── boxed_images/                   # 带人脸框的图片
```

## 🔑 API Key 安全提示

✅ **安全的做法：**
- 将 `.env` 文件添加到 `.gitignore`（不要上传到 GitHub）
- 使用个人 API Key，不要分享给他人
- 定期检查 API 使用情况

❌ **危险的做法：**
- 不要把 API Key 硬编码在代码中
- 不要把 `.env` 文件上传到公开仓库
- 不要在聊天中分享 API Key

## 🔄 工作流建议

### 第一次使用

```bash
# 1. 先用 1 张照片测试
python3 test_pipeline_vlm.py --samples 1

# 2. 如果成功，尝试 10 张
python3 test_pipeline_vlm.py --samples 10

# 3. 检查结果
cat cache/test_pipeline_vlm_result.json
```

### 开发和调试

```bash
# 使用固定种子，便于比较结果
python3 test_pipeline_vlm.py --samples 5 --seed 123

# 多次运行时使用相同的 5 张照片，便于对比
```

### 批量处理

```bash
# 处理所有 63 张照片（需要 30+ 分钟）
python3 test_pipeline_vlm.py --samples 63
```

## 🆘 常见问题

### Q: API Key 是什么样的？
A: 通常是这样：`gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

### Q: 免费额度够用吗？
A: 足够！每月 1500 次请求可以处理 150+ 张照片。

### Q: 忘记设置 --use-cache 了怎么办？
A: 不用担心，下次运行时脚本会自动检测缓存。

### Q: 能否删除缓存重新分析？
A: 可以，删除 `cache/Vyoyo.json` 后重新运行脚本。

### Q: 脚本挂起了怎么办？
A: 按 `Ctrl+C` 中断，检查网络连接后重试。

## 🚀 下一步

### 运行 VLM 分析

```bash
# 配置完 API Key 后，立即运行
python3 test_pipeline_vlm.py

# 或用参数自定义
python3 test_pipeline_vlm.py --samples 10 --seed 42
```

### 继续主流程

```bash
# VLM 分析完成后，可以进行事件提取和关系推断
python3 main.py --photos data/raw --max-photos 10 --use-cache
```

## 📚 更多文档

- **详细使用指南**：`VLM_PIPELINE_GUIDE.md`
- **脚本源代码**：`test_pipeline_vlm.py`
- **人脸识别文档**：`TEST_PIPELINE_README.md`
- **项目配置**：`config.py`

## 📞 获取帮助

如果遇到问题：

1. **检查 API Key 格式** - 是否有多余空格或换行符？
2. **检查网络连接** - VLM 分析需要访问 Google API
3. **查看错误信息** - 脚本会给出详细的错误描述
4. **查阅文档** - `VLM_PIPELINE_GUIDE.md` 的故障排除部分

---

**已准备完毕！现在只需配置 API Key，即可开始 VLM 分析！** 🎉
