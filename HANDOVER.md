# 记忆工程项目 - 问题修复交接文档

**日期**: 2026-03-06
**项目路径**: `/Users/bj/Downloads/Users 2/ouzhihao/memory_engineering`

---

## 一、沟通内容与问题发现

### 1.1 最后修改查询
- **问题**: "最后一个修改是什么"
- **操作**: 查看 `config.py` 文件，发现最后修改的是高德地图编码功能
- **结论**: 高德地图编码功能正常工作，能正确将 GPS 坐标转换为地址（如北京海淀区）

### 1.2 测试运行（10张照片）
- **操作**: 运行完整项目测试
- **输出**: 事件 8个、关系 0 个、用户画像有数据
- **文件**:
  - `output/memory_output.json`
  - `output/memory_detailed.md`
  - `cache/vlm_results.json`

### 1.3 发现问题：person_0 不是主角
- **现象**: 人脸库中 person_1 显示为"主角"，person_0 显示为"未知"
- **原因**: `reorder_protagonist()` 只重排 ID，不更新名字
- **修复**: 在 `reorder_protagonist()` 末尾添加 `self.update_names()` 调用

**修改文件**: `services/face_recognition.py`
```python
# 修改位置：reorder_protagonist 方法末尾
# 添加：
# 5. 更新名称（主角标记）
self.update_names()

# 6. 保存
self._save_face_db()
```

### 1.4 VLM 输出混乱问题
- **现象**: summary 中"【主角】"和"person_0"混用
- **原因**: Prompt 指令矛盾（"用 person_0" vs "用【主角】"）
- **修复**: 修改 `services/vlm_analyzer.py`，统一 prompt 指令

**修改内容**:
```python
# 修改 _create_prompt 方法
# 区分两种情况：
# 1. 主角在照片中：使用"【主角】"指代主角（{protagonist}）
# 2. 主角不在照片中：使用"【主角】"指代拍摄者
```

### 1.5 清空缓存重新测试
- **操作**: 清空 `cache/face_db.json` 和 `cache/vlm_results.json`
- **结果**: 识别了 5 个全新人物（之前是 16 个缓存人物）
- **问题**: person_0 和 person_1 都被标记为"主角"（都是2次）

**这是 `update_names()` 的 bug**：当多个人物有相同最大次数时都会被标记为主角。

---

## 二、代码修改记录

### 2.1 文件：`services/face_recognition.py`

#### 修改 1：`reorder_protagonist()` 方法
**位置**: 第 251-297 行

**修改前**:
```python
# 5. 保存
self._save_face_db()

# 打印映射结果
print(f"  ID重排映射: {id_mapping}")

return id_mapping
```

**修改后**:
```python
# 5. 更新名称（主角标记）
self.update_names()

# 6. 保存
self._save_face_db()

# 打印映射结果
print(f"  ID重排映射: {id_mapping}")

return id_mapping
```

**目的**: 重排 ID 后同步更新人物名称，确保 person_0 被标记为"主角"

---

### 2.2 文件：`services/vlm_analyzer.py`

#### 修改 1：`_create_prompt()` 方法
**位置**: 第 76-208 行

**修改前**: Prompt 中指令矛盾，同时要求"用 person_0"和"用【主角】"

**修改后**: 明确区分两种场景的 prompt

```python
# 主角在照片中
if protagonist_in_photo:
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
```

**目的**: 统一 VLM 输出，避免"【主角】"和"person_0"混用

---

## 三、问题发现总结

### 3.1 人脸识别问题

| 问题 | 详情 |
|------|------|
| **相似度过低** | 同一人的相似度只有 0.33（阈值 0.70） |
| **模型** | Facenet512 在角度、光照差异大的场景下不够鲁棒 |
| **阈值** | 0.70 可能过高 |

### 3.2 人脸框画错位置

| 照片 | 问题 |
|------|------|
| IMG_5205 | 框画在了背景区域，而不是人脸上 |
| **原因** | DeepFace 返回的 bbox 坐标 (2235, 4584) 在图片下方 |
| **影响** | 用户看不到框，以为没有检测到人脸 |

### 3.3 多个"主角"问题

| 问题 | 原因 |
|------|------|
| person_0 和 person_1 都被标记为"主角" | 两者都是 2 次出现 |
| **代码 bug** | `update_names()` 没有处理"相同最大次数"的情况 |

### 3.4 PNG 文件路径问题

| 问题 | 详情 |
|------|------|
| 4 张 PNG 文件人脸识别失败 | DeepFace 不支持中文路径 |
| 错误信息: `Input image must not have non-english characters` |

---

## 四、未修复的问题

### 4.1 人脸识别阈值过高
- **当前阈值**: 0.70
- **建议**: 降到 0.50-0.60

### 4.2 人脸框位置错误
- **原因**: DeepFace bbox 坐标不准确
- **影响**: 框画在错误位置

### 4.3 多个"主角"的 bug
- **文件**: `services/face_recognition.py` 的 `update_names()` 方法
- **问题**: 当多个人物有相同最大次数时，都被标记为主角
- **修复方案**: 只保留第一个最大值作为主角

---

## 五、待办事项

| 优先级 | 任务 | 说明 |
|--------|------|------|
| 高 | 修复 `update_names()` bug | 只保留一个主角 |
| 中 | 降低人脸识别阈值 | 从 0.70 降到 0.50 |
| 中 | 修复 PNG 文件路径问题 | 复制到临时目录 |
| 低 | 考虑更换人脸模型 | VGG-Face 或 ArcFace |
| 低 | 修复 bbox 坐标问题 | 确保框画在正确位置 |

---

## 六、输出文件位置

| 文件 | 路径 |
|------|------|
| JSON 输出 | `output/memory_output.json` |
| 详细报告 | `output/memory_detailed.md` |
| VLM 缓存 | `cache/vlm_results.json` |
| 人脸数据库 | `cache/face_db.json` |
| 带框图片 | `cache/boxed_images/` |

---

## 七、修改的代码位置

### `services/face_recognition.py`
- **行数**: 约 290-297
- **修改**: 添加 `self.update_names()` 调用

### `services/vlm_analyzer.py`
- **行数**: 约 100-135
- **修改**: 重写人物说明部分的 prompt

---

## 八、测试命令

### 运行项目
```bash
cd "/Users/bj/Downloads/Users 2/ouzhihao/memory_engineering"
python3 main.py --photos "/Users/bj/Downloads/测试1/测试2/11" --max-photos 10
```

### 清空缓存
```bash
rm -f cache/face_db.json cache/vlm_results.json
```

### 测试高德地图
```bash
python3 -c "
import urllib.request
import json
lng, lat = 116.397428, 39.90923
api_key = '263f3f40b5ac8921c2a98616ffa96201'
url = f'https://restapi.amap.com/v3/geocode/regeo?key={api_key}&location={lng},{lat}&extensions=base'
print('北京天安门:', urllib.request.urlopen(url, timeout=5).read().decode())
"
```
