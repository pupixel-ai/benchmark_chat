# 记忆工程数据 Schema 规范

## 1. 本数据规范

本文档是当前记忆工程各环节的标准、真实数据结构 Schema 规范。该数据规范原则上不会改变，且有且只有一个本文档一个判断标准，在记忆工程的各版本迭代，以及上下游的使用过程中都保持不变。

---

## 2. 适用范围

本文档规范的数据格式用于记忆工程的版本迭代和上下游调用。其中数据源主要来自记忆工程以下阶段：

1. 照片加载与预处理
2. 人脸识别与人物库
3. VLM 结果
4. `MemoryState` 及其内部中间结构
5. 事件、关系、圈层、画像
6. 主链路正式输出与主要调试 artifact

核心使用规范：
1. ID 与 格式的“强一致性”约束
这是系统链路追踪（Traceability）的基础，严禁在任何模块中自定义格式。

严格 ID 前缀：所有新生成的 ID 必须符合约定：用户 user_、照片 photo_、人物 Person_、事件 EVT_。

时间戳归一化：
运行时：必须保持为 datetime 对象。
序列化/落盘：必须转换为 ISO 字符串格式。

地理位置：统一使用 {"latitude": float, "longitude": float} 或 {"lat": float, "lng": float}，严禁使用格式不稳定的地名字符串作为唯一地理标识。

2. 置信度与空值的“显式处理”原则
为了避免下游 Agent 在进行逻辑推理时产生歧义：
数值标尺：confidence（置信度）与 intimacy_score（亲密度）必须严格限制在 0.0 ~ 1.0 之间。
语义化空值：单值缺失：使用 null。集合缺失：使用 []。
关键区别：在字段存在但无证据时，不得删除该字段，应设为 null 以示“已扫描但无结论”。

3. 记忆工程（Memory Engineering）的原子化约束
VLM 分析的定位：vlm_analysis 是单张照片的视觉理解结论，不具备跨照片稳定性。
关系发现的必经之路：严禁直接从 Event 跳到 Relationship。必须先建立 RelationshipDossier（关系档案）进行证据聚合，再输出最终的关系标签。
主角判定：系统主角并非单纯由人脸出现次数决定，必须通过 primary_decision 模块进行最终裁决，且需支持 photographer_mode（主角作为拍摄者，不出现在画面中）。

4. 故事 Agent（Story Agent）的调用与审计约束
可追溯证据链（Traceability）：所有生成的画像标签（TagObject）必须携带 evidence（包含 photo_ids, event_ids 等）。
对于高风险或需要审计的字段，必须保留 profile_fact_decisions 中的 tool_trace 和 reasoning。

---

## 3. 全局约定


### 3.1 ID 约定
- `user_id`: `user_{NNNNNN}`，例如 `user_000002`
- `photo_id`: `photo_{NNN}`，例如 `photo_001`
- `person_id`: `Person_{NNN}`，例如 `Person_001`
- `event_id`: `EVT_{NNN}`，例如 `EVT_001`

### 3.2 时空约定

- 运行时 `Photo.timestamp` 是 `datetime`
- 从用户数据集进入记忆工程到 VLM 缓存、JSON artifact、最终输出时，时间统一使用 ISO 字符串
- 基于经纬度的地理位置数据，统一使用包含经纬度的对象格式，例如：`{"latitude": 39.9042, "longitude": 116.4074}`（浮点数），或在具体数据结构中明确分离为 `latitude` 和 `longitude` 字段。

### 3.3 置信度约定

- `confidence` 统一使用 `0.0 ~ 1.0` 的浮点数
- `intimacy_score` 也统一使用 `0.0 ~ 1.0`

### 3.4 空值约定

- 缺失的单值字段优先使用 `null`
- 缺失的集合字段优先使用 `[]`
- 字段存在但暂无证据，不等于字段不存在

### 3.5 运行时对象与落盘对象可能不同

最重要的差异有两类：

- 运行时 dataclass 在落盘后会变成普通 JSON 对象
- `structured_profile` 中的 `evidence` 是裁剪过的 traceable evidence，不等于字段判定时使用的完整 evidence

---

## 4. 主链路总览

当前主链路正式顺序如下：

```text
Photo
  -> face_db + photo.faces
  -> vlm_results
  -> MemoryState
  -> screening
  -> primary_decision
  -> events
  -> relationship_dossiers
  -> relationships
  -> group_artifacts
  -> profile_context
  -> structured_profile
  -> profile_fact_decisions
  -> final output artifacts
```

---

## 5. 阶段 1：Photo 运行时对象

`Photo` 是主链路最上游的运行时对象。

```python
Photo = {
  "photo_id": str,
  "filename": str,
  "path": str,
  "timestamp": datetime,
  "location": {
    "lat": float,
    "lng": float,
    "name": str,
  } | {},
  "source_hash": str | None,
  "original_path": str | None,
  "compressed_path": str | None,
  "boxed_path": str | None,
  "face_image_hash": str | None,
  "primary_person_id": str | None,
  "processing_errors": dict[str, str],
  "faces": list[FaceInPhoto],
  "vlm_analysis": dict | None,
}
```

字段说明：

- `path`: 当前可处理图路径；HEIC 转换后会被更新为 JPEG 工作图
- `original_path`: 原始文件路径；仅在发生转换时保留原始路径
- `compressed_path`: VLM 使用的压缩图路径
- `boxed_path`: 带框图路径，VLM 优先读这个
- `faces`: 当前照片的人脸识别结果
- `vlm_analysis`: 当前照片的 VP1 结果

---

## 6. 阶段 2：photo.faces[] 的真实结构

当前人脸识别中，`photo.faces[]` 规范如下：

```python
FaceInPhoto = {
  "face_id": str,
  "person_id": str,
  "score": float,
  "similarity": float,
  "faiss_id": int,
  "bbox": list[int],
  "bbox_xywh": {
    "x": int,
    "y": int,
    "w": int,
    "h": int,
  },
  "kps": list,
  "quality_score": float,
  "quality_flags": list[str],
  "match_decision": str,
  "match_reason": str,
  "pose_yaw": float | None,
  "pose_pitch": float | None,
  "pose_roll": float | None,
  "pose_bucket": str | None,
  "eye_visibility_ratio": float | None,
  "landmark_detected": bool | None,
  "landmark_source": str | None,
}
```

补充说明：

- `bbox` 是映射回原图后的整数坐标框，固定为 `[x1, y1, x2, y2]`
- `kps` 当前来自检测引擎的 5 点关键点，元素通常为 `[x, y]`
- `score` 表示检测模型对“这是一张有效人脸框”的置信度
- `similarity` 表示该脸与当前最佳候选人物的相似度
- `quality_score` 用于衡量这张脸是否适合做稳定匹配，不等于检测分数
- `match_reason` 是当前归人决策的简短解释，方便排查误识别
- `match_decision` 当前正式取值包含：
  - `new_person`
  - `strong_match`
  - `profile_rescue_match`
  - `gray_match`
  - `new_person_from_ambiguity`
  - `cluster_merge_match`（仅在二次 cluster merge 后出现）
- `pose_bucket` 当前正式取值包含：
  - `frontal`
  - `left_profile`
  - `right_profile`
  - `unknown`

### 6.1 备注：FaceInPhoto 的可选扩展字段

```python
FaceInPhotoOptional = {
  "cluster_merge_reason": str | None,
}
```

说明：

- `cluster_merge_reason` 只在二次聚类合并后出现，不是每张脸都存在
- 出现该字段时，`match_decision` 可能被改写为 `cluster_merge_match`

---

## 7. 阶段 3：face_db 运行时结构

主链路里传给 `memory_pipeline` 的 `face_db` 是一个 `Dict[str, Person]`。

这里的 `Person` 是运行时缩略对象，不等于 `face_recognition_output.json` 中的 `persons[]`。

```python
Person = {
  "person_id": str,
  "name": str,
  "features": list,
  "photo_count": int,
  "first_seen": datetime | None,
  "last_seen": datetime | None,
  "avg_confidence": float,
  "avg_quality": float,
  "high_quality_face_count": int,
}
```

说明：

- `features` 是运行时 embedding 容器，不直接进入正式 JSON artifact
- `avg_confidence` 对应平均检测分数，不等于平均相似度
- 主角正式判定在后续 `primary_decision`

### 7.1 阶段 3.1：`face_recognition_output.json` 原始落盘结构

当前人脸识别阶段的原始落盘 artifact 是 `cache/face_recognition_output.json`，其正式结构如下：

```python
FaceRecognitionOutput = {
  "engine": FaceEngineMeta,
  "primary_person_id": str | None,
  "metrics": {
    "total_images": int,
    "total_faces": int,
    "total_persons": int,
  },
  "cache_metrics": {
    "total_images": int,
    "total_persons": int,
    "indexed_faces": int,
  },
  "persons": list[FacePersonArtifact],
  "images": list[FaceImageArtifact],
}

FaceEngineMeta = {
  "src_path": str,
  "model_name": str,
  "providers": list[str],
  "max_side": int,
  "det_threshold": float,
  "sim_threshold": float,
  "strong_threshold": float,
  "weak_threshold": float,
  "margin_threshold": float,
  "match_top_k": int,
  "task_version": str,
  "landmark_source": str,
  "index_path": str,
}

FacePersonArtifact = {
  "person_id": str,
  "representative_face_id": str | None,
  "photo_count": int,
  "face_count": int,
  "first_seen": str | None,
  "last_seen": str | None,
  "avg_score": float,
  "avg_similarity": float,
  "avg_quality": float,
  "high_quality_face_count": int,
  "sample_photo_ids": list[str],
  "label": str,
  "pose_counts": dict[str, int],
}

FaceImageArtifact = {
  "image_hash": str,
  "photo_id": str,
  "filename": str,
  "path": str,
  "source_hash": str | None,
  "timestamp": str,   # ISO
  "location": {
    "lat": float,
    "lng": float,
    "name": str,
  } | {},
  "width": int,
  "height": int,
  "faces": list[FaceInPhoto | FaceInPhotoOptional],
  "detection_seconds": float,
  "embedding_seconds": float,
}
```

说明：

- 这里的 `persons[]` 是正式聚合 artifact，不是运行时 dataclass `Person`
- `engine` 记录本次识别所用模型、阈值和索引配置，是复盘和对账时的入口
- `avg_score` 是人物聚合后的平均检测分数
- `avg_similarity` 是人物聚合后的平均最佳匹配相似度
- `representative_face_id` 是当前人物的代表脸 `face_id`
- `sample_photo_ids` 最多保留 10 个样本 `photo_id`
- `label` 当前正式取值为 `Primary` 或 `Person`
- `pose_counts` 的 key 当前与 `pose_bucket` 候选值保持一致
- 这里的 `images[]` 仍使用 `photo_id`，还没有切换成前端展示层的 `image_id`
- `photo_count` 是“出现过的不同照片数”，`face_count` 是“累计识别到的人脸次数”
- `image_hash` 是工作图内容哈希，用于缓存命中和去重追踪
- `path` 是本轮识别真正读取的工作图路径，不一定等于原始上传路径
- `width / height` 用于解释 `bbox` 坐标所在的图像尺寸
- `detection_seconds / embedding_seconds` 是这张图的阶段耗时，主要用于性能诊断

### 7.2 阶段 3.2：任务结果中的 `result.face_recognition`

任务 API / `result.json` 中的人脸识别返回结构，在原始 `FaceRecognitionOutput` 基础上，进一步归一化为前端与工作台消费结构：

```python
TaskFaceRecognitionPayload = {
  "engine": FaceEngineMeta,
  "primary_person_id": str | None,
  "metrics": {
    "total_images": int,
    "total_faces": int,
    "total_persons": int,
  },
  "cache_metrics": {
    "total_images": int,
    "total_persons": int,
    "indexed_faces": int,
  },
  "persons": list[FacePersonArtifact],
  "images": list[FaceImageEntry],
  "person_groups": list[PersonGroupEntry],
  "failed_images": list[FailureItem],
}

FaceImageEntry = {
  "image_id": str,
  "filename": str,
  "source_hash": str | None,
  "timestamp": str | None,
  "status": str,
  "detection_seconds": float,
  "embedding_seconds": float,
  "original_image_url": str | None,
  "display_image_url": str | None,
  "boxed_image_url": str | None,
  "compressed_image_url": str | None,
  "location": {
    "lat": float,
    "lng": float,
    "name": str,
  } | {} | None,
  "face_count": int,
  "faces": list[FaceItemForTask],
  "failures": list[FailureItem],
}

FaceItemForTask = {
  "face_id": str,
  "person_id": str,
  "score": float,
  "similarity": float,
  "faiss_id": int,
  "bbox": list[int],
  "bbox_xywh": {
    "x": int,
    "y": int,
    "w": int,
    "h": int,
  },
  "kps": list,
  "quality_score": float,
  "quality_flags": list[str],
  "match_decision": str,
  "match_reason": str,
  "pose_yaw": float | None,
  "pose_pitch": float | None,
  "pose_roll": float | None,
  "pose_bucket": str | None,
  "eye_visibility_ratio": float | None,
  "landmark_detected": bool | None,
  "landmark_source": str | None,
  "cluster_merge_reason": str | None,
  "image_id": str,
  "source_hash": str | None,
  "boxed_image_url": str | None,
}

PersonGroupEntry = {
  "person_id": str,
  "is_primary": bool,
  "photo_count": int,
  "face_count": int,
  "avg_score": float,
  "avg_quality": float,
  "high_quality_face_count": int,
  "avatar_url": str | None,
  "images": list[PersonGroupImage],
}

PersonGroupImage = {
  "image_id": str,
  "filename": str,
  "timestamp": str | None,
  "display_image_url": str | None,
  "boxed_image_url": str | None,
  "source_hash": str | None,
  "face_id": str,
  "score": float,
  "similarity": float,
  "quality_score": float,
  "quality_flags": list[str],
  "match_decision": str | None,
  "match_reason": str | None,
  "pose_yaw": float | None,
  "pose_pitch": float | None,
  "pose_roll": float | None,
  "pose_bucket": str | None,
  "eye_visibility_ratio": float | None,
  "landmark_detected": bool | None,
  "landmark_source": str | None,
}

FailureItem = {
  "image_id": str,
  "filename": str,
  "path": str,
  "step": str,
  "error": str,
}
```

说明：

- 这里的 `images[]` 是前端消费层结构，不再等于原始 `FaceImageArtifact`
- `images[]` 以图片为中心组织，适合做图片浏览、逐图检查和失败排查
- `image_id` 是任务返回层对 `photo_id` 的展示别名，语义仍然是一张图
- `status` 用于区分该图是否进入本轮识别；核心使用时先关注 `processed / skipped`
- `original_image_url / display_image_url / boxed_image_url / compressed_image_url` 是任务工作目录中文件的可访问 URL，不是源文件路径
- `original_image_url` 更接近原始工作图，`display_image_url` 优先给前端展示，`boxed_image_url` 是画框图，`compressed_image_url` 是 VLM 用压缩图
- `faces[]` 保存该图下的完整人脸明细，是逐脸排查时的主入口
- `FaceItemForTask` 在原始 face 字段外，额外补了 `image_id / source_hash / boxed_image_url` 这类任务关联信息
- `failures[]` 记录该图在加载、识别或前处理阶段的错误，不等于整任务失败
- `person_groups[]` 是面向人物浏览体验的聚合层，不是原始识别层
- `person_groups[]` 以人物为中心组织，适合看“某个人出现在哪些图里”
- `avatar_url` 来自该人物当前最佳分数的人脸裁剪图
- `person_groups[].images[]` 以“每人每图最多一条”去重，不等于该图中该人物的全部 face 明细

### 7.3 阶段 3.3：`face_report` 摘要结构

任务结果中的 `face_report` 是面向展示与运营摘要的派生结构，不等于原始识别 artifact：

```python
FaceReport = {
  "status": str,
  "generated_at": str,   # ISO
  "primary_person_id": str | None,
  "total_images": int,
  "total_faces": int,
  "total_persons": int,
  "failed_images": int,
  "ambiguous_faces": int,
  "low_quality_faces": int,
  "new_person_from_ambiguity": int,
  "no_face_images": list[{
    "image_id": str,
    "filename": str,
  }],
  "failed_items": list[{
    "image_id": str,
    "filename": str,
    "step": str,
    "error": str,
  }],
  "engine": {
    "model_name": str | None,
    "providers": list[str],
  },
  "timings": {
    "detection_seconds": float,
    "embedding_seconds": float,
    "total_seconds": float,
    "average_image_seconds": float,
  },
  "processing": {
    "original_uploads_preserved": bool,
    "preview_format": str,
    "boxed_format": str,
    "recognition_input": str,
  },
  "precision_enhancements": list[str],
  "score_guide": {
    "detection_score": str,
    "similarity": str,
  },
  "persons": list[{
    "person_id": str,
    "is_primary": bool,
    "photo_count": int,
    "face_count": int,
    "avg_score": float,
    "avg_quality": float,
    "high_quality_face_count": int,
  }],
}
```

说明：

- `face_report` 是摘要视图，方便快速判断本轮识别质量，不应替代原始 `persons[] / images[]`
- `ambiguous_faces` 统计当前落入灰区或新建歧义的人脸数
- `low_quality_faces` 统计质量分偏低的人脸数，用于提醒后续人工抽检
- `new_person_from_ambiguity` 单独统计“有候选但仍新建人物”的情况，方便观察聚类是否过碎
- `persons[]` 只保留人物级摘要，不再重复图片级细节

---

## 8. 阶段 4：VLM 结果外层 envelope

`VLMAnalyzer` 当前主链路写出的 `vlm_results[]` 外层结构如下：

```python
VLMResult = {
  "photo_id": str,
  "filename": str,
  "timestamp": str,   # ISO
  "location": {
    "lat": float,
    "lng": float,
    "name": str,
  } | {},
  "vlm_analysis": VLMAnalysis,
}
```

这是 VLM 的正式落盘结构。

说明：

- `photo_id` 与上游 `Photo.photo_id` 一一对应，是 VLM 结果回挂到原图的主键
- `filename` 主要用于人工排查、日志与调试展示，不参与主链路推理
- `timestamp` 是从上游 `Photo.timestamp` 序列化后的字符串版本
- `location` 是当前照片在进入 VLM 前已经解析好的地理信息，不是 VLM 重新生成的位置结论
- `vlm_analysis` 是“单张照片的一次结构化视觉理解结果”，不是跨照片聚合结论

---

## 9. VLMAnalysis 正式结构

正式写出的 `vlm_analysis` 结构如下：

```python
VLMAnalysis = {
  "summary": str,
  "people": list[VLMPerson],
  "relations": list[VLMRelation],
  "scene": {
    "environment_details": list[str],
    "location_detected": str,
    "location_type": str,
  },
  "event": {
    "activity": str,
    "social_context": str,
    "mood": str,
    "story_hints": list[str],
  },
  "details": list[str],
}
```

其中：

```python
VLMPerson = {
  "person_id": str,
  "appearance": str,
  "clothing": str,
  "activity": str,
  "interaction": str,
  "contact_type": str,
  "expression": str,
}

VLMRelation = {
  "subject": str,
  "relation": str,
  "object": str,
}
```

说明：

- `summary` 是这张图的最高层摘要，后续会被 LP1/LP3 直接读取
- `people[]` 是逐人物观察层，要求尽量绑定 `person_id`，但它本身不是正式关系输出
- `appearance / clothing / activity / interaction / expression` 都是单图观察字段，默认不具备跨时间稳定性
- `contact_type` 是当前比自由文本 `interaction` 更稳定的接触枚举信号，后续代码侧会更优先消费它
- `relations[]` 只是视觉层的原始主谓宾关系，不等于 LP2 的正式 `Relationship`
- `scene` 偏环境与地点线索，`event` 偏活动与社交氛围线索，两者共同组成 LP1 的事件输入

### 9.1 备注：当前允许读取但不保证写出的可选字段

以下字段在部分读取逻辑中会被消费，但不是当前 `VLMAnalyzer.add_result()` 的正式写出字段：

- `vlm_analysis.ocr_hits`
- `vlm_analysis.brands`
- `vlm_analysis.place_candidates`
- 外层 `face_person_ids`
- 外层 `media_kind`
- 外层 `is_reference_like`

这些字段当前只能视为“读时兼容的可选扩展字段”，不能当作 VP1 正式 contract。

---

## 10. 阶段 6：MemoryState

`MemoryState` 是当前主链路在 LP1/LP2/LP3 之间传递的统一运行时容器。

```python
MemoryState = {
  "photos": list[Photo],
  "face_db": dict[str, Person],
  "vlm_results": list[VLMResult],
  "screening": dict[str, PersonScreening],
  "primary_decision": dict | None,
  "primary_reflection": dict | None,
  "events": list[Event],
  "relationships": list[Relationship],
  "relationship_dossiers": list[RelationshipDossier],
  "groups": list[GroupArtifact],
  "profile_context": dict | None,
}
```

说明：

- `MemoryState` 是 LP1、LP2、LP3 共用的统一运行时容器，不是正式落盘 artifact
- `photos / face_db / vlm_results` 是进入主链路时就准备好的上游输入
- `screening / primary_decision / events / relationships / relationship_dossiers / groups / profile_context` 都是在主链路内逐步填充出来的中间状态
- 同一个 `MemoryState` 会在 rerun 场景里被继续复用，因此部分字段会被下游回流后重写

---

## 11. 阶段 7：PersonScreening

`screening` 用于在正式关系和画像之前做人物价值筛选。

```python
PersonScreening = {
  "person_id": str,
  "person_kind": str,
  "memory_value": str,
  "screening_refs": list[dict],
  "block_reasons": list[str],
}
```

当前典型取值：

- `person_kind`: `real_person / mediated_person / service_person / incidental_person`
- `memory_value`: `block / low_value / candidate / core`

说明：

- `person_kind` 解决的是“这个人到底是不是值得进入记忆主线的人物类型”
- `memory_value` 解决的是“这个人物在当前相册里值不值得继续进入关系与画像阶段”
- `screening_refs` 是筛选阶段留下来的证据引用，方便后续复盘为什么给了这个筛选等级
- `block_reasons` 是显式阻断原因；一旦命中，后续通常不会再进入正式关系判定

---

## 12. 阶段 8：PrimaryDecision

正式主角判定不是 `face_recognition` 阶段的“出现次数最多的人”，而是 `primary_decision`。

```python
PrimaryDecision = {
  "mode": str,
  "primary_person_id": str | None,
  "confidence": float,
  "evidence": {
    "photo_ids": list[str],
    "event_ids": list[str],
    "person_ids": list[str],
    "group_ids": list[str],
    "feature_names": list[str],
    "supporting_refs": list[dict],
    "contradicting_refs": list[dict],
  },
  "reasoning": str,
}
```

说明：

- `mode` 不是固定只会输出 `person_id`
- 当前主链路中，`photographer_mode` 也是正式模式之一
- `primary_person_id` 只在主角被识别成具体人物时存在；如果走 `photographer_mode`，这里可以是 `null`
- `evidence` 使用的是统一证据 payload 结构，里面的 `photo_ids / event_ids / person_ids / group_ids / feature_names` 都是可追溯锚点
- `supporting_refs` 与 `contradicting_refs` 用于主角判定复盘，不等于最终消费层看到的裁剪版 evidence
- `reasoning` 是主角最终裁决理由，而不是候选排序的完整中间痕迹

---

## 13. 阶段 9：Event 正式结构

LP1 输出的正式事件结构如下：

```python
Event = {
  "event_id": str,
  "date": str,
  "time_range": str,
  "duration": str,
  "title": str,
  "type": str,
  "participants": list[str],
  "location": str,
  "description": str,
  "photo_count": int,
  "confidence": float,
  "reason": str,
  "narrative": str,
  "narrative_synthesis": str,
  "meta_info": dict,
  "objective_fact": dict,
  "social_interaction": dict,
  "social_dynamics": list[dict],
  "lifestyle_tags": list[str],
  "tags": list[str],
  "social_slices": list[dict],
  "persona_evidence": dict,
}
```

补充约定：
- `event_id` 会被下游统一重写为 `EVT_{NNN}`
- `participants` 中若涉及主角，会先做主角别名归一化，再进入后续阶段

说明：

- `date / time_range / duration` 共同描述事件时间信息，其中 `duration` 是摘要文本，不是严格时长秒数
- `title / type / description` 是事件的主标题、类别和文字解释
- `participants` 是事件级人物列表，后续关系和画像都会直接读取它
- `location` 是 LP1 归纳后的事件地点文本，不一定等于单张图的原始 GPS 名称
- `reason` 是 LP1 当前这条事件为何成立的简短判定依据
- `narrative / narrative_synthesis` 都是叙事字段，其中 `narrative_synthesis` 更偏后续复用的浓缩版总结
- `meta_info / objective_fact / social_interaction / social_dynamics / persona_evidence` 都是扩展信息桶，允许部分为空，不要求每个事件都完整填满

---

## 14. 阶段 10：关系证据原始结构

在 `RelationshipDossier` 形成之前，关系证据原始 dict 结构应该如下：

```python
RelationshipEvidenceRaw = {
  "photo_count": int,
  "time_span": str,
  "time_span_days": int,
  "recent_gap_days": int,
  "scenes": list[str],
  "private_scene_ratio": float,
  "dominant_scene_ratio": float,
  "interaction_behavior": list[str],
  "weekend_frequency": str,
  "with_user_only": bool,
  "sample_scenes": list[dict],
  "contact_types": list[str],
  "rela_events": list[dict],
  "monthly_frequency": float,
  "trend_detail": dict,
  "co_appearing_persons": list[dict],
  "anomalies": list[dict],
}
```

说明：

- 这是 LP2 建 dossier 前的原始聚合证据层，仍然是“证据统计”，还不是正式关系结论
- `time_span` 是面向阅读的文本跨度摘要，`time_span_days` 才是后续规则更稳定使用的数值跨度
- `scenes / sample_scenes / contact_types / interaction_behavior` 都来自跨图聚合，不再是单张 VLM 原句
- `private_scene_ratio / dominant_scene_ratio / with_user_only` 用来判断这段关系更像私域关系、功能关系还是群体偶遇
- `rela_events` 是当前人与主角共享的事件切片，后续会被压缩进 `shared_events`
- `trend_detail / anomalies` 用于判断关系是稳定、升温、淡化还是有突发偏差


---

## 15. 阶段 11：RelationshipDossier

当前主链路不是直接从 `Event` 判 `Relationship`，而是先建立 `RelationshipDossier`。

```python
RelationshipDossier = {
  "person_id": str,
  "person_kind": str,
  "memory_value": str,
  "photo_count": int,
  "time_span_days": int,
  "recent_gap_days": int,
  "monthly_frequency": float,
  "scene_profile": {
    "scenes": list[str],
    "private_scene_ratio": float,
    "dominant_scene_ratio": float,
    "with_user_only": bool,
  },
  "interaction_signals": list[str],
  "shared_events": list[dict],
  "trend_detail": dict,
  "co_appearing_persons": list[dict],
  "anomalies": list[dict],
  "evidence_refs": list[dict],
  "block_reasons": list[str],
  "retention_decision": str,
  "retention_reason": str,
  "group_eligible": bool,
  "group_block_reason": str | None,
  "group_weight": float,
  "relationship_result": dict,
  "relationship_reflection": dict,
}
```

说明：

- `RelationshipDossier` 是 LP2 的正式中间层，不是调试边角料
- 当前关系 schema 的核心不是“只给一个关系标签”，而是“先建 dossier，再出最终关系”
- `person_kind / memory_value / block_reasons` 继承自 `screening`，确保 LP2 不绕过上游人物筛选
- `scene_profile` 是把原始 scene 统计压成可判定的场景画像
- `interaction_signals` 是当前人物最重要的互动线索汇总，通常由 `interaction_behavior + contact_types` 合并而来
- `shared_events` 是后续关系判断最稳定的事件锚点
- `retention_decision / retention_reason` 决定这条关系候选是否能进入正式 `relationships[]`
- `group_eligible / group_block_reason / group_weight` 是圈层检测前的准入结果，不等于最终一定进入 group
- `relationship_result` 是这份 dossier 最终归纳出的标准关系结论摘要
- `relationship_reflection` 是关系裁决后的反思结果，用于记录降级、否决和问题信号

---

## 16. 阶段 12：Relationship 正式结构

正式关系结构如下：

```python
Relationship = {
  "person_id": str,
  "relationship_type": str,
  "intimacy_score": float,
  "status": str,
  "confidence": float,
  "reasoning": str,
  "shared_events": list[{
    "event_id": str,
    "date": str,
    "narrative": str,
  }],
  "evidence": dict,
}
```

其中：

- `relationship_type` 正式候选集为：
  - `family`
  - `romantic`
  - `bestie`
  - `close_friend`
  - `friend`
  - `classmate_colleague`
  - `activity_buddy`
  - `acquaintance`

- `status` 正式候选集为：
  - `new`
  - `growing`
  - `stable`
  - `fading`
  - `gone`

说明：

- `person_id` 始终表示“主角与这个 person 的关系”，不需要再额外写主角一侧
- `relationship_type` 是当前 LP2 的正式输出标签，后续画像和下游审计都直接依赖它
- `intimacy_score` 是连续分值，和离散 `relationship_type` 同时存在，用于排序和阈值判断
- `status` 描述的是关系阶段变化，不是关系类型本身
- `shared_events` 是面向消费层保留的轻量事件摘要；完整事件证据仍要回 dossier 或 evidence 看
- `evidence` 是正式关系的 traceable evidence，后续下游 adapter 会从这里抽 `event_id / photo_id / person_id / feature_names`

---

## 17. 阶段 13：GroupArtifact

圈层不直接写进 `relationships[]`，而是单独输出 `group_artifacts[]`。

```python
GroupArtifact = {
  "group_id": str,
  "members": list[str],
  "group_type_candidate": str,
  "confidence": float,
  "strong_evidence_refs": list[dict],
  "reason": str,
}
```

说明：
- 运行时字段名是 `groups`
- 序列化 artifact 字段名是 `group_artifacts`
- `members` 是群组中的 `person_id` 列表，不包含主角自身
- `group_type_candidate` 是当前对圈层类型的候选判断，不是更高层产品化标签体系
- `strong_evidence_refs` 是当前群组成立的强锚点，通常直接回到共享事件
- `reason` 是群组生成原因摘要，便于快速理解这组人为什么被聚成一圈

---

## 18. 阶段 14：ProfileContext

LP3 不直接吃全量原始对象，而是吃整理后的 `profile_context`。

```python
ProfileContext = {
  "primary_person_id": str | None,
  "events": list[Event],
  "relationships": list[Relationship],
  "groups": list[GroupArtifact],
  "vlm_observations": list[VLMObservation],
  "feature_refs": list[dict],
  "social_media_available": bool,
  "resolved_facts": dict,
}
```

其中：

```python
VLMObservation = {
  "photo_id": str | None,
  "timestamp": str | None,
  "summary": str,
  "location": str,
  "activity": str,
  "people": list[str],
  "details": list[str],
  "ocr_hits": list[str],
  "brands": list[str],
  "place_candidates": list[str],
  "face_person_ids": list[str],
  "media_kind": str | None,
  "is_reference_like": bool,
  "subject_role": str,
}
```

说明：

- `VLMObservation` 是 LP3 的观测层，不等于原始输出
- 缺失的扩展字段会在这里被归一化成空列表、`None` 或 `False`
- `events / relationships / groups` 是 LP3 当前直接读取的正式上游事实
- `vlm_observations` 是把单图 VLM 结果压成统一观察结构后的结果，供画像字段按需取证
- `feature_refs` 是轻量上下文引用层，主要提供统计型或归纳型线索，不替代主证据
- `social_media_available` 当前主要作为表达层静默开关使用
- `resolved_facts` 是 LP3 在多 batch 执行时用于承接“已定稿事实”的容器，初始可以为空
- `subject_role` 用来区分“主角在画面里”“主角作为拍摄者视角”“只有其他人出现”三种主体归属状态

---

## 19. 阶段 15：Structured Profile 顶层结构

正式画像输出是 `structured_profile`。

其顶层结构固定如下：

```python
structured_profile = {
  "long_term_facts": ...,
  "short_term_facts": ...,
  "long_term_expression": ...,
  "short_term_expression": ...,
}
```

### 19.1 TagObject

`structured_profile` 的每个叶子字段都不是纯值，而是一个统一的 `TagObject`：

```python
TagObject = {
  "value": any | None,
  "confidence": float,
  "evidence": TraceableProfileEvidence,
  "reasoning": str,
}
```

说明：

- `value` 是字段最终值；当证据不足、主体不清或命中硬阻断时应返回 `null`
- `confidence` 仍然是 `0.0 ~ 1.0` 浮点数，不会在 `structured_profile` 里切成百分制
- `evidence` 是裁剪版可追溯证据，面向消费层和轻量审计
- `reasoning` 是字段级最终结论理由，不等于模型的完整内部思考过程

### 19.2 TraceableProfileEvidence

`structured_profile` 叶子节点里的 `evidence` 是裁剪版：

```python
TraceableProfileEvidence = {
  "photo_ids": list[str],
  "event_ids": list[str],
  "person_ids": list[str],
  "group_ids": list[str],
  "feature_names": list[str],
  "supporting_ref_count": int,
  "contradicting_ref_count": int,
  "constraint_notes": list[str],
  "summary": str,
}
```

注意：

- 这里没有完整 `supporting_refs`
- 这里没有完整 `contradicting_refs`
- 完整版本保存在 `profile_fact_decisions[].draft.evidence` 和 `profile_fact_decisions[].final.evidence`

补充说明：

- `photo_ids / event_ids / person_ids / group_ids / feature_names` 是裁剪后保留下来的锚点集合
- `supporting_ref_count / contradicting_ref_count` 只给数量，不给具体 ref 明细
- `constraint_notes` 用于保留字段级 gate、静默规则和其他约束信息
- `summary` 是这份裁剪 evidence 的简短说明，不是完整证据文本

---

## 20. 阶段 16：Structured Profile 固定字段清单

`structured_profile` 的字段层级固定如下：

```python
structured_profile = {
  "long_term_facts": {
    "identity": {
      "name": TagObject,
      "gender": TagObject,
      "age_range": TagObject,
      "role": TagObject,
      "race": TagObject,
      "nationality": TagObject,
    },
    "social_identity": {
      "education": TagObject,
      "career": TagObject,
      "career_phase": TagObject,
      "professional_dedication": TagObject,
      "language_culture": TagObject,
      "political_preference": TagObject,
    },
    "material": {
      "asset_level": TagObject,
      "spending_style": TagObject,
      "brand_preference": TagObject,
      "income_model": TagObject,
      "signature_items": TagObject,
    },
    "geography": {
      "location_anchors": TagObject,
      "mobility_pattern": TagObject,
      "cross_border": TagObject,
    },
    "time": {
      "life_rhythm": TagObject,
      "event_cycles": TagObject,
      "sleep_pattern": TagObject,
    },
    "relationships": {
      "intimate_partner": TagObject,
      "close_circle_size": TagObject,
      "social_groups": TagObject,
      "pets": TagObject,
      "parenting": TagObject,
      "living_situation": TagObject,
    },
    "hobbies": {
      "interests": TagObject,
      "frequent_activities": TagObject,
      "solo_vs_social": TagObject,
    },
    "physiology": {
      "fitness_level": TagObject,
      "health_conditions": TagObject,
      "diet_mode": TagObject,
    },
  },
  "short_term_facts": {
    "life_events": TagObject,
    "phase_change": TagObject,
    "spending_shift": TagObject,
    "current_displacement": TagObject,
    "recent_habits": TagObject,
    "recent_interests": TagObject,
    "physiological_state": TagObject,
  },
  "long_term_expression": {
    "personality_mbti": TagObject,
    "morality": TagObject,
    "philosophy": TagObject,
    "attitude_style": TagObject,
    "aesthetic_tendency": TagObject,
    "visual_creation_style": TagObject,
  },
  "short_term_expression": {
    "current_mood": TagObject,
    "mental_state": TagObject,
    "motivation_shift": TagObject,
    "stress_signal": TagObject,
    "social_energy": TagObject,
  },
}
```

这份字段树是当前 LP3 的正式输出 contract。

补充说明：

- `long_term_facts` 强调跨时间稳定模式，原则上不能被单次事件直接拉动
- `short_term_facts` 强调近期窗口变化，默认需要和长期基线做区分
- `long_term_expression` 更偏稳定表达风格、价值倾向和审美/行为模式
- `short_term_expression` 更偏近期情绪、状态和社交能量变化

---

## 21. 阶段 17：ProfileFactDecision

字段级判定的完整调试与审计信息在 `profile_fact_decisions[]`。

```python
ProfileFactDecision = {
  "field_key": str,
  "domain_name": str,
  "batch_name": str,
  "field_spec_snapshot": FieldSpec,
  "tool_trace": {
    "evidence_bundle": dict,
    "stats_bundle": dict,
    "ownership_bundle": dict,
    "counter_bundle": dict,
  },
  "draft": FieldDecisionValue,
  "final": FieldDecisionValue,
  "null_reason": str | None,
}
```

其中：

```python
FieldSpec = {
  "field_key": str,
  "risk_level": str,
  "allowed_sources": list[str],
  "strong_evidence": list[str],
  "cot_steps": list[str],
  "owner_resolution_steps": list[str],
  "time_reasoning_steps": list[str],
  "counter_evidence_checks": list[str],
  "weak_evidence": list[str],
  "hard_blocks": list[str],
  "owner_checks": list[str],
  "time_layer_rule": str,
  "null_preferred_when": list[str],
  "reflection_questions": list[str],
  "reflection_rounds": int,
  "requires_social_media": bool,
}

FieldDecisionValue = {
  "value": any | None,
  "confidence": float,
  "evidence": FullProfileEvidence,
  "reasoning": str,
}

FullProfileEvidence = {
  "photo_ids": list[str],
  "event_ids": list[str],
  "person_ids": list[str],
  "group_ids": list[str],
  "feature_names": list[str],
  "supporting_refs": list[dict],
  "contradicting_refs": list[dict],
  "events": list[dict],
  "relationships": list[dict],
  "vlm_observations": list[dict],
  "group_artifacts": list[dict],
  "feature_refs": list[dict],
  "constraint_notes": list[str],
  "summary": str,
}
```

说明：

- `profile_fact_decisions` 才是字段级全量证据正式留痕层
- `structured_profile` 是可消费输出
- `profile_fact_decisions` 是可审计、可复盘、可回流输出
- `field_spec_snapshot` 是当次判定时所用字段规则快照，避免后续规则变更后无法复盘
- `tool_trace` 记录字段判定前的横向工具摘要，不等于最终 evidence
- `draft` 是字段草案，`final` 是经过反思和约束后定稿的结果
- `null_reason` 只在字段最终被打回空值时出现，便于解释为什么没有输出标签
- `FullProfileEvidence` 是完整证据层，包含 refs、本体对象切片和约束说明，比 `structured_profile.evidence` 丰富得多

---

## 22. 阶段 18：run_memory_pipeline() 返回结构

主链路 orchestrator 的正式返回结构如下：

```python
PipelineResult = {
  "events": list[Event],
  "relationships": list[Relationship],
  "structured": structured_profile,
  "report": str,
  "debug": {
    "field_decision_count": int,
    "report_reasoning": dict,
  },
  "consistency": dict,
  "internal_artifacts": {
    "screening": dict[str, PersonScreening],
    "primary_decision": PrimaryDecision,
    "primary_reflection": dict,
    "relationship_dossiers": list[RelationshipDossier],
    "group_artifacts": list[GroupArtifact],
    "profile_fact_decisions": list[ProfileFactDecision],
  },
}
```

说明：

- `structured` 就是正式 `structured_profile`
- `report` 是给画像报告正文预留的位置，当前主链路里通常为空字符串
- `debug.field_decision_count` 用于快速看本轮 LP3 实际判了多少字段
- `consistency` 是轻量跨层一致性检查结果，当前重点看关系层与画像层是否冲突
- `internal_artifacts` 是主链路最重要的中间留痕层，后续 rerun、下游审计和 reflection 都会用到它

---

## 23. 阶段 19：正式输出文件 schema

说明：

- 人脸识别相关原始 artifact、任务 payload 与 `face_report` 已在第 7 节补充定义
- 本节继续描述主链路后半段的最终输出文件

### 23.1 `memory_output.json`

最终总输出为：

```python
FinalOutput = {
  "metadata": {
    "generated_at": str,
    "version": str,
    "total_events": int,
    "total_relationships": int,
    "models": {
      "vlm": str,
      "llm": str,
      "face": str,
    },
  },
  "events": list[dict],          # 字段同 Event，仅由 dataclass 转成普通 JSON
  "relationships": list[dict],   # 字段同 Relationship，仅由 dataclass 转成普通 JSON
  "face_db": dict[str, SerializedFaceSummary],
  "artifacts": dict[str, str | None],
}

SerializedFaceSummary = {
  "name": str,
  "photo_count": int,
  "first_seen": str | None,
  "last_seen": str | None,
  "avg_confidence": float,
}
```

说明：

- `memory_output.json` 是总输出汇总页，不等于每个模块最完整的原始 artifact
- `events / relationships` 只是把 dataclass 转成普通 JSON，不会额外补更多调试细节
- `face_db` 这里只保留人物级摘要，不保留 embedding、face 明细等运行时重对象
- `artifacts` 保存的是其他 artifact 的文件路径，是回查入口而不是正文内容

### 23.2 `relationships*.json`

```python
RelationshipsArtifact = {
  "metadata": {
    "generated_at": str,
    "version": str,
    "primary_person_id": str | None,
    "total_relationships": int,
  },
  "relationships": list[dict],   # 字段同 Relationship，仅由 dataclass 转成普通 JSON
}
```

说明：

- 这是关系层独立调试输出，主要方便单独检查 LP2 最终结果
- `metadata.primary_person_id` 用于明确这批关系是围绕哪个主角生成的
- 该文件不重复携带 dossier、group、profile 等更后续的上下文

### 23.3 通用内部 artifact 包装格式

`relationship_dossiers`、`group_artifacts`、`profile_fact_decisions`、`downstream_audit_report` 等内部 artifact 统一按以下包装格式落盘：

```python
InternalArtifact = {
  "metadata": {
    "generated_at": str,
    "version": str,
    # 其他 artifact 特定 metadata
  },
  "<artifact_name>": payload,
}
```

说明：

- 这层包装的目标是统一落盘形式，而不是改变 payload 本身的数据结构
- `metadata` 至少保证产物生成时间和版本号存在，其余字段按 artifact 类型补充
- 真正的业务对象仍然在 `payload` 里，读取时不要把 `metadata` 和业务字段混在一起

---

## 24. 当前最需要记住的结论

在所有数据规范中应记住：

当前主链路真正固定的数据核心，不是旧版 `UserProfile`，而是：

`Photo / face_db / VLMResult / MemoryState / PrimaryDecision / Event / RelationshipDossier / Relationship / GroupArtifact / structured_profile / profile_fact_decisions`
