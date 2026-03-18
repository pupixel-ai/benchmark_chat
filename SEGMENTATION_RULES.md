# 分层切分与 LLM 交付规则表

## 1. 文档目标

本文档定义一套可落地的分层切分与 LLM 交付方案，用于替代当前“把整批 `vlm_results` 一次性交给 LLM”的实现。

目标：

- 保留原始设计的核心思想：全局理解、证据闭环、人物身份谨慎绑定、跨照片关系推断。
- 支持 `1000+` 张照片级别的数据规模。
- 尽量不丢失信息，不依赖激进抽帧。
- 将“现实世界边界”和“LLM 可读窗口”拆开建模。

非目标：

- 不要求一次性重写所有现有模块。
- 不要求立即替换现有 `memory_module` 的所有下游逻辑。
- 不以最小 token 消耗为唯一目标；优先保证信息保真和可解释性。

---

## 2. 当前代码现状

当前主流程：

1. 逐图做人脸识别
2. 逐图做 VLM 分析
3. 将全部 `vlm_results` 一次性交给 LLM 提取事件
4. 基于全量 `vlm_results` 推断关系
5. 基于 `events + relationships` 生成画像

相关位置：

- `main.py`
- `services/vlm_analyzer.py`
- `services/llm_processor.py`
- `memory_module/sequencing.py`

当前问题：

- `services/llm_processor.py` 的 `extract_events()` 直接吃全部 `vlm_results`。
- 当照片规模上升到几百或上千张时，prompt 体积、响应时延、失败率和成本都会失控。
- 当前 `memory_module/sequencing.py` 已经有 `burst / session / timeline` 的雏形，但尚未成为 LLM 前置分层结构。

---

## 3. 核心设计原则

### 3.1 现实边界和模型窗口分离

- `session` 表示现实世界里的连续活动链。
- `slice` 表示为了适配 LLM 上下文限制而创建的分析窗口。
- 一个 `raw session` 可以对应多个 `session slice`。
- 不允许为了喂 LLM 而强行改变 `session` 身份。

### 3.2 先保守切分，再谨慎合并

- 宁可多切，不可乱并。
- 所有原始边界都要保留。
- 合并只能发生在证据充分的情况下。

### 3.3 压缩重复，不压缩事实

- 不优先使用“少量关键帧代表全部”的激进抽帧策略。
- 优先使用“事实去重、异常高亮、证据引用保留”的无损或近无损压缩。

### 3.4 LLM 只负责语义判断，不负责管理全部原始上下文

- 原始 photo-level facts 先进入 buffer。
- LLM 优先看结构化证据包，而不是直接看无界增长的逐图长文本。

### 3.5 所有高层结论都必须可回溯

- `event`
- `relationship`
- `profile clue`

以上对象都必须能回溯到 `photo_id`、`burst_id`、`session_id` 或明确的事实引用。

---

## 4. 术语与对象

| 对象 | 定义 | 是否代表现实边界 | 是否直接交给 LLM |
|------|------|------------------|------------------|
| `photo` | 单张照片及其 VLM/人脸结果 | 是 | 否，通常不直接大规模交付 |
| `photo fact` | 从单张照片沉淀出的结构化事实 | 否 | 第二阶段可按需回查 |
| `burst` | 高密度连续拍摄簇 | 是，局部现实片段 | 作为 slice 的基础单元 |
| `raw session` | 连续活动链，允许跨微小地点变化 | 是 | 否，不直接裸传 |
| `session slice` | 从 `raw session` 切出的 LLM 分析窗口 | 否 | 是 |
| `event candidate` | 第一阶段 LLM 在单个 slice 上提取出的事件候选 | 否 | 是，供拼接与确认 |
| `global event` | 经过拼接、确认和去重后的事件对象 | 是，语义边界 | 是，供画像和关系使用 |

---

## 5. 数据来源与信号优先级

## 5.1 EXIF / 文件层信号

- `timestamp`
- `GPS / location`
- 文件顺序
- 原始文件哈希

用途：

- 建立时间顺序
- 构建 `burst`
- 构建 `raw session skeleton`

## 5.2 人脸层信号

- `person_id`
- 人脸数量
- 主用户锚点
- 共现关系

用途：

- 人物延续判断
- 关系证据沉淀
- session continuity 修正

## 5.3 VLM 层信号

- `scene.location_detected`
- `scene.environment_description`
- `scene.environment_details`
- `event.activity`
- `event.social_context`
- `event.interaction`
- `summary`
- `details`
- `key_objects`
- OCR / 品牌 / 屏幕 / 票据等高价值线索

用途：

- session 修正
- 变化点检测
- slice 证据包构建
- event candidate 提取

## 5.4 信号优先级

用于 session 判定时，推荐优先级：

1. 时间连续性
2. 空间连续性
3. 人物延续性
4. 场景语义延续性
5. 活动逻辑延续性

说明：

- 时间和空间更接近硬约束。
- 人物、场景、活动更接近软修正。

---

## 6. Photo Fact Buffer 规则

在逐图完成 VLM 后，系统必须先构建 `photo fact buffer`，再进入 `burst / session / slice`。

每张照片至少生成以下字段：

```json
{
  "photo_id": "photo_001",
  "timestamp": "2026-03-18T10:03:21",
  "location": {"name": "XX大厦", "lat": 31.23, "lng": 121.47},
  "person_ids": ["Person_001", "Person_003"],
  "scene_hint": "办公室工位",
  "activity_hint": "办公",
  "social_hint": "同事",
  "summary": "主用户在工位前处理电脑工作",
  "details": ["显示器", "工牌", "笔记本电脑"],
  "key_objects": ["电脑", "会议桌"],
  "rare_clues": ["屏幕文字: Q2 Budget Review"],
  "confidence": {
    "vlm": 0.86
  }
}
```

### 6.1 Buffer 保留规则

- 每张图都要进 buffer。
- 不允许在此阶段删除原始照片事实。
- 即使后面生成聚合结果，也必须保留 `photo_id -> fact` 的可追踪映射。

### 6.2 Rare Clue 规则

满足以下任一条件的事实，必须提升为 `rare clue`：

- 只出现 1 次但信息价值高
- 包含 OCR / 屏幕内容 / 品牌 / 票据 / 地点名称 / 人名
- 表示活动切换、空间切换或人物切换
- 与前后照片事实冲突

---

## 7. Burst 判定规则

## 7.1 定义

`burst` 是高密度连续拍摄簇，主要用于处理同一小段场景下的大量相似照片。

`burst` 不是 `session`，也不是 `event`。

## 7.2 推荐默认阈值

| 规则 | 默认值 |
|------|--------|
| 相邻照片时间差 | `<= 90 秒` |
| burst 总持续时长 | `<= 180 秒` |
| burst 最大照片数 | `<= 30 张` |

## 7.3 建立 burst 的条件

相邻照片满足以下全部条件时，可归入同一 burst：

- 时间差 `<= 90 秒`
- 当前 burst 持续时长未超过 `180 秒`
- 当前 burst 照片数未超过 `30 张`

## 7.4 强制关闭 burst 的条件

出现以下任一情况，必须关闭当前 burst 并新开：

- 相邻照片时间差 `> 90 秒`
- burst 总持续时长 `> 180 秒`
- burst 照片数 `> 30 张`
- EXIF 地点发生明显跳变
- VLM 检测到强烈场景突变
- 人物集合出现断裂式变化

## 7.5 极端情况：500 张连续照片但不属于同一个 burst

如果 500 张在时间地点逻辑上连续，但由于：

- 总时长过长
- 变化频繁
- 已经超出 burst 限制

它们可以属于：

- 多个 `burst`
- 但仍然属于一个 `raw session`

---

## 8. Session 判定规则

## 8.1 定义

`raw session` 表示一条连续的现实活动链。

它允许：

- 微小地点变化
- 活动链上的合理转场
- 相邻场景在现实逻辑上连续

例如：

- 工位 -> 楼下咖啡馆 -> 会议室

可以是 1 个 `raw session`，但未必是 1 个 `event`。

## 8.2 Session 不能只靠 EXIF

推荐策略：

- 第一步：`EXIF-first skeleton`
- 第二步：`VLM buffer correction`

即：

1. 先用时间/GPS/地点生成 session 骨架
2. 再用 VLM 的场景、活动、人物延续性对骨架做合并或切分修正

## 8.3 推荐默认阈值

| 规则 | 默认值 |
|------|--------|
| session 时间 gap 硬阈值 | `> 4 小时` 则强制分开 |
| session 距离硬阈值 | `> 20 km` 且无连续性线索则强制分开 |
| 近距离连续性阈值 | `<= 1.5 km` 视为可连续 |
| 强逻辑转场窗口 | `<= 2 小时` 内允许楼宇内、商圈内、园区内转场 |

## 8.4 Session Skeleton 建立规则

优先在 `burst` 级别而不是 `photo` 级别建立 session。

相邻 `burst` 默认进入同一 `raw session`，当且仅当没有命中硬切分规则。

## 8.5 硬切分规则

相邻 `burst` 满足任一条件时，必须切成不同 `raw session`：

- 时间 gap `> 4 小时`
- 地点跳变 `> 20 km`，且没有旅行/移动链线索
- 日期跨越且活动链缺少连续性
- 人物、场景、活动三者同时发生断裂式变化

## 8.6 软连续性规则

相邻 `burst` 若满足以下多数条件，可保留在同一 `raw session`：

- 时间 gap 较小
- 地点在园区、商圈、楼宇或住宅区内合理移动
- 核心人物集合延续
- 场景功能连续
- 活动链逻辑连续

示例：

- 办公室 -> 楼下咖啡馆 -> 会议室
- 家里客厅 -> 小区楼下 -> 附近超市
- 商场内逛街 -> 餐厅 -> 地库取车

## 8.6.1 Session Continuity 评分表

在未命中硬切分规则时，建议对相邻 `burst` 计算连续性得分：

| 信号 | 条件 | 分值 |
|------|------|------|
| 时间连续 | gap `<= 30 分钟` | `+2` |
| 时间可连续 | gap `> 30 分钟` 且 `<= 2 小时` | `+1` |
| 空间相近 | 同地点名或距离 `<= 1.5 km` | `+2` |
| 空间可解释转场 | 同园区 / 商圈 / 楼宇链 | `+1` |
| 人物延续强 | dominant persons 重叠 `>= 50%` | `+2` |
| 人物延续弱 | 至少 1 个核心人物延续 | `+1` |
| 场景功能连续 | 办公区 -> 咖啡区 -> 会议区等 | `+1` |
| 活动链连续 | 办公 -> 午餐 -> 会议等 | `+1` |
| rare clue 提示新起点 | 机票、酒店、登机口、门牌、票据等 | `-2` |
| 语义断裂 | 人物/场景/活动同时变化 | `-3` |

推荐决策：

- 总分 `>= 3`：保留在同一 `raw session`
- 总分 `1-2`：标记为 `review-needed`，交给 VLM buffer correction
- 总分 `<= 0`：默认切成新 session

## 8.7 VLM Buffer Correction 规则

在 skeleton 建立后，必须做一次语义修正。

允许两种修正：

### Merge Back

EXIF 看起来分开，但 VLM 显示仍是同一活动链，则可合并回同一 `raw session`。

适用条件：

- 时间仍接近
- 地点变化可解释
- 人物延续明显
- 场景/活动逻辑连续

### Split Inside

EXIF 看起来连续，但 VLM 显示已经进入完全不同的活动链，则需要在 session 内再切。

适用条件：

- 场景功能突变
- 核心人物换了一批
- 活动目标明显变化
- rare clue 明确表明新的现实起点

## 8.8 Session 最大长度

`raw session` 不设置固定照片上限。

原因：

- `raw session` 是现实边界，不是 LLM 边界。
- 即使一个 session 包含 500 张照片，也不应仅因模型预算而被强行拆成多个 session。

真正需要控制的是 `session slice` 的大小，而不是 `raw session` 本身。

---

## 9. Session Slice 规则

## 9.1 定义

`session slice` 是从一个 `raw session` 中切出的 LLM 分析窗口。

它不改变 `session` 身份，只是控制 LLM 输入预算。

## 9.2 切 slice 的目的

- 控制 token 体积
- 控制延迟和失败率
- 降低一次性 prompt 风险
- 保留长 session 的整体身份

## 9.3 Slice 构建单位

推荐以 `burst` 为最小组装单元构建 `slice`。

如果没有可靠 `burst`，可以退化到：

- 连续 `photo fact` 块

## 9.4 Slice 预算

推荐默认值：

| 规则 | 默认值 |
|------|--------|
| 软 token 预算 | `6000-8000` |
| 硬 token 预算 | `10000-12000` |
| burst 数建议上限 | `<= 12` |
| burst 数硬上限 | `<= 20` |

## 9.5 Slice 构建规则

生成 `slice` 时遵守以下规则：

- 优先沿变化点切
- 如果没有明显变化点，则按 token 预算切
- 每个 slice 必须包含基础 session 头信息
- 每个 slice 必须保留 overlap

## 9.6 Slice Overlap 规则

推荐默认 overlap：

- `1-2` 个 burst
- 或 `5-10 分钟`
- 取两者中更保守的较大值

目的：

- 避免边界事件被切裂
- 提供跨 slice 对齐依据

## 9.7 长 session 示例

对于一个包含 500 张照片的 `raw session`：

- 保持它仍然是 1 个 `raw session`
- 生成多个 `session slice`

例如：

- `slice_01 = bursts 1-8`
- `slice_02 = bursts 7-14`
- `slice_03 = bursts 13-20`

这是一种窗口化分析，不是现实边界切分。

---

## 10. Slice 交付给 LLM 的数据包

LLM 不应该直接吃“裸 session 元数据”。

应交付 `session-scoped evidence packet`。

## 10.1 Evidence Packet 必备字段

```json
{
  "session_id": "session_003",
  "slice_id": "session_003_slice_02",
  "time_range": {
    "start": "2026-03-18T10:00:00",
    "end": "2026-03-18T12:30:00"
  },
  "location_chain": ["办公室", "楼下咖啡馆", "会议室"],
  "dominant_person_ids": ["Person_001", "Person_003"],
  "burst_ids": ["burst_010", "burst_011", "burst_012"],
  "fact_inventory": [],
  "rare_clues": [],
  "change_points": [],
  "conflicts": [],
  "photo_refs": ["photo_101", "photo_102", "photo_103"]
}
```

## 10.2 Fact Inventory 规则

`fact_inventory` 不是关键帧集合，而是事实集合。

每条事实必须包含：

- `fact_type`
- `value`
- `support_count`
- `photo_ids`
- `confidence`

示例：

```json
{
  "fact_type": "scene_location",
  "value": "会议室",
  "support_count": 23,
  "photo_ids": ["photo_121", "photo_122"]
}
```

## 10.3 Rare Clue 规则

以下事实必须单独保留，不能因为支持次数低而被丢掉：

- OCR
- 品牌
- 屏幕内容
- 票据
- 新人物首次出现
- 场景功能首次切换
- 活动类型首次切换

## 10.4 Change Point 规则

如果检测到以下变化，必须写入 `change_points`：

- 主要人物集合变化
- 场景功能变化
- 活动变化
- 地点链变化
- rare clue 出现

---

## 11. 两阶段 LLM 规则

## 11.1 第一阶段：Slice -> Event Candidate

第一阶段 LLM 输入：

- 单个 `session slice evidence packet`

第一阶段 LLM 输出：

- 一个或多个 `event_candidate`

每个 candidate 必须包含：

- `candidate_id`
- `session_id`
- `slice_id`
- `time_range`
- `support_photo_ids`
- `support_burst_ids`
- `dominant_person_ids`
- `location_hints`
- `activity_hints`
- `rare_clues`
- `confidence`

## 11.2 第二阶段：Candidate Confirmation

第二阶段 LLM 或规则确认输入：

- candidate 自身摘要
- 邻接 candidate 的摘要
- overlap 区域证据
- 相关 `photo-level facts`
- 相关 rare clues

输出关系：

- `same_event`
- `possible_continuation`
- `distinct`

---

## 12. Slice 之间的拼接规则

## 12.1 拼接对象

拼接不是“拼 slice 文本”，而是“拼 `event_candidate`”。

## 12.2 比较范围

只比较相邻 `slice` 的候选，不做全局乱配对。

允许比较：

- `slice_n` 与 `slice_n+1`

不建议直接比较：

- `slice_n` 与 `slice_n+3`

## 12.3 拼接证据

候选之间允许进入 `same_event` 判断的证据：

- overlap 区域存在共享 `photo_ids`
- overlap 区域存在共享 `burst_ids`
- 时间首尾连续
- dominant persons 延续
- location chain 合理延续
- activity / social context 语义延续
- rare clue 不冲突

## 12.4 默认拼接策略

推荐输出三态：

- `same_event`
- `possible_continuation`
- `distinct`

只有当以下条件全部偏强时，才直接 merge：

- overlap 对齐强
- 时间连续性强
- 人物连续性强
- 场景与活动延续强
- 没有关键冲突

如果证据不足，则保留为：

- `possible_continuation`

后续再由第二阶段确认。

## 12.4.1 Candidate Stitching 评分表

用于比较相邻 `slice` 中的两个 `event_candidate`：

| 信号 | 条件 | 分值 |
|------|------|------|
| overlap 对齐强 | 共享 `photo_ids` 或 `burst_ids` | `+3` |
| 时间紧邻 | 首尾 gap `<= 30 分钟` | `+2` |
| 人物延续强 | dominant persons 重叠 `>= 50%` | `+2` |
| 地点延续 | location chain 明显连续 | `+1` |
| 活动延续 | activity / social context 一致或可解释延续 | `+1` |
| rare clue 一致 | 关键线索互相支持 | `+1` |
| rare clue 冲突 | 关键地点/票据/OCR 明显冲突 | `-3` |
| 语义断裂 | 主要人物和活动同时变化 | `-3` |

推荐决策：

- 总分 `>= 5`：标记为 `same_event`
- 总分 `3-4`：标记为 `possible_continuation`
- 总分 `<= 2`：标记为 `distinct`

## 12.5 拼接后保留规则

即使两个 candidate 被合并成一个 `global event`，也必须保留：

- 原始 `candidate_id`
- 原始 `slice_id`
- 原始 `support_photo_ids`

以便后续调试和回滚。

---

## 13. Event 判定规则

`event` 是语义对象，不是时间窗口。

同一个 `raw session` 内可以包含多个 `event`。

例如：

- 工位办公
- 楼下喝咖啡
- 回会议室开会

可以是：

- 1 个 `raw session`
- 3 个 `event`

### 13.1 event 触发变化

在同一个 session 内，出现以下变化时，应考虑切出新的 `event`：

- 活动类型变化
- 核心人物变化
- 场景功能变化
- 互动关系变化
- 叙事目标变化

### 13.2 不应自动切 event 的情况

以下情况本身不构成新 event：

- 连续连拍
- 机位细微变化
- 姿势变化
- 表情变化
- 同一活动内的小步转场

---

## 14. 失败保护与保真策略

## 14.1 禁止激进抽帧

默认不使用“仅保留少数关键帧、丢弃其余照片”的策略。

原因：

- 高价值线索可能只出现 1 张
- OCR / 屏幕 / 票据 / 细小物体容易被漏掉
- 强压缩会破坏证据闭环

## 14.2 推荐替代方案

采用：

- 全量 `photo fact buffer`
- 重复事实聚合
- rare clue 单独高亮
- change point 保留
- 需要时二阶段回查 photo-level facts

## 14.3 可回溯性要求

任意 `global event`、`relationship`、`profile clue` 必须能追溯到：

- `photo_id`
- `burst_id`
- `session_id`
- `slice_id`

至少其中两个层级。

---

## 15. 推荐落地顺序

建议按以下顺序实现：

1. 保留现有逐图 VLM 流程
2. 新增 `photo fact buffer`
3. 将 `burst / session` 从 `memory_module` 前移为 LLM 前置结构
4. 新增 `session slice` 生成器
5. 将第一阶段事件提取改为吃 `slice evidence packet`
6. 新增 `event_candidate stitching`
7. 新增第二阶段 candidate confirmation
8. 让关系推断和画像改为读取 `global events + evidence refs`

---

## 16. 与当前代码的对应建议

### 16.1 可复用部分

- `services/vlm_analyzer.py`
- `memory_module/sequencing.py`
- `services/llm_processor.py` 中关系推断与画像生成的后半段思路

### 16.2 需要改造的重点

- `memory_module/sequencing.py`
  - `_same_session()` 当前偏 EXIF/location-first，需升级为 `EXIF-first skeleton + VLM correction`
- `services/llm_processor.py`
  - `extract_events()` 不能再直接吃全部 `vlm_results`
- `main.py`
  - VLM 后、LLM 前新增 buffer / segmentation / slice 阶段

### 16.3 关键新增对象

- `PhotoFact`
- `BurstEvidence`
- `SessionEvidence`
- `SessionSlice`
- `EventCandidate`
- `EventLink`

---

## 17. 默认规则摘要

| 层级 | 默认策略 |
|------|----------|
| `burst` | `90 秒 + 最长 180 秒 + 最多 30 张` |
| `raw session` | EXIF 骨架 + VLM 修正，不设固定照片上限 |
| `session slice` | 受 token budget 限制，不改变 session 身份 |
| `slice overlap` | `1-2` 个 burst 或 `5-10` 分钟 |
| 第一阶段 LLM | 吃 `slice evidence packet`，输出 `event_candidate` |
| 第二阶段确认 | 只对相邻候选做确认与拼接 |
| 保真策略 | 压缩重复，不压缩事实；rare clue 不可丢 |

---

## 18. 一句话结论

系统不应把“连续现实活动链”和“LLM 可读窗口”混为一谈。

正确做法是：

- 用 `burst` 和 `raw session` 表示现实结构
- 用 `session slice` 表示模型输入窗口
- 用 `event_candidate` 和二阶段确认解决 slice 拼接
- 用 `photo fact buffer` 和 rare clue 机制保证信息最大保留
