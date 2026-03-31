# v0327-db-query Query API 文档

## 1. 文档目标

本文档定义 `v0327-db-query` 版本的 Query API 对外契约，供下游系统进行在线问答、事件召回、证据展示和审慎判断集成。

这份文档只对 `v0327-db-query` 版本做正式约束。

原因是：

- `v0327-db-query` 默认依赖双层 Milvus + Neo4j + canonical query store
- 其他历史版本可能仍可调用同一路径
- 但不保证与本文档完全一致

因此，**推荐下游始终以 `v0327-db-query` 作为任务版本默认值**。

同时，本版本在原有输出基础上做的是**增量增强**：

- 原有字段继续保留
- `photo_id / photo_ids / original_photo_ids` 不删除
- 新增 `photo_url / photo_urls / cover_photo_url / photos[]`
- 下游若已有旧解析逻辑，可以继续工作；若要直接渲染图片，优先消费新增 URL 字段

## 2. 关键原则

### 2.1 权威召回结果

`matched_events` 是权威召回结果。
下游渲染图片时，应优先使用 API 直接返回的 `photo_url`，不要自行拼接图片地址。

这意味着：

- API 返回的是“所有与 query 相关的事件”
- 这些事件按最终分数排序
- `summary` 只是摘要表达，不是权威结果
- 下游如果做进一步展示、筛选、打标，应以 `matched_events` 为准

### 2.2 照片是硬约束

`v0327-db-query` 的目标不是单纯答一句话，而是返回：

- 相关事件
- 支撑照片
- 支撑事实
- 支撑关系

### 2.3 默认任务版本

创建新任务时，推荐明确指定：

```json
{
  "version": "v0327-db-query"
}
```

### 2.4 与六个检索接口的关系

`v0327-db-query` 当前对下游暴露两类能力：

1. Query API  
   `POST /api/tasks/{task_id}/memory/query`

2. 六个 task-scoped 检索接口  
   `GET /api/tasks/{task_id}/memory/faces|events|vlm|profiles|relationships|bundle`

注意：

- 这两类接口都已经部署在同一套生产服务上
- 六个检索接口现在也已经切到“数据库表优先”读取
- Query API 会先把任务物化到 canonical query store，再做 route / retrieve / answer
- 六个检索接口读取的也是同一套 canonical tables
- `faces` 接口对应的人脸数据来自 face 阶段原始快照 `face_recognition`，但线上 serving 时已经先物化进 `memory_persons / memory_faces / memory_photos` 再返回

这意味着：

- 如果某个 `v0327-db-query` 任务历史上根本没跑出 VLM / LP1 / LP2 / LP3 数据，那么六个检索接口会返回空数组
- Query API 不会瞎答；在证据不足时会返回 `insufficient_evidence` 风格的回答

## 3. 当前生产环境

| 项目 | 当前值 |
| --- | --- |
| Service | `app-v0317` |
| Backend Base URL | `http://10.60.1.243:8000` |
| Frontend | `http://10.60.1.243:3000` |
| 推荐版本 | `v0327-db-query` |

### 3.2 当前生产验证状态（2026-03-30）

| 任务版本 | task_id | Query API | 六个检索接口 |
| --- | --- | --- | --- |
| `v0327-db` | `5570845cce1a47819227b3d89fcec9cb` | 已验证 | 6/6 全通，返回完整数据 |
| `v0327-db-query` | `e79a9632744a4a7f801bacae7fddf012` | `200`，当前问题返回 `insufficient_evidence` | 6/6 全通，其中 `events / vlm / profiles / relationships` 为 0 条、`bundle` 返回空后段数据 |

### 3.1 推荐联调账号

当前线上已验证可用的默认测试账号：

| 项目 | 当前值 |
| --- | --- |
| Username | `vigar_heavy_0319` |
| Password | 由服务 owner 单独分发 |

说明：

- `app-v0317` 当前仅内网可访问
- 如果下游只是联调 Query API，建议直接使用该默认账号
- 若后续需要固定密码写入交付文档，可再补充

## 4. 鉴权

当前 API 使用 cookie session 鉴权。

### 4.1 登录

`POST /api/auth/login`

请求体：

```json
{
  "username": "your_username",
  "password": "your_password"
}
```

成功后服务端会设置 `memory_session` cookie。  
后续请求应携带该 cookie。

联调时推荐直接先登录默认测试账号，再调用 Query API。

### 4.2 登录态确认

`GET /api/auth/me`

### 4.3 联调自检

在开始联调 Query API 前，建议先做两步自检：

1. `GET /api/auth/me`
2. `GET /api/tasks`

目的：

- 确认当前 cookie 实际绑定的是预期账号
- 确认任务列表接口已经返回该账号名下任务
- 避免“登录成功但实际拿的是旧 session / 其他账号”的误判

## 5. 推荐调用流程

### 5.1 创建任务

`POST /api/tasks`

推荐请求：

```json
{
  "version": "v0327-db-query",
  "normalize_live_photos": true
}
```

### 5.2 上传并执行任务

上传和执行接口沿用现有任务接口，不在本文档展开。

### 5.3 等待任务完成

查询：

- `GET /api/tasks/{task_id}`

要求：

- `status = completed`

### 5.4 执行 query

调用：

- `POST /api/tasks/{task_id}/memory/query`

推荐顺序：

1. `POST /api/auth/login`
2. `GET /api/auth/me`
3. `GET /api/tasks`
4. 选择目标 `task_id`
5. `POST /api/tasks/{task_id}/memory/query`

## 6. Query API

### 6.1 Endpoint

`POST /api/tasks/{task_id}/memory/query`

### 6.2 请求体

```json
{
  "question": "主角养什么宠物？",
  "context_hints": {
    "client_request_id": "optional"
  },
  "time_hint": null,
  "answer_shape_hint": null
}
```

### 6.3 请求字段

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `question` | string | 是 | 用户问题 |
| `context_hints` | object | 否 | 调用方附加上下文 |
| `time_hint` | string | 否 | 额外时间提示 |
| `answer_shape_hint` | string | 否 | 额外答案形态提示 |

## 7. 响应结构

顶层结构如下：

```json
{
  "query_plan": {},
  "answer": {},
  "matched_events": [],
  "supporting_units": [],
  "supporting_evidence": [],
  "supporting_facts": [],
  "supporting_relationships": [],
  "supporting_graph_entities": [],
  "supporting_photos": [],
  "graph_support": [],
  "clause_results": [],
  "aggregation_result": {},
  "judgement_status": "supported | contradicted | insufficient_evidence",
  "abstain_reason": null
}
```

## 8. `query_plan` 字段

`query_plan` 描述 router 的结构化理解结果。

### 8.1 字段示例

```json
{
  "engine": "query_v1",
  "schema_version": "query_v1",
  "plan_type": "fact_first",
  "router_version": "route_plan_v1",
  "normalized_question": "主角养什么宠物？",
  "composition_mode": "single",
  "global_constraints": {
    "task_scope": "5570845cce1a47819227b3d89fcec9cb",
    "photo_required": true
  },
  "clauses": [],
  "router_confidence": 0.63,
  "needs_abstain_judgement": true,
  "router_source": "llm",
  "router_fallback": null
}
```

### 8.2 核心字段说明

| 字段 | 说明 |
| --- | --- |
| `plan_type` | 本次查询的总体类型 |
| `router_version` | 当前 router 契约版本 |
| `composition_mode` | 单问句或多子句组合方式 |
| `router_source` | `llm` 或 `deterministic` |
| `router_fallback` | 若 LLM 失败，是否回退到 deterministic |

### 8.3 `clauses`

每个子句字段固定包括：

| 字段 | 说明 |
| --- | --- |
| `clause_id` | 子句 ID |
| `clause_text` | 子句文本 |
| `route` | `fact-first / relationship-first / event-first / hybrid-judgement` |
| `intent` | `lookup / explain / rank / count / existence / compare / judgement` |
| `target_fact_keys` | 命中的事实字段 |
| `target_person_ids` | 命中的人物 ID |
| `target_relationship_types` | 命中的关系类型 |
| `time_windows` | 标准化后的时间窗 |
| `same_event_required` | 是否要求同一事件满足 |
| `requires_photos` | 是否要求照片证据 |
| `can_abstain` | 是否允许保守回答 |

## 9. `answer` 字段

`answer` 是结构化主结果。

### 9.1 常见字段

| 字段 | 说明 |
| --- | --- |
| `answer_type` | 结果类型 |
| `summary` | 最终摘要 |
| `confidence` | 汇总置信度 |
| `matched_event_count` | 命中事件数 |
| `matched_event_ids` | 命中事件 ID 列表 |
| `top_event_ids_for_summary` | 用于摘要生成的 top event IDs |
| `supporting_photos` | 扁平化照片证据列表，含 `photo_id + photo_url` |
| `supporting_facts` | 命中的事实 |
| `supporting_relationships` | 命中的关系 |
| `judgement_status` | `supported / contradicted / insufficient_evidence` |
| `abstain_reason` | 保守回答原因 |
| `materialization_id` | 对应 query store materialization |
| `writer_source` | `llm` 或 `template` |

### 9.2 `summary` 的定位

`summary` 的定位是：

- 供下游直接展示给用户
- 方便快速阅读
- 不是最终权威数据源

下游如果要做进一步处理，请以这些字段为准：

- `matched_events`
- `supporting_facts`
- `supporting_relationships`
- `clause_results`

### 9.3 输出兼容性

本版本对响应做的是**只增不减**的增强：

- 旧字段仍可继续使用
- 新增图片 URL 字段仅用于降低下游拼接成本
- 如果下游原先只依赖 `photo_id`，不需要改也不会报错
- 如果下游希望直接展示图片，请改为优先使用 `photo_url`

## 10. `matched_events`

### 10.1 定义

`matched_events` 表示所有与 query 相关、并通过当前 evidence gating 的事件集合。

### 10.2 结构

每条 event 结构如下：

```json
{
  "event_id": "EVT_0132",
  "title": "记录猫用化毛膏信息",
  "summary": "主角在客厅内仔细查看并拍摄了一支GimCat猫用化毛膏的包装和成分信息。",
  "start_ts": "2026-02-26",
  "end_ts": "",
  "photo_ids": ["photo_559", "photo_019", "photo_563"],
  "photo_urls": [
    "/api/assets/5570845cce1a47819227b3d89fcec9cb/uploads/559.jpg",
    "/api/assets/5570845cce1a47819227b3d89fcec9cb/uploads/019.jpg"
  ],
  "photos": [
    {
      "photo_id": "photo_559",
      "photo_url": "/api/assets/5570845cce1a47819227b3d89fcec9cb/uploads/559.jpg",
      "asset_url": "/api/assets/5570845cce1a47819227b3d89fcec9cb/uploads/559.jpg",
      "object_key": "tasks/5570845cce1a47819227b3d89fcec9cb/uploads/559.jpg",
      "captured_at": "2026-02-26T20:13:11",
      "content_type": "image/jpeg",
      "width": 3024,
      "height": 4032,
      "supporting_event_id": "EVT_0132",
      "is_cover": true,
      "support_strength": 1.0,
      "sort_order": 0
    }
  ],
  "cover_photo_id": "photo_559",
  "cover_photo_url": "/api/assets/5570845cce1a47819227b3d89fcec9cb/uploads/559.jpg",
  "person_ids": [],
  "place_refs": ["私人住宅客厅"],
  "confidence": 0.9,
  "score": 0.95,
  "reasons": ["canonical_support"],
  "clause_ids": ["clause_1"]
}
```

### 10.3 下游处理建议

推荐下游：

1. 直接展示 top N 事件
2. 保留完整 `matched_events` 供“查看更多”或二次筛选
3. 事件卡片优先使用 `cover_photo_url`
4. 证据图或图库优先使用 `supporting_photos[].photo_url`

### 10.4 图片字段使用建议

- 如果你在渲染事件卡片：优先使用 `matched_events[].cover_photo_url`
- 如果你在渲染单个事件的全部相关图：使用 `matched_events[].photos[]`
- 如果你在做全局证据画廊：使用 `supporting_photos[]`
- 不要只拿 `photo_id` 后再去猜 URL；API 已直接返回 `photo_url`

## 11. `supporting_facts`

适用于画像型、标量型、判断型问题。

示例结构：

```json
{
  "fact_id": "profilefact_xxx",
  "field_key": "structured_profile.long_term_facts.relationships.pets",
  "value": "猫",
  "confidence": 0.8,
  "source_level": "structured",
  "match_score": 0.93,
  "evidence_event_ids": [],
  "evidence_photo_ids": ["photo_019"]
}
```

## 12. `supporting_relationships`

适用于伴侣、亲密关系、relationship ranking、romantic 线索等问题。

示例结构：

```json
{
  "entity_type": "relationship",
  "relationship_id": "REL_Person_012",
  "target_person_id": "Person_012",
  "relationship_type": "romantic",
  "status": "growing",
  "confidence": 0.94,
  "intimacy_score": 0.95,
  "shared_event_count": 9,
  "photo_count": 74,
  "reasoning": "..."
}
```

## 12.1 `supporting_photos`

适用于下游直接渲染证据图，不需要再次拼接图片地址。

示例结构：

```json
{
  "photo_id": "photo_003",
  "photo_url": "/assets/uploads/003_exhibition.png",
  "asset_url": "/assets/uploads/003_exhibition.png",
  "object_key": "uploads/003_exhibition.png",
  "captured_at": "2026-01-18T15:00:00",
  "content_type": "image/png",
  "width": 1200,
  "height": 900,
  "supporting_event_id": "EVT_EXHIBITION_001"
}
```

## 13. `clause_results`

`clause_results` 用于解释多子句题如何被拆解和执行。

每条 `clause_result` 包括：

| 字段 | 说明 |
| --- | --- |
| `clause_id` | 子句 ID |
| `route` | 本子句走的执行链 |
| `intent` | 子句意图 |
| `matched_event_ids` | 本子句命中的事件 ID |
| `matched_event_count` | 本子句命中事件数 |
| `supporting_fact_keys` | 本子句命中的事实字段 |
| `supporting_relationship_ids` | 本子句命中的关系 ID |
| `supporting_photo_ids` | 本子句相关照片 |
| `supporting_photo_urls` | 本子句相关照片 URL |
| `judgement_status` | 本子句判断结果 |
| `abstain_reason` | 本子句保守原因 |
| `aggregation_result` | 本子句聚合结果 |
| `result_hint` | 子句级结果提示 |

## 14. 路由类型

### 14.1 `fact-first`

适用于：

- 宠物
- 地点锚点
- 亲密圈大小
- 身份阶段
- 职业判断前置事实

### 14.2 `relationship-first`

适用于：

- 伴侣是谁
- 和谁关系最好
- 是否有多条 romantic 线索

### 14.3 `event-first`

适用于：

- 某天做了什么
- 与某人一起经历过哪些事件
- 既看过演唱会也看过展览吗

### 14.4 `hybrid-judgement`

适用于：

- 能不能判断已经工作
- 能不能判断仍在上学
- 是否存在某类关系或某类事件

## 15. Judgement 语义

`judgement_status` 只允许以下取值：

- `supported`
- `contradicted`
- `insufficient_evidence`

推荐解释：

| 值 | 含义 |
| --- | --- |
| `supported` | 当前证据支持该结论 |
| `contradicted` | 当前证据反向支持另一结论 |
| `insufficient_evidence` | 当前证据不足，不能贸然判断 |

## 16. 示例

### 16.1 示例一：事实查询

请求：

```http
POST /api/tasks/5570845cce1a47819227b3d89fcec9cb/memory/query
```

```json
{
  "question": "主角养什么宠物？"
}
```

结果特征：

- `plan_type = fact_first`
- `route = fact-first`
- `judgement_status = supported`
- `supporting_facts[0].field_key = structured_profile.long_term_facts.relationships.pets`
- `matched_events` 中仍会返回与该事实相关的事件与照片

### 16.2 示例二：审慎判断

请求：

```json
{
  "question": "如果只看可审计证据，能不能判断主角已经工作？如果不能，请说明为什么。"
}
```

结果特征：

- `plan_type = composite`
- `composition_mode = judge_then_explain`
- 子句 1 走 `hybrid-judgement`
- 子句 2 走 `event-first`
- `judgement_status = contradicted`
- `supporting_facts` 会给出职业阶段、身份角色、收入模式等依据

## 17. HTTP 状态码

| 状态码 | 说明 |
| --- | --- |
| `200` | 成功 |
| `400` | 请求参数错误 |
| `401` | 未登录或登录失效 |
| `404` | 任务不存在，或任务未具备可查询结果 |
| `500` | 服务内部错误 |

## 18. 版本约束

对下游的正式推荐如下：

1. 创建新任务时始终指定 `version = v0327-db-query`
2. 不要默认依赖历史版本的 query 行为
3. 如果需要对旧任务做统一查询，请先确认是否已完成 `query_v1` materialization

## 19. 非契约实现说明

以下信息仅用于帮助理解当前实现，不应作为长期稳定契约：

- 当前 router 与 summary writer 都会调用 LLM
- 当前 `app-v0317` 线上 `OPENROUTER_LLM_MODEL` 为 `nvidia/nemotron-3-super-120b-a12b`
- 未来模型可能更换，但不应影响本 API 文档定义的字段结构

## 20. 对下游的最终建议

如果下游只需要一句话摘要，可直接展示：

- `answer.summary`

如果下游要做严肃业务处理，请使用：

- `matched_events`
- `supporting_facts`
- `supporting_relationships`
- `clause_results`
- `judgement_status`

其中最重要的一条是：

**`matched_events` 才是最终召回的权威结果，`summary` 只是对它的压缩表达。**
