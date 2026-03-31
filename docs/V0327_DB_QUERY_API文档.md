# v0327-db-query Query API 文档

默认对外版本：`v0327-db-query`  
文档语言：中文  
最后更新：`2026-03-31`

## Swagger

仓库内同时提供静态 Swagger 页面与 OpenAPI 快照：

- `docs/swagger/index.html`
- `docs/swagger/openapi-v0327-db-query.json`

如果仓库启用 GitHub Pages，可直接把 `docs/swagger/` 作为静态页面目录对外展示。

## 1. 文档目标

本文档定义 `v0327-db-query` 的 Query API 正式契约，供下游系统做：

- 在线问答
- 事件召回
- 支撑照片展示
- 画像 / 关系 / judgement 类问题消费

## 2. 身份模型

### 2.1 `user_id` 的正式语义

在 `v0327-db-query` 中：

- `user_id`
  - 指照片主人 / 被分析对象 / subject user
- `operator_user_id`
  - 指代上传照片或代操作的后台管理员
  - 只是可选元信息
  - 不是 Query API 主键

一句话：

> Query API 的正式主键是照片主人的 `user_id`，不是后台管理员账号。

### 2.2 历史保留样例

历史任务 `5570845cce1a47819227b3d89fcec9cb` 保留原有 `task_id / user_id` 作为只读样例。

这条任务可以继续用于联调，但不代表新任务继续沿用旧语义。

## 3. 正式路由

### 3.1 任务列表

```http
GET /api/users/{user_id}/tasks
GET /api/users/{user_id}/tasks/{task_id}
```

### 3.2 Query API

```http
POST /api/users/{user_id}/tasks/{task_id}/memory/query
```

说明：

- path 里的 `user_id` 是照片主人
- path 里的 `task_id` 是该用户名下某次任务
- 当前登录账号只负责会话鉴权和操作记录

## 4. 默认版本与创建任务

### 4.1 默认版本

下游新建任务时，默认版本应固定为：

```json
{
  "version": "v0327-db-query"
}
```

### 4.2 创建任务的正式语义

当前创建任务仍走：

```http
POST /api/tasks
```

但请求体里的 `user_id` 语义已经是：

- `user_id = 照片主人`
- 不是后台管理员

管理员代上传示例：

```json
{
  "version": "v0327-db-query",
  "user_id": "jennie_user_id",
  "normalize_live_photos": true
}
```

self-upload 示例：

```json
{
  "version": "v0327-db-query",
  "normalize_live_photos": true
}
```

此时服务端会把当前登录用户自动作为 subject user，且不写 `operator_user_id`。

## 5. 关键原则

### 5.1 权威召回结果

`matched_events` 是权威召回结果。

- API 返回的是所有与 query 相关的事件
- `summary` 只是压缩表达
- 下游如果要做后续筛选、展示或打分，应以 `matched_events` 为准

### 5.2 照片是硬约束

`v0327-db-query` 返回的不只是答案文本，而是：

- 相关事件
- 支撑照片
- 支撑 facts
- 支撑 relationships

### 5.3 图片 URL 直接消费

下游应优先使用 API 已返回的：

- `photo_url`
- `photo_urls`
- `cover_photo_url`
- `photos[]`

不要自己根据 `photo_id` 拼路径。

## 6. 与六个检索接口的关系

`v0327-db-query` 对下游暴露两类能力：

1. Query API  
   `POST /api/users/{user_id}/tasks/{task_id}/memory/query`

2. 六个检索接口  
   `GET /api/users/{user_id}/tasks/{task_id}/memory/faces|events|vlm|profiles|relationships|bundle`

这两类接口共享同一套结构化数据来源：

- `memory_events`
- `memory_event_photos`
- `memory_event_people`
- `memory_event_places`
- `memory_evidence`
- `memory_relationships`
- `memory_relationship_support`
- `memory_profile_facts`
- `memory_photos`
- `memory_persons`
- `memory_faces`

## 7. 鉴权

当前 API 使用 cookie session。

### 7.1 登录

```http
POST /api/auth/login
```

成功后会设置 session cookie。

### 7.2 登录态自检

```http
GET /api/auth/me
```

### 7.3 推荐联调顺序

1. `POST /api/auth/login`
2. `GET /api/auth/me`
3. `GET /api/users/{user_id}/tasks`
4. 选择目标 `task_id`
5. `POST /api/users/{user_id}/tasks/{task_id}/memory/query`

## 8. 请求

### 8.1 Endpoint

```http
POST /api/users/{user_id}/tasks/{task_id}/memory/query
```

### 8.2 请求体

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

### 8.3 字段说明

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `question` | string | 是 | 用户问题 |
| `context_hints` | object | 否 | 调用方附加上下文 |
| `time_hint` | string | 否 | 额外时间提示 |
| `answer_shape_hint` | string | 否 | 额外答案形态提示 |

## 9. 响应结构

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

## 10. `query_plan`

`query_plan` 表示 router 的结构化理解结果。

### 10.1 常见字段

| 字段 | 说明 |
| --- | --- |
| `plan_type` | 本次查询总体类型 |
| `router_version` | 当前 router 契约版本 |
| `composition_mode` | 单问句或多子句组合方式 |
| `router_source` | `llm` 或 `deterministic` |
| `router_fallback` | 若 LLM 失败，是否回退到 deterministic |

### 10.2 `clauses`

每个 clause 常见字段：

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

## 11. `answer`

`answer` 是结构化主结果。

### 11.1 常见字段

| 字段 | 说明 |
| --- | --- |
| `answer_type` | 结果类型 |
| `summary` | 最终摘要 |
| `confidence` | 汇总置信度 |
| `matched_event_count` | 命中事件数 |
| `matched_event_ids` | 命中事件 ID 列表 |
| `top_event_ids_for_summary` | 用于摘要生成的 top event IDs |
| `supporting_photos` | 扁平化照片证据列表 |
| `supporting_facts` | 命中的事实 |
| `supporting_relationships` | 命中的关系 |
| `judgement_status` | `supported / contradicted / insufficient_evidence` |
| `abstain_reason` | 保守回答原因 |
| `materialization_id` | 对应 materialization ID |
| `writer_source` | `llm` 或 `template` |

### 11.2 `summary` 的定位

`summary` 供直接展示，但不是权威结果。

下游如果要继续处理，请以这些字段为准：

- `matched_events`
- `supporting_facts`
- `supporting_relationships`
- `clause_results`

## 12. `matched_events`

### 12.1 定义

`matched_events` 表示所有与 query 相关、并通过 evidence gating 的事件集合。

### 12.2 使用建议

推荐下游：

1. 直接展示 top N 事件
2. 保留完整 `matched_events` 用于“查看更多”
3. 事件卡片优先使用 `cover_photo_url`
4. 全局证据画廊优先使用 `supporting_photos[]`

## 13. `supporting_facts`

适用于画像型、标量型、判断型问题。

常见字段：

- `fact_id`
- `field_key`
- `value`
- `confidence`
- `source_level`
- `match_score`
- `evidence_event_ids`
- `evidence_photo_ids`

## 14. `supporting_relationships`

适用于伴侣、亲密关系、relationship ranking、romantic 线索等问题。

常见字段：

- `relationship_id`
- `target_person_id`
- `relationship_type`
- `status`
- `confidence`
- `intimacy_score`
- `shared_event_count`
- `photo_count`
- `reasoning`

## 15. 路由类型

### 15.1 `fact-first`

适用于：

- 宠物
- 地点锚点
- 亲密圈大小
- 身份阶段
- 职业判断前置事实

### 15.2 `relationship-first`

适用于：

- 伴侣是谁
- 和谁关系最好
- 是否有多条 romantic 线索

### 15.3 `event-first`

适用于：

- 某天做了什么
- 与某人一起经历过哪些事件
- 既看过演唱会也看过展览吗

### 15.4 `hybrid-judgement`

适用于：

- 能不能判断已经工作
- 是否存在某类关系
- 是否从未发生某类事件

## 16. 当前验证结论

### 16.1 历史完整样例

任务：`5570845cce1a47819227b3d89fcec9cb`

- 可用于只读联调
- 保留历史 `task_id / user_id`

### 16.2 当前推荐口径

新任务、新脚本、新对接文档，一律按以下口径写：

- 默认版本：`v0327-db-query`
- 主键：`user_id = 照片主人`
- Query 路由：`/api/users/{user_id}/tasks/{task_id}/memory/query`

## 17. 下游接入建议

如果是管理后台代上传：

1. 管理员登录
2. `POST /api/tasks` 时传 `user_id = 照片主人`
3. 后续所有读取都走照片主人的 `user_id`

如果是用户自己上传：

1. 用户自己登录
2. `POST /api/tasks` 可不传 `user_id`
3. 后续仍用该用户自己的 `user_id` 读取任务和 Query API
