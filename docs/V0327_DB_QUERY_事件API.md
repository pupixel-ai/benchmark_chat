# v0327-db-query Events API 对外文档

默认对外版本：`v0327-db-query`  
文档语言：中文  
最后更新：`2026-03-31`

Swagger 页面：

- `docs/swagger/index.html`
- `docs/swagger/openapi-v0327-db-query.json`

## 1. 文档目标

本文档定义 `Events API` 的正式对外契约，供下游系统读取**某个照片主人某次任务**的事件结果。

## 2. 身份与路由口径

### 2.1 `user_id` 的正式语义

- `user_id`
  - 指照片主人 / subject user
- `operator_user_id`
  - 指代上传的后台管理员
  - 不是本接口主键

### 2.2 正式路由

```http
GET /api/users/{user_id}/tasks/{task_id}/memory/events
```

示例：

```http
GET /api/users/1a7c10412d724597adc4b6f5514ef7b1/tasks/5570845cce1a47819227b3d89fcec9cb/memory/events
GET /api/users/1a7c10412d724597adc4b6f5514ef7b1/tasks/5570845cce1a47819227b3d89fcec9cb/memory/events?include_raw=true
GET /api/users/1a7c10412d724597adc4b6f5514ef7b1/tasks/5570845cce1a47819227b3d89fcec9cb/memory/events?include_artifacts=true
```

## 3. 设计结论

`Events API` 的正式原则只有一句话：

> 接口返回事件列表，以及每个事件自己的照片；不返回整条任务的全量拼盘数据。

这意味着：

- 顶层只返回 `data.events`
- 不再顶层返回整条任务的全量 `photos`
- 不再顶层返回整条任务的全量 `persons`
- 不再顶层返回 `event_people`
- 不再顶层返回 `event_places`
- 不再顶层返回 `evidence`

如果某些人物、照片、地点、VLM 线索属于某个事件，它们会折叠进该事件自己的对象里。

## 4. 认证

该接口要求先登录，再携带会话 Cookie。

推荐联调顺序：

1. `POST /api/auth/login`
2. `GET /api/auth/me`
3. `GET /api/users/{user_id}/tasks`
4. 选择目标 `task_id`
5. 调用 `GET /api/users/{user_id}/tasks/{task_id}/memory/events`

## 5. 请求参数

### 5.1 Path 参数

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | ---: | --- |
| `user_id` | string | 是 | 照片主人 / subject user |
| `task_id` | string | 是 | 任务 ID |

### 5.2 Query 参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | ---: | --- | --- |
| `include_raw` | boolean | 否 | `false` | 是否返回 `data.raw_events` |
| `include_artifacts` | boolean | 否 | `false` | 是否返回 `artifacts` |
| `include_traces` | boolean | 否 | `false` | 预留字段，当前事件接口不返回 traces 主体 |

## 6. 响应壳

```json
{
  "query": {
    "user_id": "subject_user_id",
    "task_id": "task_xxx",
    "include_raw": false,
    "include_artifacts": false,
    "include_traces": false
  },
  "task": {
    "task_id": "task_xxx",
    "user_id": "subject_user_id",
    "operator_user_id": "admin_user_id_or_null",
    "version": "v0327-db-query",
    "status": "completed",
    "stage": "completed"
  },
  "data": {
    "events": []
  },
  "artifacts": null
}
```

## 7. `data.events[]` 字段约定

### 7.1 稳定字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `event_id` | string | 事件 ID |
| `title` | string \| null | 事件标题 |
| `date` | string \| null | 事件日期 |
| `started_at` | string \| null | 开始时间 |
| `ended_at` | string \| null | 结束时间 |
| `person_ids` | string[] | 该事件关联的人物 ID |
| `photo_ids` | string[] | 该事件关联的照片 ID |
| `photo_urls` | string[] | 该事件关联的照片 URL |
| `photos` | object[] | 该事件关联的照片对象 |
| `cover_photo_id` | string \| null | 封面照片 ID |
| `cover_photo_url` | string \| null | 封面照片 URL |

### 7.2 继承字段

除稳定字段外，接口也会保留事件业务字段，例如：

- `description`
- `duration`
- `location`
- `participants`
- `type`
- `reason`
- `tags`
- `llm_summary`
- `narrative`
- `objective_fact`
- `persona_evidence`
- `social_dynamics`
- `meta_info`
- `vlm`
- `confidence`

这些字段不是每条 event 都保证存在，下游应把它们视为事件补充字段。

## 8. `photos[]` 子对象约定

`event.photos[]` 只包含该事件自己的照片。

典型字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `photo_id` | string | 任务内照片 ID |
| `source_photo_id` | string \| null | 原始阶段照片 ID |
| `original_filename` | string \| null | 原始文件名 |
| `stored_filename` | string \| null | 存储文件名 |
| `asset_path` | string \| null | 资源路径 |
| `photo_url` | string \| null | 可直接展示的图片 URL |
| `captured_at` | string \| null | 拍摄时间 |
| `width` | integer \| null | 宽度 |
| `height` | integer \| null | 高度 |
| `content_type` | string \| null | 媒体类型 |

下游展示规则：

1. 封面图优先使用 `cover_photo_url`
2. 事件图集使用 `photos[]`
3. 不要根据 `photo_id` 自己拼 URL

## 9. `include_raw=true`

当 `include_raw=true` 时，响应会额外返回：

```json
{
  "data": {
    "events": [...],
    "raw_events": [...]
  }
}
```

`raw_events` 适用于联调或排查映射问题；正式 UI 消费仍建议以 `events` 为主。

## 10. 数据来源说明

当前 `Events API` 已经改成：

1. LP1 事件阶段完成
2. 事件结果立即写入数据库表
3. `Events API` 直接读表
4. 接口层补齐 `photo_urls / photos / cover_photo_url`

一句话：

> 现在是“事件生成后立刻写表，再读表返回”，不是“先存一大坨，第一次读接口时再拆”。

当前读取的主表包括：

- `memory_events`
- `memory_event_photos`
- `memory_event_people`
- `memory_event_places`
- `memory_photos`

## 11. 当前线上实测

### 11.1 历史完整样例

任务：`5570845cce1a47819227b3d89fcec9cb`  
说明：保留为历史只读样例

- HTTP：`200`
- 非本机耗时：约 `5.533s`
- 响应体大小：约 `902,201 bytes`
- 顶层 `data` 只有 `events`
- `event_count = 152`

### 11.2 历史不完整任务

任务：`e79a9632744a4a7f801bacae7fddf012`

- HTTP：`200`
- `event_count = 0`

这说明：

- 接口本身是通的
- 只是该任务本身没有产出事件数据

## 12. 不建议的使用方式

下面这些预期不再成立：

- 期待 `Events API` 顶层返回全量 `photos`
- 期待 `Events API` 顶层返回全量 `persons`
- 期待 `Events API` 同时替代 `vlm / profiles / relationships`
- 根据 `photo_id` 自己猜图片 URL

## 13. 与其他接口的边界

| 接口 | 职责 |
| --- | --- |
| `/memory/events` | 返回事件，以及每个事件自己的照片 |
| `/memory/vlm` | 返回照片级 VLM 观察 |
| `/memory/profiles` | 返回画像结果 |
| `/memory/relationships` | 返回关系结果 |
| `/memory/bundle` | 返回全量聚合包 |

## 14. 推荐下游接入方式

1. 以 `user_id` 为主键组织数据
2. `GET /api/users/{user_id}/tasks`
3. 选定 `task_id`
4. 调用 `GET /api/users/{user_id}/tasks/{task_id}/memory/events`
5. 事件卡片直接使用 `cover_photo_url`
6. 事件详情直接使用 `photos[]`
