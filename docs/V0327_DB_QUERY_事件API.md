# v0327-db-query Events API 对外文档

默认对外版本：`v0327-db-query`  
文档语言：中文  
面向对象：下游系统、接口调用方、数据消费方  
最后校验时间：`2026-03-30`

## 1. 文档目标

本文档定义 `Events API` 的正式对外契约，供下游系统读取单个任务的事件结果。

这份文档只约束当前最新口径：

- 默认任务版本：`v0327-db-query`
- 路由：`task-scoped`
- 返回主体：`events`
- 每个 event 自带自己的照片

## 2. 设计结论

`Events API` 现在的设计原则只有一句话：

> 接口返回事件列表，不返回整条任务的全量拼盘数据。

这意味着：

- 顶层只返回 `data.events`
- 不再顶层返回整条任务的全量 `photos`
- 不再顶层返回整条任务的全量 `persons`
- 不再顶层返回 `event_people`
- 不再顶层返回 `event_places`
- 不再顶层返回 `evidence`

如果某些人物、照片、地点、VLM 线索属于某个事件，它们会折叠进**该事件自己的对象**里。

## 3. 路由

```http
GET /api/tasks/{task_id}/memory/events
```

示例：

```http
GET /api/tasks/5570845cce1a47819227b3d89fcec9cb/memory/events
GET /api/tasks/5570845cce1a47819227b3d89fcec9cb/memory/events?include_raw=true
GET /api/tasks/5570845cce1a47819227b3d89fcec9cb/memory/events?include_artifacts=true
```

## 4. 认证

该接口要求先登录，再携带会话 Cookie。

推荐联调顺序：

1. `POST /api/auth/login`
2. `GET /api/auth/me`
3. `GET /api/tasks`
4. 选择目标 `task_id`
5. 调用 `GET /api/tasks/{task_id}/memory/events`

## 4.1 响应压缩

当前生产服务已开启 `gzip` 响应压缩。

- 客户端默认无需额外改动
- 常见客户端如浏览器、`requests`、`curl --compressed` 会自动处理
- `events` 返回体较大时，服务端会自动返回 `Content-Encoding: gzip`
- 如果下游排查慢链路问题，请同时记录 `Content-Encoding` 和 `Content-Length`

## 5. 请求参数

### 5.1 Path 参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `task_id` | string | 是 | 任务 ID，唯一决定本次事件读取范围 |

### 5.2 Query 参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `include_raw` | boolean | 否 | `false` | 是否返回 `data.raw_events` |
| `include_artifacts` | boolean | 否 | `false` | 是否返回 `artifacts` |
| `include_traces` | boolean | 否 | `false` | 预留字段，当前事件接口不返回 traces 主体 |

## 6. 响应壳

```json
{
  "query": {
    "task_id": "5570845cce1a47819227b3d89fcec9cb",
    "include_raw": false,
    "include_artifacts": false,
    "include_traces": false
  },
  "task": {
    "task_id": "5570845cce1a47819227b3d89fcec9cb",
    "version": "v0327-db",
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

下面这些字段是当前接口层会稳定补齐的字段，下游应优先消费它们：

| 字段 | 类型 | 说明 |
|---|---|---|
| `event_id` | string | 事件 ID |
| `title` | string \| null | 事件标题 |
| `date` | string \| null | 事件日期，通常为 `YYYY-MM-DD` |
| `started_at` | string \| null | 事件开始时间 |
| `ended_at` | string \| null | 事件结束时间 |
| `person_ids` | string[] | 与该事件关联的人物 ID 列表 |
| `photo_ids` | string[] | 与该事件关联的照片 ID 列表 |
| `photo_urls` | string[] | 与该事件关联的照片 URL 列表 |
| `photos` | object[] | 与该事件关联的照片对象列表 |
| `cover_photo_id` | string \| null | 事件封面照片 ID |
| `cover_photo_url` | string \| null | 事件封面照片 URL |

### 7.2 继承字段

除了上面的稳定字段，接口还会尽量保留 LP1 事件 payload 里的业务字段。

当前线上实测中，常见字段包括：

- `description`
- `duration`
- `location`
- `participants`
- `type`
- `reason`
- `tags`
- `lifestyle_tags`
- `llm_summary`
- `narrative`
- `narrative_synthesis`
- `objective_fact`
- `persona_evidence`
- `social_dynamics`
- `social_interaction`
- `social_slices`
- `time_range`
- `meta_info`
- `vlm`
- `evidence_photos`
- `photo_count`
- `confidence`

说明：

- 这些字段不是每条 event 都保证存在
- 下游不应把它们全部当作强契约字段
- 推荐把它们视为“事件业务内容补充字段”

## 8. `photos[]` 子对象约定

`event.photos[]` 只包含**该事件自己的照片**。

它不是整条任务的全量 `photos`。

典型字段如下：

| 字段 | 类型 | 说明 |
|---|---|---|
| `photo_id` | string | 任务内照片 ID |
| `source_photo_id` | string \| null | 原始阶段里的照片 ID |
| `original_filename` | string \| null | 原始文件名 |
| `stored_filename` | string \| null | 存储后的文件名 |
| `asset_path` | string \| null | 资源路径 |
| `photo_url` | string \| null | 可直接展示的图片 URL |
| `captured_at` | string \| null | 拍摄时间 |
| `width` | integer \| null | 宽度 |
| `height` | integer \| null | 高度 |
| `content_type` | string \| null | 媒体类型 |

下游图片消费规则：

1. 优先直接使用 `event.cover_photo_url`
2. 展示事件图片列表时使用 `event.photos[]`
3. 不要根据 `photo_id` 自己拼 URL

## 9. 示例响应

> 说明：下面示例使用当前线上完整样例任务 `5570845cce1a47819227b3d89fcec9cb` 说明返回结构。  
> 该任务本身的历史版本是 `v0327-db`，但当前响应结构来自最新部署的 `v0327-db-query` 服务代码。

```json
{
  "query": {
    "task_id": "5570845cce1a47819227b3d89fcec9cb",
    "include_raw": true,
    "include_artifacts": false,
    "include_traces": false
  },
  "task": {
    "task_id": "5570845cce1a47819227b3d89fcec9cb",
    "version": "v0327-db",
    "status": "completed",
    "stage": "completed"
  },
  "data": {
    "events": [
      {
        "event_id": "EVT_001",
        "title": "Concert Night",
        "date": "2026-01-18",
        "description": "An evening concert with a close friend.",
        "llm_summary": "Concert Night",
        "location": "Hangzhou",
        "participants": ["Person_001", "Person_012"],
        "person_ids": ["Person_001", "Person_012"],
        "photo_ids": [
          "5570845cce1a47819227b3d89fcec9cb:photo_001",
          "5570845cce1a47819227b3d89fcec9cb:photo_002"
        ],
        "photo_urls": [
          "/api/assets/5570845cce1a47819227b3d89fcec9cb/uploads/001_MVIMG_20260118_135245.jpg",
          "/api/assets/5570845cce1a47819227b3d89fcec9cb/uploads/002_MVIMG_20260118_142210.jpg"
        ],
        "cover_photo_id": "5570845cce1a47819227b3d89fcec9cb:photo_001",
        "cover_photo_url": "/api/assets/5570845cce1a47819227b3d89fcec9cb/uploads/001_MVIMG_20260118_135245.jpg",
        "photos": [
          {
            "photo_id": "5570845cce1a47819227b3d89fcec9cb:photo_001",
            "source_photo_id": "photo_001",
            "original_filename": "MVIMG_20260118_135245.jpg",
            "photo_url": "/api/assets/5570845cce1a47819227b3d89fcec9cb/uploads/001_MVIMG_20260118_135245.jpg"
          },
          {
            "photo_id": "5570845cce1a47819227b3d89fcec9cb:photo_002",
            "source_photo_id": "photo_002",
            "original_filename": "MVIMG_20260118_142210.jpg",
            "photo_url": "/api/assets/5570845cce1a47819227b3d89fcec9cb/uploads/002_MVIMG_20260118_142210.jpg"
          }
        ]
      }
    ],
    "raw_events": [
      {
        "source_temp_event_id": "tmp_evt_001",
        "supporting_photo_ids": ["photo_001", "photo_002"]
      }
    ]
  }
}
```

## 10. `include_raw=true` 的含义

当 `include_raw=true` 时，响应会额外返回：

```json
{
  "data": {
    "events": [...],
    "raw_events": [...]
  }
}
```

`raw_events` 的用途是：

- 联调阶段查看 LP1 原始事件 payload
- 排查结构映射问题
- 对照稳定字段和原始字段

下游正式消费 UI 时，建议仍以 `events` 为主。

## 11. 当前线上实测结果

以下结果基于 `2026-03-30` 从**非本机**机器 `vm-cc-bedrock` 调用 `app-v0317` 得到。

### 11.1 完整样例任务

任务：`5570845cce1a47819227b3d89fcec9cb`

- HTTP：`200`
- 耗时：`5.533s`
- 响应体大小：`902,201 bytes`
- `data` 顶层键：只有 `events`
- `event_count = 152`
- 首个 event 自带 `photos`
- 首个 event 的照片数：`2`

### 11.2 历史不完整任务

任务：`e79a9632744a4a7f801bacae7fddf012`

- HTTP：`200`
- 耗时：`0.143s`
- 响应体大小：`467 bytes`
- `event_count = 0`

说明：

- 这表示接口本身是通的
- 只是该任务本身没有产出事件数据

## 12. 数据来源说明

当前 `Events API` 不再依赖“第一次读接口时临时拆 JSON”。

现在的逻辑是：

1. LP1 事件阶段完成
2. 事件、事件-照片、事件-人物、事件-地点等结果直接写入数据库表
3. `Events API` 直接读这些表
4. 接口层补齐 `photo_urls`、`photos`、`cover_photo_url`

一句话说：

> 现在是“事件生成后立刻写表，再读表返回”，不是“先存一大坨，读的时候再拆”。

## 13. 下游接入建议

下游消费这个接口时，建议按下面方式处理：

1. 先读 `data.events`
2. 列表卡片封面直接用 `cover_photo_url`
3. 事件详情页的图片列表直接用 `photos[]`
4. 如果需要原始 LP1 内容，再开 `include_raw=true`
5. 如果需要全量聚合数据，调用 `/memory/bundle`
6. 如果只需要 VLM / profiles / relationships，不要再从 `events` 里硬拆，直接调用各自接口

## 14. 不要这样用

下面这些用法不建议继续使用：

- 期待 `data.photos` 在 `Events API` 顶层返回
- 期待 `data.persons` 在 `Events API` 顶层返回
- 期待 `Events API` 同时替代 `vlm / profiles / relationships`
- 根据 `photo_id` 自己拼接图片 URL

## 15. 和其他接口的边界

| 接口 | 职责 |
|---|---|
| `/memory/events` | 只返回事件，以及每个事件自己的照片 |
| `/memory/vlm` | 返回照片级 VLM 观察 |
| `/memory/profiles` | 返回画像结果 |
| `/memory/relationships` | 返回关系结果 |
| `/memory/bundle` | 返回全量聚合包 |

如果下游问“为什么 `events` 里不再有顶层全量 `photos/persons`”，答案就是：

> 因为这些已经不是 `Events API` 的职责了。
