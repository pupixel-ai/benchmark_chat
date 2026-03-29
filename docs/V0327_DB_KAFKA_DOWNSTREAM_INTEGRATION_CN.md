# v0327-db Kafka 下游自动推送对接文档

## 1. 文档目标

本文档用于说明 `v0327-db` 生产环境的 Kafka 终态推送集成方式，供下游系统进行消费、落库、告警和二次处理。

本文档覆盖的是“任务终态自动推送”链路，不包含 `memory/query` 在线召回接口。在线召回接口请使用另一份文档：

- [V0327_DB_QUERY_API_CN.md](/Users/ziyan/Downloads/memory_engineering_v1.0.1_20260310_v0327_db_query/docs/V0327_DB_QUERY_API_CN.md)

## 2. 适用范围

- 适用环境：`office-internal vpc`
- 当前生产者服务：`app-v0317`
- 当前 control-plane backend 地址：`http://10.60.1.243:8000`
- 当前 frontend 地址：`http://10.60.1.243:3000`
- 当前 Kafka topic：`prod.memory.task.terminal.v1`

说明：

- 对 Kafka 下游消费者来说，`app-v0317` 的服务地址不是“消费接入必填项”。
- 但建议在对接文档中保留该地址，作为联调、回查、问题定位时的生产者来源信息。

## 3. 链路概览

当前自动推送链路如下：

1. worker 或 control-plane 将任务状态推进到终态
2. control-plane 生成标准 terminal event
3. terminal event 进入 outbox
4. Kafka publisher 从 outbox 取出事件并推送到 Kafka
5. 下游消费者从 `prod.memory.task.terminal.v1` 订阅并处理

推荐把这条链路理解为：

- Kafka 是对下游公开的集成面
- control-plane HTTP 地址是内部排障面

## 4. 是否需要写入 `app-v0317` 服务地址

推荐写入，但定位要准确。

推荐写法：

- `app-v0317` backend 地址用于“生产者来源说明、联调回查、故障排查”
- Kafka 消费侧的接入契约仍然以 `topic + message schema + consumer group` 为准
- 不建议把 `app-v0317` 地址写成 Kafka 消费者的必填参数

建议文档中至少保留以下字段：

| 字段 | 推荐值 | 用途 |
| --- | --- | --- |
| Producer Service | `app-v0317` | 标识消息来源 |
| Backend Address | `http://10.60.1.243:8000` | 联调、排障、回查 |
| Health Check | `GET /api/health` | 验证 control-plane 是否存活 |
| Kafka Topic | `prod.memory.task.terminal.v1` | 下游实际订阅入口 |

## 5. Kafka 接入参数

### 5.1 必填参数

| 参数 | 说明 |
| --- | --- |
| Bootstrap Servers | 由平台侧发放 |
| Topic | `prod.memory.task.terminal.v1` |
| Consumer Group | 由下游系统自行定义 |
| Security Protocol | 由平台侧发放 |
| SASL Mechanism | 由平台侧发放 |
| 用户名 / 密码 | 由平台侧发放 |

### 5.2 当前生产环境约定

当前 `app-v0317` 线上配置已启用 Kafka terminal publisher，topic 为：

```text
prod.memory.task.terminal.v1
```

## 6. 消息模型

### 6.1 事件类型

当前只推送两类终态事件：

- `task.completed`
- `task.failed`

### 6.2 顶层 envelope

每条消息顶层结构如下：

```json
{
  "event_id": "string",
  "event_type": "task.completed | task.failed",
  "schema_version": "v1",
  "occurred_at": "ISO-8601 datetime",
  "task_id": "string",
  "task_version": "string",
  "snapshot_mode": "full | reduced",
  "producer": {
    "service": "memory-engineering",
    "version": "string"
  },
  "payload": {}
}
```

### 6.3 顶层字段说明

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `event_id` | string | 单条事件唯一 ID |
| `event_type` | string | 终态类型 |
| `schema_version` | string | 当前固定为 `v1` |
| `occurred_at` | string | 事件时间 |
| `task_id` | string | 任务 ID |
| `task_version` | string | 任务版本 |
| `snapshot_mode` | string | 负载快照模式 |
| `producer.service` | string | 当前为 `memory-engineering` |
| `producer.version` | string | 当前服务版本 |

## 7. `payload` 结构

业务 payload 结构如下：

```json
{
  "task": {},
  "summary": {},
  "memory_core": {},
  "steps": {},
  "reviews": {},
  "artifacts": {},
  "failure": {}
}
```

### 7.1 `payload.task`

```json
{
  "task_id": "string",
  "user_id": "string",
  "version": "string",
  "status": "completed | failed",
  "stage": "string",
  "upload_count": 0,
  "worker_status": "string | null",
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

### 7.2 `payload.summary`

- 对应任务的 `result_summary`
- 用于下游快速展示摘要信息
- 该字段可能为空

### 7.3 `payload.memory_core`

- 对应任务的 memory core 视图
- 适合下游做轻量检索、摘要展示、关系概览
- 构建失败时可能为 `null`

### 7.4 `payload.steps`

- 对应 LP steps 输出
- 当前仅部分版本支持
- 构建失败时可能为 `null`

### 7.5 `payload.reviews`

- 包含 face review 与 policy 反馈
- 典型用途：下游审核、标注闭环

### 7.6 `payload.artifacts`

```json
{
  "artifact_count": 0,
  "files": [],
  "named_urls": {}
}
```

### 7.7 `payload.failure`

- 仅失败任务或存在错误信息时有值
- 成功任务通常为 `null`

## 8. `snapshot_mode` 说明

Kafka 消息会根据大小自动降级：

- `full`
  - 尽可能携带完整业务 payload
- `reduced`
  - 当消息过大时，对 `artifacts`、`steps`、`reviews` 做裁剪

下游必须兼容：

- 同一个 `event_type` 可能出现不同 `snapshot_mode`
- `memory_core`、`steps`、`failure` 等字段可能为 `null`
- `artifacts.files` 在 `reduced` 模式下可能只保留抽样

## 9. 幂等与去重建议

推荐下游以以下策略进行幂等：

1. 优先使用 `event_id` 做消息去重
2. 如需业务去重，可额外使用 `(task_id, event_type)`
3. 如遇重复推送，以最新 `occurred_at` 或最新消费时间覆盖

## 10. 消费建议

推荐下游消费策略：

1. 首先按 `event_type` 分流
2. 按 `task_id` 落幂等键
3. 保留原始 envelope
4. 将 `payload.task`、`payload.summary`、`payload.memory_core` 分字段落库
5. 如果 `snapshot_mode=reduced`，不要假设缺失字段代表业务为空

## 11. 推荐错误处理

### 11.1 消息字段缺失

- 视为兼容性问题，不应导致消费者整体中断
- 建议落原始消息并记录 schema warning

### 11.2 `memory_core` 或 `steps` 为 `null`

- 视为合法情况
- 推荐继续消费 `task / summary / artifacts / failure`

### 11.3 大消息裁剪

- 通过 `snapshot_mode` 判断
- 不应将 `reduced` 解读为任务异常

## 12. 联调与排障

### 12.1 生产者回查入口

| 项目 | 当前值 |
| --- | --- |
| Producer Service | `app-v0317` |
| Backend Address | `http://10.60.1.243:8000` |
| Health Check | `GET http://10.60.1.243:8000/api/health` |
| Kafka Topic | `prod.memory.task.terminal.v1` |

### 12.2 建议排障顺序

1. 检查 `app-v0317` backend 健康状态
2. 检查任务是否已进入终态
3. 检查 outbox / publisher 日志
4. 检查 Kafka topic 是否有对应 `task_id`
5. 检查下游 consumer group lag

## 13. 不对下游公开的内部接口

以下接口属于内部 worker/control-plane 协作，不建议作为下游接入面：

- `POST /internal/tasks/{task_id}/terminal-update`

这类接口是内部回调链路的一部分，不是 Kafka 消费对接面的正式契约。

## 14. 推荐写法摘要

如果你要把这份文档发给下游，我推荐首页摘要写成下面这样：

```text
Memory Engineering 在任务终态时会向 Kafka topic `prod.memory.task.terminal.v1`
自动推送 `task.completed` / `task.failed` 事件。Kafka 是对下游的正式集成面。

当前生产者服务为 `app-v0317`，backend 地址 `http://10.60.1.243:8000`。
该地址用于联调与排障，不是 Kafka 消费接入的必填参数。
```
