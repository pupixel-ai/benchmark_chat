## AWS Parallel Rollout Plan for Full Relational Persistence

### Goal
在 `v0327-db` 上推进一套“工作流不变、结果双写入库、API 可任意召回”的平行环境。

这轮的目标不是重写现有任务系统，而是：
- 保留当前 `/api/tasks/*`、上传流程、worker 编排和 task-dir 输出
- 在任务完成后把融合后的 runtime contract 镜像到 PostgreSQL
- 提供新的 `/api/users/{user_id}/memory/*` 检索 API
- 不影响现网已有版本：`v0317`、`v0321`、`v0323`、`v0325`

### Current Implementation Reality
当前 `v0327-db` 已经不是最初设想里的 `run_id / stage_run_id` 模型，而是下面这套：

- `task_id` 是唯一结果键
- `dataset_id` 是同一组照片的固定集合键
- 每次重跑都创建一个新的 `task`
- `version` 仍保留原始文本，例如 `v0327-exp`
- 数据库里额外保存数字版版本矩阵：
  - `pipeline_version = 327`
  - `pipeline_channel = "exp"`
  - `face_version / vlm_version / lp1_version / lp2_version / lp3_version / judge_version`
- 阶段级镜像表使用 `task_stage_records`
- 不再使用：
  - `pipeline_runs`
  - `stage_runs`
  - `pipeline_run_stage_bindings`
  - `run_id`

### Hard Rules
- 现有上传和处理工作流不变。
- 历史 task bundle 不是数据库 schema 真源。
- 数据库以当前代码里的融合输出 contract 为语义真源。
- task-dir JSON 和 sidecar 继续保留，但它们是 debug / audit / download 资产，不再是唯一结构化真源。
- 新增 DB mirror 不得破坏老的 `/api/tasks/*`。
- 所有 retrieval API 默认按：
  - `task_id` 精确命中
  - 否则 `dataset_id`
  - 否则最近 dataset 的最近 task

### Canonical Contract
数据库设计以当前融合后的 pipeline contract 为准，而不是旧样例文件。

关键点：
- `memory` 同时保留 `lp1_events` 和 `lp1_events_raw`
- `lp2_relationships` 和 `lp3_profile` 是正式语义对象
- `lp3_profile` 同时包含正式字段和兼容字段
- sidecar artifact 使用 `metadata + payload` 包装
- `result.json` 是兼容 DTO，不是数据库 schema 真源

### Chosen Defaults
- AWS 网络：`office-internal`
- 数据库：PostgreSQL
- 向量扩展：`pgvector`
- 二进制资产：S3
- 部署模式：新的平行 app 环境
- 首轮回填：只处理新任务，旧任务按需 lazy mirror
- 不在本轮落地：
  - Milvus
  - Neo4j
  - 历史任务全量 backfill

### What Ships In v0327-db
当前实现已经落地的能力：

#### 1. Control-plane compatibility
继续保留：
- `users`
- `sessions`
- `tasks`
- `artifacts`
- `face_reviews`
- `face_recognition_image_policies`

`tasks` 已扩成 superset，新增：
- `dataset_id`
- `dataset_fingerprint`
- `pipeline_version`
- `pipeline_channel`
- `face_version`
- `vlm_version`
- `lp1_version`
- `lp2_version`
- `lp3_version`
- `judge_version`

#### 2. Normalized memory-domain tables
已经接入的主表：
- `datasets`
- `task_stage_records`
- `task_photo_items`
- `binary_assets`
- `photos`
- `photo_exif`
- `photo_assets`
- `persons`
- `person_revisions`
- `face_observations`
- `face_embeddings`
- `person_face_links`
- `vlm_observation_revisions`
- `vlm_observation_people`
- `vlm_observation_relations`
- `vlm_observation_clues`
- `event_roots`
- `event_revisions`
- `event_participants`
- `event_photo_links`
- `event_detail_units`
- `relationship_roots`
- `relationship_dossier_revisions`
- `relationship_revisions`
- `relationship_shared_events`
- `group_roots`
- `group_revisions`
- `group_members`
- `profile_context_revisions`
- `profile_revisions`
- `profile_field_values`
- `profile_fact_decisions`
- `judge_decision_revisions`
- `consistency_check_revisions`
- `ground_truth_revisions`
- `ground_truth_assertions`
- `agent_runs`
- `agent_messages`
- `agent_tool_calls`
- `agent_trace_events`
- `agent_outputs`
- `object_registry`
- `object_links`
- `user_heads`

#### 3. Stable asset URLs
任何返回照片/人脸的 API 都走稳定平台 URL，而不是一次性签名链接：
- `/api/assets/photos/{photo_id}/raw`
- `/api/assets/photos/{photo_id}/display`
- `/api/assets/photos/{photo_id}/boxed`
- `/api/assets/photos/{photo_id}/compressed`
- `/api/assets/faces/{face_id}/crop`

### Persistence Strategy In The Current Implementation
当前实现不是“worker 直接写 Postgres”，而是 control-plane 做 DB mirror。

#### Local run path
本地执行结束后：
- pipeline 先照旧写 task-dir / result / artifact manifest
- control-plane 再调用 `MemoryDBSyncService.sync_task_snapshot(...)`
- 把 task 结果镜像到 normalized tables

#### Worker path
worker 完成后：
- control-plane 先把 remote result / artifact manifest 拉回本地 task
- 然后在 control-plane 上调用同一个 `MemoryDBSyncService.sync_task_snapshot(...)`
- 所以当前首轮实现仍然是 control-plane 统一镜像入库

这意味着：
- 首轮不要求 worker 直接访问 Postgres
- 首轮不要求 worker 直接访问 S3 的结构化镜像层
- worker 只需继续完成现有产物输出

### Dataset Semantics
`dataset_id` 由任务上传集合推导：
- 基于排序后的 `source_hash` 集合生成 `dataset_fingerprint`
- 相同照片集的 rerun task 共享同一个 `dataset_id`
- 每个 dataset 记录：
  - `first_task_id`
  - `latest_task_id`
  - `photo_count`

### Version Semantics
数据库查询统一使用数字版版本字段：
- `pipeline_version`
- `pipeline_channel`
- `face_version`
- `vlm_version`
- `lp1_version`
- `lp2_version`
- `lp3_version`
- `judge_version`

解析规则示例：
- `v0327-exp -> pipeline_version=327, pipeline_channel="exp"`
- `v0325 -> pipeline_version=325`

当前实现里，各阶段数字版默认从 task 的 pipeline version 派生；后续如果阶段独立版本真正拆开，可以直接复用同一列而不用改 API 形态。

### Retrieval API Model
新读路径统一走：
- `/api/users/{user_id}/datasets`
- `/api/users/{user_id}/tasks`
- `/api/users/{user_id}/versions`
- `/api/users/{user_id}/memory/faces`
- `/api/users/{user_id}/memory/events`
- `/api/users/{user_id}/memory/vlm`
- `/api/users/{user_id}/memory/profiles`
- `/api/users/{user_id}/memory/relationships`
- `/api/users/{user_id}/memory/photos`
- `/api/users/{user_id}/memory/bundle`

规则：
- `task_id` 最高优先级
- 默认返回最近 dataset 的最近 task
- `all=true` 返回当前筛选范围内全部 task
- `scope=user` 才跨 dataset
- 默认返回 normalized payload
- `include_raw / include_artifacts / include_traces` 才返回重字段

### Lazy Mirror Strategy
为了不阻塞旧任务兼容，当前 retrieval service 会在读取前检查并补齐镜像：
- 老 task 可以在第一次被 `/api/users/{user_id}/memory/*` 读取时补入 normalized tables
- 所以首轮不需要先做历史任务全量 backfill

### Embedding Strategy
当前实现里：
- `face_embeddings.embedding` 使用 `vector(512)`
- 本地运行时如果能从 FAISS / index store 拿到真实向量，就同步入库
- 如果是 worker 回传路径且当前 control-plane 拿不到向量，只会保留：
  - `faiss_id`
  - `embedding_model`
  - `embedding_version`
  - `source_backend`

这意味着当前首轮的“Postgres 作为 embedding 真源”是部分达成：
- 本地链路可达成
- 远端 worker 链路还需要后续补全或回填策略

### Delete Semantics
当前 `v0327-db` 已补上 task 删除时的 DB 清理：
- 删除 `tasks` 旧记录
- 删除该 task 的 normalized mirror rows
- 回收 dataset head / user head
- 保持删除后 `/api/users/{user_id}/memory/*?task_id=...` 不再返回旧镜像数据

### AWS Deployment Shape
- 新建一套平行 app 环境，不覆盖现有版本
- 新建私有 PostgreSQL（启用 `pgvector`）
- 新建独立 S3 bucket 或 prefix
- 新环境只服务 `v0327-db`
- 不修改现有 `app-v0317` 默认行为

### Required Runtime Changes Before AWS
- 增加 PostgreSQL 驱动依赖
- 增加 `pgvector` 运行时依赖
- 为新环境提供正式 migration 入口
- 保留当前 `ensure_schema()` 作为 legacy 补列兜底，而不是长期主迁移手段

### Validation Checklist
- 现有 `/api/tasks/*` 行为不回归
- 新任务完成后，DB 中出现：
  - `datasets`
  - `tasks`
  - `task_stage_records`
  - 各阶段 normalized rows
- `/api/users/{user_id}/memory/*` 可以按：
  - `task_id`
  - `dataset_id`
  - `pipeline_version`
  - `pipeline_channel`
  - 各 stage 数字版本
  精确取数
- 照片和人脸 URL 可直接展示
- 删除 task 后，新 retrieval API 不再读到脏数据

### Out Of Scope For Phase 1
- `run_id` / `stage_run_id` 体系
- worker 直写 Postgres
- Milvus projection schema
- Neo4j projection schema
- 历史任务全量 backfill
