# Memory Graph vNext

## Goal
- 面向 agent 调用和答案准确性，而不是 graph 展示
- `Neo4j = facts + hypotheses`
- `Milvus = semantic evidence`
- `Artifacts = raw outputs`

## Current vNext scope
- `Photo` 不进入 Neo4j
- `OperatorPlan` 取代持久化 template
- `Event` / `RelationshipHypothesis` / `MoodStateHypothesis` / `PeriodHypothesis` 默认是 hypothesis
- 统一对外输出 `AnswerDTO`

## Implemented building blocks
- Canonical graph schema:
  - `User`
  - `Person`
  - `PlaceAnchor`
  - `Session`
  - `DayTimeline`
  - `Event`
  - `RelationshipHypothesis`
  - `MoodStateHypothesis`
  - `PrimaryPersonHypothesis`
  - `PeriodHypothesis`
  - `Concept`
- Relationship revision model:
  - `draft / active / cooling / superseded / rejected`
  - 历史指标继承
  - 升降级阈值
- Agent query path:
  - `Natural language -> OperatorPlan -> Concept/Entity recall -> QueryDSL -> graph execution -> evidence fill -> AnswerDTO`
- Backend API:
  - `POST /api/tasks/{task_id}/memory/query`

## Evidence binding
- Neo4j 与 Milvus 通过 canonical IDs 绑定：
  - `event_uuid`
  - `session_uuid`
  - `person_uuid`
  - `relationship_uuid`
  - `concept_uuid`

## Validation focus
- 时间类查询
- 关系探索
- 关系排序
- 最近心情
- 可解释证据链
