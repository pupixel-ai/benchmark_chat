# 反思自动化审批流程交接文档

## 1. 版本目标与当前状态

本版本已经把反思链路拆成两层：

- `ReflectionAgent`：负责 badcase 反思、改面建议、实验计划与实验结果解读。
- `MemoryEngineerAgent`：负责在审批通过后执行工程落地（双路径：overlay 直复用 / 工程师重写）。

并且已经接入飞书群审批：

- 疑难 case 群：`记忆疑难杂症会诊`
- 审批群：`记忆反思中`

## 2. 端到端流程（运行态）

### 2.1 线上产物采集

主流程产出 `ObservationCase / CaseFact`，并落到用户隔离资产目录：

- `memory/reflection/case_facts_{user}.jsonl`
- `memory/reflection/profile_field_trace_index_{user}.jsonl`
- `memory/reflection/profile_field_trace_payloads/{user}/{case_id}.json`

其中 `LP3 + GT` 对齐信息会写入：

- `comparison_result`
- `comparison_grade`
- `accuracy_gap_status`
- `causality_route`

### 2.2 离线反思生成

入口脚本：

```bash
PYTHONPATH=/Users/vigar07/Desktop/benchmark_chat \
python3 /Users/vigar07/Desktop/benchmark_chat/scripts/run_reflection_tasks.py --user-name <user>
```

核心执行函数：

- `services/reflection/tasks.py::run_reflection_task_generation`

执行内容：

1. 读取 case facts
2. 叠加 GT 对齐（`gt.py` / `gt_matcher.py`）
3. `UpstreamTriageScorer` enrich
4. `UpstreamReflectionAgent` 生成根因与改面建议
5. 聚类生成 `PatternCluster`
6. 生成 `DecisionReviewItem`
7. 生成实验、提案与工程改码请求
8. 按严格通知策略发飞书卡

### 2.3 实验与提案

实验规划与执行在：

- `services/reflection/upstream_agent.py::ExperimentPlanner`
- `services/reflection/upstream_agent.py::ProposalBuilder`

关键规则：

- 已放开：`LP3 + high + mismatch + 单 case` 可进实验链（不再强依赖 `support_count>=2`）。
- 默认 evaluator 新逻辑：
  - 若能从当前 case 读到 `GT + 当前输出`，返回 `completed`（`gt_alignment_without_replay` 模式）。
  - 若读不到，才返回 `need_revision`。

这保证了“有 GT 的真实 case”可以产出反思审批卡，不会被默认 `need_revision` 卡死。

## 3. 双 Agent 职责边界

## 3.1 ReflectionAgent（不改代码）

职责：

- 判断根因家族
- 判断推荐改面
- 请求工具（`trace_diagnose` / `history_recall`）
- 输出中文、可审批结论

不负责：

- 直接改代码
- 直接批准改码

实现位置：

- `services/reflection/upstream_agent.py::UpstreamReflectionAgent`

## 3.2 MemoryEngineerAgent（审批后执行）

职责：

- 读取批准后的提案
- 生成工程改码请求（极简摘要）
- 走执行审批后触发 `MutationExecutor`

双路径：

- `overlay_direct_apply`：实验结果达标时，直接复用 overlay 意图
- `engineer_rewrite_apply`：实验不达标但人工仍批准时，按审批建议工程重写

实现位置：

- `services/reflection/upstream_agent.py::MemoryEngineerAgent`
- `services/reflection/upstream_agent.py::MutationExecutor`

## 4. 飞书审批状态机

### 4.1 卡片类型

- `upstream_decision_task`：方向不稳的人审任务
- `proposal_review`：反思提案审批
- `engineering_execute_review`：执行前最终审批
- `difficult_case`：疑难会诊告警

### 4.2 自动推进

回调入口：

- `backend/app.py` -> `POST /api/integrations/feishu/callback`
- 内部处理：`services/reflection/feishu.py::handle_feishu_callback`

关键动作：

1. `proposal_review` 批准
2. 自动生成 `engineering_execute_review`
3. 自动推送第二张卡到审批群
4. 第二张卡批准后才执行真实改码

## 5. 群路由配置（当前生效）

配置项在 `.env` / `config.py`：

- `FEISHU_APPROVAL_RECEIVE_ID`
- `FEISHU_APPROVAL_RECEIVE_ID_TYPE`
- `FEISHU_DIFFICULT_CASE_RECEIVE_ID`
- `FEISHU_DIFFICULT_CASE_RECEIVE_ID_TYPE`

路由策略：

- 审批类卡片（含第二张执行卡） -> 审批群
- 疑难 case 卡片 -> 会诊群

## 6. 数据资产与可追溯性

用户隔离目录：

- `memory/reflection/*_{user}.jsonl|json`

关键文件：

- `tasks_{user}.jsonl`
- `task_actions_{user}.jsonl`
- `proposals_{user}.jsonl`
- `proposal_actions_{user}.jsonl`
- `engineering_change_requests_{user}.jsonl`
- `difficult_cases_{user}.jsonl`
- `reflection_feedback_{user}.jsonl`
- `upstream_experiments_{user}.json`
- `upstream_outcomes_{user}.json`

所有审批动作都可回放到：

- 谁在何时点了什么按钮
- 对应哪条 task/proposal/change_request

## 7. 运维排障手册

### 7.1 `sent_task_count=0` 常见原因

1. 没有新 task（旧 task 已是 `sent/acted`）
2. pattern 不满足提案条件（当前已放宽单 case mismatch）
3. evaluator 产出 `need_revision`（当前有 GT 时已改为 `completed`）
4. 接收人配置缺失（`FEISHU_*_RECEIVE_ID`）

### 7.2 快速检查命令

```bash
# 看任务与状态
python3 - <<'PY'
import json, pathlib
p=pathlib.Path('/Users/vigar07/Desktop/benchmark_chat/memory/reflection/tasks_youruixun.jsonl')
rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
print(len(rows))
for r in rows[-5:]:
    print(r.get('task_id'), r.get('task_type'), r.get('status'), r.get('feishu_status'))
PY
```

```bash
# 重新生成一轮反思
PYTHONPATH=/Users/vigar07/Desktop/benchmark_chat \
python3 /Users/vigar07/Desktop/benchmark_chat/scripts/run_reflection_tasks.py --user-name youruixun
```

## 8. 当前关键约束

1. 反思结论默认使用中文、人话输出，避免术语堆叠。
2. 审批前不执行真实改码。
3. 第二张执行审批卡不通过，不会写入 repo-tracked 规则资产。
4. 单位粒度仍是“单字段 + 单改面”。

## 9. 交接建议

后续接手同学先按这个顺序：

1. 确认 `.env` 的飞书群路由配置
2. 用一个真实 user 跑 `run_reflection_tasks.py`
3. 检查 `tasks/proposals/change_requests` 是否更新
4. 在飞书走一遍两阶段审批（proposal -> execute）
5. 检查规则资产是否在执行审批通过后才变更

