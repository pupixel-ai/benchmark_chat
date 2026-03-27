# Memory Pipeline 自进化闭环（Proposal-First）

## 目标

把主链路运行结果结构化记录为 `run trace`，并通过 nightly 生成 `insight/proposal`，由人工确认后再应用到 `rule_assets`。

原则：

- 自动采集、自动评测
- 不自动改规则（proposal-first）
- 用户隔离（按 `user_name` 分桶）

## Trace 采集入口

以下入口现在会自动写出 `memory_pipeline_run_trace.json`，并追加到 `memory/evolution/traces/{user}/{date}.jsonl`：

- `main.py` 主链路
- `services/memory_pipeline/precomputed_bundle_runner.py`
- `services/memory_pipeline/reusable_smoke_runner.py`

## Nightly 评测

```bash
python3 scripts/run_memory_nightly_eval.py --user-name "vigar" --date "2026-03-27"
```

新模式（多用户集 + GT 字段循环）：

```bash
python3 scripts/run_memory_nightly_eval.py \
  --user-names "youruixun,default" \
  --date "2026-03-27" \
  --top-k-fields 3
```

字段循环逻辑：

- 每个用户优先读取 `memory/reflection/gt_comparisons_{user}.jsonl`
- 按 GT 对齐问题分数筛出 TopK 字段（默认 3 个）
- 预算优先给活跃字段：`cooldown_remaining > 0` 的字段会被标记为 throttled，仅在活跃字段不足时补位
- 活跃字段内部会优先「最近有新线索/新规则候选」且 `no_new_signal_streak` 更低的字段
- 每个字段进入 nightly 循环状态机（cycle 递增，不会每天重置）
- 连续 `N` 轮无新线索会自动触发降频（throttle），进入 cooldown 窗口，把预算集中到仍有增益字段
- 每轮输出：
  - `focus_fields`（本轮关注字段）
  - `field_cycles`（失败模式、忽视线索、启发、状态、streak/cooldown）
  - `proposals`（仅有明确规则候选时输出 patch_preview）

循环状态保存：

- `memory/evolution/field_loop_state/{user}.json`

可选环境变量（默认值）：

- `FIELD_LOOP_NO_SIGNAL_STREAK_THRESHOLD=2`：连续多少轮无新线索后触发降频
- `FIELD_LOOP_THROTTLE_COOLDOWN_ROUNDS=2`：降频窗口持续多少轮

输出：

- `memory/evolution/reports/{user}/{date}.json`
- `memory/evolution/insights/{user}/{date}.json`
- `memory/evolution/proposals/{user}/{date}.json`
- `memory/evolution/focus_fields/{user}/{date}.json`
- `memory/evolution/field_cycles/{user}/{date}.json`
- 多用户聚合：`memory/evolution/reports/_user_set/{date}.json`

## 人工应用 Proposal

```bash
python3 scripts/apply_memory_evolution_proposal.py \
  --proposal-file "memory/evolution/proposals/vigar/2026-03-27.json" \
  --proposal-id "proposal_20260327_001" \
  --actor "manual_cli"
```

说明：

- 若 `patch_preview` 为空，则只记录 action，不改规则
- 若 `patch_preview` 有内容，会写入 `services/memory_pipeline/rule_assets/*.json`
- 行为会记录到 `memory/evolution/proposal_actions_{user}.jsonl`
