# 反思与审批链路 README

## 1. 这部分代码在解决什么

这一套链路不是主记忆生成链路本身，而是主链路跑完之后的“问题沉淀、审计回流、人工审批、工程落地”系统。

它主要负责四件事：

1. 把主链路和下游审计里暴露出来的问题沉淀成可追踪 case
2. 把 case 归类成 pattern、任务、疑难问题和工程告警
3. 把适合继续推进的问题送进审批链
4. 在审批通过后，把批准的规则修改真正落到仓库规则资产里

换句话说，这不是“再跑一遍记忆生成”，而是把 badcase 变成一条能闭环的工程工作流。

---

## 2. 代码范围

这条链跨两个目录：

- `services/memory_pipeline/downstream_audit.py`
- `services/memory_pipeline/profile_agent_adapter.py`
- `services/reflection/`

其中分工很明确：

- `memory_pipeline/downstream_audit.py`
  - 负责把主工程输出映射到下游 `profile_agent`
  - 跑 `Critic -> Judge`
  - 生成 `downstream_audit_report`
  - 按裁决把结果回流到主工程对象
- `reflection/`
  - 负责把主工程和下游审计暴露的问题沉淀成 case
  - 做 triage、聚类、任务生成、Feishu 通知、审批回调和规则落地

---

## 3. 全链路总览

```text
主链路输出
  -> downstream_audit
  -> backflow 回主工程
  -> persist_mainline_reflection_assets
  -> persist_downstream_audit_reflection_assets
  -> observation_cases / case_facts
  -> GT 对齐 + upstream triage
  -> pattern clustering
  -> decision tasks / difficult cases / engineering alerts
  -> strategy experiments
  -> proposal review
  -> engineering execute review
  -> apply_repo_rule_patch
  -> compile validation + outcome 落盘
```

这条链里真正的中枢对象只有一个：

- `CaseFact`

所有后续的 pattern、task、proposal、change request，都是围绕 `CaseFact` 继续加工出来的。

---

## 4. 在线主流程里的入口

主流程入口在 `main.py` 的 `run_memory_pipeline_entry()`。

它的顺序是：

1. 跑 `run_memory_pipeline()`
2. 调 `run_downstream_profile_agent_audit()`
3. 按 audit 结果依次做三类 backflow
4. 必要时重跑主链路局部阶段
5. 把主链路内部产物写入 mainline reflection
6. 把 downstream audit 裁决写入 downstream reflection

这里有一个重要事实：

- 反思链路不是在 backflow 之前抓快照
- 它是在“主工程已经接受下游裁决并完成必要 rerun”之后再把结果沉淀成 case

所以它记录的是“本轮最终真实工程状态”，不是最早的原始状态。

---

## 5. 下游审计链路

## 5.1 适配层

入口：

- `services/memory_pipeline/profile_agent_adapter.py`

作用：

- 把主工程的 `primary_decision / relationships / structured_profile`
  映射成下游 `profile_agent` 能消费的三路 `ExtractorOutput`

三路固定是：

- `protagonist`
- `relationship`
- `profile`

适配层只做“格式对齐 + 证据抽取”，不做新的业务判断。

## 5.2 审计层

入口：

- `services/memory_pipeline/downstream_audit.py`

作用：

1. 调 `Critic`
2. 对被 challenge 的维度做 V2 定向补证
3. 调 `Judge`
4. 生成 `downstream_audit_report`
5. 生成 backflow action

关键约束：

- 只有 `Judge` 的 `nullify / downgrade` 才会在 reflection 里变成 downstream case
- `accept` 不会变成 downstream 反思 case

## 5.3 三类回流

入口都在 `services/memory_pipeline/downstream_audit.py`：

- `apply_downstream_protagonist_backflow()`
- `apply_downstream_relationship_backflow()`
- `apply_downstream_profile_backflow()`

行为差异：

- 主角回流：
  - 可能把主角改成 `photographer_mode`
  - 可能触发“关系清空 -> 画像重跑”
- 关系回流：
  - 先改正式 `Relationship[]`
  - 再重跑 `groups -> LP3 -> downstream audit`
- 画像回流：
  - 直接改 `structured_profile`
  - 同步回写 `profile_fact_decisions.final`
  - 保留 `final_before_backflow + backflow`

所以这条链不是只做“报警”，它是真的会改变主工程最终产物。

---

## 6. Reflection 资产捕获

## 6.1 mainline capture

入口：

- `services/reflection/mainline_capture.py`

抓三类东西：

1. `primary_decision`
2. `relationship_dossiers`
3. `profile_fact_decisions`

输出两层资产：

- `ObservationCase`
- `CaseFact`

此外还会额外写：

- `profile_field_trace_index`
- `profile_field_trace_payloads/<case_id>.json`

这部分最重要的作用是把 LP3 字段判定时的完整上下文单独存出来，否则后面做 GT 对账和提案审批时会丢证据。

## 6.2 downstream capture

入口：

- `services/reflection/downstream_capture.py`

它只捕获两类问题：

1. 下游 audit 初始化失败
2. `Judge` 把某个维度判成 `nullify / downgrade`

也就是说：

- 下游 audit 不是把所有 tag 都转成 case
- 只有真正出现分歧、需要回看和纠偏的裁决才会进入 reflection

---

## 7. Reflection 的核心数据对象

定义都在：

- `services/reflection/types.py`

最重要的对象如下。

## 7.1 `ObservationCase`

它是“观测记录”。

适合回答：

- 这条问题是什么
- 它第一次在哪个阶段出现
- 它是在什么时候被 surfacing 出来的
- 当时原始 payload 长什么样

它更像 case 卡片，而不是工程决策表。

## 7.2 `CaseFact`

它是“工作中枢对象”。

后续的 triage、pattern、task、proposal、change request 基本都围绕它展开。

它会逐步被补充这些信息：

- 路由结果 `routing_result`
- 业务优先级 `business_priority`
- 准确性缺口状态 `accuracy_gap_status`
- 根因家族 `root_cause_family`
- 修复面置信度 `fix_surface_confidence`
- 解决路径 `resolution_route`
- GT 对比结果
- downstream disagreement

## 7.3 其他重要对象

- `PatternCluster`
  - 多个相似 case 的聚类结果
- `DecisionReviewItem`
  - 需要人工点按钮的审批任务
- `DifficultCaseRecord`
  - 方向不清、证据不足或需要人工判断的疑难问题
- `EngineeringAlert`
  - 运行时 / 工程层面的问题告警
- `StrategyExperiment`
  - 已经进入策略实验阶段的问题
- `ProposalReviewRecord`
  - 实验完成后等待产品/策略批准的提案
- `EngineeringChangeRequest`
  - 提案批准后、等待工程执行批准的改码包
- `ExperimentOutcome`
  - 最终执行后的结果记录

---

## 8. 路由与分流逻辑

## 8.1 第一层路由：`route_case_fact`

入口：

- `services/reflection/triage.py`

这是最粗粒度的初分流。

它会先把 case 分成几类：

- `engineering_issue`
- `audit_disagreement`
- `strategy_candidate`
- `expected_uncertainty`
- `pending_triage`

核心规则：

- `system_runtime` 直接进 `engineering_issue`
- `downstream_audit` 来源直接进 `audit_disagreement`
- `mainline_primary / mainline_relationship / mainline_profile` 会进 `strategy_candidate`
- 被判为“正常不确定性”的关系 suppression 会进 `expected_uncertainty`

## 8.2 第二层路由：upstream triage

入口：

- `services/reflection/upstream_triage.py`

只重点处理：

- `mainline_profile` 的上游画像 badcase

它会基于 GT、工具命中、历史复现和 LLM scorer，判断根因家族：

- `field_reasoning`
- `evidence_packaging`
- `tool_retrieval`
- `tool_selection_policy`
- `orchestration_guardrail`
- `engineering_issue`
- `watch_only`

然后再把 case 送去：

- `strategy_fix`
- `engineering_fix`
- `difficult_case`

这里的核心思想是：

- 先判断“错在哪一层”
- 再决定“要不要进入策略修复”

---

## 9. Pattern、任务与疑难问题

入口：

- `services/reflection/tasks.py`

## 9.1 pattern clustering

函数：

- `build_pattern_clusters()`

作用：

- 把多个同类 `CaseFact` 聚成一个 `PatternCluster`

聚类结果会分成两条 lane：

- `upstream`
- `downstream`

它还会把 `engineering_issue` 直接转成 `EngineeringAlert`，不走 pattern。

## 9.2 decision task

函数：

- `build_decision_tasks()`

它只给“方向还不清、但足够重要”的 pattern 生成人工决策任务。

典型任务类型：

- `upstream_decision_task`
- `downstream_decision_task`

这类任务不是让人直接改代码，而是让人先决定“应该修哪一面”。

## 9.3 difficult case

函数：

- `build_difficult_cases()`

它把那些：

- 方向不清
- 证据不足
- GT 只部分匹配
- 需要人工确认

的问题单独拉出来，进入疑难问题通道。

---

## 10. 实验、提案与审批链

这是新增链路里最重要的部分。

## 10.1 从 pattern 到 experiment

函数：

- `build_strategy_experiments()`
- `_plan_strategy_experiments_with_proposals()`

只有同时满足这些条件的 upstream pattern，才会进入实验：

- `resolution_route == strategy_fix`
- 高优先级
- 方向清晰
- 推荐修复面属于 `field_cot / tool_rule / call_policy`

实验对象是：

- `StrategyExperiment`

实验会生成：

- `override_bundle.json`
- `experiment_report.json`

当前默认 evaluator 还是保守模式：

- 如果 support case 有 GT 对齐信息，就做“GT 对齐但不回放”的轻评估
- 如果缺少可重跑离线 bundle，就返回 `need_revision`

## 10.2 从 experiment 到 proposal

类：

- `ProposalBuilder`

只有实验结果 `status == completed` 才会生成提案。

提案对象是：

- `ProposalReviewRecord`

同时还会生成一个人工审批任务：

- `DecisionReviewItem(task_type="proposal_review")`

这一步的含义是：

- 实验已经说明“往哪个方向改可能有价值”
- 但还没有得到人工批准

## 10.3 proposal approve 后发生什么

入口：

- `services/reflection/feishu.py`
  - `apply_reflection_task_action()`

如果人工在 proposal review 上点了批准：

1. proposal 状态变成 `approved_for_engineering`
2. `MemoryEngineerAgent` 生成 `EngineeringChangeRequest`
3. 同时生成新的审批任务：
   - `DecisionReviewItem(task_type="engineering_execute_review")`
4. 这条新的执行审批任务会再次发到 Feishu

也就是说：

- proposal approval 不是直接改代码
- proposal approval 只是把它推进到“工程执行审批”

## 10.4 engineering execute approve 后发生什么

类：

- `MutationExecutor`

如果人工在 engineering execute review 上点了批准：

1. `MutationExecutor.execute_change_request()` 执行
2. `apply_repo_rule_patch()` 把 patch 写到规则资产
3. 跑最小 compile 验证
4. 写 `ExperimentOutcome`

当前它改的是 repo-tracked 规则资产，不是直接让 agent 改业务 Python 逻辑。

所以这套链当前更像：

- 策略规则改动审批链

而不是：

- 任意代码变更审批链

---

## 11. 通知系统（待重新设计）

飞书通知逻辑已从反思链路中完全移除（`dispatch_strict_reflection_notifications` 等函数已删除）。
`services/reflection/feishu.py` 等文件保留代码壳，待后续重新设计通知方案。

当前行为：反思任务和疑难 case 只会落盘到 JSONL 文件，不会自动推送通知。

---

## 12. 落盘目录

所有 reflection 资产都在：

- `memory/reflection/`

关键文件如下：

- `observation_cases_{user}.jsonl`
- `case_facts_{user}.jsonl`
- `profile_field_gt_{user}.jsonl`
- `gt_comparisons_{user}.jsonl`
- `profile_field_trace_index_{user}.jsonl`
- `profile_field_trace_payloads/{user}/{case_id}.json`
- `upstream_patterns_{user}.json`
- `downstream_audit_patterns_{user}.json`
- `engineering_alerts_{user}.jsonl`
- `difficult_cases_{user}.jsonl`
- `tasks_{user}.jsonl`
- `task_actions_{user}.jsonl`
- `proposals_{user}.jsonl`
- `proposal_actions_{user}.jsonl`
- `engineering_change_requests_{user}.jsonl`
- `reflection_feedback_{user}.jsonl`
- `upstream_experiments_{user}.json`
- `upstream_outcomes_{user}.json`
- `experiments/{user}/{experiment_id}/override_bundle.json`
- `experiments/{user}/{experiment_id}/experiment_report.json`

补充说明：

- `decision_review_items_{user}.json` 这个路径当前在 `storage.py` 里有定义，但主流程现在实际使用的是 `tasks_{user}.jsonl`

其中最关键的几层分别是：

- 原始 case 层：`observation_cases / case_facts`
- 诊断层：`profile_field_trace_* / gt_comparisons`
- 聚类与任务层：`*_patterns / tasks / difficult_cases / engineering_alerts`
- 审批层：`proposals / engineering_change_requests / *_actions / reflection_feedback`
- 执行层：`upstream_outcomes / experiments/*`

---

## 13. 这条链的真实边界

当前这套系统已经做到：

- 主工程问题可沉淀成 case
- 下游审计分歧可回流成 case
- GT 可参与上游 badcase 判断
- pattern 可自动生成
- 任务可发 Feishu
- 提案批准后可继续走工程执行审批
- 批准后的规则 patch 可真正写回 repo 规则资产

但它当前还不是：

- 通用工单系统
- 通用代码审批系统
- 任意 Python 改动自动合并系统

它现在最适合做的是：

- LP3 规则资产
- 证据打包规则
- 调用策略
- 与 badcase 直接相关的受控策略修改

---

## 14. 阅读顺序建议

如果要继续改这套链，建议按这个顺序读代码：

1. `main.py`
2. `services/memory_pipeline/downstream_audit.py`
3. `services/reflection/mainline_capture.py`
4. `services/reflection/downstream_capture.py`
5. `services/reflection/triage.py`
6. `services/reflection/upstream_triage.py`
7. `services/reflection/tasks.py`
8. `services/reflection/upstream_agent.py`
9. `services/reflection/feishu.py`
10. `services/reflection/types.py`

这样最容易先建立“主流程”，再进入“审批细节”。

---

## 15. 一句话总结

这套“反思 + 审批链”本质上是在做一件事：

把主记忆工程里出现的问题，从一次性的 badcase，变成一条可追踪、可审批、可落规则、可记录结果的工程闭环。
