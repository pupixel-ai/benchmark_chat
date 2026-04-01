# Harness Engineering 完整策略

> 更新时间：2026-03-28
> 版本：v1.0

---

## 概述

Harness Engineering 是记忆工程的反思子系统，包含两条独立但共享底层能力的链路：

1. **单用户自回归**（Nightly Evolution）：每晚对单个用户的数据集反复评测，逐字段找到最优补丁
2. **跨用户聚合分析**（Cross-User Harness）：聚合多用户 bad case，发现系统性问题模式

两条链路共享同一个**诊断引擎**（`services/reflection/engine/`），确保诊断标准一致。

---

## 架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│                    共享诊断引擎 (engine/)                         │
│  field_diagnostics.py — 字段诊断 (failure_mode + root_cause)     │
│  signal_extractor.py  — 信号提取 (overlooked_signals)            │
│  patch_planner.py     — 补丁规划 (override_bundle)               │
└──────────────┬───────────────────────────┬───────────────────────┘
               │                           │
    ┌──────────▼──────────┐     ┌──────────▼──────────────────┐
    │  链路 A: 单用户自回归  │     │  链路 B: 跨用户聚合分析       │
    │  (Nightly Evolution) │     │  (Cross-User Harness)       │
    │                      │     │                              │
    │  输入: 1 用户 × N 轮  │     │  输入: M 用户 × 各自 case     │
    │  目标: 逐字段迭代优化  │     │  目标: 发现系统性问题模式      │
    └──────────────────────┘     └──────────────────────────────┘
```

---

## 链路 A：单用户自回归（Nightly Evolution）

### 目标

给定一个用户的数据集和 GT，通过多轮次自回归，为每个问题字段找到最优补丁。

### 执行流程

```
每晚定时触发 run_memory_nightly_evaluation()
      │
      ▼
[加载] traces + GT 比对结果 + 字段循环持久状态
      │
      ▼
[选择] Top-K 问题字段（按 issue_score 排序）
      │  issue_score = grade_weight + severity_weight + score_penalty
      │                + field_risk_weight + recency_bonus
      │
      ▼
[循环] 对每个 focus 字段执行一轮迭代：
      │
      ├─ cooldown > 0 → 跳过（降频中）
      │
      ├─ engine.diagnose_field()
      │  → failure_mode (missing_signal / wrong_value / overclaim / partial_coverage)
      │  → root_cause_family (field_reasoning / tool_retrieval / ...)
      │  → coverage_gap (4 类结构性缺口检测)
      │
      ├─ engine.extract_signals()
      │  → 从 GT 差异中提取遗漏线索
      │  → 与 seen_signal_keys 比对判断是否为新信号
      │
      ├─ 收敛检测
      │  ├─ 新信号 → streak = 0，继续
      │  └─ 无新信号 → streak += 1
      │     └─ streak ≥ 阈值 → cooldown = N 轮（降频，让位给其他字段）
      │
      ├─ engine.plan_patch()
      │  → 生成 patch_preview (call_policy / tool_rule / field_spec)
      │
      └─ 输出 cycle_entry + proposal（如有新信号 + patch）
      │
      ▼
[持久化] field_loop_state（跨天累积）+ 日报 + 提案
      │
      ▼
[人在平台上]
  ├─ 查看字段迭代详情（系统输出 vs GT + 证据 + 亮点线索）
  ├─ 点击「执行实验」→ 沙盒验证补丁效果
  └─ Approve → apply_repo_rule_patch() 写入规则变更
```

### 关键机制

| 机制 | 说明 |
|------|------|
| **Top-K 字段选择** | 按 issue_score 排序，优先处理问题最大的字段 |
| **信号去重** | 每轮提取的信号与 seen_signal_keys 比对，只有新信号才推进迭代 |
| **收敛降频** | 连续 N 轮无新信号 → 暂停该字段 M 轮，预算让给其他字段 |
| **cooldown 衰减** | 每晚自动 cooldown -= 1，到 0 后恢复活跃 |
| **半自动验证** | 人在平台上手动触发实验，看 baseline vs candidate 对比后再 approve |

### 相关文件

| 文件 | 职责 |
|------|------|
| `services/memory_pipeline/evolution.py` | Nightly 编排：`run_memory_nightly_evaluation()` |
| `services/reflection/engine/field_diagnostics.py` | 字段诊断（共享） |
| `services/reflection/engine/signal_extractor.py` | 信号提取（共享） |
| `services/reflection/engine/patch_planner.py` | 补丁规划（共享） |
| `scripts/run_memory_nightly_eval.py` | CLI 入口 |

---

## 链路 B：跨用户聚合分析（Cross-User Harness）

### 目标

聚合多个用户的 bad case，找出在多用户中重复出现的系统性问题，由 EngineeringCritic（LLM）做深度批评分析。

### 执行流程

```
run_harness_engineering(user_names)
      │
      ▼
[加载] 所有用户的 case_facts
      │
      ▼
[聚类] build_cross_user_patterns()
      │  聚类 key: (lane, normalized_dimension, failure_mode, root_cause)
      │  lane: protagonist / relationship / profile
      │  relationship 按关系类型聚类（不按 Person_ID）
      │
      │  输出:
      │  - affected_users: 哪些用户受影响
      │  - user_coverage: 影响百分之几的用户
      │  - per_user_examples: 每个用户的典型 case
      │  - cross_user_consistency: 跨用户根因一致性
      │
      ▼
[缺失能力检测] detect_missing_capabilities()
      │  规则检测 3 类缺失：
      │  1. 域级信息荒漠：某字段域在 >50% 用户中全空
      │  2. 修复面反复失败：同一 fix_surface 被推荐 ≥3 次但从未成功
      │  3. 结构性缺口：CoverageProbe 检测到的 gap 跨多字段出现
      │
      ▼
[疾病分类] detect_diseases()
      │  将问题归类为有名字的"病"：
      │
      │  主角类：
      │  - 脸部缺失症（相册无人脸）
      │  - 主角模糊症（多人频率接近）
      │
      │  关系类：
      │  - 场景缺失症（某关系类型缺场景证据）
      │  - 自我误识症（用户本人被误识为他人）
      │  - 虚拟人物症（AI 生图/截图被当作真实关系）
      │
      │  画像类：
      │  - 信息荒漠（某字段域跨用户为空）
      │  - COT 无效循环（fix_surface 反复推荐但无效）
      │  - 姓名不可得症（照片不含姓名信息）
      │
      ▼
[专家批评] EngineeringCritic（LLM，O(M) 次调用）
      │
      │  对 top-10 跨用户 pattern 做分步对话：
      │
      │  Step 1: 输入 pattern 摘要 + policy + field_index
      │         → 初步诊断 + evidence_requests（Critic 说它想看什么证据）
      │
      │  Step 2: 按请求给具体证据
      │         → 系统性诊断 + 分级建议 + 知识更新提案
      │
      │  Step 3: (可选) 若 Critic 请求外部知识工具
      │         → 当前返回 placeholder，未来接 MCP/RAG
      │
      ▼
[输出] harness_engineering_report.json
      ├─ cross_user_patterns（跨用户模式列表）
      ├─ critic_reports（专家批评报告）
      ├─ missing_capabilities（缺失能力清单）
      └─ diseases（疾病分类）
```

### EngineeringCritic 详细设计

#### 分级质疑

| 等级 | 触发条件 | 输出 |
|------|---------|------|
| **Level 1 策略** | 所有 pattern（默认） | 具体的 COT/tool_rule/call_policy 修改建议 |
| **Level 2 工具** | field_index 历史失败 ≥2 次，或 user_coverage ≥ 50%，或 CoverageProbe 有 gap | 新工具/新数据源需求 |
| **Level 3 架构** | Level 2 也被人否决过，或同一问题跨 3+ 轮未解决 | Agent 流程变更建议 |

#### 知识架构

```
Layer 1: System Policy (critic_policy.md)
  ├─ 工程链路理解（VP1→LP1→LP2→LP3 各环节能力边界）
  ├─ Agent 设计原则（证据闭环、代码做重活等）
  ├─ 行为宪法（质疑等级规则）
  └─ 通用经验（Critic 自己写入，受约束）
  加载: 每次完整加载（~2K tokens，稳定不膨胀）
  更新: Agent 可自更新，但每周 ≤3 条 + 需 ≥2 个 pattern 支撑 + 7 天冷却期

Layer 2: Field Index (critic_field_index.json)
  ├─ {field_key → [{diagnosis, weight, human_verdict, outcome}]}
  └─ 每字段最多 5 条，weight < 0.2 自动淘汰
  加载: 只加载当前 pattern 涉及的字段（~200 tokens/字段）
  更新: 每次 Critic 运行后自动写入
  衰减: 每 30 天 weight × 0.9

Layer 3: External Tools (预留接口)
  ├─ search_industry_knowledge（行业知识/论文/基准）
  └─ 仅 Level 3 可请求，当前返回 placeholder
```

#### 学习闭环

```
Critic 运行 → 输出诊断 + 建议 → 写入 field_index
      │
      ▼
人类在平台审批 → approve/reject + 备注
      │
      ├─ approve → field_index.weight += 0.15
      ├─ reject → field_index.weight -= 0.20
      └─ need_revision → field_index.weight -= 0.05
      │
      ▼
下次 Nightly 运行 → 新 GT 对照 → 验证效果
      │
      ├─ grade 改善 → field_index.weight += 0.10
      └─ 未改善 → field_index.weight -= 0.10
      │
      ▼
下次 Critic 运行 → prompt 自动包含上次经验
```

### 相关文件

| 文件 | 职责 |
|------|------|
| `services/reflection/harness_engineering.py` | 跨用户入口：`run_harness_engineering()` |
| `services/reflection/engineering_critic.py` | EngineeringCritic（分步对话 + 分级质疑） |
| `services/reflection/critic_knowledge.py` | 知识管理：Policy + Field Index + 自进化 |
| `services/reflection/difficult_case_taxonomy.py` | 疾病分类检测 |
| `services/reflection/harness_types.py` | 数据结构：CrossUserPattern, CriticReport 等 |
| `memory/reflection/critic_policy.md` | 通用知识库 |
| `memory/reflection/critic_field_index.json` | 字段级索引 |
| `scripts/run_reflection_full_chain.py` | 全链路自动化脚本 |

---

## 两条链路的关系

```
                     ┌──────────────────────┐
                     │  共享诊断引擎 engine/ │
                     │  diagnose_field()    │
                     │  extract_signals()   │
                     │  plan_patch()        │
                     └──────┬───────┬───────┘
                            │       │
              ┌─────────────▼─┐   ┌─▼─────────────────┐
              │ 链路 A: 单用户 │   │ 链路 B: 跨用户      │
              │ Nightly       │   │ Harness            │
              │               │   │                    │
              │ 逐字段迭代     │   │ 跨用户聚类          │
              │ 信号提取       │   │ EngineeringCritic  │
              │ 收敛检测       │   │ 疾病分类            │
              │ 补丁生成       │   │ 缺失能力检测        │
              └───────┬───────┘   └─────────┬──────────┘
                      │                     │
                      │   ┌─────────────┐   │
                      └──▶│  提案审批队列  │◀──┘
                          │  (统一)      │
                          └──────┬──────┘
                                 │
                          ┌──────▼──────┐
                          │ approve     │
                          │ → apply     │
                          │ rule patch  │
                          └─────────────┘
```

**链路 A → 链路 B 的数据流**：
- 链路 A 的 case_facts（单用户 bad case）是链路 B 的输入
- 链路 A 的 Evolution field_loop_state 被链路 B 的 Critic 引用为 evolution_context

**链路 B → 链路 A 的数据流**：
- 链路 B 的 CriticReport 建议（Level 1 补丁）可以通过提案审批后被 apply
- apply 后下次链路 A 运行时，该字段的 rule_assets 已更新，评测结果会反映改善

---

## 平台展示

| 页面 | URL | 对应链路 |
|------|-----|---------|
| 总览（热力图 + 用户卡片） | `/` | 跨链路 |
| 用户详情（日报 + 字段迭代） | `/user/:name` | 链路 A |
| Harness（画像/关系/主角 Tab） | `/harness` | 链路 B |
| 疑难杂症（三栏病症卡片） | `/difficult` | 链路 B |
| 提案审批（合并队列） | `/proposals` | 跨链路 |

---

## 自动化运行

```bash
# 完整链路（Nightly + 单用户 Harness + 跨用户 Harness）
python scripts/run_reflection_full_chain.py --date 2026-03-28

# 只跑 Nightly
python scripts/run_memory_nightly_eval.py --user-name youruixun --date 2026-03-28

# 只跑跨用户分析
python scripts/run_reflection_full_chain.py --skip-nightly --skip-harness
```

---

## 未来扩展

| 方向 | 说明 | 状态 |
|------|------|------|
| **外部知识工具** | Critic Level 3 接入行业知识/论文/基准 | 接口预留 |
| **沙盒自动验证** | approve 后自动重跑目标字段验证效果 | placeholder |
| **EngineeringCritic 多轮对话** | 当前 2 步，未来可扩展为多轮工具调用 | 架构支持 |
| **跨用户 pre_cluster** | 聚类前置（Phase 2 A2），LLM O(N)→O(K) | 待实现 |
