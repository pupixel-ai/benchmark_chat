# 记忆工程 v2.0 - Codex 工作指南

## Agent 行为规范（必须遵守）

### 开局
每次新对话开始，先读本文件 + 下方「核心知识速览」，根据用户任务按需加载 `.knowledge/` 中的相关文件。

### 过程中
遇到不确定的项目知识，先查 `.knowledge/` 再问用户。

### 收尾 — 知识维护 Checklist
每次对话结束前，逐条过一遍，**不需要用户提醒**：

- □ 新的设计决策或架构变更？→ 更新 `.knowledge/base/` 或 `roles/`
- □ 新的可复用操作技巧？→ 写入 `skills/`
- □ 新的设计红线或原则？→ 写入 `principles/`
- □ 新的跨模块深层规律？→ 写入 `insights/`
- □ 新的未解决问题？→ 写入 `questions/open_questions.md`
- □ 已有知识过时或错误？→ 直接更新或删除对应文件
- □ 核心知识有变动？→ 同步更新下方「核心知识速览」和 `memory/MEMORY.md`

### 知识库完整索引
`.knowledge/INDEX.md`

---

## 核心知识速览（自动加载，无需跳转）

**设计原则 (principles)**
- **代码做重活，LLM 只做判定**: 能用确定性计算的不交给 LLM（如亲密度打分、频率统计、异常检测）
- **证据闭环**: 所有推断必须基于客观可观测证据，拒绝无根据猜测，低置信度标"疑似/待观察"
- **用户数据隔离**: 不同用户的缓存/输出/人脸库严格隔离，不允许交叉污染
- **阶段优先级明确**: 当前阶段优先服务产品经理跑通链路、调试全流程策略；完整标注层、全链路追溯、回写融合属于下一阶段工程化建设
- **北极星目标**: 记忆系统最终需要具备“可标注、可追溯、可回写、可约束”四个能力，但现阶段以可观察、可调试、可复盘为第一目标
- **方案评估思维链**: 影响核心策略？→ 真/伪问题？→ 数据流追踪 → 最简解是否删除 → 不被原方案框住

**深层洞察 (insights)**
- **结构化枚举 > 自由文本**: VLM→代码数据流转中，枚举字段（如 contact_type）可靠性远高于自由文本（如 interaction），新增字段优先设计为枚举
- **事件召回策略**: 首尾必选（标记关系时间线）+ 中间按重要性排序（抓高光时刻）+ 硬上限 10 条（防 prompt 膨胀）

**待解决问题 (questions)**
- 数据 Schema 标准化方案待定（dataset_id 引入方式、ID 全局唯一）
- VLM 修复 boxed_path 后，需观察 VLM 是否因看到人脸框而输出"框"相关噪声描述

**待实施方案 (plans/)**
- P0: 观测性增强（关键中间产物可检查、关键字段可比对、问题可定位）
- P2: LP1 分段传输（30天窗口 + 50条阈值 + 上下文滚动）
- P3: LP3 已切到 Gemini 对齐架构：`1 个 ProfileAgent + domain batch + horizontal tools + 动态 COT 注入`，并已接入 reusable 数据离线双轮评测（Round1→自动写回 COT→Round2）
- P4: 多 Agent 方案已合入主链路，主实现目录为 `services/memory_pipeline/`
- P5-P6: 亲密度模式修正 / LLM-代码矛盾仲裁 / 趋势分析分层降级（待评估）

**LLM 角色体系 (roles)**

Pipeline 中有 4 个 LLM 角色，各司其职，详见 `.knowledge/roles/`：

| 角色 | 阶段 | 模型 | 职责 |
|:---|:---|:---|:---|
| VP1 | [7/9] VLM 分析 | Gemini 2.0 Flash（HTTP 代理转发 Gemini API） | 每张照片→结构化视觉档案（6 个顶层字段：summary / people / relations / scene / event / details） |
| LP1 | [8/9] LLM 事件 | Gemini 2.5 Flash（HTTP 代理转发 Gemini API） | 全量 VLM 结果→事件提取（全量照片一次调用） |
| LP2 | [8/9] LLM 关系 | Gemini 2.5 Flash（HTTP 代理转发 Gemini API） | `RelationshipDossier` → 8 标签关系判定 + 类型反思 + retention |
| LP3 | [8/9] LLM 画像 | Gemini 3.1 Flash Lite Preview（OpenRouter） | `ProfileAgent` 按 domain batch 调度字段，动态注入字段级 COT，产出结构化画像 |

数据流：`VP1 → LP1 → LP2 → LP3`，每一步的输出是下一步的输入。

**当前主链路关键约束**
- **主角识别**: 按“自拍高频优先 → 明确身份锚点优先 → 他拍高频候选 → photographer_mode 兜底”执行；他拍候选必须警惕“用户主要在拍别人”
- **关系识别**: 8 类 `RelationshipTypeSpec` 是关系标签 COT 的单一真源，所有正式关系输出都带 `evidence + reasoning`
- **画像生成**: LP3 只有一个 `ProfileAgent`；按 `domain batch` 调度，2×2 分层（long/short × facts/expression）只作为输出挂载结构，不再作为执行 phase
- **动态 COT 注入**: 每个 batch 只注入当前字段的 `FieldSpec/COT`、证据摘要、统计摘要、归属摘要和反证摘要，不再把整份画像规则一次塞给 LLM
- **LP3 主判边界**: LP3 主判直接读原始 `events / relationships / groups / vlm_observations`，仅保留轻量 `feature_refs`（计数型上下文）
- **LP3 证据收敛**: 字段进入模型前必须先做字段级 evidence 过滤与 top-k 收敛，避免“全量散证据”污染 LP3 判定
- **LP3 Tool Contract**: 运行时固定只有 `5` 个 horizontal tool，且统一拆成 `debug/full + llm/compact` 双层输出；prompt 只能读 compact，不得再注入完整 refs
- **长期事实去偏**: 长期事实字段统一防止低频事件、高频但非日常事件把长期标签带偏
- **表达层静默**: 强依赖社媒的表达标签在缺少社媒模态时直接静默为 `null`
- **下游裁判机制**: 主工程输出不会全量镜像到下游，而是经 `services/memory_pipeline/profile_agent_adapter.py` 直接产出下游 `ExtractorOutput`，再送入 `profile_agent` 的 `Critic -> Judge` 审计链
- **多锚点证据**: 下游 `profile_agent` 的 `Evidence` 已扩展支持 `event_id / photo_id / person_id / feature_names`，主工程 evidence 要尽量保留这些 refs
- **下游审计产物**: 下游裁判结果会先按原生 `Judge -> merged_outputs` 逻辑写入下游 `profiles.json`，并同步生成 `downstream_audit_report`；当前主角 `nullify/downgrade` 会回退成 `photographer_mode`，直接清空正式 `relationships / relationship_dossiers / groups`，再重跑 `LP3 -> downstream audit`；关系 `nullify/downgrade` 会先改正式 `Relationship[]` 再重跑 `groups -> LP3 -> downstream audit`，最后才把已映射的 profile facts 字段安全回流到 `structured_profile`，并同步更新 `profile_fact_decisions.final`，保留 `final_before_backflow + backflow`

**自回归循环（Evolution Loop）**
- 入口: `scripts/run_hardness_pipeline.py evolve --user {name}` 或前端审批触发
- 完整链路:
  1. `cmd_run` → pipeline + GT 对比 → 初始化 field_loop_state
  2. 人工校准 GT（错题集页面 → 保存 → 点"开始自回归循环"）
  3. `cmd_evolve` → 选 top_k 焦点字段 → reflect agent 反思 → 生成提案 → 飞书通知
  4. 人工审批（网页 or 飞书）:
     - **批准** → apply rule_patch → rerun-fields(单字段重跑 LP3) → auto-evolve(下一轮)
     - **拒绝/修订** → reset field state → evolve(换方向，reflect 看到 reviewer_note)
     - **已修正** → 标记 monitoring + human_resolved，停止该字段循环
  5. 收敛 → 生成复盘报告
- 并发保护: 文件锁 `_evolve_lock_{user}.lock` + 队列 `_rerun_queue_{user}.txt`，同一用户同一时间只有一个 rerun+evolve 在跑
- 停止条件（三选一）:
  - 所有字段收敛（monitoring/exhausted/human_resolved，throttle 不算收敛）
  - 单日版本上限（10 轮）→ 强制复盘
  - autoloop max_rounds → 强制复盘
- 上下文注入（每次 reflect 调用，单次注入不持久化）:
  - `evolution_context`: 上轮 grade/score/patch_effect/last_proposed_direction（cycle > 1 时）
  - `human_reviewer_note`: 审批备注（用完即弃）
  - `history_feedback`: 同字段历史人工反馈
- 规则约束: reflect agent 只能输出 cot_hint（已移除 weak_evidence_caution 选项）
- album_id 对齐: sync_gt_to_reflection 自动从最新 run_meta.json 读取

**跨用户分析（Harness Engineering）+ 通用规则提炼**
- 入口: `scripts/run_hardness_pipeline.py harness` 或前端"运行跨用户分析"按钮
- Agent 架构: GT Matcher → UpstreamTriageScorer → BadcasePacketAssembler → UpstreamReflectionAgent → ProposalBuilder → EngineeringCritic → extract_difficult_cases
- 核心原则: 人工覆写优先、用户数据隔离、GT 对比为主数据源、字段标签映射唯一来自 `docs/reflection_field_bilingual_table.csv`
- 通用规则提炼（三层综合，跨用户分析后自动触发）:
  1. **bad case 聚类**: 从 harness_engineering_report 读跨 2+ 用户的 cross_user_patterns
  2. **知识库经验整合**: 对每个 pattern 字段查 critic_field_index.json，收集有效/无效方向
  3. **综合提炼**: pattern + 知识库双重满足 → 生成通用规则提案（只用 cot_hint）
- 通用规则审批（双入口）:
  - 网页: Harness 知识库 Tab 审批区
  - 飞书: 卡片带批准/拒绝按钮，点击跳转网页自动执行
  - approve → 写 field_specs.overrides.json(全局生效) + critic_policy.md(政策积累)
  - reject → 降 weight，不写规则
- 数据存储: `memory/evolution/general_proposals.json`（通用提案）、`memory/reflection/critic_field_index.json`（知识库）、`memory/reflection/critic_policy.md`（政策层）

**疑难杂症沉淀**
- 来源 A（跨用户）: EngineeringCritic 判定需要 new_tool/architecture_change → `summary.difficult_cases`
- 来源 B（单用户）: Reflect Agent 多轮信号累积（watch_only/needs_review/低 confidence）→ `exhausted_fields/{user}.json`
- 关键原则: LLM 信号驱动而非硬编码规则、门槛严格、分类信息来自 LLM 诊断

---

## 工程信息

**项目名称**: Memory Engineering v2.0
**路径**: `/Users/vigar07/Desktop/agent_ME_0324`
**创建时间**: 2026-03-19

### 产品背景
记忆工程是一个高质量记忆库。最终生成的记忆事件、关系和画像，服务于：当看到用户的新照片时，能讲好照片上的故事——把故事讲得有趣、让用户满意、让朋友愿意互动。

**目标用户**：美国高中生和大学生（社交活跃，恋爱关系变动快，"搭子文化"突出）

### 当前阶段定位
当前这份工程文档首先服务于产品经理 Vigar 跑通链路、调试全流程策略，而不是一次性完成完整的标注产品和记忆基础设施。

现阶段优先目标：
1. 跑通 `照片 -> 人脸 -> VLM -> 事件 -> 关系 -> 画像 -> 输出` 的主链路
2. 让每一步输出足够可观察、可检查、可复盘，便于定位策略问题
3. 快速验证哪些策略有效，哪些策略会造成误判、漏判或链路断裂

现阶段非目标：
1. 完整的用户标注界面
2. 全量数据回写融合系统
3. 完整的跨模块追溯基础设施
4. 通用化生产级 memory store

下一阶段北极星目标：
- **可标注**：用户可对人物、关系、画像字段做显式修正
- **可追溯**：每条结论能回到照片、事件和证据
- **可回写**：用户修正能覆盖旧结论并沉淀为系统事实
- **可约束**：画像层不能绕过关系层，推测层不能冒充事实层

### 技术栈
- **人脸识别**: InsightFace (buffalo_l) + FAISS k-NN + 4层递阶匹配 + MediaPipe 姿态估计
- **VLM分析**: Gemini 2.0 Flash（HTTP 代理转发 Gemini API）
- **LLM处理（LP1/LP2）**: Gemini 2.5 Flash（HTTP 代理转发 Gemini API）
- **画像生成（LP3）**: Gemini 3.1 Flash Lite Preview（OpenRouter）
- **下游审计（Judge）**: Gemini 3.1 Flash Lite Preview（OpenRouter，fallback: Gemini 2.0 Flash Exp / Llama 3.3 70B）
- **图像处理**: Pillow + pillow-heif（HEIC支持）
- **地址解析**: 高德地图API

### v2.0 核心变更（相对 v1.0.1）
1. **人脸识别引擎整体替换**: DeepFace/Facenet512 → InsightFace/buffalo_l + FAISS + 4层匹配
2. **person_id 格式全链路统一**: `person_0/person_1` → `Person_001/Person_002` + `primary_person_id`
3. **Pipeline 顺序调整**: 去重提前到人脸识别前，删除 `photo_deduplicator.py`

### ⚠️ v2.0 注意事项

1. **缓存兼容性**: v1.x 缓存全部作废。InsightFace 和 Facenet512 的 embedding 空间完全不同，FAISS index 不兼容。首次运行必须使用 `--reset-cache`。
2. **VLM person_id 格式回退风险**: VLM prompt 中的示例已全部更新为 `Person_001` 格式，但 Gemini 可能偶尔回退到 `person_0` 格式。上线后首次运行需检查 VLM 输出格式是否正确。
3. **boxed_path 修复副作用**: 老版 VLM 实际读的是无人脸框的 `compressed_path`。v2.0 修复为 `boxed_path`（带人脸框），VLM 首次能看到框图。需留意 VLM 输出是否因为看到框而产生"框"相关的噪声描述。
4. **新依赖安装耗时**: `insightface`、`onnxruntime`、`faiss-cpu`、`mediapipe` 安装较慢，首次 `pip install` 预计 3-5 分钟。MediaPipe 模型首次运行自动下载。

### 核心流程（9步）
```
[1/9] 加载照片 → 读取EXIF（时间、GPS），按时间排序
[2/9] 转换HEIC → 转换为标准朝向JPEG（人脸识别用）
[3/9] 照片去重 → source_hash精确去重 + 时间窗口内感知哈希近重复
[4/9] 人脸识别 → InsightFace (buffalo_l) + FAISS 4层递阶匹配
[5/9] 主角推断 → 四层主角信号（自拍/身份锚点/他拍候选/拍摄者兜底）+ 绘制人脸框
[6/9] 压缩照片 → 压缩到1024px（VLM用）
[7/9] VLM分析 → Gemini 2.0 Flash理解照片内容（使用boxed_path带框图）
[8/9] LLM处理 → 提取事件、推断关系、生成 2×2 分层结构化画像
[9/9] 保存结果 → JSON + Markdown报告
```

### 关键文件
| 文件 | 说明 |
|------|------|
| `main.py` | 主入口，9步处理流程 |
| `config.py` | 全局配置（API、人脸、VLM/LLM、关系） |
| `models/__init__.py` | 数据模型（Photo, Person, Event, Relationship, UserProfile） |
| `services/image_processor.py` | 图片处理（HEIC转JPEG、去重、GPS解析、画框） |
| `services/face_recognition.py` | 人脸识别（InsightFace + FAISS） |
| `services/face_precision.py` | 4层递阶匹配决策 |
| `services/face_landmarks.py` | MediaPipe 姿态估计 |
| `services/vlm_analyzer.py` | VLM分析（Gemini 2.0 Flash） |
| `services/llm_processor.py` | 公共 LLM 调用与主线 prompt 构建 |
| `services/memory_pipeline/` | 主线编排、主角/关系/画像多 Agent 策略实现 |
| `vendor/face_recognition_src/` | vendored InsightFace engine |
| `utils/__init__.py` | 工具函数 |

---

## 工作规则（必须遵守）

### 规则1：称呼约定
**每次回复都称呼用户为 Vigar**

### 规则2：决策确认
**遇到不确定的代码设计问题，必须先询问 Vigar，不得直接行动**

### 规则3：兼容性代码
**不能写兼容性代码，除非 Vigar 主动要求**

### 规则4：改配置前检查消费端
**修改任何配置变量、常量、函数签名或参数默认值时，必须先 grep 所有使用方，确认消费逻辑兼容新值后再改。不要只看定义处就动手。**

---

## 推荐使用流程

### 场景1：完整处理新数据集
```bash
python3 main.py --photos "/path/to/photos" --user-name "username" --reset-cache
```

### 场景2：已有VLM缓存，仅运行LLM
```bash
python3 main.py --photos "/path/to/photos" --user-name "username" --use-cache
```

---

## 数据隔离规范

### 自动隔离
- `cache/{USERNAME}/faces.index` - FAISS人脸索引
- `cache/{USERNAME}/face_recognition_state.json` - 人脸识别状态
- `cache/{USERNAME}/{USERNAME}.json` - VLM缓存
- `output/{USERNAME}_记忆测试_{TIMESTAMP}/` - 所有输出文件
