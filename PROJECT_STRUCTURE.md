# 项目结构分类

> 更新时间：2026-03-27
> 本文件记录三个独立子系统的边界划分，以及根目录文件的保留 / 清理决策。

---

## Part 1：图像 → 关系 & 画像生成管道（含 Critic/Judge 下游审计）

核心能力：从相册照片生成结构化画像和关系，带下游质疑裁决链路。

### 入口
| 文件 | 说明 |
|------|------|
| `main.py` | 本地单用户执行入口（8步流程） |
| `config.py` | 全局配置（API、路径、阈值） |
| `backend/app.py` | FastAPI 服务入口 |
| `backend/worker_app.py` | 异步 Worker 进程 |

### 核心 Pipeline
| 路径 | 说明 |
|------|------|
| `services/pipeline_service.py` | 主调度引擎 |
| `services/image_processor.py` | HEIC→JPEG、EXIF/GPS 提取 |
| `services/face_recognition.py` | InsightFace 人脸识别 |
| `services/face_landmarks.py` | 面部关键点检测 |
| `services/face_precision.py` | 人脸对齐评分 |
| `services/vlm_analyzer.py` | Gemini 2.0 Flash VLM 分析 |
| `services/vlm_feature_matcher.py` | VLM 特征匹配 |
| `services/llm_processor.py` | 事件提取、关系推断、画像生成 |
| `services/asset_store.py` | 资产缓存 |
| `services/consistency_checker.py` | 数据一致性检查 |
| `services/relationship_rules.py` | 关系推断规则 |
| `services/bedrock_runtime.py` | AWS Bedrock 备用 |

### Memory Pipeline（含 LP1/LP2/LP3 + 下游审计）
| 路径 | 说明 |
|------|------|
| `services/memory_pipeline/orchestrator.py` | Pipeline 协调器 |
| `services/memory_pipeline/profile_agent.py` | LP3 画像生成 Agent |
| `services/memory_pipeline/profile_agent_adapter.py` | Agent 适配器 |
| `services/memory_pipeline/profile_fields.py` | 53 个字段定义（FIELD_SPECS） |
| `services/memory_pipeline/profile_llm.py` | LLM 调用封装 |
| `services/memory_pipeline/profile_tools.py` | 画像 Tool 工具集 |
| `services/memory_pipeline/events.py` | LP1 事件提取 |
| `services/memory_pipeline/relationships.py` | LP2 关系发现 |
| `services/memory_pipeline/groups.py` | 社群检测 |
| `services/memory_pipeline/primary_person.py` | 主角识别 |
| `services/memory_pipeline/person_screening.py` | 人物相关性过滤 |
| `services/memory_pipeline/evidence_utils.py` | 证据收集工具 |
| `services/memory_pipeline/other_photo_signals.py` | 视觉信号提取 |
| `services/memory_pipeline/keyword_matcher.py` | 关键词匹配 |
| `services/memory_pipeline/downstream_audit.py` | Critic + Judge 下游审计 |
| `services/memory_pipeline/rule_asset_loader.py` | 规则资产加载 |
| `services/memory_pipeline/rule_assets/` | field_specs、tool_rules、call_policies JSON |
| `services/memory_pipeline/precomputed_loader.py` | 预计算数据加载 |
| `services/memory_pipeline/precomputed_bundle_runner.py` | 预计算 Bundle 执行 |
| `services/memory_pipeline/reusable_smoke_loader.py` | Smoke 测试数据加载 |
| `services/memory_pipeline/reusable_smoke_llm.py` | Smoke 测试 LLM |
| `services/memory_pipeline/reusable_smoke_runner.py` | Smoke 测试执行 |
| `services/memory_pipeline/offline_profile_eval.py` | 离线画像评估 |
| `services/memory_pipeline/types.py` | Pipeline 类型定义 |

### Memory Module（存储与检索）
| 路径 | 说明 |
|------|------|
| `memory_module/service.py` | 记忆服务主入口 |
| `memory_module/query.py` | 查询构建与执行 |
| `memory_module/adapters.py` | 适配器实现 |
| `memory_module/embeddings.py` | 向量嵌入与相似度 |
| `memory_module/sequencing.py` | 结果排序 |
| `memory_module/domain.py` | 域模型 |
| `memory_module/dto.py` | 数据传输对象 |
| `memory_module/vo.py` | 值对象 |
| `memory_module/records.py` | 记录管理 |
| `memory_module/ontology.py` | 记忆本体 |
| `memory_module/views.py` | 视图定义 |
| `memory_module/evaluation.py` | 评估工具 |

### Backend API 层
| 路径 | 说明 |
|------|------|
| `backend/models.py` | SQLAlchemy 数据库模型 |
| `backend/db.py` | 数据库连接 |
| `backend/task_store.py` | 任务存储 |
| `backend/artifact_store.py` | 产物存储 |
| `backend/face_review_store.py` | 人脸审核存储 |
| `backend/worker_manager.py` | Worker 生命周期管理 |
| `backend/worker_client.py` | Worker 客户端 |
| `backend/progress_utils.py` | 进度追踪 |
| `backend/upload_utils.py` | 文件上传 |
| `backend/task_download_bundle.py` | 结果打包 |
| `backend/auth.py` | 认证授权 |
| `backend/memory_full_retrieval.py` | 记忆全量检索 |
| `backend/memory_step_retrieval.py` | 记忆分步检索 |
| `backend/reflection_api.py` | 反思系统 API |

### 相关测试
```
tests/test_image_processor.py
tests/test_face_landmarks.py
tests/test_face_precision.py
tests/test_consistency_checker.py
tests/test_relationship_rules.py
tests/test_memory_pipeline.py
tests/test_memory_pipeline_evolution.py
tests/test_downstream_profile_agent_integration.py
tests/test_profile_agent_offline_fallback.py
tests/test_profile_entrypoint_chain.py
tests/test_profile_rule_assets.py
tests/test_primary_person_inference.py
tests/test_precomputed_bundle_pipeline.py
tests/test_reusable_smoke_pipeline.py
tests/test_memory_module.py
tests/test_memory_adapters.py
tests/test_memory_query.py
tests/test_pipeline_memory.py
tests/test_lp3_openrouter_bundle.py
tests/test_llm_chunking.py
tests/test_model_providers.py
tests/test_artifact_store.py
tests/test_progress_utils.py
tests/test_task_api.py
tests/test_worker_manager.py
tests/test_repo_mainline_layout.py
tests/test_reflection_api.py
tests/test_main.py
```

---

## Part 2：下游反思 Harness Agent 系统

独立的 bad case 诊断与提案系统。与 Part 1 通过 `case_facts` JSONL 文件解耦（离线运行）。

### 核心模块
| 路径 | 说明 |
|------|------|
| `services/reflection/types.py` | CaseFact、PatternCluster、ProposalReviewRecord 等核心类型 |
| `services/reflection/storage.py` | 反思资产路径管理 |
| `services/reflection/tasks.py` | `run_reflection_task_generation()` 主入口；A0/A1/B/A4 harness 逻辑 |
| `services/reflection/upstream_agent.py` | BadcasePacketAssembler、CoverageProbe、UpstreamReflectionAgent、ExperimentPlanner、ProposalBuilder |
| `services/reflection/upstream_triage.py` | UpstreamTriageScorer，root_cause_family 枚举，fix_surface 映射 |
| `services/reflection/triage.py` | 通用分诊逻辑 |
| `services/reflection/downstream_capture.py` | 下游审计 bad case 捕获 |
| `services/reflection/mainline_capture.py` | 主线 bad case 捕获 |
| `services/reflection/gt.py` | GT 加载、比对、auto_generate_pseudo_gt（A4） |
| `services/reflection/gt_matcher.py` | 字段值比对算法 |
| `services/reflection/gt_excel.py` | GT Excel 导入/导出 |
| `services/reflection/labels.py` | 字段双语标签 |

### Harness Phase 1 实现状态（已完成）
- **A0 DataSufficiencyGate**：策略问题 vs 非策略问题（数据不足）分诊，防止无解 case 进入反思链路
- **A1 ResolutionSignal**：`history_recall()` 返回 `already_fixed/failed_before/open`，已修复字段跳过 LLM
- **B CoverageProbe**：4类结构性 gap 检查（`source_unconfigured / tool_called_no_hit / index_path_suspect / tool_rule_blocked`）
- **A4 PseudoGT**：高置信度枚举字段自动生成 GT

### 待完成（Phase 2/3）
- A2：pre_cluster() 聚类前置（LLM O(N)→O(K)）
- A3：沙盒重跑 → outcomes 自动写入

### 相关测试
```
tests/test_harness_phase1.py          ← Phase 1 新增（24个测试）
tests/test_reflection_tasks.py        ← 核心流程测试（14个测试）
tests/test_reflection_gt_matcher.py
tests/test_reflection_labels.py
tests/test_reflection_skeleton.py
```

### 通知系统（待重新设计）
飞书通知已从代码中完全移除，以下文件保留 shell 但逻辑为空，待后续重新设计：
- `services/reflection/feishu.py` — 1284行，完整保留但已无调用方
- `services/reflection/feishu_long_connection.py` — 长连接守护进程

---

## Part 3：夜间模式系统（独立字段迭代）

"多用户集 + 每用户 Top3 问题字段 + 字段级循环反思"，可独立运行，不依赖 Part 2。

### 核心模块
| 路径 | 说明 |
|------|------|
| `services/memory_pipeline/evolution.py` | `run_memory_nightly_evaluation()` + `run_memory_nightly_user_set_evaluation()` + `_run_gt_field_loop()` |

### 入口脚本
| 路径 | 说明 |
|------|------|
| `scripts/run_memory_nightly_eval.py` | CLI 入口，支持单用户 / 用户集两种模式 |
| `scripts/apply_memory_evolution_proposal.py` | 人工确认后应用 proposal 到 rule_assets |

### 数据目录
```
memory/evolution/
  field_cycles/{user}/         ← 每日字段循环状态
  field_loop_state/{user}.json ← 字段级收敛状态
  focus_fields/{user}/         ← Top3 问题字段
  insights/{user}/             ← 当日洞察
  proposals/{user}/            ← 当日提案
  reports/{user}/              ← 汇总报告
  reports/_user_set/           ← 多用户汇总
```

### 相关测试
```
tests/test_memory_pipeline_evolution.py
```

---

## 根目录文件清理决策

### ✅ 保留（核心宪法 / Schema / 策略）
| 文件 | 理由 |
|------|------|
| `CLAUDE.md` | Agent 工作指南（规则 1/2/3） |
| `CONSTITUTION.md` | 系统宪法，约束 Agent 行为 |
| `AGENTS.md` | Agent 角色说明 |
| `memory_engineering_STATA_SCHEMA.md` | 核心数据 Schema 定义 |
| `SEGMENTATION_RULES.md` | 事件分割规则 |

### 🗑️ 清理清单（根目录）
| 文件 | 原因 |
|------|------|
| `README.md` | 已过期，描述的是 v1.0 旧版本 |
| `PLAN - v0-1.md` | 历史规划文档，已被实现取代 |
| `PROJECT.md` | 重复内容，与 CLAUDE.md 重叠 |
| `HANDOVER.md` | 移交文档，已无现实意义 |
| `DEPLOY.md` | 部署文档，与 `deploy/` 目录重复 |
| `LOCAL_MEMORY_SERVICES.md` | 本地服务文档，内容已过期 |
| `PROXY_CONFIGURED.md` | 一次性配置记录 |
| `prompts.md` | 散落的 prompt 片段，已内化到代码 |
| `SETUP_VLM.md` | 初始配置文档，已过期 |
| `VLM_PIPELINE_GUIDE.md` | 已内化到代码注释 |
| `TEST_PIPELINE_README.md` | 测试流程说明，已过期 |

### 🗑️ 清理清单（根目录 Python 文件）
| 文件 | 原因 |
|------|------|
| `test_pipeline.py` | 根目录散落测试，已有 `tests/` 目录 |
| `test_pipeline_vlm.py` | 同上 |
| `test_event_extraction.py` | 同上 |
| `benchmark_lfw.py` | 人脸识别 benchmark 工具，不属于主链路 |
| `diagnose_face_pair.py` | 诊断工具，不属于主链路 |

### 🗑️ 清理清单（测试文件 — 测试已删除功能）
| 文件 | 原因 |
|------|------|
| `tests/test_reflection_feishu.py` | 测试飞书卡片构建（已删除），整个文件失效 |
| `tests/test_v0321_3_pipeline.py` | 测试历史版本 v0321.3 pipeline |
| `tests/test_v0323_pipeline.py` | 测试历史版本 v0323 pipeline |

### 🗑️ 清理清单（历史版本代码）
| 路径 | 原因 |
|------|------|
| `services/v0321_2/` | 历史版本，已被当前主线取代 |
| `services/v0321_3/` | 历史版本 |
| `services/v0323/` | 历史版本 |

### 🗑️ 清理清单（飞书相关，待重新设计后可删）
| 文件 | 原因 |
|------|------|
| `services/reflection/feishu.py` | 通知逻辑已移除，此文件现为孤岛，待重新设计后替换 |
| `services/reflection/feishu_long_connection.py` | 飞书长连接守护进程，依赖 feishu.py |
| `backend/feishu_api.py` | backend 侧的飞书 API 封装 |
| `scripts/run_feishu_long_connection.py` | 启动飞书长连接的脚本 |
| `docs/feishu_worker_keepalive.md` | 飞书 Worker 心跳文档 |
| `docs/reflection_approval_handoff.md` | 飞书审批交接文档 |

### ℹ️ 暂存（可按需处理）
| 文件 | 说明 |
|------|------|
| `scripts/run_v0321_2_scale_replay.py` | v0321.2 规模回放脚本 |
| `scripts/resume_v0323_from_vp1.py` | v0323 恢复脚本 |
| `scripts/backfill_artifact_catalog.py` | 一次性数据回填 |
| `scripts/aws_staging_preflight.py` | AWS staging 预检 |
| `scripts/upgrade_vlm_cache_format.py` | 一次性缓存格式迁移 |
| `deploy/` | AWS 部署文档（保留供运维参考） |
