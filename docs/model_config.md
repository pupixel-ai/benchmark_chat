# Model & API Key Configuration

All model and API key configuration通过 `.env` 文件统一管理。

## API Keys

| 环境变量 | 用途 | 说明 |
|---------|------|------|
| `OPENROUTER_API_KEY` | 通用 OpenRouter key | Pipeline、Reflect、GT Matcher 共用 |
| `REFLECTION_AGENT_OPENROUTER_API_KEY` | Reflect Agent 专用 | 默认回退到 OPENROUTER_API_KEY |
| `ANTHROPIC_API_KEY` | Engineering Critic | Harness 阶段的深度分析 |
| `GEMINI_API_KEY` | VLM 图像分析 | Google Gemini 直连 |

## Model 配置

| 环境变量 | 默认值 | 用途 |
|---------|--------|------|
| `PROFILE_LLM_MODEL` | `deepseek/deepseek-chat-v3-0324` | Pipeline Run / Rerun — 53 个字段推理 |
| `REFLECTION_AGENT_MODEL` | `anthropic/claude-opus-4.6` | Reflect Agent — evolve 循环中的反思和提案生成 |
| `GT_MATCHER_MODEL` | `anthropic/claude-opus-4.6` | GT Matcher — LLM fallback 语义打分 |
| `CRITIC_MODEL` | `anthropic/claude-opus-4.6` | Engineering Critic — Harness 跨用户深度分析 |
| `SMOKE_LLM_MODEL` | `google/gemini-3.1-flash-lite-preview` | Smoke Test / Bundle Runner — 快速验证 |
| `VLM_MODEL` (GEMINI_MODEL) | `gemini-2.0-flash` | VLM 照片分析 |

## Provider 配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `PROFILE_LLM_PROVIDER` | `openrouter` | Pipeline LLM provider |
| `REFLECTION_AGENT_PROVIDER` | `openrouter` | Reflect Agent provider |
| `CRITIC_PROVIDER` | `anthropic` | Engineering Critic provider |

## 各环节数据流

```
Stage 1: Pipeline Run
  模型: PROFILE_LLM_MODEL (via OpenRouter)
  输入: 用户照片 VLM 结果 + 事件
  输出: 53 个画像字段

Stage 2: GT Comparison
  模型: GT_MATCHER_MODEL (via OpenRouter, LLM fallback)
  输入: 系统输出 vs GT 标注
  输出: grade + score

Stage 3: Evolve (自循环)
  模型: REFLECTION_AGENT_MODEL (via OpenRouter)
  输入: case_facts + evolution_context + 新线索
  输出: 反思结论 + 提案 + rule_patch

Stage 3.5: Rerun (审批后重跑)
  模型: PROFILE_LLM_MODEL (via OpenRouter)
  输入: 同 Stage 1，仅重跑指定字段
  输出: 更新后的字段值

Stage 4: Harness (跨用户)
  模型: CRITIC_MODEL (via Anthropic)
  输入: 全用户 GT 对比 + 知识库
  输出: 工程报告 + 知识库更新
```

## 修改模型

只需修改 `.env` 文件中对应的环境变量，无需改代码。所有模型引用都从 `.env` 或 `config.py` 读取。
