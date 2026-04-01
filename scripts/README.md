# 脚本索引

## 当前保留

- `run_lp3_fresh.py`
  - 官方 Gemini 直连的 LP3 运行脚本
- `run_lp3_bundle_openrouter.py`
  - 使用 OpenRouter 跑 bundle 的 LP3 脚本
- `upgrade_vlm_cache_format.py`
  - VLM 缓存格式升级工具
- `run_precomputed_bundle_pipeline.py`
  - 运行 precomputed bundle 主链，并输出 stage report + run trace
- `run_reusable_smoke_test.py`
  - 运行 reusable smoke 链路，并输出 comparison + run trace

## 进化闭环

- `run_memory_nightly_eval.py`
  - 聚合 `memory/evolution/traces/{user}/{date}.jsonl`，生成 insights/proposals 报告
- `apply_memory_evolution_proposal.py`
  - 手动应用 proposal 的 `patch_preview` 到 `services/memory_pipeline/rule_assets/`

## 调试

- `scripts/debug/debug_llm_response.py`
  - 单字段 LLM 响应调试脚本
  - 运行前需要配置 `GEMINI_API_KEY` 或 `GOOGLE_API_KEY`

## 维护参考

- `scripts/maintenance/restructure_generated_20260325.sh`
  - 2026-03-25 Claude 生成的重组脚本留档
  - 仅作参考，不应在当前仓库直接照跑
