#!/bin/bash
# PostToolUse hook: 当核心策略文件被修改时，提醒更新 me_reflection/AGENTS.md
#
# 监控范围：
#   - services/reflection/         (Reflect/Harness Agent 链路)
#   - services/memory_pipeline/    (evolution, profile_fields, profile_agent)
#   - config.py                    (模型/Provider 配置)
#   - rule_assets/                 (字段规则)
#   - docs/model_config.md         (模型配置文档)

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# 没有 file_path 的工具调用直接跳过
if [ -z "$FILE_PATH" ]; then
  exit 0
fi

# 核心策略文件匹配模式
WATCHED_PATTERNS=(
  "services/reflection/"
  "services/memory_pipeline/evolution.py"
  "services/memory_pipeline/profile_fields.py"
  "services/memory_pipeline/profile_agent.py"
  "services/memory_pipeline/types.py"
  "services/memory_pipeline/upstream_agent.py"
  "config.py"
  "rule_assets/"
  "docs/model_config.md"
)

MATCHED=""
for pattern in "${WATCHED_PATTERNS[@]}"; do
  if [[ "$FILE_PATH" == *"$pattern"* ]]; then
    MATCHED="$pattern"
    break
  fi
done

if [ -n "$MATCHED" ]; then
  echo "[AGENTS.md sync] $FILE_PATH (matched: $MATCHED) was modified. Check if me_reflection/AGENTS.md needs updating." >&2
fi

exit 0
