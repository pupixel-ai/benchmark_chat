#!/bin/bash
# 飞书会议纪要自动总结机器人
# 监听会议结束事件 → 获取纪要 → Claude总结 → 发送卡片给一佳
#
# 用法: bash scripts/meeting_notes_bot.sh
# 停止: Ctrl+C

set -euo pipefail

USER_ID="ou_b3739f5a4d0b6ae787e75708e2905d4c"
LOG_FILE="/tmp/meeting_notes_bot.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

send_card() {
  local title="$1"
  local summary="$2"
  local todos="$3"
  local meeting_link="$4"

  # 构建飞书卡片 JSON
  local card_json
  card_json=$(jq -n \
    --arg title "$title" \
    --arg summary "$summary" \
    --arg todos "$todos" \
    --arg link "$meeting_link" \
    '{
      "config": {"wide_screen_mode": true},
      "header": {
        "title": {"tag": "plain_text", "content": ("📋 " + $title)},
        "template": "blue"
      },
      "elements": [
        {
          "tag": "markdown",
          "content": ("**🎯 核心议题**\n" + $summary)
        },
        {"tag": "hr"},
        {
          "tag": "markdown",
          "content": ("**✅ 你的待办**\n" + $todos)
        },
        {"tag": "hr"},
        {
          "tag": "action",
          "actions": [
            {
              "tag": "button",
              "text": {"tag": "plain_text", "content": "查看完整纪要"},
              "type": "primary",
              "url": $link
            }
          ]
        }
      ]
    }')

  lark-cli im +messages-send \
    --user-id "$USER_ID" \
    --msg-type interactive \
    --content "$card_json" \
    --as bot 2>&1 | tee -a "$LOG_FILE"
}

process_meeting() {
  local meeting_id="$1"
  log "处理会议: $meeting_id"

  # 1. 获取会议详情
  local meeting_info
  meeting_info=$(lark-cli vc meeting get --params "{\"meeting_id\": \"$meeting_id\", \"with_participants\": true}" --format json 2>&1) || true

  local meeting_topic
  meeting_topic=$(echo "$meeting_info" | jq -r '.data.meeting.topic // "未知会议"' 2>/dev/null || echo "未知会议")
  log "会议主题: $meeting_topic"

  # 2. 获取会议纪要（通过 meeting_id）
  local notes_result
  notes_result=$(lark-cli vc +notes --meeting-ids "$meeting_id" --format json 2>&1) || true

  # 提取 minute_token（如果有）
  local minute_token
  minute_token=$(echo "$notes_result" | jq -r '.data.notes[0].minute_token // empty' 2>/dev/null || echo "")

  local note_doc_token
  note_doc_token=$(echo "$notes_result" | jq -r '.data.notes[0].note_doc_token // empty' 2>/dev/null || echo "")

  local summary_text=""
  local todos_text=""
  local meeting_link=""

  # 3. 尝试获取 AI 产物（总结 + 待办）
  if [ -n "$minute_token" ]; then
    local artifacts_result
    artifacts_result=$(lark-cli vc +notes --minute-tokens "$minute_token" --format json 2>&1) || true

    # 提取 AI 总结
    local raw_summary
    raw_summary=$(echo "$artifacts_result" | jq -r '.data[0].artifacts.summary // empty' 2>/dev/null || echo "")

    # 提取待办
    local raw_todos
    raw_todos=$(echo "$artifacts_result" | jq -r '.data[0].artifacts.todos // empty' 2>/dev/null || echo "")

    meeting_link="https://feishu.cn/minutes/$minute_token"

    # 4. 用 Claude 整理核心议题和个人待办
    if [ -n "$raw_summary" ] || [ -n "$raw_todos" ]; then
      local claude_prompt="你是飞书会议助手。根据以下会议纪要信息，整理出：
1. 核心议题（3-5个要点，每个一句话）
2. 一佳的个人待办（从待办中筛选与一佳相关的，如果无法确定，列出所有待办）

会议主题: $meeting_topic
AI总结: $raw_summary
待办事项: $raw_todos

输出格式：
---核心议题---
- 议题1
- 议题2
---待办事项---
- [ ] 待办1
- [ ] 待办2"

      local claude_result
      claude_result=$(claude -p "$claude_prompt" 2>/dev/null) || true

      # 分割 Claude 输出
      summary_text=$(echo "$claude_result" | sed -n '/---核心议题---/,/---待办事项---/p' | grep -v '---' || echo "$claude_result")
      todos_text=$(echo "$claude_result" | sed -n '/---待办事项---/,$ p' | grep -v '---' || echo "暂无待办")
    fi
  fi

  # 如果没有拿到纪要，尝试用纪要文档
  if [ -z "$summary_text" ] && [ -n "$note_doc_token" ]; then
    local doc_content
    doc_content=$(lark-cli docs +fetch --doc "$note_doc_token" 2>&1) || true

    if [ -n "$doc_content" ]; then
      local claude_prompt="你是飞书会议助手。根据以下会议纪要文档内容，整理出：
1. 核心议题（3-5个要点，每个一句话）
2. 一佳的个人待办

文档内容:
$doc_content

输出格式：
---核心议题---
- 议题1
---待办事项---
- [ ] 待办1"

      local claude_result
      claude_result=$(claude -p "$claude_prompt" 2>/dev/null) || true
      summary_text=$(echo "$claude_result" | sed -n '/---核心议题---/,/---待办事项---/p' | grep -v '---' || echo "$claude_result")
      todos_text=$(echo "$claude_result" | sed -n '/---待办事项---/,$ p' | grep -v '---' || echo "暂无待办")
    fi
    meeting_link="https://feishu.cn/docx/$note_doc_token"
  fi

  # 兜底
  [ -z "$summary_text" ] && summary_text="纪要尚未生成，请稍后查看"
  [ -z "$todos_text" ] && todos_text="暂无待办"
  [ -z "$meeting_link" ] && meeting_link="https://feishu.cn"

  # 5. 发送卡片
  log "发送卡片: $meeting_topic"
  send_card "$meeting_topic" "$summary_text" "$todos_text" "$meeting_link"
  log "完成: $meeting_id"
}

# === 主流程 ===
log "🚀 会议纪要机器人启动"
log "监听事件: vc.meeting.meeting_ended_v1"
log "目标用户: $USER_ID"

# 监听会议结束事件，逐行处理
lark-cli event +subscribe \
  --event-types vc.meeting.meeting_ended_v1 \
  --compact --quiet --as bot 2>>"$LOG_FILE" \
  | while IFS= read -r line; do
      log "收到事件: $line"

      # 提取 meeting_id
      meeting_id=$(echo "$line" | jq -r '.meeting_id // .meeting.id // .id // empty' 2>/dev/null || echo "")

      if [ -z "$meeting_id" ]; then
        log "⚠️ 无法提取 meeting_id，跳过"
        continue
      fi

      # 会议结束后纪要可能需要时间生成，等待 30 秒
      log "等待 30s 让纪要生成..."
      sleep 30

      process_meeting "$meeting_id" &
    done

log "❌ 事件监听意外退出"
