"""Critic knowledge management: Policy (Layer 1) + Field Index (Layer 2).

Policy: stable general knowledge, loaded every run, rarely updated.
Field Index: per-field lightweight records, auto-accumulated and decayed.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from config import PROJECT_ROOT

REFLECTION_DIR = Path(PROJECT_ROOT) / "memory" / "reflection"
POLICY_PATH = REFLECTION_DIR / "critic_policy.md"
FIELD_INDEX_PATH = REFLECTION_DIR / "critic_field_index.json"
FIELD_INDEX_CHANGELOG_PATH = REFLECTION_DIR / "field_index_changelog.jsonl"
POLICY_LOG_PATH = REFLECTION_DIR / "critic_policy_log.jsonl"

# ── Policy constraints ──
MAX_POLICY_UPDATES_PER_WEEK = 3
MIN_EVIDENCE_PATTERNS = 2
COOLDOWN_DAYS_PER_ENTRY = 7


# ═══════════════════════════════════════════════════
# Layer 1: Policy
# ═══════════════════════════════════════════════════

def load_policy() -> str:
    """Load the full policy text. Always fits in ~2K tokens."""
    if POLICY_PATH.exists():
        return POLICY_PATH.read_text(encoding="utf-8")
    return _DEFAULT_POLICY


def propose_policy_update(
    *,
    content: str,
    reason: str,
    source_pattern_ids: List[str],
) -> Dict[str, Any]:
    """Propose a policy update. Returns success/rejection with reason."""
    if len(source_pattern_ids) < MIN_EVIDENCE_PATTERNS:
        return {"accepted": False, "reason": f"需要至少 {MIN_EVIDENCE_PATTERNS} 个 pattern 支撑"}

    # Check weekly limit
    recent_updates = _count_recent_policy_updates(days=7)
    if recent_updates >= MAX_POLICY_UPDATES_PER_WEEK:
        return {"accepted": False, "reason": f"本周已更新 {recent_updates} 次，达到上限 {MAX_POLICY_UPDATES_PER_WEEK}"}

    # Check cooldown for this content
    if _is_content_in_cooldown(content):
        return {"accepted": False, "reason": f"相似内容 {COOLDOWN_DAYS_PER_ENTRY} 天内已更新过"}

    # Apply update
    current = load_policy()
    marker = "\n## 通用经验\n"
    if marker in current:
        before, after = current.split(marker, 1)
        new_entry = f"- [{datetime.now().strftime('%Y-%m-%d')}] {content}（来源: {', '.join(source_pattern_ids[:3])}）\n"
        updated = before + marker + new_entry + after
    else:
        updated = current + f"\n\n## 通用经验\n- [{datetime.now().strftime('%Y-%m-%d')}] {content}\n"

    POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
    POLICY_PATH.write_text(updated, encoding="utf-8")

    # Log
    _append_policy_log({
        "action": "update",
        "content": content,
        "reason": reason,
        "source_patterns": source_pattern_ids,
        "timestamp": datetime.now().isoformat(),
    })

    return {"accepted": True}


def _count_recent_policy_updates(days: int) -> int:
    logs = _read_policy_log()
    cutoff = datetime.now() - timedelta(days=days)
    return sum(
        1 for log in logs
        if log.get("action") == "update"
        and datetime.fromisoformat(log.get("timestamp", "2000-01-01")) > cutoff
    )


def _is_content_in_cooldown(content: str) -> bool:
    logs = _read_policy_log()
    cutoff = datetime.now() - timedelta(days=COOLDOWN_DAYS_PER_ENTRY)
    content_lower = content.lower().strip()
    for log in reversed(logs):
        if log.get("action") != "update":
            continue
        if datetime.fromisoformat(log.get("timestamp", "2000-01-01")) < cutoff:
            break
        if content_lower in str(log.get("content", "")).lower():
            return True
    return False


def _read_policy_log() -> List[Dict]:
    if not POLICY_LOG_PATH.exists():
        return []
    records = []
    for line in POLICY_LOG_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _append_policy_log(record: Dict) -> None:
    POLICY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(POLICY_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ═══════════════════════════════════════════════════
# Layer 2: Field Index
# ═══════════════════════════════════════════════════

def load_field_index(field_key: str) -> Dict[str, Any]:
    """Load index entries for a specific field. Returns aggregate + entries."""
    all_data = _read_field_index()
    field_data = all_data.get(field_key, {})
    _decay_weights(field_data)
    return field_data


def update_field_index(
    *,
    field_key: str,
    diagnosis: str,
    recommendation_level: int = 1,
    pattern_id: str = "",
    user_name: str = "",
    proposal_id: str = "",
    root_cause_family: str = "",
    fix_surface: str = "",
    human_verdict: str | None = None,
    outcome_improved: bool | None = None,
) -> None:
    """Add or update a field index entry. user_name is required for traceability."""
    if not user_name:
        return  # User isolation: refuse anonymous entries

    all_data = _read_field_index()
    field_data = all_data.setdefault(field_key, {"entries": [], "aggregate": {}})
    entries = field_data.get("entries", [])

    # Match by proposal_id first, then pattern_id, then diagnosis similarity
    existing = None
    for e in entries:
        if proposal_id and e.get("proposal_id") == proposal_id:
            existing = e
            break
        if pattern_id and e.get("pattern_id") == pattern_id and e.get("user_name") == user_name:
            existing = e
            break
    # Dedup: same user + similar diagnosis → update instead of append
    if existing is None and diagnosis and user_name:
        for e in entries:
            if e.get("user_name") != user_name:
                continue
            old_diag = e.get("diagnosis", "")
            if _diagnosis_similar(old_diag, diagnosis):
                existing = e
                existing["diagnosis"] = diagnosis  # use newer wording
                existing["proposal_id"] = proposal_id  # update reference
                break

    if existing:
        if human_verdict:
            existing["human_verdict"] = human_verdict
            if human_verdict == "approve":
                existing["weight"] = min(0.95, existing.get("weight", 0.6) + 0.15)
            elif human_verdict == "reject":
                existing["weight"] = max(0.1, existing.get("weight", 0.6) - 0.20)
            elif human_verdict == "need_revision":
                existing["weight"] = max(0.1, existing.get("weight", 0.6) - 0.05)
            elif human_verdict == "field_resolved":
                existing["weight"] = min(0.95, existing.get("weight", 0.6) + 0.20)
        if outcome_improved is not None:
            existing["outcome_improved"] = outcome_improved
            if outcome_improved:
                existing["weight"] = min(0.95, existing.get("weight", 0.6) + 0.10)
            else:
                existing["weight"] = max(0.1, existing.get("weight", 0.6) - 0.10)
        existing["updated_at"] = datetime.now().isoformat()
    else:
        entries.append({
            "diagnosis": diagnosis,
            "user_name": user_name,
            "proposal_id": proposal_id,
            "pattern_id": pattern_id,
            "recommendation_level": recommendation_level,
            "root_cause_family": root_cause_family,
            "fix_surface": fix_surface,
            "weight": 0.60,
            "human_verdict": human_verdict,
            "outcome_improved": outcome_improved,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        })

    # Prune: max 10 entries per field, keep highest weight
    entries.sort(key=lambda e: e.get("weight", 0), reverse=True)
    field_data["entries"] = [e for e in entries[:10] if e.get("weight", 0) >= 0.2]

    # Update aggregate
    valid = field_data["entries"]
    reviewed = [e for e in valid if e.get("human_verdict")]
    field_data["aggregate"] = {
        "avg_weight": round(sum(e.get("weight", 0) for e in valid) / len(valid), 3) if valid else 0,
        "total_reviews": len(reviewed),
        "approve_rate": round(
            sum(1 for e in reviewed if e["human_verdict"] == "approve") / max(1, len(reviewed)), 2
        ),
        "dominant_diagnosis": valid[0].get("diagnosis", "") if valid else "",
        "source_users": sorted(set(e.get("user_name", "") for e in valid if e.get("user_name"))),
    }

    all_data[field_key] = field_data
    _write_field_index(all_data)

    # Append changelog
    _append_field_index_changelog({
        "action": "update",
        "field_key": field_key,
        "user_name": user_name,
        "proposal_id": proposal_id,
        "diagnosis": diagnosis[:120],
        "human_verdict": human_verdict,
        "timestamp": datetime.now().isoformat(),
    })


def _diagnosis_similar(a: str, b: str, threshold: float = 0.5) -> bool:
    """Check if two diagnosis texts are similar enough to be considered duplicates.

    Uses character-level overlap ratio. Threshold 0.6 catches paraphrases
    of the same diagnosis while allowing genuinely different ones through.
    """
    if not a or not b:
        return False
    a, b = a.strip(), b.strip()
    if a == b:
        return True
    # Character bigram overlap (language-agnostic, works for Chinese)
    def bigrams(s: str) -> set:
        return {s[i:i+2] for i in range(len(s) - 1)} if len(s) >= 2 else {s}
    ba, bb = bigrams(a), bigrams(b)
    if not ba or not bb:
        return False
    overlap = len(ba & bb)
    return overlap / min(len(ba), len(bb)) >= threshold


def _decay_weights(field_data: Dict) -> None:
    """Decay weights by 0.9x per 30 days since last update."""
    now = datetime.now()
    for entry in field_data.get("entries", []):
        updated = entry.get("updated_at") or entry.get("created_at", "")
        if not updated:
            continue
        try:
            age_days = (now - datetime.fromisoformat(updated)).days
        except (ValueError, TypeError):
            continue
        if age_days > 30:
            periods = age_days // 30
            entry["weight"] = round(entry.get("weight", 0.6) * (0.9 ** periods), 3)


def _read_field_index() -> Dict[str, Any]:
    if not FIELD_INDEX_PATH.exists():
        return {}
    try:
        return json.loads(FIELD_INDEX_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_field_index(data: Dict) -> None:
    FIELD_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIELD_INDEX_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_field_index_changelog(record: Dict) -> None:
    FIELD_INDEX_CHANGELOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FIELD_INDEX_CHANGELOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ═══════════════════════════════════════════════════
# Layer 2b: Field Knowledge Summary (for LLM consumption)
# ═══════════════════════════════════════════════════


def summarize_field_knowledge(field_key: str) -> str:
    """Compact summary of field knowledge for LLM prompt injection (< 200 tokens)."""
    data = load_field_index(field_key)
    entries = data.get("entries", [])
    if not entries:
        return ""
    agg = data.get("aggregate", {})
    verdict_zh = {"approve": "有效", "reject": "无效", "need_revision": "需修改"}
    lines = [
        f"该字段历史经验（{len(entries)} 条，审批通过率 {agg.get('approve_rate', 0):.0%}）：",
    ]
    for e in entries[:3]:
        v = verdict_zh.get(e.get("human_verdict", ""), "待审")
        u = e.get("user_name", "?")
        lines.append(f"  - [{v}] {e.get('diagnosis', '')[:80]} (来源: {u}, 权重: {e.get('weight', 0):.2f})")
    return "\n".join(lines)


def load_all_field_knowledge() -> Dict[str, Any]:
    """Load entire field index for API consumption."""
    return _read_field_index()


# ═══════════════════════════════════════════════════
# Default policy content
# ═══════════════════════════════════════════════════

_DEFAULT_POLICY = """# 记忆工程 Critic 通用知识库

## 工程链路理解
- VP1 (VLM): Gemini 2.0 Flash 做场景描述，不做 OCR/文字识别，不识别具体品牌型号
- LP1 (事件): 从时间+地点+人物聚合照片为事件，标题为英文，narrative 为英文
- LP2 (关系): 从共现频率+场景类型推断关系类型和亲密度，依赖 VLM 的 interaction 字段
- LP3 (画像): 53 个字段逐一 COT 推理，每个字段从 evidence_bundle 中最多引用 N 条证据
- 下游审计: Critic + Judge 对主角/关系/画像三维度做质疑裁决，可回流修改上游结果

## Agent 设计原则
- 证据闭环：任何推断必须有可追溯的照片/事件证据
- 代码做重活，LLM 只判定：检索和过滤用规则，LLM 只做最终语义判断
- 结构化枚举优于自由文本：能用枚举不用自由文本
- 数据不足不强推：confidence 低时宁可输出 null 也不编造

## Critic 行为宪法

### 质疑等级
Level 1（策略质疑）— 默认
  触发: 所有 pattern
  输出: 具体的 COT/tool_rule/call_policy 修改建议
  约束: 建议必须可直接执行

Level 2（工具质疑）— 条件升级
  触发: field_index 中该字段历史修复失败 ≥2 次，或 CoverageProbe 检测到结构性缺口，或 user_coverage ≥ 50%
  输出: 新工具/新数据源需求
  约束: 必须说明"为什么改规则不够"

Level 3（架构质疑）— 高门槛
  触发: Level 2 建议也被人否决过，或同一问题跨 3+ 轮 Critic 分析仍未解决
  输出: Agent 流程变更建议
  约束: 必须引用至少 3 个用户的具体 case 作为论据
  外部工具: 仅 Level 3 可请求 search_industry_knowledge

## 通用经验
"""
