"""EngineeringCritic: multi-step dialogue LLM agent for cross-user pattern analysis.

Architecture:
  Step 1: Give pattern summary + policy + field_index → get preliminary diagnosis + evidence requests
  Step 2: Give requested evidence → get final diagnosis + recommendations + knowledge updates
  Step 3: (Optional) If Critic requests external tool → placeholder response

Uses Claude via Anthropic API. Falls back to OpenRouter if Anthropic unavailable.
"""

from __future__ import annotations

import json
import traceback
from typing import Any, Dict, List

from config import (
    ANTHROPIC_API_KEY,
    CRITIC_MODEL,
    CRITIC_PROVIDER,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    PROJECT_ROOT,
)

from .critic_knowledge import load_field_index, load_policy, propose_policy_update, update_field_index
from .harness_types import CriticReport

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


class EngineeringCritic:
    """Cross-user pattern critic with multi-step dialogue."""

    def __init__(self, *, model: str = "", temperature: float = 0.2):
        self.model = model or CRITIC_MODEL or "claude-sonnet-4-6"
        self.temperature = temperature
        self.provider = CRITIC_PROVIDER

    def analyze_pattern(
        self,
        *,
        pattern: Dict[str, Any],
        total_users: int,
        rule_assets: Dict[str, Any],
        evolution_context: Dict[str, Any] | None = None,
        labels: Dict[str, str] | None = None,
    ) -> CriticReport:
        """Run multi-step analysis on a single cross-user pattern."""
        dimension = pattern.get("dimension", "")
        labels = labels or {}
        dimension_zh = labels.get(dimension, dimension)

        # Load knowledge layers
        policy = load_policy()
        field_index = load_field_index(dimension)

        # Determine critique level
        critique_level = self._determine_level(pattern, field_index)

        try:
            # Step 1: Preliminary diagnosis
            step1_result = self._step1_diagnosis(
                pattern=pattern,
                dimension_zh=dimension_zh,
                policy=policy,
                field_index=field_index,
                critique_level=critique_level,
                total_users=total_users,
            )

            # Gather evidence per Step 1 requests
            evidence = self._gather_evidence(
                requests=step1_result.get("evidence_requests", []),
                pattern=pattern,
                rule_assets=rule_assets,
                evolution_context=evolution_context,
            )

            # Step 2: Deep analysis
            step2_result = self._step2_analysis(
                pattern=pattern,
                dimension_zh=dimension_zh,
                step1=step1_result,
                evidence=evidence,
                critique_level=critique_level,
            )

            # Apply knowledge updates
            self._apply_knowledge_updates(
                dimension=dimension,
                pattern_id=pattern.get("pattern_id", ""),
                step2=step2_result,
            )

            return CriticReport(
                pattern_id=pattern.get("pattern_id", ""),
                dimension=dimension,
                system_diagnosis=step2_result.get("system_diagnosis", ""),
                root_cause_depth=str(step2_result.get("root_cause_depth", "")),
                recommendations=step2_result.get("recommendations", []),
                referenced_users=pattern.get("affected_users", []),
                referenced_evolution_context=evolution_context or {},
                confidence=float(step2_result.get("confidence", 0.5)),
                raw_llm_response={"step1": step1_result, "step2": step2_result},
            )

        except Exception as e:
            return CriticReport(
                pattern_id=pattern.get("pattern_id", ""),
                dimension=dimension,
                system_diagnosis=f"Critic 分析失败: {e}",
                confidence=0.0,
                raw_llm_response={"error": str(e), "traceback": traceback.format_exc()},
            )

    # ── Step 1: Preliminary diagnosis ──

    def _step1_diagnosis(
        self,
        *,
        pattern: Dict,
        dimension_zh: str,
        policy: str,
        field_index: Dict,
        critique_level: int,
        total_users: int,
    ) -> Dict:
        field_index_text = ""
        entries = field_index.get("entries", [])
        if entries:
            agg = field_index.get("aggregate", {})
            field_index_text = (
                f"\n该字段历史记录（{len(entries)} 条）:\n"
                f"  主要诊断: {agg.get('dominant_diagnosis', '无')}\n"
                f"  平均权重: {agg.get('avg_weight', 0)}\n"
                f"  审批通过率: {agg.get('approve_rate', 0)}\n"
            )
            for e in entries[:3]:
                field_index_text += f"  - {e.get('diagnosis', '')} (权重={e.get('weight', 0)}, 人类={e.get('human_verdict', '未审')})\n"

        prompt = f"""{policy}

---

你是记忆工程系统的首席技术评审官。当前质疑等级: Level {critique_level}。

以下是一个跨用户问题模式：
- 字段: {dimension_zh} ({pattern.get('dimension', '')})
- 失败模式: {pattern.get('failure_mode', '')}
- 影响范围: {len(pattern.get('affected_users', []))}/{total_users} 个用户，共 {pattern.get('total_case_count', 0)} cases
- 根因候选: {', '.join(pattern.get('root_cause_candidates', []))}
- 跨用户一致性: {pattern.get('cross_user_consistency', 0):.0%}
- 平均置信度: {pattern.get('avg_confidence', 0):.2f}
{field_index_text}

请给出初步诊断，并告诉我你需要看哪些具体证据来做最终判断。

只返回 JSON:
{{
    "preliminary_diagnosis": "一句话初步判断",
    "critique_level": {critique_level},
    "evidence_requests": [
        {{"type": "user_case", "user": "用户名", "reason": "想看什么"}},
        {{"type": "rule_asset", "asset": "field_spec/call_policy/tool_rules", "reason": "想看什么"}},
        {{"type": "evolution_history", "field": "字段名", "reason": "想看什么"}}
    ]
}}"""

        return self._call_llm(prompt)

    # ── Step 2: Deep analysis ──

    def _step2_analysis(
        self,
        *,
        pattern: Dict,
        dimension_zh: str,
        step1: Dict,
        evidence: Dict,
        critique_level: int,
    ) -> Dict:
        evidence_text = json.dumps(evidence, ensure_ascii=False, indent=2)
        if len(evidence_text) > 6000:
            evidence_text = evidence_text[:6000] + "\n...(截断)"

        level_instruction = ""
        if critique_level == 1:
            level_instruction = "请输出具体的规则修改建议（COT/tool_rule/call_policy），建议必须可直接执行。"
        elif critique_level == 2:
            level_instruction = "除了规则修改建议，还要分析是否需要新工具或新数据源。必须说明为什么改规则不够。"
        elif critique_level >= 3:
            level_instruction = "除了规则和工具建议，还要评估是否需要 Agent 架构变更。必须引用具体用户 case 作为论据。"

        prompt = f"""你的初步诊断: {step1.get('preliminary_diagnosis', '')}

以下是你要求的证据:
{evidence_text}

当前质疑等级: Level {critique_level}
{level_instruction}

请输出最终分析（JSON）:
{{
    "system_diagnosis": "系统性问题的深度分析（不是标签，是具体的技术分析）",
    "root_cause_depth": {{
        "surface": "表层根因",
        "deep": "深层根因（系统设计层面）"
    }},
    "confidence": 0.0到1.0,
    "recommendations": [
        {{
            "level": 1或2或3,
            "type": "cot_rule_update / tool_rule_update / call_policy_update / new_tool_needed / architecture_change",
            "description": "具体建议描述",
            "expected_impact": "预计影响范围",
            "confidence": 0.0到1.0
        }}
    ],
    "policy_update_proposal": null 或 {{"content": "一句话通用知识", "reason": "为什么这是通用的"}},
    "field_index_update": {{"adjustment": "描述", "weight_delta": 0.0}}
}}"""

        return self._call_llm(prompt)

    # ── Evidence gathering ──

    def _gather_evidence(
        self,
        *,
        requests: List[Dict],
        pattern: Dict,
        rule_assets: Dict,
        evolution_context: Dict | None,
    ) -> Dict:
        evidence: Dict[str, Any] = {}
        per_user = pattern.get("per_user_examples", {})

        for req in requests[:5]:  # Max 5 requests
            req_type = req.get("type", "")
            if req_type == "user_case":
                user = req.get("user", "")
                cases = per_user.get(user, [])
                if cases:
                    evidence[f"user_case:{user}"] = cases[:2]
                elif per_user:
                    first_user = list(per_user.keys())[0]
                    evidence[f"user_case:{first_user}"] = per_user[first_user][:2]

            elif req_type == "rule_asset":
                asset_name = req.get("asset", "")
                for key in ("field_spec_overrides", "call_policies", "tool_rules"):
                    if key in asset_name or asset_name in key:
                        dim = pattern.get("dimension", "")
                        asset_data = (rule_assets.get(key) or {}).get(dim, {})
                        if asset_data:
                            evidence[f"rule:{key}.{dim}"] = asset_data

            elif req_type == "evolution_history":
                if evolution_context:
                    field = req.get("field", pattern.get("dimension", ""))
                    evidence[f"evolution:{field}"] = evolution_context.get(field, {})

        # Always include a compact pattern summary
        evidence["pattern_summary"] = {
            "dimension": pattern.get("dimension"),
            "affected_users": pattern.get("affected_users"),
            "total_case_count": pattern.get("total_case_count"),
            "summary": pattern.get("summary"),
        }

        return evidence

    # ── Knowledge updates ──

    def _apply_knowledge_updates(self, *, dimension: str, pattern_id: str, step2: Dict) -> None:
        # Policy update proposal
        policy_proposal = step2.get("policy_update_proposal")
        if isinstance(policy_proposal, dict) and policy_proposal.get("content"):
            propose_policy_update(
                content=policy_proposal["content"],
                reason=policy_proposal.get("reason", ""),
                source_pattern_ids=[pattern_id],
            )

        # Field index update
        field_update = step2.get("field_index_update")
        if isinstance(field_update, dict):
            update_field_index(
                field_key=dimension,
                diagnosis=step2.get("system_diagnosis", ""),
                recommendation_level=int(step2.get("recommendations", [{}])[0].get("level", 1)) if step2.get("recommendations") else 1,
                pattern_id=pattern_id,
            )

    # ── Critique level determination ──

    def _determine_level(self, pattern: Dict, field_index: Dict) -> int:
        """Determine critique level based on pattern + history."""
        level = 1

        # Level 2 triggers
        entries = field_index.get("entries", [])
        failed_fixes = sum(1 for e in entries if e.get("human_verdict") == "reject")
        if failed_fixes >= 2:
            level = max(level, 2)
        if pattern.get("user_coverage", 0) >= 0.5:
            level = max(level, 2)

        # Check for coverage gap in any case
        for user_cases in (pattern.get("per_user_examples") or {}).values():
            for c in (user_cases or []):
                gap = (c.get("tool_usage_summary") or {}).get("coverage_gap") or {}
                if gap.get("has_gap"):
                    level = max(level, 2)
                    break

        # Level 3 triggers
        level2_rejected = sum(
            1 for e in entries
            if e.get("recommendation_level", 1) >= 2 and e.get("human_verdict") == "reject"
        )
        if level2_rejected >= 1:
            level = max(level, 3)

        return level

    # ── LLM calling ──

    def _call_llm(self, prompt: str) -> Dict:
        """Call LLM and parse JSON response."""
        if self.provider == "anthropic" and HAS_ANTHROPIC and ANTHROPIC_API_KEY:
            return self._call_anthropic(prompt)
        return self._call_openrouter(prompt)

    def _call_anthropic(self, prompt: str) -> Dict:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text if response.content else ""
        return self._parse_json(text)

    def _call_openrouter(self, prompt: str) -> Dict:
        """Fallback to OpenRouter."""
        import httpx
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        resp = httpx.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=body,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return self._parse_json(text)

    def _parse_json(self, text: str) -> Dict:
        """Extract JSON from LLM response text."""
        text = text.strip()
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try extracting from markdown fences
        for fence in ("```json", "```"):
            if fence in text:
                start = text.index(fence) + len(fence)
                end = text.index("```", start) if "```" in text[start:] else len(text)
                try:
                    return json.loads(text[start:end].strip())
                except json.JSONDecodeError:
                    pass
        # Try finding first { ... }
        brace_start = text.find("{")
        if brace_start >= 0:
            depth = 0
            for i in range(brace_start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start : i + 1])
                    except json.JSONDecodeError:
                        break
        return {"error": "failed_to_parse", "raw": text[:500]}
