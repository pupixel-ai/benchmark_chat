from __future__ import annotations

import compileall
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    PROFILE_LLM_MODEL,
    PROJECT_ROOT,
    REFLECTION_AGENT_MODEL,
    REFLECTION_AGENT_OPENROUTER_API_KEY,
    REFLECTION_AGENT_PROVIDER,
    REFLECTION_AGENT_TEMPERATURE,
)
from services.memory_pipeline.profile_llm import OpenRouterProfileLLMProcessor
from services.memory_pipeline.rule_asset_loader import apply_repo_rule_patch

from .storage import build_reflection_asset_paths, ensure_reflection_root
from .types import CaseFact, DecisionReviewItem, EngineeringChangeRequest, ExperimentOutcome, PatternCluster, ProposalReviewRecord, StrategyExperiment
from .upstream_triage import ALLOWED_ROOT_CAUSE_FAMILIES, ROOT_CAUSE_TO_FIX_SURFACE


PROPOSAL_TASK_OPTIONS = ["approve", "reject", "need_revision"]
ENGINEERING_EXECUTE_TASK_OPTIONS = ["approve", "reject", "need_revision"]
DEFAULT_EXPECTED_GAIN = {
    "exact_or_close_delta": 0,
    "mismatch_delta": 0,
}


class BadcasePacketAssembler:
    def __init__(self, *, project_root: str = PROJECT_ROOT) -> None:
        self.project_root = project_root

    def assemble(self, case_fact: CaseFact | Dict[str, Any]) -> Dict[str, Any]:
        fact = case_fact if isinstance(case_fact, CaseFact) else CaseFact(**case_fact)
        paths = build_reflection_asset_paths(project_root=self.project_root, user_name=fact.user_name)
        ensure_reflection_root(paths)

        trace_payload = _load_json_object(fact.trace_payload_path)
        historical_patterns = _read_json_array(paths.upstream_patterns_path)
        historical_experiments = _read_json_array(paths.upstream_experiments_path)
        historical_outcomes = _read_json_array(paths.upstream_outcomes_path)
        historical_feedback = _read_jsonl_records(paths.reflection_feedback_path)
        filtered_patterns = [
            payload
            for payload in historical_patterns
            if str(payload.get("dimension") or "") == fact.dimension
        ][:5]
        pattern_ids = {str(payload.get("pattern_id") or "") for payload in filtered_patterns}
        filtered_experiments = [
            payload
            for payload in historical_experiments
            if str(payload.get("pattern_id") or "") in pattern_ids
        ][:5]
        experiment_ids = {str(payload.get("experiment_id") or "") for payload in filtered_experiments}
        filtered_outcomes = [
            payload
            for payload in historical_outcomes
            if str(payload.get("experiment_id") or "") in experiment_ids
        ][:5]
        return {
            "case_fact": fact.to_dict(),
            "comparison_result": dict(fact.comparison_result or {}),
            "pre_audit_comparison_result": dict(fact.pre_audit_comparison_result or {}),
            "final_before_backflow": dict(trace_payload.get("final_before_backflow") or fact.pre_audit_output or {}),
            "final_after_backflow": dict(trace_payload.get("final") or fact.upstream_output or {}),
            "null_reason": str((trace_payload.get("final") or {}).get("null_reason") or (fact.upstream_output or {}).get("null_reason") or ""),
            "tool_trace": dict(trace_payload.get("tool_trace") or {}),
            "llm_batch_debug": _normalize_llm_batch_debug(trace_payload.get("llm_batch_debug")),
            "history_patterns": filtered_patterns,
            "history_experiments": filtered_experiments,
            "history_outcomes": filtered_outcomes,
            "history_feedback": [
                payload
                for payload in historical_feedback
                if str(payload.get("field_key") or "").strip() == fact.dimension
            ][:5],
            "trace_payload": trace_payload,
        }

    def trace_diagnose(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        case_fact = dict(packet.get("case_fact") or {})
        tool_trace = dict(packet.get("tool_trace") or {})
        final_before = dict(packet.get("final_before_backflow") or {})
        final_after = dict(packet.get("final_after_backflow") or {})
        evidence = dict(final_after.get("evidence") or {})
        supporting_refs = list(evidence.get("supporting_refs") or [])
        contradicting_refs = list(evidence.get("contradicting_refs") or [])
        retrieval_hit_count = _count_tool_supporting_refs(tool_trace)
        tool_called = bool(tool_trace) or retrieval_hit_count > 0
        diagnostic_flags: List[str] = []
        if not tool_called:
            diagnostic_flags.append("no_tool_call_but_tool_expected")
        elif retrieval_hit_count == 0:
            diagnostic_flags.append("tool_called_but_no_hits")
        elif str(case_fact.get("comparison_grade") or "").strip() in {"mismatch", "missing_prediction"}:
            diagnostic_flags.append("hits_exist_but_semantic_mapping_suspect")
        causality_route = str(case_fact.get("causality_route") or "").strip()
        if causality_route in {"downstream_caused", "downstream_exacerbated"}:
            diagnostic_flags.append("downstream_backflow_degraded")
        llm_batch_debug = _normalize_llm_batch_debug(packet.get("llm_batch_debug"))
        if any(str(item.get("error") or "").strip() or str(item.get("status") or "").strip().lower() not in {"", "ok", "success"} for item in llm_batch_debug):
            diagnostic_flags.append("runtime_or_parse_issue")
        return {
            "field_key": str(case_fact.get("dimension") or case_fact.get("entity_id") or "").strip(),
            "pre_audit_value": _truncate_text(final_before.get("value"), limit=160),
            "post_audit_value": _truncate_text(final_after.get("value"), limit=160),
            "null_reason": str(packet.get("null_reason") or final_after.get("null_reason") or ""),
            "audit_action_type": str(case_fact.get("audit_action_type") or ""),
            "tool_called": tool_called,
            "retrieval_hit_count": retrieval_hit_count,
            "supporting_ref_count": len(supporting_refs) or retrieval_hit_count,
            "contradicting_ref_count": len(contradicting_refs),
            "top_supporting_refs": _compact_evidence_refs(supporting_refs or _extract_tool_supporting_refs(tool_trace)),
            "top_contradicting_refs": _compact_evidence_refs(contradicting_refs or _extract_tool_contradicting_refs(tool_trace)),
            "llm_reason_excerpt": _truncate_text(final_after.get("reasoning") or final_before.get("reasoning"), limit=220),
            "diagnostic_flags": diagnostic_flags,
        }

    def history_recall(
        self,
        packet: Dict[str, Any],
        *,
        root_cause_candidates: Iterable[str] | None = None,
        fix_surface_candidates: Iterable[str] | None = None,
    ) -> Dict[str, Any]:
        case_fact = dict(packet.get("case_fact") or {})
        field_key = str(case_fact.get("dimension") or case_fact.get("entity_id") or "").strip()
        requested_root_causes = {str(item).strip() for item in list(root_cause_candidates or []) if str(item).strip()}
        requested_fix_surfaces = [str(item).strip() for item in list(fix_surface_candidates or []) if str(item).strip()]

        patterns = [dict(item or {}) for item in list(packet.get("history_patterns") or [])]
        experiments = [dict(item or {}) for item in list(packet.get("history_experiments") or [])]
        outcomes = [dict(item or {}) for item in list(packet.get("history_outcomes") or [])]
        feedback_records = [dict(item or {}) for item in list(packet.get("history_feedback") or [])]

        relevant_patterns = []
        for pattern in patterns:
            same_field = str(pattern.get("dimension") or "").strip() == field_key
            root_cause_overlap = bool(requested_root_causes.intersection({str(item).strip() for item in list(pattern.get("root_cause_candidates") or [])}))
            if same_field or root_cause_overlap:
                relevant_patterns.append(pattern)
        relevant_patterns = relevant_patterns[:3]
        pattern_ids = {str(item.get("pattern_id") or "").strip() for item in relevant_patterns if str(item.get("pattern_id") or "").strip()}

        relevant_experiments = [
            item for item in experiments
            if str(item.get("pattern_id") or "").strip() in pattern_ids
        ][:5]
        experiment_ids = {str(item.get("experiment_id") or "").strip() for item in relevant_experiments if str(item.get("experiment_id") or "").strip()}
        outcome_by_experiment = {
            str(item.get("experiment_id") or "").strip(): dict(item or {})
            for item in outcomes
            if str(item.get("experiment_id") or "").strip() in experiment_ids
        }

        successful_surfaces: List[str] = []
        failed_surfaces: List[str] = []
        for experiment in relevant_experiments:
            fix_surface = str(experiment.get("fix_surface") or experiment.get("recommended_option") or "").strip()
            outcome_status = str((outcome_by_experiment.get(str(experiment.get("experiment_id") or "").strip(), {}) or {}).get("status") or "").strip().lower()
            if not fix_surface:
                continue
            if outcome_status in {"success", "applied"}:
                if fix_surface not in successful_surfaces:
                    successful_surfaces.append(fix_surface)
            elif outcome_status in {"failed", "need_revision", "rejected"}:
                if fix_surface not in failed_surfaces:
                    failed_surfaces.append(fix_surface)

        recommended_surface_prior = ""
        for candidate in requested_fix_surfaces:
            if candidate in successful_surfaces:
                recommended_surface_prior = candidate
                break
        if not recommended_surface_prior and requested_fix_surfaces:
            recommended_surface_prior = requested_fix_surfaces[0]

        feedback_summary = {
            "approve_count": 0,
            "reject_count": 0,
            "need_revision_count": 0,
        }
        recent_feedback_notes: List[Dict[str, Any]] = []
        for feedback in feedback_records[:5]:
            human_action = str(feedback.get("human_action") or "").strip()
            if human_action == "proposal_approve":
                feedback_summary["approve_count"] += 1
            elif human_action in {"proposal_reject", "engineering_execute_reject"}:
                feedback_summary["reject_count"] += 1
            elif human_action in {"proposal_need_revision", "engineering_execute_need_revision"}:
                feedback_summary["need_revision_count"] += 1
            note = str(feedback.get("reviewer_note") or "").strip()
            if note:
                recent_feedback_notes.append(
                    {
                        "human_action": human_action,
                        "reviewer_note": note,
                    }
                )

        resolution_signal = "open"
        if any(s for s in successful_surfaces if s in (requested_fix_surfaces or successful_surfaces)):
            resolution_signal = "already_fixed"
        elif failed_surfaces and not successful_surfaces:
            resolution_signal = "failed_before"

        return {
            "field_key": field_key,
            "similar_pattern_count": len(relevant_patterns),
            "same_field_successful_surfaces": successful_surfaces,
            "same_field_failed_surfaces": failed_surfaces,
            "recommended_surface_prior": recommended_surface_prior,
            "resolution_signal": resolution_signal,
            "feedback_summary": feedback_summary,
            "recent_feedback_notes": recent_feedback_notes,
            "history_patterns": _compact_history_items(relevant_patterns),
            "recent_experiments": _compact_history_items(relevant_experiments),
            "recent_outcomes": _compact_history_items(
                [outcome_by_experiment[exp_id] for exp_id in experiment_ids if exp_id in outcome_by_experiment]
            ),
        }


class UpstreamReflectionAgent:
    def __init__(
        self,
        *,
        llm_processor: Any | None = None,
        provider: str = REFLECTION_AGENT_PROVIDER,
        model: str | None = None,
        temperature: float = REFLECTION_AGENT_TEMPERATURE,
        project_root: str = PROJECT_ROOT,
    ) -> None:
        self.project_root = project_root
        self.provider = str(provider or "").strip().lower() or "openrouter"
        self.model = str(model or REFLECTION_AGENT_MODEL or PROFILE_LLM_MODEL or "").strip()
        self.temperature = float(temperature)
        self.llm_processor = llm_processor or self._build_llm_processor()

    def reflect(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        case_fact = dict(packet.get("case_fact") or {})
        comparison_grade = str(case_fact.get("comparison_grade") or "").strip()
        causality_route = str(case_fact.get("causality_route") or "").strip()
        accuracy_gap_status = str(case_fact.get("accuracy_gap_status") or "").strip()

        if accuracy_gap_status != "open":
            return self._failed_result("case_not_open", packet)
        if causality_route in {"downstream_caused", "downstream_exacerbated"}:
            return self._difficult_case_result(packet, "downstream_backflow_degraded_gt_alignment")
        if comparison_grade in {"exact_match", "close_match"}:
            return self._failed_result("comparison_already_resolved", packet)
        if comparison_grade == "partial_match":
            return self._difficult_case_result(packet, "partial_match_requires_manual_confirmation")

        heuristic = self._heuristic_result(packet)
        if self.llm_processor is None:
            return {
                **heuristic,
                "status": "failed",
                "error": "reflection_agent_model_unavailable",
            }

        prompt = self._build_prompt(packet)
        try:
            response = self.llm_processor._call_llm_via_official_api(
                prompt,
                response_mime_type="application/json",
                model_override=self.model or None,
            )
        except Exception as exc:
            return {
                **heuristic,
                "status": "failed",
                "error": f"reflection_agent_call_failed:{exc}",
            }
        if not isinstance(response, dict):
            return {
                **heuristic,
                "status": "failed",
                "error": "reflection_agent_invalid_response",
            }

        tool_outputs = self._execute_requested_tools(packet, response)
        if tool_outputs:
            followup_prompt = self._build_followup_prompt(packet, tool_outputs)
            try:
                response = self.llm_processor._call_llm_via_official_api(
                    followup_prompt,
                    response_mime_type="application/json",
                    model_override=self.model or None,
                )
            except Exception as exc:
                return {
                    **heuristic,
                    "status": "failed",
                    "error": f"reflection_agent_followup_call_failed:{exc}",
                }
            if not isinstance(response, dict):
                return {
                    **heuristic,
                    "status": "failed",
                    "error": "reflection_agent_invalid_followup_response",
                }

        root_cause_family = str(response.get("root_cause_family") or heuristic["root_cause_family"]).strip()
        if root_cause_family not in ALLOWED_ROOT_CAUSE_FAMILIES:
            return {
                **heuristic,
                "status": "failed",
                "error": "reflection_agent_invalid_root_cause_family",
            }
        recommended_fix_surface = str(
            response.get("recommended_fix_surface")
            or ROOT_CAUSE_TO_FIX_SURFACE.get(root_cause_family, heuristic["recommended_fix_surface"])
        ).strip()
        status = str(response.get("status") or "ok").strip() or "ok"
        if status not in {"ok", "needs_review"}:
            status = "ok"
        judgment_summary_zh = str(response.get("judgment_summary_zh") or "").strip()
        key_evidence_zh = [str(item).strip() for item in list(response.get("key_evidence_zh") or []) if str(item).strip()]
        why_this_surface_zh = str(response.get("why_this_surface_zh") or "").strip()
        why_not_other_surfaces_zh = str(response.get("why_not_other_surfaces_zh") or "").strip()
        if not judgment_summary_zh or not key_evidence_zh or not why_this_surface_zh or not why_not_other_surfaces_zh:
            return {
                **heuristic,
                "status": "failed",
                "error": "reflection_agent_missing_required_reasoning",
            }
        return {
            "status": status,
            "root_cause_family": root_cause_family,
            "recommended_fix_surface": recommended_fix_surface,
            "confidence": _safe_float(response.get("confidence"), default=heuristic["confidence"]),
            "judgment_summary_zh": judgment_summary_zh,
            "key_evidence_zh": key_evidence_zh,
            "why_this_surface_zh": why_this_surface_zh,
            "why_not_other_surfaces_zh": why_not_other_surfaces_zh,
            "decision_tree_path": heuristic["decision_tree_path"],
            "patch_intent": _normalize_patch_intent(
                response.get("patch_intent"),
                default=dict(heuristic["patch_intent"]),
                field_key=str((heuristic.get("patch_intent") or {}).get("field_key") or ""),
                fix_surface=recommended_fix_surface,
            ),
            "expected_metric_gain": dict(heuristic["expected_metric_gain"]),
        }

    def _execute_requested_tools(self, packet: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
        tool_requests = list(response.get("tool_requests") or [])
        if not tool_requests:
            return {}
        assembler = BadcasePacketAssembler(project_root=self.project_root)
        tool_outputs: Dict[str, Any] = {}
        for tool_request in tool_requests[:2]:
            request = dict(tool_request or {})
            tool_name = str(request.get("tool_name") or "").strip()
            arguments = dict(request.get("arguments") or {})
            if tool_name == "trace_diagnose" and "trace_diagnosis" not in tool_outputs:
                tool_outputs["trace_diagnosis"] = assembler.trace_diagnose(packet)
            elif tool_name == "history_recall" and "history_recall" not in tool_outputs:
                tool_outputs["history_recall"] = assembler.history_recall(
                    packet,
                    root_cause_candidates=list(arguments.get("root_cause_candidates") or []),
                    fix_surface_candidates=list(arguments.get("fix_surface_candidates") or []),
                )
        return tool_outputs

    def _build_llm_processor(self) -> Any | None:
        if self.provider != "openrouter":
            return None
        api_key = (REFLECTION_AGENT_OPENROUTER_API_KEY or OPENROUTER_API_KEY or "").strip()
        if not api_key:
            return None
        try:
            return OpenRouterProfileLLMProcessor(
                api_key=api_key,
                base_url=OPENROUTER_BASE_URL,
                model=self.model or PROFILE_LLM_MODEL,
            )
        except Exception:
            return None

    def _heuristic_result(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        case_fact = dict(packet.get("case_fact") or {})
        root_cause_family = str(case_fact.get("root_cause_family") or "field_reasoning").strip() or "field_reasoning"
        recommended_fix_surface = ROOT_CAUSE_TO_FIX_SURFACE.get(root_cause_family, "watch_only")
        field_key = str(case_fact.get("dimension") or case_fact.get("entity_id") or "").strip()
        return {
            "status": "ok",
            "root_cause_family": root_cause_family,
            "recommended_fix_surface": recommended_fix_surface,
            "confidence": _safe_float(case_fact.get("fix_surface_confidence"), default=0.75),
            "judgment_summary_zh": str((case_fact.get("decision_trace") or {}).get("reason") or "").strip() or "当前更像既有 triage 推荐的同类问题。",
            "key_evidence_zh": [
                "当前证据先按既有 triage 结论处理。",
            ],
            "why_this_surface_zh": f"当前更适合先按 {recommended_fix_surface} 这一改面推进。",
            "why_not_other_surfaces_zh": "当前证据先按既有 triage 推荐面处理，其它改面暂未看到更强证据。",
            "decision_tree_path": _build_decision_tree_path(case_fact),
            "patch_intent": {
                "field_key": field_key,
                "fix_surface": recommended_fix_surface,
                "change_summary_zh": f"按 {recommended_fix_surface} 方向继续收口该字段策略。",
            },
            "expected_metric_gain": dict(DEFAULT_EXPECTED_GAIN),
        }

    def _difficult_case_result(self, packet: Dict[str, Any], reason: str) -> Dict[str, Any]:
        case_fact = dict(packet.get("case_fact") or {})
        return {
            "status": "needs_review",
            "root_cause_family": "watch_only",
            "recommended_fix_surface": "watch_only",
            "confidence": 0.0,
            "judgment_summary_zh": "当前结论还不够稳定，更适合进入 difficult_case / 人工判断。",
            "key_evidence_zh": [reason],
            "why_this_surface_zh": "因为这条 case 目前不适合直接推进具体策略改面。",
            "why_not_other_surfaces_zh": "当前更适合进入 difficult_case，而不是直接推动策略改面。",
            "decision_tree_path": _build_decision_tree_path(case_fact, tail=["difficult_case"]),
            "patch_intent": {
                "field_key": str(case_fact.get("dimension") or case_fact.get("entity_id") or "").strip(),
                "fix_surface": "watch_only",
                "change_summary_zh": "暂不直接改策略，先进入 difficult_case 做进一步人工判断。",
            },
            "expected_metric_gain": dict(DEFAULT_EXPECTED_GAIN),
        }

    def _failed_result(self, error: str, packet: Dict[str, Any]) -> Dict[str, Any]:
        heuristic = self._heuristic_result(packet)
        heuristic["status"] = "failed"
        heuristic["error"] = error
        return heuristic

    def _build_prompt(self, packet: Dict[str, Any]) -> str:
        allowed_fix_surfaces = ["field_cot", "tool_rule", "call_policy", "engineering_issue", "watch_only"]
        compact_packet = _build_compact_prompt_packet(packet)
        return f"""你是一个擅长找线索、分析严谨的反思 Agent。

你的背景目标：
我们在做一个记忆工程，我们会从用户的图片出发，经过 VLM、事件提取、人物关系推断，最后生成用户画像字段。
你的任务不是直接改代码，而是判断某条画像 badcase 为什么错、最可能该改哪一层策略，并给人工审批提供最关键的判断依据。

你只负责：
- 基于 badcase 进行精准的上游定位、反思
- 为人工审批提供清晰、精炼、可判断的结论
- 产出可直接应用的 rule_patch 配置（JSON 配置，非 Python 代码）
- 必要时请求 tool 来补充定位信息

你不负责：
- 直接修改 Python 代码
- 直接批准代码变更
- 总结底层调试日志

你要理解的当前工程链路：
1. 图片与 VLM 先形成事件和人物上下文
2. 主角判断产物：primary_decision
3. 关系推断产物：relationship_dossiers
4. LP3 画像字段主产物：profile_fact_decisions
5. LP3 批次调试信息：profile_llm_batch_debug
6. 单字段完整 trace：profile_field_trace_payload
7. GT 对比与反思资产：case_fact / comparison_result / pre_audit_comparison_result
8. 下游裁决产物：downstream_audit_report

核心原则（必须遵守）：
- 严禁通过 GT 答案直接限制或指导输出方向。GT 只用于发现问题，不用于推导答案。
- 你必须假设自己不知道 GT 的具体值。你只知道 grade（偏差类型）和 severity（严重程度）。思考"如果没有 GT，系统应该从哪些证据中发现正确的规律"。
- 你的提案必须是通用规则改进，不能针对让系统输出某个特定值。
- 如果系统输出和 GT 都是对的（比如都是 null），不要强行提修改建议。
- 如果 grade 是 missing_gt（GT 没有标注值但系统有输出），这是召回场景。此时你应该评估"系统输出的值是否有证据支撑且合理"。如果合理，就建议保留当前输出，提案方向是"确认系统发现的新信号有价值"。只有系统输出明显不合理时才建议修改。
- 不要倾向于让系统输出空值来回避问题。应该用 cot_hint 或 tool_rules 来改进推理质量。
- 反思时优先发散思考：先思考"证据里还有没有被系统忽略的线索"，再思考"推理逻辑是否合理"，最后才考虑"是否需要收敛条件"。
- 提案的 rule_patch 应该帮助系统找到更多证据或改进推理方式，而不是通过收紧 null 条件来回避错误。

你的定位决策树（按顺序快速判断，不需要逐步展开每一步）：
1. grade 是 exact_match/close_match → 不需要修改，直接输出 status=ok
2. causality_route 是 downstream_caused/exacerbated → difficult_case，不推上游改面
3. 看 final/draft/null_reason：一开始就空？有值但跑偏？被下游改坏？→ 定位问题阶段
4. 看 tool_trace 的 evidence_bundle：关键证据是否被取到？是否被正确呈现？→ 区分取证问题 vs 归纳问题
5. 看 reasoning + field_spec：归纳逻辑是否合理？→ 定位具体根因

如果你在第 3-4 步无法判断，请求 trace_diagnose tool。如果需要知道历史修复记录，请求 history_recall。最多 2 个 tool。

你的核心任务：找到系统忽略了什么线索。case card 里的证据可能包含系统没注意到的关键信息，把它指出来。

问题分类（root_cause_family，5 种，每种对应明确的修复手段）：
- field_reasoning：LLM 看到了关键证据，但归纳逻辑/理解口径有问题 → 修复：cot_hint 引导推理
- evidence_packaging：关键证据在系统中存在但被埋没，LLM 没看到应该看到的线索 → 修复：调整证据过滤排序（tool_rules），或用 cot_hint 告诉 LLM 主动关注被忽略的证据类型
- tool_retrieval：关键证据不在系统中，取证/召回本身失败，或该调的数据源没配置 → 修复：tool_rules 或 call_policies
- engineering_issue：运行错误、解析失败、上下游状态不一致 → 标记给工程
- watch_only：证据不够稳定或多根因难排除，暂不推动改面

什么时候用 tool：
- 如果你无法区分“没取到证据”还是“取到了证据但归纳错了”，请求 trace_diagnose
- 如果你需要知道“这个字段过去改哪一面更容易成功”，请求 history_recall
- 如果当前 case card 已经足够清楚，不要请求 tool
- 最多请求 2 个 tool

如果要请求 tool，只输出 JSON：
{{
  "status": "need_tool",
  "tool_requests": [
    {{
      "tool_name": "trace_diagnose",
      "arguments": {{}}
    }}
  ],
  "why_need_tool_zh": "需要先确认关键证据是否已经被正确召回，才能区分是 tool 问题还是字段归纳问题。"
}}

如果信息足够，输出最终 JSON：
{{
  "status": "ok" | "needs_review",
  "root_cause_family": "{' | '.join(ALLOWED_ROOT_CAUSE_FAMILIES)}",
  "recommended_fix_surface": "{' | '.join(allowed_fix_surfaces)}",
  "confidence": 0.0,
  "judgment_summary_zh": "一句话结论，直接说明这条 case 更像什么问题，建议改哪里。",
  "key_evidence_zh": [
    "2-3 条最关键证据，只写辅助人工判断的人话，不要写底层计数器或调试字段名。"
  ],
  "why_this_surface_zh": "为什么是这个改面。",
  "why_not_other_surfaces_zh": "为什么不是另外两个常见改面。",
  “patch_intent”: {{
    “field_key”: “字段 key”,
    “fix_surface”: “{' | '.join(allowed_fix_surfaces)}”,
    “change_summary_zh”: “用人话说想改什么策略。”,
    “rule_patch”: {{
      “tool_rules”: {{
        “字段key”: {{
          “max_refs_per_source”: 20,
          “required_evidence_types”: [“event”, “vlm_observations”]
        }}
      }},
      “call_policies”: {{
        “字段key”: {{
          “append_allowed_sources”: [“event”, “vlm_observations”, “relationship”]
        }}
      }},
      “field_spec_overrides”: {{
        “字段key”: {{
          “cot_hint”: “归纳引导”,
          “field_boundary”: “字段边界定义”,
          “cross_field_caution”: “跨字段隔离提示”
        }}
      }}
    }}
  }}
}}

rule_patch 槽位说明：

field_spec_overrides 内可用的字段：

增量槽位（优先使用，不影响默认行为）：
1. cot_hint (str)：归纳引导——帮助 LLM 理解字段含义、区分容易混淆的概念
2. field_boundary (str)：字段边界——“本字段是 X 而非 Y，和 Z 字段的区别是...”
3. cross_field_caution (str)：跨字段隔离——“不要从 A 字段的结论推导本字段的值”

替换槽位（仅当默认步骤本身有问题时使用，必须产出完整内容）：
4. cot_steps (List[str])：替换该字段的完整推理步骤
   注意：这会覆盖默认的 3-4 条推理步骤，你必须产出完整的步骤列表（不是补丁），考虑该字段已知的所有 case 场景，保留默认步骤中仍然有效的部分
5. strong_evidence (List[str])：替换高权重信号列表
6. owner_resolution_steps (List[str])：替换归属校验步骤
7. time_reasoning_steps (List[str])：替换时间维度判断步骤
8. counter_evidence_checks (List[str])：替换反证检查项

谨慎使用（容易导致系统偏向输出 null）：
9. weak_evidence_caution (List[str])：弱证据条件——触发时系统倾向输出 null，尽量少用

tool_rules 和 call_policies 在证据确实缺失时使用（格式见上方 JSON）
- 具体的条件描述用自然语言中文写，不要写代码

输出要求：
- 所有解释性文字必须是简体中文
- 请用通俗易懂的人话，像在给不懂技术的人解释结论
- 不要解释术语，也不要直接复述 field_cot、tool_rule、call_policy 这类词；直接说”判断口径””找证据规则””调用时机”
- 不要输出 retrieval_hit_count、comparison_score、comparison_method、stats_bundle_keys 之类的低价值调试信息
- 不要把日志当结论复述
- 重点写：
  - GT 偏差方向（grade 和 severity 告诉你的）
  - 当前输出成了什么
  - 关键线索是什么
  - 为什么判断是这一层问题
  - 为什么不是另外几个常见改面
- 如果判断不稳定，输出 needs_review，并更偏 watch_only / difficult_case

严格禁止：
- 禁止根据单个用户或单个字段的特殊情况做硬编码优化
- 规则修改必须是通用的，适用于所有用户
- 不要在 rule_patch 里引用具体的 photo_id、EVT_id、Person_id、人名、地名、品牌名

{"如果 case card 中包含 human_reviewer_note，这是上一轮审批者的修订意见，你必须认真考虑这个反馈，在本轮反思中回应它。但不要把它当作固定规则，只作为本轮的参考方向。" if packet.get("human_reviewer_note") else ""}

{"如果 case card 中包含 evolution_context，说明这个字段已经在之前的轮次中被反思过。" + chr(10) + "evolution_context 包含：cycle_index（当前第几轮）、previous_grade（上轮评分）、score_trend（分数趋势）、patch_effect（上次规则修改后的效果）、last_proposed_direction（上次提出的改进方向）。" + chr(10) + "你应该：" + chr(10) + "1) 如果 patch_effect 显示 degraded（恶化），必须换一个不同的改进方向，不要重复上次的建议" + chr(10) + "2) 如果 patch_effect 显示 improved 但还未达标，在上次方向的基础上深化" + chr(10) + "3) 如果 score_trend 是 stable 且 no_new_signal_streak > 2，考虑 watch_only" + chr(10) + "4) 不要重复提出和 last_proposed_direction 完全相同的建议，除非你有新的证据" if packet.get("evolution_context") else ""}

这是当前 case card：
{json.dumps(compact_packet, ensure_ascii=False)}"""

    def _build_followup_prompt(self, packet: Dict[str, Any], tool_outputs: Dict[str, Any]) -> str:
        allowed_fix_surfaces = ["field_cot", "tool_rule", "call_policy", "engineering_issue", "watch_only"]
        compact_packet = _build_compact_prompt_packet(packet)
        return f"""你已经拿到了精简 tool 诊断结果。请基于 case card + tool 输出最终结论，不要再请求额外 tool。

输出格式保持不变：
{{
  "status": "ok" | "needs_review",
  "root_cause_family": "{' | '.join(ALLOWED_ROOT_CAUSE_FAMILIES)}",
  "recommended_fix_surface": "{' | '.join(allowed_fix_surfaces)}",
  "confidence": 0.0,
  "judgment_summary_zh": "一句话结论",
  "key_evidence_zh": ["2-3 条关键证据"],
  "why_this_surface_zh": "为什么是这个改面",
  "why_not_other_surfaces_zh": "为什么不是另外两个常见改面",
  “patch_intent”: {{
    “field_key”: “字段 key”,
    “fix_surface”: “{' | '.join(allowed_fix_surfaces)}”,
    “change_summary_zh”: “用人话说想改什么策略”,
    “rule_patch”: {{
      “tool_rules”: {{}},
      “call_policies”: {{}},
      “field_spec_overrides”: {{}}
    }}
  }}
}}

rule_patch 优先用增量槽位（cot_hint / field_boundary / cross_field_caution），默认步骤有问题时才用替换槽位（cot_steps 等，必须产出完整内容），证据缺失时用 tool_rules / call_policies。禁止根据单个用户/字段做硬编码优化。

仍然要求：
- 所有解释性文字必须是简体中文
- 用通俗人话，不复述术语
- 不要输出低价值调试字段
- 重点写 GT、当前输出、关键线索、改面判断和排除理由

{json.dumps(
    {
        "case_card": compact_packet,
        "trace_diagnosis": dict(tool_outputs.get("trace_diagnosis") or {}),
        "history_recall": dict(tool_outputs.get("history_recall") or {}),
    },
    ensure_ascii=False,
)}"""


class ExperimentPlanner:
    def __init__(self, *, project_root: str = PROJECT_ROOT, evaluator: Any | None = None) -> None:
        self.project_root = project_root
        self.evaluator = evaluator

    def plan(
        self,
        *,
        pattern: PatternCluster,
        agent_result: Dict[str, Any],
    ) -> StrategyExperiment:
        experiment_id = _stable_id("exp", f"{pattern.pattern_id}|{agent_result.get('recommended_fix_surface')}")
        experiment_dir = self._experiment_dir(pattern.user_name, experiment_id)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        override_bundle_path = experiment_dir / "override_bundle.json"
        experiment_report_path = experiment_dir / "experiment_report.json"
        patch_intent = dict(agent_result.get("patch_intent") or {})
        overlay_bundle = _build_overlay_bundle(
            fix_surface=str(agent_result.get("recommended_fix_surface") or ""),
            field_key=patch_intent.get("field_key") or pattern.dimension,
            patch_intent=patch_intent,
        )
        override_bundle_path.write_text(json.dumps(overlay_bundle, ensure_ascii=False, indent=2), encoding="utf-8")
        return StrategyExperiment(
            experiment_id=experiment_id,
            pattern_id=pattern.pattern_id,
            user_name=pattern.user_name,
            lane=pattern.lane,
            fix_surface=str(agent_result.get("recommended_fix_surface") or ""),
            field_key=str(patch_intent.get("field_key") or pattern.dimension or ""),
            change_scope="single_field_single_surface",
            hypothesis=str(agent_result.get("reason") or pattern.summary or ""),
            status="proposed",
            override_bundle_path=str(override_bundle_path),
            experiment_report_path=str(experiment_report_path),
            proposal_status="not_built",
            approval_required=True,
            evidence_refs=list(pattern.evidence_refs),
            history_pattern_ids=list(pattern.history_pattern_ids),
            history_experiment_ids=list(pattern.history_experiment_ids),
            learning_summary=dict(pattern.history_summary or {}),
            metrics={
                "support_count": pattern.support_count,
                "fix_surface_confidence": pattern.fix_surface_confidence,
            },
        )

    def execute(
        self,
        *,
        pattern: PatternCluster,
        experiment: StrategyExperiment,
        agent_result: Dict[str, Any],
        support_cases: List[CaseFact] | None = None,
    ) -> Dict[str, Any]:
        report_path = Path(experiment.experiment_report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        evaluator = self.evaluator or self._default_evaluator
        report = evaluator(
            pattern=pattern,
            experiment=experiment,
            agent_result=agent_result,
            support_cases=support_cases or [],
        )
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    def _default_evaluator(
        self,
        *,
        pattern: PatternCluster,
        experiment: StrategyExperiment,
        agent_result: Dict[str, Any],
        support_cases: List[CaseFact] | None = None,
    ) -> Dict[str, Any]:
        patch_preview = _load_json_object(experiment.override_bundle_path)
        representative = support_cases[0] if support_cases else None
        if representative is not None:
            comparison = dict(representative.comparison_result or {})
            gt_value = comparison.get("gt_value")
            if gt_value in (None, ""):
                gt_value = (representative.gt_payload or {}).get("gt_value")
            current_output = comparison.get("output_value")
            if current_output in (None, ""):
                current_output = (representative.upstream_output or {}).get("value")
            current_grade = str(comparison.get("grade") or representative.comparison_grade or "").strip()
            if gt_value not in (None, "") and current_output not in (None, ""):
                return {
                    "status": "completed",
                    "reason": "gt_alignment_without_replay",
                    "evaluation_mode": "gt_alignment_without_replay",
                    "baseline_metrics": {
                        "comparison_grade": current_grade,
                    },
                    "candidate_metrics": {
                        "comparison_grade": current_grade,
                    },
                    "metric_gain": dict(DEFAULT_EXPECTED_GAIN),
                    "is_significant_improvement": False,
                    "current_case_result": {
                        "gt_value": gt_value,
                        "current_output": current_output,
                        "candidate_output": current_output,
                        "comparison_grade": current_grade,
                        "candidate_comparison_grade": current_grade,
                    },
                    "patch_preview": patch_preview,
                    "diff_summary": [
                        "当前已基于 GT 完成对齐评估；因未接入可重跑离线样本，候选输出先按当前输出保守记录，等待审批后再做正式回放。"
                    ],
                }
        return {
            "status": "need_revision",
            "reason": "missing_offline_case_dir",
            "baseline_metrics": {},
            "candidate_metrics": {},
            "metric_gain": dict(DEFAULT_EXPECTED_GAIN),
            "is_significant_improvement": False,
            "patch_preview": patch_preview,
            "diff_summary": ["当前 support cases 没有绑定可重跑的离线 bundle，实验仍需补齐 case_dir。"],
        }

    def _experiment_dir(self, user_name: str, experiment_id: str) -> Path:
        paths = build_reflection_asset_paths(project_root=self.project_root, user_name=user_name)
        ensure_reflection_root(paths)
        return Path(paths.experiments_dir) / experiment_id


class ProposalBuilder:
    def build(
        self,
        *,
        pattern: PatternCluster,
        experiment: StrategyExperiment,
        agent_result: Dict[str, Any],
        experiment_report: Dict[str, Any],
        support_cases: List[CaseFact] | None = None,
    ) -> Dict[str, Any] | None:
        if str(experiment_report.get("status") or "").strip() != "completed":
            return None
        reason = str(agent_result.get("reason") or "").strip()
        why_not_other_surfaces = str(agent_result.get("why_not_other_surfaces") or "").strip()
        if not reason or not why_not_other_surfaces:
            return None

        proposal_id = _stable_id("proposal", experiment.experiment_id)
        task_id = _stable_id("task", f"proposal_review|{proposal_id}")
        now = _utcnow_iso()
        representative = support_cases[0] if support_cases else None
        current_case_result = dict(experiment_report.get("current_case_result") or {})
        current_output = current_case_result.get("current_output")
        if current_output in (None, "") and representative is not None:
            current_output = (representative.upstream_output or {}).get("value")
        gt_value = current_case_result.get("gt_value")
        if gt_value in (None, "") and representative is not None:
            gt_value = (representative.gt_payload or {}).get("gt_value")
        candidate_output = current_case_result.get("candidate_output")
        execution_path_recommendation = _determine_execution_path_recommendation(experiment_report)
        result_delta_summary = _build_result_delta_summary(
            gt_value=gt_value,
            current_output=current_output,
            candidate_output=candidate_output,
            execution_path_recommendation=execution_path_recommendation,
        )
        is_significant_improvement = bool(experiment_report.get("is_significant_improvement"))
        summary = (
            f"{pattern.dimension} 实验显著改善，等待你确认是否落正式规则"
            if is_significant_improvement
            else f"{pattern.dimension} 已完成 GT 对齐评估，等待你确认是否按该方向推进工程收口"
        )
        proposal = ProposalReviewRecord(
            proposal_id=proposal_id,
            task_id=task_id,
            experiment_id=experiment.experiment_id,
            pattern_id=pattern.pattern_id,
            user_name=pattern.user_name,
            lane=pattern.lane,
            field_key=experiment.field_key or pattern.dimension,
            fix_surface=experiment.fix_surface,
            summary=summary,
            detail_url=f"/review/task/{task_id}",
            status="pending_review",
            approval_required=True,
            recommended_option="approve",
            options=list(PROPOSAL_TASK_OPTIONS),
            agent_reasoning_summary=reason,
            key_evidence_zh=[str(item).strip() for item in list(agent_result.get("key_evidence_zh") or []) if str(item).strip()],
            why_this_surface_zh=str(agent_result.get("why_this_surface_zh") or "").strip(),
            why_not_other_surfaces=why_not_other_surfaces,
            decision_tree_path=list(agent_result.get("decision_tree_path") or []),
            patch_intent=dict(agent_result.get("patch_intent") or {}),
            patch_preview=dict(experiment_report.get("patch_preview") or {}),
            diff_summary=list(experiment_report.get("diff_summary") or []),
            baseline_metrics=dict(experiment_report.get("baseline_metrics") or {}),
            candidate_metrics=dict(experiment_report.get("candidate_metrics") or {}),
            metric_gain=dict(experiment_report.get("metric_gain") or {}),
            execution_path_recommendation=execution_path_recommendation,
            gt_value=gt_value,
            current_output=current_output,
            candidate_output=candidate_output,
            result_delta_summary=result_delta_summary,
            override_bundle_path=experiment.override_bundle_path,
            experiment_report_path=experiment.experiment_report_path,
            proposal_status="pending_review",
            created_at=now,
            updated_at=now,
        )
        task = DecisionReviewItem(
            task_id=task_id,
            task_type="proposal_review",
            pattern_id=pattern.pattern_id,
            user_name=pattern.user_name,
            lane=pattern.lane,
            priority=pattern.business_priority,
            summary=proposal.summary,
            detail_url=proposal.detail_url,
            support_case_ids=list(pattern.support_case_ids),
            options=list(PROPOSAL_TASK_OPTIONS),
            recommended_option="approve",
            status="new",
            feishu_status="not_triggered",
            created_at=now,
            updated_at=now,
            album_id=pattern.album_id,
            proposal_id=proposal_id,
            experiment_id=experiment.experiment_id,
            evidence_refs=list(pattern.evidence_refs),
        )
        return {
            "proposal": proposal.to_dict(),
            "task": task,
        }


class MemoryEngineerAgent:
    def __init__(
        self,
        *,
        llm_processor: Any | None = None,
        provider: str = REFLECTION_AGENT_PROVIDER,
        model: str | None = None,
        temperature: float = REFLECTION_AGENT_TEMPERATURE,
    ) -> None:
        self.provider = str(provider or "").strip().lower() or "openrouter"
        self.model = str(model or REFLECTION_AGENT_MODEL or PROFILE_LLM_MODEL or "").strip()
        self.temperature = float(temperature)
        self.llm_processor = llm_processor or self._build_llm_processor()

    def build_change_request(self, *, proposal: Dict[str, Any], reviewer_note: str = "") -> Dict[str, Any]:
        field_key = str(proposal.get("field_key") or "").strip()
        fix_surface = str(proposal.get("fix_surface") or "watch_only").strip()
        execution_path = str(proposal.get("execution_path_recommendation") or "engineer_rewrite_apply").strip() or "engineer_rewrite_apply"
        patch_preview = (
            dict(proposal.get("patch_preview") or {})
            if execution_path == "overlay_direct_apply"
            else self._build_engineer_patch_preview(proposal=proposal, reviewer_note=reviewer_note)
        )
        fallback_change_summary = self._build_change_summary(
            field_key=field_key,
            fix_surface=fix_surface,
            execution_path=execution_path,
            reviewer_note=reviewer_note,
            proposal=proposal,
        )
        fallback_short_reason = self._build_short_reason(proposal=proposal)
        llm_result = self._generate_engineer_summary(
            proposal=proposal,
            reviewer_note=reviewer_note,
            fallback_change_summary=fallback_change_summary,
            fallback_short_reason=fallback_short_reason,
        )
        change_summary = str(llm_result.get("change_summary_zh") or fallback_change_summary).strip() or fallback_change_summary
        short_reason = str(llm_result.get("short_reason_zh") or fallback_short_reason).strip() or fallback_short_reason
        change_request_id = _stable_id(
            "change_request",
            f"{proposal.get('proposal_id') or ''}|{execution_path}|{field_key}|{fix_surface}",
        )
        task_id = _stable_id("task", f"engineering_execute_review|{change_request_id}")
        now = _utcnow_iso()
        return EngineeringChangeRequest(
            change_request_id=change_request_id,
            task_id=task_id,
            proposal_id=str(proposal.get("proposal_id") or ""),
            experiment_id=str(proposal.get("experiment_id") or ""),
            pattern_id=str(proposal.get("pattern_id") or ""),
            user_name=str(proposal.get("user_name") or ""),
            field_key=field_key,
            fix_surface=fix_surface,
            execution_path=execution_path,
            approved_scope="single_field_single_surface",
            detail_url=f"/review/task/{task_id}",
            change_summary_zh=change_summary,
            short_reason_zh=short_reason,
            gt_value=proposal.get("gt_value"),
            current_output=proposal.get("current_output"),
            candidate_output=proposal.get("candidate_output"),
            overlay_experiment_summary=str(proposal.get("result_delta_summary") or ""),
            key_evidence_zh=[str(item).strip() for item in list(proposal.get("key_evidence_zh") or []) if str(item).strip()],
            support_case_ids=list(proposal.get("support_case_ids") or []),
            evidence_refs=list(proposal.get("evidence_refs") or []),
            patch_preview=patch_preview,
            status="pending_review",
            created_at=now,
            updated_at=now,
        ).to_dict()

    def _generate_engineer_summary(
        self,
        *,
        proposal: Dict[str, Any],
        reviewer_note: str,
        fallback_change_summary: str,
        fallback_short_reason: str,
    ) -> Dict[str, Any]:
        if str(proposal.get("execution_path_recommendation") or "").strip() == "overlay_direct_apply":
            return {
                "change_summary_zh": fallback_change_summary,
                "short_reason_zh": fallback_short_reason,
            }
        if self.llm_processor is None:
            return {
                "change_summary_zh": fallback_change_summary,
                "short_reason_zh": fallback_short_reason,
            }

        prompt = self._build_prompt(
            proposal=proposal,
            reviewer_note=reviewer_note,
            fallback_change_summary=fallback_change_summary,
            fallback_short_reason=fallback_short_reason,
        )
        try:
            response = self.llm_processor._call_llm_via_official_api(
                prompt,
                response_mime_type="application/json",
                model_override=self.model or None,
            )
        except Exception:
            return {
                "change_summary_zh": fallback_change_summary,
                "short_reason_zh": fallback_short_reason,
            }
        if not isinstance(response, dict):
            return {
                "change_summary_zh": fallback_change_summary,
                "short_reason_zh": fallback_short_reason,
            }

        change_summary = str(response.get("change_summary_zh") or "").strip()
        short_reason = str(response.get("short_reason_zh") or "").strip()
        if not change_summary or not short_reason:
            return {
                "change_summary_zh": fallback_change_summary,
                "short_reason_zh": fallback_short_reason,
            }
        return {
            "change_summary_zh": change_summary,
            "short_reason_zh": short_reason,
        }

    def _build_prompt(
        self,
        *,
        proposal: Dict[str, Any],
        reviewer_note: str,
        fallback_change_summary: str,
        fallback_short_reason: str,
    ) -> str:
        engineer_packet = {
            "field_key": str(proposal.get("field_key") or "").strip(),
            "fix_surface": str(proposal.get("fix_surface") or "").strip(),
            "execution_path_recommendation": str(proposal.get("execution_path_recommendation") or "").strip(),
            "root_cause_family": str(proposal.get("root_cause_family") or "").strip(),
            "judgment_summary_zh": str(proposal.get("agent_reasoning_summary") or "").strip(),
            "key_evidence_zh": list(proposal.get("key_evidence_zh") or []),
            "why_this_surface_zh": str(proposal.get("why_this_surface_zh") or "").strip(),
            "why_not_other_surfaces_zh": str(proposal.get("why_not_other_surfaces") or "").strip(),
            "reviewer_note": str(reviewer_note or "").strip(),
            "gt_value": proposal.get("gt_value"),
            "current_output": proposal.get("current_output"),
            "candidate_output": proposal.get("candidate_output"),
            "overlay_experiment_summary": str(proposal.get("result_delta_summary") or "").strip(),
            "patch_intent": dict(proposal.get("patch_intent") or {}),
            "overlay_patch_preview": dict(proposal.get("patch_preview") or {}),
        }
        return f"""你是 MemoryEngineerAgent，是一个非常专业的记忆工程师。

你处理的是一个单用户记忆工程系统。这个系统从图片出发，经过 VLM、事件提取、关系推断，最后在 LP3 生成画像字段。
你当前只负责 LP3 profile fields 的工程落地。

你的职责不是重新反思 badcase，也不是重新判断问题根因。
你的职责是：在人工已经批准反思方向之后，把“批准后的工程改码包”翻译成最合适的正式工程改动表达。

你的输出会直接进入飞书审批，所以必须让人一眼看懂。
你不需要长篇分析，也不要复述底层调试信息。
请用通俗易懂的人话，不要解释术语。
不要直接写 field_cot、tool_rule、call_policy 这类词，直接说“判断口径”“找证据规则”“调用时机”。

你只会收到一份精简工程包，里面包含：
- field_key
- fix_surface
- execution_path_recommendation
- root_cause_family
- judgment_summary_zh
- key_evidence_zh
- why_this_surface_zh
- why_not_other_surfaces_zh
- 人工审批建议
- GT
- 当前输出
- overlay 实验结果摘要
- patch_intent
- overlay patch_preview（如果有）

你的输出要求：
- 只输出 JSON
- 所有解释字段都必须是简体中文
- 必须极简，只保留人工快速审批需要看的关键信息
- 重点输出：
  - 一句改动说明
  - 一句极短理由
- 不要输出大段分析，不要复述 trace，不要输出调试字段

注意：
- execution_path 不能改写，必须沿用输入里的 execution_path_recommendation
- 如果 execution_path_recommendation 是 overlay_direct_apply，就按实验通过的 overlay 来表达正式改动
- 如果 execution_path_recommendation 是 engineer_rewrite_apply，就结合人工建议，把工程改法表达得更具体
- 只能处理“单字段 + 单改面”

输出格式：
{{
  "execution_path": "overlay_direct_apply | engineer_rewrite_apply",
  "change_summary_zh": "一句改动说明，直接说把哪个字段的什么规则改成什么方向。",
  "short_reason_zh": "一句极短理由，直接说为什么要这样改。",
  "target_files": [
    "rule_assets/field_specs.overrides.json"
  ],
  "patch_plan": {{
    "field_key": "字段 key",
    "fix_surface": "field_cot | tool_rule | call_policy | engineering_issue | watch_only",
    "change_intent_zh": "一句话说明具体想改什么"
  }}
}}

如果你觉得表述仍然不够清楚，请至少保证不比下面这组 fallback 更差：
{{
  "change_summary_zh": {json.dumps(fallback_change_summary, ensure_ascii=False)},
  "short_reason_zh": {json.dumps(fallback_short_reason, ensure_ascii=False)}
}}

这是当前的精简工程包：
{json.dumps(engineer_packet, ensure_ascii=False)}"""

    def _build_llm_processor(self) -> Any | None:
        if self.provider != "openrouter":
            return None
        api_key = (REFLECTION_AGENT_OPENROUTER_API_KEY or OPENROUTER_API_KEY or "").strip()
        if not api_key:
            return None
        try:
            return OpenRouterProfileLLMProcessor(
                api_key=api_key,
                base_url=OPENROUTER_BASE_URL,
                model=self.model or PROFILE_LLM_MODEL,
            )
        except Exception:
            return None

    def build_execute_task(self, *, change_request: Dict[str, Any]) -> DecisionReviewItem:
        now = _utcnow_iso()
        return DecisionReviewItem(
            task_id=str(change_request.get("task_id") or ""),
            task_type="engineering_execute_review",
            pattern_id=str(change_request.get("pattern_id") or ""),
            user_name=str(change_request.get("user_name") or ""),
            lane="upstream",
            priority="high",
            summary=str(change_request.get("change_summary_zh") or ""),
            detail_url=str(change_request.get("detail_url") or ""),
            support_case_ids=list(change_request.get("support_case_ids") or []),
            options=list(ENGINEERING_EXECUTE_TASK_OPTIONS),
            recommended_option="approve",
            status="new",
            feishu_status="not_triggered",
            proposal_id=str(change_request.get("proposal_id") or ""),
            experiment_id=str(change_request.get("experiment_id") or ""),
            change_request_id=str(change_request.get("change_request_id") or ""),
            evidence_refs=list(change_request.get("evidence_refs") or []),
            created_at=now,
            updated_at=now,
        )

    def _build_engineer_patch_preview(self, *, proposal: Dict[str, Any], reviewer_note: str) -> Dict[str, Any]:
        existing_patch = dict(proposal.get("patch_preview") or {})
        if existing_patch:
            patched = json.loads(json.dumps(existing_patch, ensure_ascii=False))
        else:
            patched = _build_overlay_bundle(
                fix_surface=str(proposal.get("fix_surface") or ""),
                field_key=str(proposal.get("field_key") or ""),
                patch_intent=dict(proposal.get("patch_intent") or {}),
            )
        field_key = str(proposal.get("field_key") or "").strip()
        fix_surface = str(proposal.get("fix_surface") or "").strip()
        note = str(reviewer_note or "").strip()
        if not note:
            return patched
        if fix_surface == "field_cot":
            overrides = patched.setdefault("field_spec_overrides", {})
            field_override = overrides.setdefault(field_key, {})
            cot_steps = list(field_override.get("cot_steps") or [])
            if note not in cot_steps:
                cot_steps.append(note)
            field_override["cot_steps"] = cot_steps
        elif fix_surface == "tool_rule":
            tool_rules = patched.setdefault("tool_rules", {})
            field_rule = tool_rules.setdefault(field_key, {})
            field_rule["engineer_note"] = note
        elif fix_surface == "call_policy":
            call_policies = patched.setdefault("call_policies", {})
            field_policy = call_policies.setdefault(field_key, {})
            field_policy["engineer_note"] = note
        return patched

    def _build_change_summary(
        self,
        *,
        field_key: str,
        fix_surface: str,
        execution_path: str,
        reviewer_note: str,
        proposal: Dict[str, Any],
    ) -> str:
        candidate_output = str(proposal.get("candidate_output") or "").strip()
        if execution_path == "overlay_direct_apply" and candidate_output:
            return f"把 {field_key} 的 {fix_surface} 直接按实验通过的 overlay 落成正式规则，目标输出改为“{candidate_output}”。"
        if reviewer_note:
            return f"把 {field_key} 的 {fix_surface} 按人工建议改成：{reviewer_note}"
        return f"把 {field_key} 的 {fix_surface} 调整到更贴近 GT 的实现。"

    def _build_short_reason(self, *, proposal: Dict[str, Any]) -> str:
        gt_value = proposal.get("gt_value")
        current_output = proposal.get("current_output")
        candidate_output = proposal.get("candidate_output")
        if gt_value not in (None, "") and current_output not in (None, "") and candidate_output not in (None, ""):
            return f"因为 GT 要的是“{gt_value}”，当前输出是“{current_output}”，实验后已能逼近“{candidate_output}”。"
        if gt_value not in (None, "") and current_output not in (None, ""):
            return f"因为 GT 要的是“{gt_value}”，当前输出是“{current_output}”，需要把语义方向拉回 GT。"
        return "因为当前输出与 GT 仍有偏差，需要按批准的方向落成正式工程修改。"


class MutationExecutor:
    def __init__(self, *, project_root: str = PROJECT_ROOT, validator: Any | None = None) -> None:
        self.project_root = project_root
        self.validator = validator or self._default_validator

    def execute(self, *, proposal: Dict[str, Any]) -> Dict[str, Any]:
        if str(proposal.get("status") or "").strip() != "approved":
            return {
                "status": "blocked_unapproved",
                "proposal_id": str(proposal.get("proposal_id") or ""),
            }

        patch_preview = dict(proposal.get("patch_preview") or {})
        asset_paths = apply_repo_rule_patch(project_root=self.project_root, patch_preview=patch_preview)
        validation_summary = self.validator(proposal=proposal)
        outcome = self._persist_outcome(proposal=proposal, validation_summary=validation_summary)
        return {
            "status": "applied",
            "asset_paths": asset_paths,
            "validation_summary": validation_summary,
            "outcome": outcome,
        }

    def execute_change_request(self, *, change_request: Dict[str, Any]) -> Dict[str, Any]:
        if str(change_request.get("status") or "").strip() != "approved":
            return {
                "status": "blocked_unapproved",
                "change_request_id": str(change_request.get("change_request_id") or ""),
            }

        patch_preview = dict(change_request.get("patch_preview") or {})
        asset_paths = apply_repo_rule_patch(project_root=self.project_root, patch_preview=patch_preview)
        validation_summary = self.validator(proposal=change_request)
        outcome = self._persist_outcome(
            proposal={
                "proposal_id": str(change_request.get("proposal_id") or change_request.get("change_request_id") or ""),
                "experiment_id": str(change_request.get("experiment_id") or ""),
                "user_name": str(change_request.get("user_name") or ""),
            },
            validation_summary=validation_summary,
        )
        return {
            "status": "applied",
            "asset_paths": asset_paths,
            "validation_summary": validation_summary,
            "outcome": outcome,
        }

    def _persist_outcome(self, *, proposal: Dict[str, Any], validation_summary: Dict[str, Any]) -> Dict[str, Any]:
        user_name = str(proposal.get("user_name") or "").strip()
        paths = build_reflection_asset_paths(project_root=self.project_root, user_name=user_name)
        ensure_reflection_root(paths)
        existing = _read_json_array(paths.upstream_outcomes_path)
        outcome = ExperimentOutcome(
            outcome_id=_stable_id("outcome", str(proposal.get("proposal_id") or "")),
            experiment_id=str(proposal.get("experiment_id") or ""),
            user_name=user_name,
            status=str(validation_summary.get("status") or "success"),
            summary=str(validation_summary.get("summary") or "proposal applied"),
            metrics=dict(validation_summary.get("metrics") or {}),
        ).to_dict()
        merged = {str(item.get("outcome_id") or ""): item for item in existing if str(item.get("outcome_id") or "")}
        merged[outcome["outcome_id"]] = outcome
        Path(paths.upstream_outcomes_path).write_text(
            json.dumps(list(merged.values()), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return outcome

    def _default_validator(self, *, proposal: Dict[str, Any]) -> Dict[str, Any]:
        success = compileall.compile_dir(
            str(Path(self.project_root) / "services"),
            quiet=1,
            maxlevels=10,
        )
        compileall.compile_dir(
            str(Path(self.project_root) / "backend"),
            quiet=1,
            maxlevels=10,
        )
        return {
            "status": "success" if success else "failed",
            "summary": "repo-tracked 规则资产已写入，并完成最小 compile 验证",
            "metrics": {
                "compile_success": bool(success),
            },
        }


def _build_overlay_bundle(*, fix_surface: str, field_key: str, patch_intent: Dict[str, Any]) -> Dict[str, Any]:
    if fix_surface == "field_cot":
        # 优先使用 agent 输出的 field_spec_overrides（可能包含任意槽位）
        agent_overrides = dict(patch_intent.get("field_spec_overrides") or {})
        if not agent_overrides:
            # 兼容旧格式：agent 直接输出 cot_steps 等顶层字段
            from services.memory_pipeline.types import FieldSpec
            allowed = set(FieldSpec.__dataclass_fields__.keys())
            agent_overrides = {field_key: {
                k: v for k, v in patch_intent.items()
                if k in allowed and v and k not in ("field_key", "fix_surface", "change_summary_zh")
            }}
        return {
            "field_spec_overrides": agent_overrides,
            "tool_rules": {},
            "call_policies": {},
        }
    if fix_surface == "tool_rule":
        return {
            "field_spec_overrides": {},
            "tool_rules": dict(patch_intent.get("tool_rules") or {field_key: dict(patch_intent.get("tool_rule_patch") or {})}),
            "call_policies": {},
        }
    if fix_surface == "call_policy":
        return {
            "field_spec_overrides": {},
            "tool_rules": {},
            "call_policies": dict(patch_intent.get("call_policies") or {field_key: dict(patch_intent.get("call_policy_patch") or {})}),
        }
    return {
        "field_spec_overrides": {},
        "tool_rules": {},
        "call_policies": {},
    }


def _build_decision_tree_path(case_fact: Dict[str, Any], tail: Iterable[str] | None = None) -> List[str]:
    path = [
        str(case_fact.get("accuracy_gap_status") or "unknown_gap"),
        str(case_fact.get("comparison_grade") or "unknown_grade"),
        str(case_fact.get("root_cause_family") or case_fact.get("triage_reason") or "unknown_root_cause"),
        str(ROOT_CAUSE_TO_FIX_SURFACE.get(str(case_fact.get("root_cause_family") or ""), "watch_only")),
    ]
    if tail:
        path.extend(str(item).strip() for item in tail if str(item).strip())
    return path


def _load_json_object(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_json_array(path: str) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []


def _read_jsonl_records(path: str) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _stable_id(prefix: str, raw_key: str) -> str:
    digest = hashlib.sha1(raw_key.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _utcnow_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _determine_execution_path_recommendation(experiment_report: Dict[str, Any]) -> str:
    current_case_result = dict(experiment_report.get("current_case_result") or {})
    candidate_grade = str(
        current_case_result.get("candidate_comparison_grade")
        or current_case_result.get("comparison_grade")
        or dict(experiment_report.get("candidate_metrics") or {}).get("comparison_grade")
        or ""
    ).strip()
    if candidate_grade in {"exact_match", "close_match"}:
        return "overlay_direct_apply"
    return "engineer_rewrite_apply"


def _build_result_delta_summary(
    *,
    gt_value: Any,
    current_output: Any,
    candidate_output: Any,
    execution_path_recommendation: str,
) -> str:
    gt_text = _truncate_text(gt_value, limit=100)
    current_text = _truncate_text(current_output, limit=100)
    candidate_text = _truncate_text(candidate_output, limit=100)
    if gt_text and current_text and candidate_text:
        if execution_path_recommendation == "overlay_direct_apply":
            return f"GT 是“{gt_text}”，当前输出是“{current_text}”，实验后输出已收敛到“{candidate_text}”。"
        return f"GT 是“{gt_text}”，当前输出是“{current_text}”，实验后候选输出是“{candidate_text}”，但还需要工程师再收口。"
    if gt_text and current_text:
        return f"GT 是“{gt_text}”，当前输出是“{current_text}”，需要继续把结果拉回 GT。"
    return "当前 badcase 已完成实验，但还需要结合 GT 与工程实现继续收口。"


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_compact_prompt_packet(packet: Dict[str, Any]) -> Dict[str, Any]:
    case_fact = dict(packet.get("case_fact") or {})
    return {
        "case_fact": {
            "case_id": str(case_fact.get("case_id") or ""),
            "album_id": str(case_fact.get("album_id") or ""),
            "dimension": str(case_fact.get("dimension") or ""),
            "entity_type": str(case_fact.get("entity_type") or ""),
            "comparison_grade": str(case_fact.get("comparison_grade") or ""),
            "comparison_score": _safe_float(case_fact.get("comparison_score"), 0.0),
            "badcase_source": str(case_fact.get("badcase_source") or ""),
            "triage_reason": str(case_fact.get("triage_reason") or ""),
            "root_cause_family": str(case_fact.get("root_cause_family") or ""),
            "fix_surface_confidence": _safe_float(case_fact.get("fix_surface_confidence"), 0.0),
            "auto_confidence": _safe_float(case_fact.get("auto_confidence"), 0.0),
        },
        "comparison_result": _compact_comparison_result(packet.get("comparison_result")),
        "pre_audit_comparison_result": _compact_comparison_result(packet.get("pre_audit_comparison_result")),
        "final_before_backflow": _compact_final_payload(packet.get("final_before_backflow")),
        "final_after_backflow": _compact_final_payload(packet.get("final_after_backflow")),
        "null_reason": str(packet.get("null_reason") or ""),
        "tool_trace_summary": _compact_tool_trace(packet.get("tool_trace")),
        "llm_batch_debug_summary": _compact_llm_batch_debug(packet.get("llm_batch_debug")),
        "history_summary": {
            "history_pattern_count": len(list(packet.get("history_patterns") or [])),
            "history_experiment_count": len(list(packet.get("history_experiments") or [])),
            "history_outcome_count": len(list(packet.get("history_outcomes") or [])),
            "history_patterns": _compact_history_items(packet.get("history_patterns")),
            "history_experiments": _compact_history_items(packet.get("history_experiments")),
            "history_outcomes": _compact_history_items(packet.get("history_outcomes")),
        },
        # 人工审批备注（临时注入，本轮用完即弃）
        **({"human_reviewer_note": packet["human_reviewer_note"]} if packet.get("human_reviewer_note") else {}),
        # 自循环上下文（上一轮结果、趋势、改进效果）
        **({"evolution_context": dict(packet["evolution_context"])} if packet.get("evolution_context") else {}),
    }


def _compact_comparison_result(payload: Any) -> Dict[str, Any]:
    result = dict(payload or {})
    gt_val = result.get("gt_value")
    return {
        "grade": str(result.get("grade") or ""),
        "score": _safe_float(result.get("score"), 0.0),
        "severity": str(result.get("severity") or ""),
        "output_value": _truncate_text(result.get("output_value"), limit=120),
        # gt_value 遮蔽：只告诉 LLM GT 是否有值，不暴露具体内容
        "gt_has_value": gt_val not in (None, "", []),
        "method": str(result.get("method") or ""),
    }


def _compact_final_payload(payload: Any) -> Dict[str, Any]:
    final_payload = dict(payload or {})
    evidence = dict(final_payload.get("evidence") or {})
    supporting_refs = list(evidence.get("supporting_refs") or [])
    contradicting_refs = list(evidence.get("contradicting_refs") or [])
    return {
        "value": _truncate_text(final_payload.get("value"), limit=160),
        "confidence": _safe_float(final_payload.get("confidence"), 0.0),
        "reasoning_excerpt": _truncate_text(final_payload.get("reasoning"), limit=280),
        "supporting_ref_count": len(supporting_refs),
        "contradicting_ref_count": len(contradicting_refs),
        "supporting_refs_sample": _compact_evidence_refs(supporting_refs),
        "contradicting_refs_sample": _compact_evidence_refs(contradicting_refs),
        "summary": _truncate_text(evidence.get("summary"), limit=180),
    }


def _compact_tool_trace(payload: Any) -> Dict[str, Any]:
    tool_trace = dict(payload or {})
    evidence_bundle = dict(tool_trace.get("evidence_bundle") or {})
    supporting_refs = evidence_bundle.get("supporting_refs") or {}
    total_supporting_refs = 0
    sample_refs: List[Dict[str, Any]] = []
    if isinstance(supporting_refs, dict):
        for refs in supporting_refs.values():
            refs_list = list(refs or [])
            total_supporting_refs += len(refs_list)
            if len(sample_refs) < 3:
                sample_refs.extend(refs_list[: max(0, 3 - len(sample_refs))])
    elif isinstance(supporting_refs, list):
        total_supporting_refs = len(supporting_refs)
        sample_refs = list(supporting_refs[:3])
    return {
        "has_tool_trace": bool(tool_trace),
        "supporting_ref_count": total_supporting_refs,
        "supporting_refs_sample": _compact_evidence_refs(sample_refs),
        "stats_bundle_keys": sorted(dict(tool_trace.get("stats_bundle") or {}).keys())[:8],
        "ownership_bundle_keys": sorted(dict(tool_trace.get("ownership_bundle") or {}).keys())[:8],
        "counter_bundle_keys": sorted(dict(tool_trace.get("counter_bundle") or {}).keys())[:8],
    }


def _count_tool_supporting_refs(tool_trace: Dict[str, Any]) -> int:
    evidence_bundle = dict(tool_trace.get("evidence_bundle") or {})
    supporting_refs = evidence_bundle.get("supporting_refs") or {}
    if isinstance(supporting_refs, dict):
        return sum(len(list(refs or [])) for refs in supporting_refs.values())
    if isinstance(supporting_refs, list):
        return len(supporting_refs)
    return 0


def _extract_tool_supporting_refs(tool_trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    evidence_bundle = dict(tool_trace.get("evidence_bundle") or {})
    supporting_refs = evidence_bundle.get("supporting_refs") or {}
    return _flatten_tool_refs(supporting_refs)


def _extract_tool_contradicting_refs(tool_trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    evidence_bundle = dict(tool_trace.get("evidence_bundle") or {})
    contradicting_refs = evidence_bundle.get("contradicting_refs") or {}
    return _flatten_tool_refs(contradicting_refs)


def _flatten_tool_refs(payload: Any) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        for refs in payload.values():
            for item in list(refs or []):
                if isinstance(item, dict):
                    flattened.append(item)
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                flattened.append(item)
    return flattened


def _compact_llm_batch_debug(payload: Any) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    raw_items = _normalize_llm_batch_debug(payload)
    for raw_item in raw_items[:3]:
        item = dict(raw_item or {})
        records.append(
            {
                "batch_name": str(item.get("batch_name") or ""),
                "status": str(item.get("status") or ""),
                "raw_response_preview": _truncate_text(item.get("raw_response_preview"), limit=500),
                "error": _truncate_text(item.get("error"), limit=240),
            }
        )
    return records


def _compact_history_items(items: Any) -> List[Dict[str, Any]]:
    compacted: List[Dict[str, Any]] = []
    for raw_item in list(items or [])[:5]:
        item = dict(raw_item or {})
        compacted.append(
            {
                "id": str(item.get("pattern_id") or item.get("experiment_id") or item.get("outcome_id") or ""),
                "status": str(item.get("status") or ""),
                "summary": _truncate_text(item.get("summary"), limit=180),
                "recommended_option": str(item.get("recommended_option") or item.get("fix_surface") or ""),
            }
        )
    return compacted


def _compact_evidence_refs(refs: Any) -> List[Dict[str, Any]]:
    compacted: List[Dict[str, Any]] = []
    for raw_ref in list(refs or [])[:3]:
        ref = dict(raw_ref or {})
        compacted.append(
            {
                "source_type": str(ref.get("source_type") or ""),
                "source_id": str(ref.get("source_id") or ref.get("event_id") or ""),
                "description": _truncate_text(ref.get("description"), limit=120),
            }
        )
    return compacted


def _truncate_text(value: Any, *, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _normalize_llm_batch_debug(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    return []


def _normalize_patch_intent(payload: Any, *, default: Dict[str, Any], field_key: str, fix_surface: str) -> Dict[str, Any]:
    if isinstance(payload, dict):
        normalized = dict(payload)
        normalized.setdefault("field_key", field_key)
        normalized.setdefault("fix_surface", fix_surface)
        if not str(normalized.get("change_summary_zh") or "").strip():
            legacy_summary = str(normalized.get("summary") or "").strip()
            if legacy_summary:
                normalized["change_summary_zh"] = legacy_summary
        # 清理 weak_evidence_caution — 不再作为输出选项
        rule_patch = normalized.get("rule_patch")
        if isinstance(rule_patch, dict):
            fso = rule_patch.get("field_spec_overrides")
            if isinstance(fso, dict):
                for _fk, spec in fso.items():
                    if isinstance(spec, dict):
                        spec.pop("weak_evidence_caution", None)
                        spec.pop("null_preferred_when", None)
        return normalized
    if isinstance(payload, str):
        text = payload.strip()
        if text:
            return {
                "field_key": field_key,
                "fix_surface": fix_surface,
                "change_summary_zh": text,
            }
    return dict(default)


def _normalize_expected_metric_gain(payload: Any, *, default: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, str):
        text = payload.strip()
        if text:
            return {
                **dict(default),
                "summary": text,
            }
    return dict(default)


class CoverageProbe:
    """
    Purely rule-based structural check: does this field's tool setup have gaps?
    Runs before UpstreamReflectionAgent. Zero LLM calls.

    Output coverage_gap dict injected into CaseFact.tool_usage_summary["coverage_gap"].
    """

    def probe(
        self,
        *,
        field_key: str,
        tool_trace: Dict[str, Any],
        allowed_sources: List[str],
        call_policies: Dict[str, Any],
        tool_rules: Dict[str, Any],
        group_tool_traces: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        evidence_bundle = dict(tool_trace.get("evidence_bundle") or {})
        tool_called = bool(tool_trace)

        gap_type = "none"
        affected_sources: List[str] = []
        detail = ""

        # Rule 1: tool_rule_blocked — max_refs_per_source == 0 or allowed_sources empty
        field_tool_rule = dict((tool_rules or {}).get(field_key) or {})
        if not allowed_sources:
            gap_type = "tool_rule_blocked"
            detail = "allowed_sources is empty"
        elif field_tool_rule.get("max_refs_per_source") is not None and int(field_tool_rule["max_refs_per_source"]) == 0:
            gap_type = "tool_rule_blocked"
            detail = f"tool_rules[{field_key}].max_refs_per_source == 0"

        if gap_type == "none" and tool_called:
            field_call_policy = dict((call_policies or {}).get(field_key) or {})
            appended_sources = list(field_call_policy.get("append_allowed_sources") or [])
            effective_sources = list(set(allowed_sources + appended_sources))

            for source in effective_sources:
                source_bundle = evidence_bundle.get(source)
                hit_count = 0
                if isinstance(source_bundle, list):
                    hit_count = len(source_bundle)
                elif isinstance(source_bundle, dict):
                    hit_count = int(source_bundle.get("hit_count") or 0)

                if hit_count > 0:
                    continue

                # Rule 2: source_unconfigured — source in allowed_sources but no call_policy entry for this field
                if source not in appended_sources and field_key not in (call_policies or {}):
                    gap_type = "source_unconfigured"
                    affected_sources.append(source)
                    detail = f"source '{source}' in allowed_sources but call_policy not configured for {field_key}"
                    break

                # Rule 3: tool_called_no_hit — tool was called but returned 0 hits for this source
                if source in evidence_bundle:
                    gap_type = "tool_called_no_hit"
                    affected_sources.append(source)
                    detail = f"source '{source}' called but hit_count == 0"

        # Rule 4: index_path_suspect — only this field has 0 refs while same-group fields have refs
        if gap_type == "none" and group_tool_traces:
            this_total = sum(
                len(v) if isinstance(v, list) else int((v or {}).get("hit_count") or 0)
                for v in evidence_bundle.values()
            )
            if this_total == 0:
                group_totals = []
                for gt in group_tool_traces:
                    gb = dict((gt or {}).get("evidence_bundle") or {})
                    group_totals.append(sum(
                        len(v) if isinstance(v, list) else int((v or {}).get("hit_count") or 0)
                        for v in gb.values()
                    ))
                if any(c > 0 for c in group_totals):
                    gap_type = "index_path_suspect"
                    detail = f"field {field_key} has 0 refs while same-group fields have evidence"

        return {
            "has_gap": gap_type != "none",
            "gap_type": gap_type,
            "affected_sources": affected_sources,
            "detail": detail,
        }
