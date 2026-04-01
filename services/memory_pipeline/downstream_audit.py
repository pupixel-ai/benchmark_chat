from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
from datetime import datetime
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Tuple

from models import Relationship
from dataclasses import replace

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, PROFILE_AGENT_ROOT
from .evidence_utils import extract_ids_from_refs
from .relationships import (
    RELATIONSHIP_TYPE_SPECS,
    _determine_group_eligibility,
    _determine_retention,
)
from .profile_agent_adapter import (
    PROFILE_DIMENSION_FIELD_PATHS,
    RELATIONSHIP_DIMENSION_MAP,
    build_profile_agent_extractor_outputs,
)
from .types import RelationshipDossier

DEFAULT_PROFILE_AGENT_MODEL_CANDIDATES = tuple(
    m.strip() for m in os.environ.get(
        "PROFILE_LLM_MODEL_CANDIDATES",
        "deepseek/deepseek-chat-v3-0324,qwen/qwen-2.5-72b-instruct",
    ).split(",") if m.strip()
)


class MainOutputExtractor:
    """主工程同构审计编排层：V1 适配输出 + challenge-targeted V2 补证。"""

    def __init__(
        self,
        *,
        primary_decision: Dict[str, Any] | None,
        relationships: Iterable[Relationship | Dict[str, Any]],
        structured_profile: Dict[str, Any] | None,
        profile_fact_decisions: Iterable[Dict[str, Any]] | None = None,
    ) -> None:
        self._primary_decision = primary_decision or {}
        self._relationships = [_coerce_relationship_dict(item) for item in relationships]
        self._relationship_by_person = {
            str(item.get("person_id") or ""): item
            for item in self._relationships
            if str(item.get("person_id") or "")
        }
        self._structured_profile = structured_profile or {}
        self._profile_fact_decisions_by_field = {
            str(item.get("field_key") or ""): item
            for item in list(profile_fact_decisions or [])
            if isinstance(item, dict) and str(item.get("field_key") or "")
        }

    def build_v1_outputs(self) -> Dict[str, Dict[str, Any]]:
        return build_profile_agent_extractor_outputs(
            primary_decision=self._primary_decision,
            relationships=self._relationships,
            structured_profile=self._structured_profile,
        )

    def build_v2_output(
        self,
        *,
        agent_type: str,
        extractor_v1_output: Dict[str, Any],
        critic_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        challenges = list((critic_output or {}).get("challenges") or [])
        if not challenges:
            cloned = deepcopy(extractor_v1_output)
            cloned["v2_targeted_dimensions"] = []
            return cloned

        challenge_dimensions = _collect_challenge_dimensions(challenges)
        if not challenge_dimensions:
            cloned = deepcopy(extractor_v1_output)
            cloned["v2_targeted_dimensions"] = []
            return cloned

        refreshed_tags: List[Dict[str, Any]] = []
        targeted_dimensions: List[str] = []
        for tag in list(extractor_v1_output.get("tags") or []):
            refreshed_tag = deepcopy(tag)
            dimension = str(tag.get("dimension") or "")
            if not dimension or dimension not in challenge_dimensions:
                refreshed_tags.append(refreshed_tag)
                continue

            targeted_dimensions.append(dimension)
            matched_challenges = [
                challenge
                for challenge in challenges
                if _challenge_matches_dimension(challenge, dimension)
            ]
            supplemental_evidence = self._build_supplemental_evidence(
                agent_type=agent_type,
                tag=refreshed_tag,
                challenges=matched_challenges,
            )
            refreshed_tag["evidence"] = _merge_evidence_lists(
                list(refreshed_tag.get("evidence") or []),
                supplemental_evidence,
            )
            refreshed_tag["extraction_gap"] = _append_extraction_gap(
                current_gap=refreshed_tag.get("extraction_gap"),
                challenges=matched_challenges,
                supplemental_count=len(supplemental_evidence),
            )
            refreshed_tag["v2_targeted"] = True
            refreshed_tags.append(refreshed_tag)

        cloned = deepcopy(extractor_v1_output)
        cloned["tags"] = refreshed_tags
        cloned["v2_targeted_dimensions"] = sorted(dict.fromkeys(targeted_dimensions))
        challenge_note = ", ".join(cloned["v2_targeted_dimensions"]) or "none"
        existing_reasoning = str(cloned.get("reasoning_trace") or "").strip()
        v2_note = f"[V2补证] targeted_dimensions={challenge_note}"
        cloned["reasoning_trace"] = f"{existing_reasoning}\n{v2_note}".strip()
        return cloned

    def _build_supplemental_evidence(
        self,
        *,
        agent_type: str,
        tag: Dict[str, Any],
        challenges: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if agent_type == "protagonist":
            payload = dict(self._primary_decision.get("evidence") or {})
            supplemental = _build_evidence_from_main_payload(
                payload,
                fallback_description="主角判定补证",
            )
            primary_person_id = str(self._primary_decision.get("primary_person_id") or "")
            selfie_count = int(payload.get("selfie_count", 0) or 0)
            anchor_count = int(payload.get("identity_anchor_count", 0) or 0)
            photo_count = int(payload.get("photo_count", 0) or 0)
            label_count = int(payload.get("protagonist_label_count", 0) or 0)
            stats_parts = []
            if selfie_count:
                stats_parts.append(f"自拍 {selfie_count} 次")
            if anchor_count:
                stats_parts.append(f"身份锚点 {anchor_count} 个")
            if photo_count:
                stats_parts.append(f"出现在 {photo_count} 张照片中")
            if label_count:
                stats_parts.append(f"VLM 主角标记 {label_count} 次")
            if primary_person_id:
                stats_desc = f"主角识别统计: {', '.join(stats_parts)}" if stats_parts else f"主角候选 {primary_person_id}"
                supplemental.append(
                    _normalize_evidence_item(
                        {
                            "person_id": primary_person_id,
                            "description": stats_desc,
                            "evidence_type": "direct",
                            "inference_depth": 1,
                            "feature_names": ["selfie_count", "identity_anchor_count", "photo_count", "protagonist_label_count"],
                        }
                    )
                )
            rel_cross_parts: List[str] = []
            for rel in self._relationships:
                r = _coerce_relationship_dict(rel)
                rtype = r.get("relationship_type")
                if rtype in {"romantic", "family", "bestie", "close_friend"}:
                    rel_cross_parts.append(f"{r.get('person_id', '?')}({rtype})")
            if rel_cross_parts:
                supplemental.append(
                    _normalize_evidence_item(
                        {
                            "description": f"[跨层参考] 关系层核心关系: {', '.join(rel_cross_parts[:5])}",
                            "evidence_type": "cross_layer_reference",
                            "inference_depth": 2,
                            "feature_names": ["relationship_cross_reference"],
                        }
                    )
                )
            return _merge_evidence_lists([], supplemental)

        if agent_type == "relationship":
            person_id = str(tag.get("value") or "")
            relationship = self._relationship_by_person.get(person_id, {})
            payload = dict(relationship.get("evidence") or {})
            shared_events = [
                event.get("event_id")
                for event in list(relationship.get("shared_events") or [])
                if isinstance(event, dict) and event.get("event_id")
            ]
            if shared_events:
                merged_event_ids = list(
                    dict.fromkeys(list(payload.get("event_ids") or []) + shared_events)
                )
                payload["event_ids"] = merged_event_ids
            supplemental = _build_evidence_from_main_payload(
                payload,
                fallback_description=f"关系判定补证:{person_id}",
            )
            if person_id:
                supplemental.append(
                    _normalize_evidence_item(
                        {
                            "person_id": person_id,
                            "description": f"关系候选人物 {person_id} 定向补证",
                            "evidence_type": "inferred",
                            "inference_depth": 2,
                        }
                    )
                )
            profile_cross_parts: List[str] = []
            for fk in ("long_term_facts.identity.role", "long_term_facts.social_identity.education"):
                obj = self._structured_profile
                for seg in fk.split("."):
                    obj = obj.get(seg) if isinstance(obj, dict) else None
                if isinstance(obj, dict) and obj.get("value") is not None:
                    profile_cross_parts.append(f"{fk.split('.')[-1]}={obj['value']}")
            if profile_cross_parts:
                supplemental.append(
                    _normalize_evidence_item(
                        {
                            "description": f"[跨层参考] 画像层: {', '.join(profile_cross_parts)}",
                            "evidence_type": "cross_layer_reference",
                            "inference_depth": 2,
                            "feature_names": ["profile_cross_reference"],
                        }
                    )
                )
            return _merge_evidence_lists([], supplemental)

        if agent_type == "profile":
            dimension = str(tag.get("dimension") or "")
            field_key = PROFILE_DIMENSION_FIELD_PATHS.get(dimension)
            fallback_description = f"画像字段补证:{field_key or dimension}"
            supplemental: List[Dict[str, Any]] = []
            if field_key:
                field_object = _get_nested(self._structured_profile, field_key)
                if isinstance(field_object, dict):
                    supplemental = _merge_evidence_lists(
                        supplemental,
                        _build_evidence_from_main_payload(
                            dict(field_object.get("evidence") or {}),
                            fallback_description=fallback_description,
                        ),
                    )
                field_decision = self._profile_fact_decisions_by_field.get(field_key, {})
                if isinstance(field_decision, dict):
                    final_payload = field_decision.get("final")
                    if isinstance(final_payload, dict):
                        supplemental = _merge_evidence_lists(
                            supplemental,
                            _build_evidence_from_main_payload(
                                dict(final_payload.get("evidence") or {}),
                                fallback_description=fallback_description,
                            ),
                        )
                    final_before_backflow_payload = field_decision.get("final_before_backflow")
                    if isinstance(final_before_backflow_payload, dict):
                        supplemental = _merge_evidence_lists(
                            supplemental,
                            _build_evidence_from_main_payload(
                                dict(final_before_backflow_payload.get("evidence") or {}),
                                fallback_description=fallback_description,
                            ),
                        )
            challenge_requests = _collect_challenge_requests(challenges)
            if challenge_requests:
                supplemental.append(
                    _normalize_evidence_item(
                        {
                            "description": f"V2针对质疑请求: {'；'.join(challenge_requests)}",
                            "feature_names": ["challenge_targeted_refinement"],
                            "evidence_type": "inferred",
                            "inference_depth": 2,
                        }
                    )
                )
            rel_cross_parts_profile: List[str] = []
            for rel in self._relationships:
                r = _coerce_relationship_dict(rel)
                rtype = r.get("relationship_type")
                if rtype in {"romantic", "family", "bestie", "close_friend"}:
                    pid = r.get("person_id", "?")
                    conf = r.get("confidence", 0)
                    rel_cross_parts_profile.append(f"{pid}({rtype},conf={conf})")
            if rel_cross_parts_profile:
                supplemental.append(
                    _normalize_evidence_item(
                        {
                            "description": f"[跨层参考] 关系层: {', '.join(rel_cross_parts_profile[:5])}",
                            "evidence_type": "cross_layer_reference",
                            "inference_depth": 2,
                            "feature_names": ["relationship_cross_reference"],
                        }
                    )
                )
            primary_mode = str(self._primary_decision.get("mode") or "")
            primary_pid = str(self._primary_decision.get("primary_person_id") or "")
            if primary_pid:
                supplemental.append(
                    _normalize_evidence_item(
                        {
                            "description": f"[跨层参考] 主角: {primary_pid} (mode={primary_mode})",
                            "evidence_type": "cross_layer_reference",
                            "inference_depth": 2,
                            "feature_names": ["protagonist_cross_reference"],
                        }
                    )
                )
            return _merge_evidence_lists([], supplemental)

        return []


def inspect_profile_agent_runtime_health() -> Dict[str, Any]:
    root = Path(PROFILE_AGENT_ROOT)
    checks: List[Dict[str, Any]] = []
    required_paths = [
        root / "__init__.py",
        root / "storage.py",
        root / "config.py",
        root / "agents" / "critic.py",
        root / "agents" / "judge.py",
        root / "agents" / "rule_evolver.py",
        root / "feishu" / "notify.py",
    ]
    missing: List[str] = []
    for path in required_paths:
        exists = path.exists()
        checks.append({"path": str(path), "exists": exists})
        if not exists:
            missing.append(str(path))
    status = "ok" if not missing else "unhealthy"
    return {
        "status": status,
        "error_code": "ok" if status == "ok" else "missing_required_files",
        "profile_agent_root": str(root),
        "checked_at": datetime.now().isoformat(),
        "checks": checks,
        "missing_paths": missing,
    }


def run_downstream_profile_agent_audit(
    *,
    primary_decision: Dict[str, Any] | None,
    relationships: Iterable[Relationship | Dict[str, Any]],
    structured_profile: Dict[str, Any] | None,
    profile_fact_decisions: Iterable[Dict[str, Any]] | None = None,
    album_id: str = "memory_pipeline_downstream_audit",
    runtime_health: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    runtime_health = runtime_health or inspect_profile_agent_runtime_health()
    if runtime_health.get("status") != "ok":
        error_code = str(runtime_health.get("error_code") or "runtime_unhealthy")
        raise RuntimeError(f"downstream_runtime_unhealthy:{error_code}")

    relationship_list = list(relationships)
    main_output_extractor = MainOutputExtractor(
        primary_decision=primary_decision,
        relationships=relationship_list,
        structured_profile=structured_profile or {},
        profile_fact_decisions=profile_fact_decisions or [],
    )
    adapter_outputs = main_output_extractor.build_v1_outputs()
    audit_outputs, merged_outputs, storage_saved = _run_profile_agent_judges(
        adapter_outputs,
        main_output_extractor=main_output_extractor,
        album_id=album_id,
    )
    return build_downstream_audit_report(
        album_id=album_id,
        primary_decision=primary_decision or {},
        relationships=relationship_list,
        structured_profile=structured_profile or {},
        adapter_outputs=adapter_outputs,
        audit_outputs=audit_outputs,
        merged_outputs=merged_outputs,
        storage_saved=storage_saved,
        runtime_health=runtime_health,
    )


def _detect_cross_domain_issues(
    *,
    primary_decision: Dict[str, Any] | None,
    relationships: List[Dict[str, Any]],
    structured_profile: Dict[str, Any] | None,
) -> List[Dict[str, str]]:
    """纯代码跨域一致性检查（遵循'代码做重活'原则）。"""
    issues: List[Dict[str, str]] = []
    sp = structured_profile or {}

    def _nested(d: Any, dotted: str) -> Any:
        cur = d
        for seg in dotted.split("."):
            cur = cur.get(seg) if isinstance(cur, dict) else None
        return cur

    rel_types = {
        _coerce_relationship_dict(r).get("relationship_type")
        for r in relationships
    }
    rel_by_type: Dict[str, List[str]] = {}
    for r in relationships:
        rd = _coerce_relationship_dict(r)
        rtype = rd.get("relationship_type")
        if rtype:
            rel_by_type.setdefault(rtype, []).append(str(rd.get("person_id", "")))

    education_obj = _nested(sp, "long_term_facts.social_identity.education")
    education_val = education_obj.get("value") if isinstance(education_obj, dict) else None
    role_obj = _nested(sp, "long_term_facts.identity.role")
    role_val = role_obj.get("value") if isinstance(role_obj, dict) else None
    partner_obj = _nested(sp, "long_term_facts.relationships.intimate_partner")
    partner_val = partner_obj.get("value") if isinstance(partner_obj, dict) else None

    if education_val and "classmate" not in rel_types and "classmate_colleague" not in rel_types:
        if any(kw in str(education_val).lower() for kw in ("college", "university", "大学", "高中", "school")):
            issues.append({
                "type": "education_without_classmate",
                "detail": f"画像 education={education_val}，但关系层无 classmate/classmate_colleague 类型关系",
            })

    if role_val and str(role_val).lower() in ("employee", "freelancer", "worker"):
        school_rel_count = len(rel_by_type.get("classmate", []) + rel_by_type.get("classmate_colleague", []))
        work_rel_count = 0
        if school_rel_count > 0 and work_rel_count == 0:
            issues.append({
                "type": "role_relationship_mismatch",
                "detail": f"画像 role={role_val}，但关系层只有 {school_rel_count} 段 classmate 关系，无工作关系",
            })

    if partner_val:
        romantic_pids = rel_by_type.get("romantic", [])
        if partner_val not in romantic_pids:
            issues.append({
                "type": "partner_relationship_conflict",
                "detail": f"画像 intimate_partner={partner_val}，但关系层无该 person_id 的 romantic 关系",
            })

    if "romantic" in rel_types:
        living_obj = _nested(sp, "long_term_facts.relationships.living_situation")
        living_val = living_obj.get("value") if isinstance(living_obj, dict) else None
        if living_val and "alone" in str(living_val).lower():
            issues.append({
                "type": "living_romantic_conflict",
                "detail": f"画像 living_situation={living_val}，但关系层存在 romantic 关系",
            })

    return issues


def build_downstream_audit_report(
    *,
    album_id: str,
    primary_decision: Dict[str, Any],
    relationships: Iterable[Relationship | Dict[str, Any]],
    structured_profile: Dict[str, Any],
    adapter_outputs: Dict[str, Dict[str, Any]],
    audit_outputs: Dict[str, Dict[str, Any]],
    merged_outputs: Dict[str, Dict[str, Any]],
    storage_saved: bool,
    runtime_health: Dict[str, Any],
) -> Dict[str, Any]:
    relationship_list = [_coerce_relationship_dict(item) for item in relationships]
    protagonist_v1 = (audit_outputs.get("protagonist") or {}).get("extractor_v1_output") or adapter_outputs.get("protagonist", {})
    relationship_v1 = (audit_outputs.get("relationship") or {}).get("extractor_v1_output") or adapter_outputs.get("relationship", {})
    profile_v1 = (audit_outputs.get("profile") or {}).get("extractor_v1_output") or adapter_outputs.get("profile", {})
    protagonist_v2 = (audit_outputs.get("protagonist") or {}).get("extractor_v2_output") or protagonist_v1
    relationship_v2 = (audit_outputs.get("relationship") or {}).get("extractor_v2_output") or relationship_v1
    profile_v2 = (audit_outputs.get("profile") or {}).get("extractor_v2_output") or profile_v1

    protagonist_flags, protagonist_not_audited = _build_protagonist_flags(
        primary_decision=primary_decision,
        extractor_output=protagonist_v1,
        audit_output=audit_outputs.get("protagonist", {}),
    )
    relationship_flags, relationship_not_audited = _build_relationship_flags(
        relationships=relationship_list,
        extractor_output=relationship_v1,
        audit_output=audit_outputs.get("relationship", {}),
    )
    profile_flags, profile_not_audited = _build_profile_flags(
        structured_profile=structured_profile,
        extractor_output=profile_v1,
        audit_output=audit_outputs.get("profile", {}),
    )

    all_flags = protagonist_flags + relationship_flags + profile_flags
    summary = {
        "total_audited_tags": len(all_flags),
        "challenged_count": sum(1 for flag in all_flags if flag.get("was_challenged")),
        "accepted_count": sum(1 for flag in all_flags if flag.get("audit_status") == "accepted"),
        "downgraded_count": sum(1 for flag in all_flags if flag.get("audit_status") == "downgraded"),
        "rejected_count": sum(1 for flag in all_flags if flag.get("audit_status") == "rejected"),
        "not_audited_count": len(protagonist_not_audited) + len(relationship_not_audited) + len(profile_not_audited),
    }

    return {
        "metadata": {
            "downstream_engine": "profile_agent",
            "audit_mode": "selective_profile_domain_rules_facts_only",
            "audit_cycle_mode": "full_v1_critic_v2_judge",
            "profile_agent_root": PROFILE_AGENT_ROOT,
            "runtime_health": runtime_health,
            "reflection_observation_cases_written": 0,
            "reflection_case_facts_written": 0,
            "reflection_capture_status": {
                "phase": "downstream_audit_capture",
                "state": "not_triggered",
                "capture_error": "",
                "written_observation_count": 0,
                "written_case_fact_count": 0,
            },
        },
        "summary": summary,
        "backflow": _build_backflow_payload(
            album_id=album_id,
            primary_decision=primary_decision,
            relationships=relationship_list,
            structured_profile=structured_profile,
            adapter_outputs=adapter_outputs,
            audit_outputs=audit_outputs,
            merged_outputs=merged_outputs,
            storage_saved=storage_saved,
        ),
        "protagonist": {
            "extractor_output": protagonist_v1,
            "extractor_v2_output": protagonist_v2,
            "critic_output": audit_outputs.get("protagonist", {}).get("critic_output", {"challenges": []}),
            "judge_output": audit_outputs.get("protagonist", {}).get("judge_output", {"decisions": [], "hard_cases": []}),
            "v2_targeted_dimensions": audit_outputs.get("protagonist", {}).get("v2_targeted_dimensions", []),
            "audit_flags": protagonist_flags,
            "not_audited": protagonist_not_audited,
        },
        "relationship": {
            "extractor_output": relationship_v1,
            "extractor_v2_output": relationship_v2,
            "critic_output": audit_outputs.get("relationship", {}).get("critic_output", {"challenges": []}),
            "judge_output": audit_outputs.get("relationship", {}).get("judge_output", {"decisions": [], "hard_cases": []}),
            "v2_targeted_dimensions": audit_outputs.get("relationship", {}).get("v2_targeted_dimensions", []),
            "audit_flags": relationship_flags,
            "not_audited": relationship_not_audited,
        },
        "profile": {
            "extractor_output": profile_v1,
            "extractor_v2_output": profile_v2,
            "critic_output": audit_outputs.get("profile", {}).get("critic_output", {"challenges": []}),
            "judge_output": audit_outputs.get("profile", {}).get("judge_output", {"decisions": [], "hard_cases": []}),
            "v2_targeted_dimensions": audit_outputs.get("profile", {}).get("v2_targeted_dimensions", []),
            "audit_flags": profile_flags,
            "not_audited": profile_not_audited,
        },
        "cross_domain_issues": _detect_cross_domain_issues(
            primary_decision=primary_decision,
            relationships=relationship_list,
            structured_profile=structured_profile,
        ),
    }


def _run_profile_agent_judges(
    adapter_outputs: Dict[str, Dict[str, Any]],
    *,
    main_output_extractor: MainOutputExtractor,
    album_id: str = "memory_pipeline_downstream_audit",
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], bool]:
    Storage, CriticAgent, JudgeAgent = _load_profile_agent_runtime()
    storage = Storage()
    critic = CriticAgent(storage)
    judge = JudgeAgent(storage)

    audit_outputs: Dict[str, Dict[str, Any]] = {}
    merged_outputs: Dict[str, Dict[str, Any]] = {}
    for agent_type, extractor_output in adapter_outputs.items():
        if not extractor_output.get("tags"):
            audit_outputs[agent_type] = {
                "extractor_v1_output": deepcopy(extractor_output),
                "extractor_v2_output": deepcopy(extractor_output),
                "v2_targeted_dimensions": [],
                "critic_output": {"challenges": []},
                "judge_output": {"decisions": [], "hard_cases": []},
            }
            merged_outputs[agent_type] = _empty_merged_output(agent_type)
            continue
        critic_result = critic.run(extractor_output)
        critic_output = asdict(critic_result)
        extractor_v2_output = main_output_extractor.build_v2_output(
            agent_type=agent_type,
            extractor_v1_output=extractor_output,
            critic_output=critic_output,
        )
        if critic_result.has_challenges():
            judge_result = judge.run(extractor_v2_output, critic_output, album_id=album_id)
        else:
            judge_result = judge.run_no_challenges(extractor_output)
        audit_outputs[agent_type] = {
            "extractor_v1_output": deepcopy(extractor_output),
            "extractor_v2_output": deepcopy(extractor_v2_output),
            "v2_targeted_dimensions": list(extractor_v2_output.get("v2_targeted_dimensions") or []),
            "critic_output": critic_output,
            "judge_output": asdict(judge_result),
        }
        merged_outputs[agent_type] = _merge_judge_decisions(
            extractor_v2_output,
            audit_outputs[agent_type]["judge_output"],
        )

    storage.save_profile(
        album_id,
        merged_outputs.get("protagonist", _empty_merged_output("protagonist")),
        merged_outputs.get("relationship", _empty_merged_output("relationship")),
        merged_outputs.get("profile", _empty_merged_output("profile")),
    )
    return audit_outputs, merged_outputs, True


def _load_profile_agent_runtime():
    _prepare_profile_agent_runtime_env()
    profile_agent_root = Path(PROFILE_AGENT_ROOT)
    _ensure_profile_agent_package_loaded(profile_agent_root)
    api_module = importlib.import_module("profile_agent.api")
    config_module = importlib.import_module("profile_agent.config")
    _install_profile_agent_api_runtime_patch(
        api_module=api_module,
        config_module=config_module,
    )
    Storage = importlib.import_module("profile_agent.storage").Storage
    CriticAgent = importlib.import_module("profile_agent.agents.critic").CriticAgent
    JudgeAgent = importlib.import_module("profile_agent.agents.judge").JudgeAgent

    return Storage, CriticAgent, JudgeAgent


def _prepare_profile_agent_runtime_env() -> None:
    api_key = (os.environ.get("OPENROUTER_API_KEY") or OPENROUTER_API_KEY or "").strip()
    if not api_key:
        raise ValueError("下游 profile_agent 缺少 OPENROUTER_API_KEY")

    base_url = (
        os.environ.get("OPENROUTER_BASE_URL")
        or OPENROUTER_BASE_URL
        or "https://openrouter.ai/api/v1"
    ).strip() or "https://openrouter.ai/api/v1"

    os.environ["OPENROUTER_API_KEY"] = api_key
    os.environ["OPENROUTER_BASE_URL"] = base_url

    for legacy_key in (
        "GEMINI_API_KEY",
        "GEMINI_API_BASE_URL",
        "GOOGLE_GEMINI_BASE_URL",
    ):
        os.environ.pop(legacy_key, None)


def _install_profile_agent_api_runtime_patch(*, api_module: Any, config_module: Any) -> None:
    model_candidates = _resolve_profile_agent_model_candidates(default_model=str(getattr(config_module, "OPENROUTER_MODEL", "") or ""))
    max_retries = _read_positive_int_env("PROFILE_AGENT_REQUEST_MAX_RETRIES", default=1)
    timeout_seconds = _read_positive_int_env("PROFILE_AGENT_REQUEST_TIMEOUT_SECONDS", default=45)
    retry_delay = _read_positive_float_env("PROFILE_AGENT_REQUEST_RETRY_DELAY_SECONDS", default=1.0)
    extract_text_fn = getattr(api_module, "_extract_text")

    def _request_with_model_fallback(payload: dict) -> str | None:
        api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
        if not api_key:
            raise ValueError("未配置 OPENROUTER_API_KEY")

        base_url = (os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip() or "https://openrouter.ai/api/v1"
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        base_payload = dict(payload or {})
        requested_model = str(base_payload.get("model") or "").strip()
        if requested_model:
            per_call_models = _dedupe_preserve_order([requested_model, *model_candidates])
        else:
            per_call_models = list(model_candidates)

        failure_samples: List[str] = []
        for model in per_call_models:
            model_payload = dict(base_payload)
            model_payload["model"] = model
            data = json.dumps(model_payload).encode("utf-8")

            for attempt in range(1, max_retries + 1):
                try:
                    req = api_module.urllib.request.Request(url, data=data, headers=headers)
                    with api_module.urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                        result = json.loads(resp.read().decode("utf-8"))
                    text = extract_text_fn(result)
                    if text is not None:
                        api_module.OPENROUTER_MODEL = model
                        if hasattr(config_module, "OPENROUTER_MODEL"):
                            config_module.OPENROUTER_MODEL = model
                        return text
                    failure_samples.append(f"{model}@attempt{attempt}:empty_response")
                except Exception as exc:
                    failure_samples.append(f"{model}@attempt{attempt}:{exc}")
                if attempt < max_retries:
                    if failure_samples and "429" in failure_samples[-1]:
                        time.sleep(retry_delay * attempt)
                    else:
                        time.sleep(retry_delay)

        if failure_samples:
            preview = " | ".join(failure_samples[:6])
            more = len(failure_samples) - 6
            suffix = f" (+{more} more)" if more > 0 else ""
            print(f"  [API] all model fallbacks failed: {preview}{suffix}")
        return None

    api_module._request = _request_with_model_fallback
    api_module.MAX_RETRIES = max_retries
    api_module.RETRY_DELAY = retry_delay
    if model_candidates:
        api_module.OPENROUTER_MODEL = model_candidates[0]
        if hasattr(config_module, "OPENROUTER_MODEL"):
            config_module.OPENROUTER_MODEL = model_candidates[0]


def _resolve_profile_agent_model_candidates(*, default_model: str) -> List[str]:
    configured = str(os.environ.get("PROFILE_AGENT_MODEL_CANDIDATES") or "").strip()
    if configured:
        configured_models = [
            item.strip() for item in configured.split(",") if item.strip()
        ]
    else:
        configured_models = []

    ordered = _dedupe_preserve_order(
        [default_model, *configured_models, *DEFAULT_PROFILE_AGENT_MODEL_CANDIDATES]
    )
    return [model for model in ordered if model]


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        key = str(item or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def _read_positive_int_env(env_key: str, *, default: int) -> int:
    raw_value = str(os.environ.get(env_key) or "").strip()
    if not raw_value:
        return default
    try:
        parsed = int(raw_value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _read_positive_float_env(env_key: str, *, default: float) -> float:
    raw_value = str(os.environ.get(env_key) or "").strip()
    if not raw_value:
        return default
    try:
        parsed = float(raw_value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _ensure_profile_agent_package_loaded(profile_agent_root: Path) -> None:
    package_name = "profile_agent"
    init_path = profile_agent_root / "__init__.py"
    if not init_path.exists():
        raise FileNotFoundError(f"未找到下游 profile_agent 包入口: {init_path}")

    existing = sys.modules.get(package_name)
    if existing is not None:
        existing_file = getattr(existing, "__file__", "")
        if existing_file and Path(existing_file).resolve() == init_path.resolve():
            return
        for module_name in list(sys.modules):
            if module_name == package_name or module_name.startswith(f"{package_name}."):
                sys.modules.pop(module_name, None)

    spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[str(profile_agent_root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为下游 profile_agent 构建 importlib spec: {profile_agent_root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)


def _build_protagonist_flags(
    *,
    primary_decision: Dict[str, Any],
    extractor_output: Dict[str, Any],
    audit_output: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    tags = list(extractor_output.get("tags", []))
    if not tags:
        reason = "photographer_mode_not_audited" if primary_decision.get("mode") == "photographer_mode" else "no_mapped_protagonist_tag"
        return [], [{"target_id": "primary_decision", "reason": reason}]

    challenges = list((audit_output.get("critic_output") or {}).get("challenges", []))
    decisions = list((audit_output.get("judge_output") or {}).get("decisions", []))
    flags: List[Dict[str, Any]] = []
    for tag in tags:
        dimension = str(tag.get("dimension") or "")
        matched_challenges = _matched_challenges(dimension, challenges)
        decision = _find_decision(dimension, decisions)
        flags.append(
            {
                "target_id": "primary_decision",
                "mapped_dimension": dimension,
                "audit_status": _decision_to_status(decision),
                "was_challenged": bool(matched_challenges),
                "critic_challenges": matched_challenges,
                "judge_verdict": decision.get("verdict") if decision else None,
                "judge_reason": decision.get("reason") if decision else "",
                "value": tag.get("value"),
            }
        )
    return flags, []


def _build_relationship_flags(
    *,
    relationships: List[Dict[str, Any]],
    extractor_output: Dict[str, Any],
    audit_output: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    tags = list(extractor_output.get("tags", []))
    challenges = list((audit_output.get("critic_output") or {}).get("challenges", []))
    decisions = list((audit_output.get("judge_output") or {}).get("decisions", []))
    flags: List[Dict[str, Any]] = []
    audited_person_ids = set()
    for tag in tags:
        dimension = str(tag.get("dimension") or "")
        person_id = str(tag.get("value") or "")
        matched_challenges = _matched_challenges(dimension, challenges)
        decision = _find_decision(dimension, decisions)
        source_relationship = next((rel for rel in relationships if str(rel.get("person_id")) == person_id), {})
        audited_person_ids.add(person_id)
        flags.append(
            {
                "person_id": person_id,
                "source_relationship_type": source_relationship.get("relationship_type"),
                "mapped_dimension": dimension,
                "audit_status": _decision_to_status(decision),
                "was_challenged": bool(matched_challenges),
                "critic_challenges": matched_challenges,
                "judge_verdict": decision.get("verdict") if decision else None,
                "judge_reason": decision.get("reason") if decision else "",
            }
        )

    not_audited = []
    for relationship in relationships:
        person_id = str(relationship.get("person_id") or "")
        if person_id in audited_person_ids:
            continue
        not_audited.append(
            {
                "person_id": person_id,
                "relationship_type": relationship.get("relationship_type"),
                "reason": "unsupported_relationship_type_for_downstream_plugin",
            }
        )
    return flags, not_audited


def _build_profile_flags(
    *,
    structured_profile: Dict[str, Any],
    extractor_output: Dict[str, Any],
    audit_output: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    tags = list(extractor_output.get("tags", []))
    challenges = list((audit_output.get("critic_output") or {}).get("challenges", []))
    decisions = list((audit_output.get("judge_output") or {}).get("decisions", []))
    audited_field_keys = set()
    flags: List[Dict[str, Any]] = []
    for tag in tags:
        dimension = str(tag.get("dimension") or "")
        field_key = _profile_field_key_for_dimension(dimension, structured_profile, tag.get("value"))
        if field_key:
            audited_field_keys.add(field_key)
        matched_challenges = _matched_challenges(dimension, challenges)
        decision = _find_decision(dimension, decisions)
        flags.append(
            {
                "field_key": field_key,
                "mapped_dimension": dimension,
                "audit_status": _decision_to_status(decision),
                "was_challenged": bool(matched_challenges),
                "critic_challenges": matched_challenges,
                "judge_verdict": decision.get("verdict") if decision else None,
                "judge_reason": decision.get("reason") if decision else "",
                "value": tag.get("value"),
            }
        )

    not_audited: List[Dict[str, Any]] = []
    for field_key, value in _iter_non_null_profile_fields(structured_profile):
        if field_key in audited_field_keys:
            continue
        not_audited.append(
            {
                "field_key": field_key,
                "value": value,
                "reason": "field_not_covered_by_downstream_plugin",
            }
        )
    return flags, not_audited


def _build_backflow_payload(
    *,
    album_id: str,
    primary_decision: Dict[str, Any],
    relationships: List[Dict[str, Any]],
    structured_profile: Dict[str, Any],
    adapter_outputs: Dict[str, Dict[str, Any]],
    audit_outputs: Dict[str, Dict[str, Any]],
    merged_outputs: Dict[str, Dict[str, Any]],
    storage_saved: bool,
) -> Dict[str, Any]:
    return {
        "album_id": album_id,
        "storage_saved": storage_saved,
        "protagonist": {
            "official_output_applied": False,
            "merged_output": merged_outputs.get("protagonist", _empty_merged_output("protagonist")),
            "actions": _build_protagonist_backflow_actions(
                primary_decision=primary_decision,
                adapter_output=(audit_outputs.get("protagonist") or {}).get("extractor_v2_output")
                or adapter_outputs.get("protagonist", {}),
                audit_output=audit_outputs.get("protagonist", {}),
            ),
        },
        "relationship": {
            "official_output_applied": False,
            "merged_output": merged_outputs.get("relationship", _empty_merged_output("relationship")),
            "actions": _build_relationship_backflow_actions(
                relationships=relationships,
                adapter_output=(audit_outputs.get("relationship") or {}).get("extractor_v2_output")
                or adapter_outputs.get("relationship", {}),
                audit_output=audit_outputs.get("relationship", {}),
            ),
        },
        "profile": {
            "official_output_applied": True,
            "merged_output": merged_outputs.get("profile", _empty_merged_output("profile")),
            "field_actions": _build_profile_backflow_actions(
                structured_profile=structured_profile,
                adapter_output=(audit_outputs.get("profile") or {}).get("extractor_v2_output")
                or adapter_outputs.get("profile", {}),
                audit_output=audit_outputs.get("profile", {}),
            ),
        },
    }


def _build_protagonist_backflow_actions(
    *,
    primary_decision: Dict[str, Any],
    adapter_output: Dict[str, Any],
    audit_output: Dict[str, Any],
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    decisions = list((audit_output.get("judge_output") or {}).get("decisions", []))
    tag_by_dimension = {
        str(tag.get("dimension") or ""): tag for tag in list(adapter_output.get("tags", []))
    }
    for decision in decisions:
        dimension = str(decision.get("dimension") or "")
        if not dimension:
            continue
        source_tag = tag_by_dimension.get(dimension, {})
        verdict = str(decision.get("verdict") or "accept")
        actions.append(
            {
                "target_id": "primary_decision",
                "mapped_dimension": dimension,
                "verdict": verdict,
                "judge_reason": decision.get("reason") or "",
                "value_before": source_tag.get("value"),
                "value_after": decision.get("value"),
                "applied_change": _backflow_change_name(verdict, is_profile=False),
                "requires_upstream_rerun": verdict in {"nullify", "downgrade"},
            }
        )
    return actions


def _build_relationship_backflow_actions(
    *,
    relationships: List[Dict[str, Any]],
    adapter_output: Dict[str, Any],
    audit_output: Dict[str, Any],
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    decisions = list((audit_output.get("judge_output") or {}).get("decisions", []))
    tag_by_dimension = {
        str(tag.get("dimension") or ""): tag for tag in list(adapter_output.get("tags", []))
    }
    rel_by_person = {
        str(rel.get("person_id") or ""): rel for rel in relationships
    }
    for decision in decisions:
        dimension = str(decision.get("dimension") or "")
        if not dimension:
            continue
        source_tag = tag_by_dimension.get(dimension, {})
        person_id = str(source_tag.get("value") or "")
        source_relationship = rel_by_person.get(person_id, {})
        verdict = str(decision.get("verdict") or "accept")
        actions.append(
            {
                "person_id": person_id or None,
                "mapped_dimension": dimension,
                "source_relationship_type": source_relationship.get("relationship_type"),
                "verdict": verdict,
                "judge_reason": decision.get("reason") or "",
                "value_before": source_tag.get("value"),
                "value_after": decision.get("value"),
                "applied_change": _backflow_change_name(verdict, is_profile=False),
                "requires_upstream_rerun": verdict in {"nullify", "downgrade"},
            }
        )
    return actions


def _build_profile_backflow_actions(
    *,
    structured_profile: Dict[str, Any],
    adapter_output: Dict[str, Any],
    audit_output: Dict[str, Any],
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    decisions = list((audit_output.get("judge_output") or {}).get("decisions", []))
    tag_by_dimension = {
        str(tag.get("dimension") or ""): tag for tag in list(adapter_output.get("tags", []))
    }
    for decision in decisions:
        dimension = str(decision.get("dimension") or "")
        field_key = _profile_field_key_for_dimension(dimension, structured_profile, decision.get("value"))
        if not field_key:
            continue
        verdict = str(decision.get("verdict") or "accept")
        source_tag = tag_by_dimension.get(dimension, {})
        if verdict == "accept":
            continue
        actions.append(
            {
                "field_key": field_key,
                "mapped_dimension": dimension,
                "verdict": verdict,
                "judge_reason": decision.get("reason") or "",
                "value_before": source_tag.get("value"),
                "value_after": decision.get("value"),
                "applied_change": _backflow_change_name(verdict, is_profile=True),
            }
        )
    return actions


def apply_downstream_profile_backflow(
    structured_profile: Dict[str, Any] | None,
    downstream_audit_report: Dict[str, Any] | None,
    *,
    field_decisions: List[Dict[str, Any]] | None = None,
) -> Tuple[Dict[str, Any] | None, List[Dict[str, Any]]]:
    if not isinstance(structured_profile, dict):
        return structured_profile, list(field_decisions or [])
    updated_structured_profile = deepcopy(structured_profile)
    profile_backflow = (
        (downstream_audit_report or {}).get("backflow", {}).get("profile", {})
    )
    actions = list(profile_backflow.get("field_actions") or [])
    updated_field_decisions = deepcopy(list(field_decisions or []))
    decision_index = {
        str(item.get("field_key") or ""): item
        for item in updated_field_decisions
        if isinstance(item, dict) and str(item.get("field_key") or "")
    }
    for action in actions:
        field_key = str(action.get("field_key") or "")
        if not field_key:
            continue
        field_object = _get_nested(updated_structured_profile, field_key)
        if not isinstance(field_object, dict):
            continue
        evidence = field_object.get("evidence")
        if not isinstance(evidence, dict):
            evidence = {}
            field_object["evidence"] = evidence
        notes = evidence.setdefault("constraint_notes", [])
        if not isinstance(notes, list):
            notes = [str(notes)]
            evidence["constraint_notes"] = notes

        verdict = str(action.get("verdict") or "accept")
        reason = str(action.get("judge_reason") or "").strip()
        note = f"downstream_judge:{verdict}" + (f":{reason}" if reason else "")
        if note not in notes:
            notes.append(note)

        if verdict == "nullify":
            field_object["value"] = None
            field_object["confidence"] = 0.0
            field_object["reasoning"] = _append_reasoning_note(
                field_object.get("reasoning"),
                f"下游 Judge 否决：{reason}" if reason else "下游 Judge 否决",
            )
        elif verdict == "downgrade":
            field_object["reasoning"] = _append_reasoning_note(
                field_object.get("reasoning"),
                f"downstream Judge 建议降为 short_term：{reason}" if reason else "downstream Judge 建议降为 short_term",
            )
        decision = decision_index.get(field_key)
        if isinstance(decision, dict):
            _apply_profile_backflow_to_decision(decision, field_object, action)
    return updated_structured_profile, updated_field_decisions


def _apply_profile_backflow_to_decision(
    decision: Dict[str, Any],
    updated_final: Dict[str, Any],
    action: Dict[str, Any],
) -> None:
    if "final_before_backflow" not in decision and isinstance(decision.get("final"), dict):
        decision["final_before_backflow"] = deepcopy(decision["final"])
    decision["final"] = deepcopy(updated_final)
    decision["backflow"] = {
        "verdict": str(action.get("verdict") or "accept"),
        "judge_reason": str(action.get("judge_reason") or "").strip(),
        "applied_change": str(action.get("applied_change") or ""),
    }


def _should_execute_protagonist_nullify(primary_decision: Dict[str, Any], judge_action: Dict[str, Any]) -> bool:
    """代码硬门槛 — Judge 可以自由质疑，但执行推翻必须过关。"""
    evidence = primary_decision.get("evidence") or {}
    if int(evidence.get("selfie_count", 0) or 0) >= 3:
        return False
    if int(evidence.get("identity_anchor_count", 0) or 0) >= 1:
        return False
    if float(primary_decision.get("confidence", 0) or 0) >= 0.80:
        return False
    reason = str(judge_action.get("judge_reason") or "").strip()
    if len(reason) < 10:
        return False
    return True


def apply_downstream_protagonist_backflow(
    primary_decision: Dict[str, Any] | None,
    downstream_audit_report: Dict[str, Any] | None,
) -> Tuple[Dict[str, Any], bool]:
    current = deepcopy(primary_decision or {})
    actions = list(
        ((downstream_audit_report or {}).get("backflow", {}).get("protagonist", {}) or {}).get("actions", [])
    )
    triggered_actions = [
        action for action in actions if str(action.get("verdict") or "accept") in {"nullify", "downgrade"}
    ]
    if not triggered_actions:
        return current, False

    executable_actions = [a for a in triggered_actions if _should_execute_protagonist_nullify(current, a)]
    blocked_actions = [a for a in triggered_actions if a not in executable_actions]

    if blocked_actions:
        blocked_report = current.setdefault("protagonist_nullify_blocked", [])
        for action in blocked_actions:
            blocked_report.append({
                "verdict": str(action.get("verdict") or ""),
                "judge_reason": str(action.get("judge_reason") or ""),
                "blocked_by": "code_gate",
            })

    if not executable_actions:
        return current, False

    evidence = current.get("evidence")
    if not isinstance(evidence, dict):
        evidence = {}
        current["evidence"] = evidence
    notes = evidence.setdefault("constraint_notes", [])
    if not isinstance(notes, list):
        notes = [str(notes)]
        evidence["constraint_notes"] = notes
    original_primary_person_id = current.get("primary_person_id")
    if original_primary_person_id:
        evidence["rejected_primary_person_id"] = original_primary_person_id

    summary_reasons: List[str] = []
    for action in executable_actions:
        verdict = str(action.get("verdict") or "accept")
        reason = str(action.get("judge_reason") or "").strip()
        note = f"downstream_judge:{verdict}" + (f":{reason}" if reason else "")
        if note not in notes:
            notes.append(note)
        if reason:
            summary_reasons.append(reason)

    current["mode"] = "photographer_mode"
    current["primary_person_id"] = None
    current["confidence"] = 0.0
    reason_text = "；".join(summary_reasons) if summary_reasons else "下游 Judge 未接受当前主角身份"
    current["reasoning"] = _append_reasoning_note(
        current.get("reasoning"),
        f"下游 Judge 触发主角回流：{reason_text}",
    )
    return current, True


def apply_downstream_relationship_backflow(
    relationships: Iterable[Relationship | Dict[str, Any]],
    dossiers: Iterable[RelationshipDossier],
    downstream_audit_report: Dict[str, Any] | None,
) -> Tuple[List[Relationship], List[RelationshipDossier], bool]:
    current_relationships = [_coerce_relationship_dataclass(item) for item in relationships]
    current_dossiers = [deepcopy(dossier) for dossier in dossiers]
    actions = list(
        ((downstream_audit_report or {}).get("backflow", {}).get("relationship", {}) or {}).get("actions", [])
    )
    actionable = {
        str(action.get("person_id") or ""): action
        for action in actions
        if str(action.get("verdict") or "accept") in {"nullify", "downgrade"} and str(action.get("person_id") or "")
    }
    if not actionable:
        return current_relationships, current_dossiers, False

    dossier_by_person = {dossier.person_id: dossier for dossier in current_dossiers}
    updated_relationships: List[Relationship] = []
    changed = False

    for relationship in current_relationships:
        action = actionable.get(relationship.person_id)
        if not action:
            updated_relationships.append(relationship)
            continue
        verdict = str(action.get("verdict") or "accept")
        reason = str(action.get("judge_reason") or "").strip()
        dossier = dossier_by_person.get(relationship.person_id)
        changed = True

        if verdict == "nullify":
            if dossier is not None:
                dossier.retention_decision = "drop"
                dossier.retention_reason = f"downstream_judge_nullify:{reason}" if reason else "downstream_judge_nullify"
                dossier.group_eligible = False
                dossier.group_block_reason = "downstream_judge_nullify"
                dossier.group_weight = 0.0
                dossier.relationship_result = {
                    "relationship_type": None,
                    "status": relationship.status,
                    "confidence": 0.0,
                    "reasoning": _append_reasoning_note(relationship.reasoning, f"下游 Judge 否决：{reason}" if reason else "下游 Judge 否决"),
                }
            continue

        downgrade_target = RELATIONSHIP_TYPE_SPECS.get(relationship.relationship_type)
        new_type = (
            downgrade_target.downgrade_target
            if downgrade_target and downgrade_target.downgrade_target
            else relationship.relationship_type
        )
        updated_evidence = dict(relationship.evidence or {})
        notes = updated_evidence.setdefault("constraint_notes", [])
        if not isinstance(notes, list):
            notes = [str(notes)]
            updated_evidence["constraint_notes"] = notes
        note = f"downstream_judge:downgrade" + (f":{reason}" if reason else "")
        if note not in notes:
            notes.append(note)

        updated_relationship = replace(
            relationship,
            relationship_type=new_type,
            confidence=min(relationship.confidence, 0.7),
            reasoning=_append_reasoning_note(
                relationship.reasoning,
                f"downstream Judge 建议降级为 {new_type}：{reason}" if reason else f"downstream Judge 建议降级为 {new_type}",
            ),
            evidence=updated_evidence,
        )
        if dossier is not None:
            dossier.relationship_result = {
                "relationship_type": updated_relationship.relationship_type,
                "status": updated_relationship.status,
                "confidence": updated_relationship.confidence,
                "reasoning": updated_relationship.reasoning,
            }
            dossier.retention_decision, dossier.retention_reason = _determine_retention(dossier, updated_relationship)
            dossier.group_eligible, dossier.group_block_reason, dossier.group_weight = _determine_group_eligibility(
                dossier,
                updated_relationship,
            )
        updated_relationships.append(updated_relationship)

    return updated_relationships, current_dossiers, changed


def _append_reasoning_note(original: Any, note: str) -> str:
    base = str(original or "").strip()
    if not base:
        return note
    if note in base:
        return base
    return f"{base} {note}"


def _backflow_change_name(verdict: str, *, is_profile: bool) -> str:
    if verdict == "nullify":
        return "nullify_value" if is_profile else "requires_rerun_or_manual_apply"
    if verdict == "downgrade":
        return "annotate_short_term_downgrade" if is_profile else "requires_rerun_or_manual_apply"
    return "no_change"


def _merge_judge_decisions(
    extractor_output: Dict[str, Any],
    judge_output: Dict[str, Any],
) -> Dict[str, Any]:
    result = {
        "agent_type": extractor_output.get("agent_type"),
        "tags": [],
    }
    decision_map = {
        str(decision.get("dimension") or ""): decision
        for decision in list(judge_output.get("decisions") or [])
    }
    for tag in list(extractor_output.get("tags") or []):
        merged_tag = deepcopy(tag)
        dimension = str(tag.get("dimension") or "")
        decision = decision_map.get(dimension)
        if decision:
            verdict = str(decision.get("verdict") or "accept")
            if verdict == "nullify":
                merged_tag["value"] = None
            elif verdict == "downgrade":
                merged_tag["stability"] = "short_term"
        result["tags"].append(merged_tag)
    return result


def _empty_merged_output(agent_type: str) -> Dict[str, Any]:
    return {"agent_type": agent_type, "tags": []}


def _iter_non_null_profile_fields(structured_profile: Dict[str, Any]) -> List[Tuple[str, Any]]:
    results: List[Tuple[str, Any]] = []

    def walk(node: Any, prefix: str) -> None:
        if not isinstance(node, dict):
            return
        if {"value", "confidence", "evidence", "reasoning"}.issubset(node.keys()):
            value = node.get("value")
            if value not in (None, "", []):
                results.append((prefix, value))
            return
        for key, child in node.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            walk(child, child_prefix)

    walk(structured_profile, "")
    return results


def _profile_field_key_for_dimension(dimension: str, structured_profile: Dict[str, Any], tag_value: Any) -> str | None:
    return PROFILE_DIMENSION_FIELD_PATHS.get(dimension)


def _matched_challenges(dimension: str, challenges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [challenge for challenge in challenges if str(challenge.get("target_tag") or "").startswith(dimension)]


def _find_decision(dimension: str, decisions: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    for decision in decisions:
        if str(decision.get("dimension") or "") == dimension:
            return decision
    return None


def _decision_to_status(decision: Dict[str, Any] | None) -> str:
    if not decision:
        return "accepted"
    verdict = str(decision.get("verdict") or "accept")
    if verdict == "accept":
        return "accepted"
    if verdict == "downgrade":
        return "downgraded"
    if verdict == "nullify":
        return "rejected"
    return "accepted"


def _collect_challenge_dimensions(challenges: List[Dict[str, Any]]) -> set[str]:
    dimensions: set[str] = set()
    for challenge in challenges:
        target_tag = str(challenge.get("target_tag") or "").strip()
        if target_tag:
            dimension = target_tag.split(":", 1)[0].strip()
            if dimension:
                dimensions.add(dimension)
    return dimensions


def _challenge_matches_dimension(challenge: Dict[str, Any], dimension: str) -> bool:
    target_tag = str(challenge.get("target_tag") or "").strip()
    if not target_tag:
        return False
    return target_tag.split(":", 1)[0].strip() == dimension


def _collect_challenge_requests(challenges: List[Dict[str, Any]]) -> List[str]:
    requests: List[str] = []
    for challenge in challenges:
        evidence_request = str(challenge.get("evidence_request") or "").strip()
        if evidence_request and evidence_request not in requests:
            requests.append(evidence_request)
    return requests


def _append_extraction_gap(
    *,
    current_gap: Any,
    challenges: List[Dict[str, Any]],
    supplemental_count: int,
) -> str:
    base = str(current_gap or "").strip()
    requests = _collect_challenge_requests(challenges)
    request_text = "；".join(requests) if requests else "按 Critic 质疑执行定向补证"
    note = f"V2补证：{request_text}（新增证据 {supplemental_count} 条）"
    if not base:
        return note
    if note in base:
        return base
    return f"{base}；{note}"


def _build_evidence_from_main_payload(
    payload: Dict[str, Any],
    *,
    fallback_description: str,
) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    synthesized: List[Dict[str, Any]] = []

    supporting_refs = list(payload.get("supporting_refs") or [])
    for ref in supporting_refs:
        if not isinstance(ref, dict):
            continue
        ids = extract_ids_from_refs([ref])
        description = str(
            ref.get("signal")
            or ref.get("description")
            or fallback_description
            or "主工程证据补充"
        )
        synthesized.append(
            _normalize_evidence_item(
                {
                    "event_id": ids["event_ids"][0] if ids["event_ids"] else None,
                    "photo_id": ids["photo_ids"][0] if ids["photo_ids"] else None,
                    "person_id": ids["person_ids"][0] if ids["person_ids"] else None,
                    "feature_names": list(ids["feature_names"]),
                    "description": description,
                    "evidence_type": "direct"
                    if ids["event_ids"] or ids["photo_ids"]
                    else "inferred",
                    "inference_depth": 1
                    if ids["event_ids"] or ids["photo_ids"]
                    else 2,
                }
            )
        )

    event_ids = list(payload.get("event_ids") or [])
    photo_ids = list(payload.get("photo_ids") or [])
    person_ids = list(payload.get("person_ids") or [])
    feature_names = list(payload.get("feature_names") or [])
    max_len = max(len(event_ids), len(photo_ids), len(person_ids), 1 if feature_names else 0)
    for index in range(max_len):
        synthesized.append(
            _normalize_evidence_item(
                {
                    "event_id": event_ids[index] if index < len(event_ids) else None,
                    "photo_id": photo_ids[index] if index < len(photo_ids) else None,
                    "person_id": person_ids[index] if index < len(person_ids) else None,
                    "feature_names": feature_names if index == 0 else [],
                    "description": fallback_description or "主工程证据补充",
                    "evidence_type": "direct"
                    if (index < len(event_ids) or index < len(photo_ids))
                    else "inferred",
                    "inference_depth": 1
                    if (index < len(event_ids) or index < len(photo_ids))
                    else 2,
                }
            )
        )
    return _merge_evidence_lists([], synthesized)


def _normalize_evidence_item(raw_item: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inference_depth = int(raw_item.get("inference_depth") or 1)
    except (TypeError, ValueError):
        inference_depth = 1
    feature_names = []
    for item in list(raw_item.get("feature_names") or []):
        normalized = str(item or "").strip()
        if normalized and normalized not in feature_names:
            feature_names.append(normalized)
    return {
        "event_id": str(raw_item.get("event_id") or "").strip() or None,
        "photo_id": str(raw_item.get("photo_id") or "").strip() or None,
        "person_id": str(raw_item.get("person_id") or "").strip() or None,
        "feature_names": feature_names,
        "description": str(raw_item.get("description") or "").strip(),
        "evidence_type": str(raw_item.get("evidence_type") or "direct"),
        "inference_depth": inference_depth,
    }


def _has_any_evidence_anchor(item: Dict[str, Any]) -> bool:
    return bool(
        item.get("event_id")
        or item.get("photo_id")
        or item.get("person_id")
        or item.get("feature_names")
    )


def _merge_evidence_lists(
    base_items: List[Dict[str, Any]],
    new_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    for item in list(base_items or []) + list(new_items or []):
        if not isinstance(item, dict):
            continue
        normalized = _normalize_evidence_item(item)
        if not _has_any_evidence_anchor(normalized):
            continue
        identity = (
            normalized.get("event_id"),
            normalized.get("photo_id"),
            normalized.get("person_id"),
            tuple(normalized.get("feature_names") or []),
            normalized.get("description"),
        )
        if identity in seen:
            continue
        seen.add(identity)
        merged.append(normalized)
    return merged


def _coerce_relationship_dict(relationship: Relationship | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(relationship, Relationship):
        return {
            "person_id": relationship.person_id,
            "relationship_type": relationship.relationship_type,
            "intimacy_score": relationship.intimacy_score,
            "status": relationship.status,
            "confidence": relationship.confidence,
            "reasoning": relationship.reasoning,
            "shared_events": relationship.shared_events,
            "evidence": relationship.evidence,
        }
    return dict(relationship)


def _coerce_relationship_dataclass(relationship: Relationship | Dict[str, Any]) -> Relationship:
    if isinstance(relationship, Relationship):
        return relationship
    payload = dict(relationship)
    return Relationship(
        person_id=str(payload.get("person_id") or ""),
        relationship_type=str(payload.get("relationship_type") or ""),
        intimacy_score=float(payload.get("intimacy_score", 0.0) or 0.0),
        status=str(payload.get("status") or ""),
        confidence=float(payload.get("confidence", 0.0) or 0.0),
        reasoning=str(payload.get("reasoning") or ""),
        shared_events=list(payload.get("shared_events") or []),
        evidence=dict(payload.get("evidence") or {}),
    )


def _get_nested(payload: Dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


def _stringify_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if str(item or "").strip())
    return str(value)
