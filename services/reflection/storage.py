from __future__ import annotations

import os

from .types import ReflectionAssetPaths


def build_reflection_asset_paths(*, project_root: str, user_name: str) -> ReflectionAssetPaths:
    root_dir = os.path.join(project_root, "memory", "reflection")
    return ReflectionAssetPaths(
        root_dir=root_dir,
        observation_cases_path=os.path.join(root_dir, f"observation_cases_{user_name}.jsonl"),
        case_facts_path=os.path.join(root_dir, f"case_facts_{user_name}.jsonl"),
        profile_field_gt_path=os.path.join(root_dir, f"profile_field_gt_{user_name}.jsonl"),
        gt_comparisons_path=os.path.join(root_dir, f"gt_comparisons_{user_name}.jsonl"),
        profile_field_trace_index_path=os.path.join(root_dir, f"profile_field_trace_index_{user_name}.jsonl"),
        profile_field_trace_payload_dir=os.path.join(root_dir, "profile_field_trace_payloads", user_name),
        human_review_records_path=os.path.join(root_dir, f"human_review_records_{user_name}.jsonl"),
        upstream_patterns_path=os.path.join(root_dir, f"upstream_patterns_{user_name}.json"),
        upstream_experiments_path=os.path.join(root_dir, f"upstream_experiments_{user_name}.json"),
        upstream_outcomes_path=os.path.join(root_dir, f"upstream_outcomes_{user_name}.json"),
        downstream_audit_patterns_path=os.path.join(root_dir, f"downstream_audit_patterns_{user_name}.json"),
        downstream_audit_experiments_path=os.path.join(root_dir, f"downstream_audit_experiments_{user_name}.json"),
        engineering_alerts_path=os.path.join(root_dir, f"engineering_alerts_{user_name}.jsonl"),
        difficult_cases_path=os.path.join(root_dir, f"difficult_cases_{user_name}.jsonl"),
        difficult_case_actions_path=os.path.join(root_dir, f"difficult_case_actions_{user_name}.jsonl"),
        decision_review_items_path=os.path.join(root_dir, f"decision_review_items_{user_name}.json"),
        tasks_path=os.path.join(root_dir, f"tasks_{user_name}.jsonl"),
        task_actions_path=os.path.join(root_dir, f"task_actions_{user_name}.jsonl"),
        proposals_path=os.path.join(root_dir, f"proposals_{user_name}.jsonl"),
        proposal_actions_path=os.path.join(root_dir, f"proposal_actions_{user_name}.jsonl"),
        engineering_change_requests_path=os.path.join(root_dir, f"engineering_change_requests_{user_name}.jsonl"),
        reflection_feedback_path=os.path.join(root_dir, f"reflection_feedback_{user_name}.jsonl"),
        experiments_dir=os.path.join(root_dir, "experiments", user_name),
    )


def ensure_reflection_root(paths: ReflectionAssetPaths) -> str:
    os.makedirs(paths.root_dir, exist_ok=True)
    os.makedirs(paths.profile_field_trace_payload_dir, exist_ok=True)
    os.makedirs(paths.experiments_dir, exist_ok=True)
    return paths.root_dir
