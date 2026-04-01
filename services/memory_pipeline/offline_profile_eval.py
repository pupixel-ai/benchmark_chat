from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .precomputed_loader import load_precomputed_memory_state
from .profile_fields import FIELD_SPECS, apply_runtime_field_spec_updates, generate_structured_profile


EXTRA_EVAL_FIELDS = {
    "long_term_facts.geography.location_anchors",
    "short_term_facts.recent_interests",
}


@dataclass
class FieldEvalRow:
    field_key: str
    status: str
    predicted_value: Any
    gt_value: Any
    blocked_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_key": self.field_key,
            "status": self.status,
            "predicted_value": self.predicted_value,
            "gt_value": self.gt_value,
            "blocked_reason": self.blocked_reason,
        }


def run_two_round_offline_eval(
    *,
    case_dir: str | Path,
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    case_path = Path(case_dir)
    report_dir = Path(output_dir) if output_dir else case_path / "offline_eval_output"
    report_dir.mkdir(parents=True, exist_ok=True)

    gt_structured = _load_gt_structured_profile(case_path)
    state = load_precomputed_memory_state(case_path)

    round1 = generate_structured_profile(state, llm_processor=None)
    round1_eval = _evaluate_structured_profile(
        predicted=round1.get("structured", {}),
        gt=gt_structured,
        state=state,
    )

    cot_updates = _propose_cot_updates(round1_eval)
    cot_update_path = report_dir / "round2_cot_updates.json"
    cot_update_path.write_text(json.dumps(cot_updates, ensure_ascii=False, indent=2), encoding="utf-8")
    apply_runtime_field_spec_updates(cot_updates)

    round2 = generate_structured_profile(state, llm_processor=None)
    round2_eval = _evaluate_structured_profile(
        predicted=round2.get("structured", {}),
        gt=gt_structured,
        state=state,
    )

    html_path = report_dir / "lp3_offline_eval_report.html"
    html_path.write_text(
        _build_html_report(
            case_name=case_path.name,
            round1_eval=round1_eval,
            round2_eval=round2_eval,
            cot_updates=cot_updates,
            generated_at=datetime.now().isoformat(),
        ),
        encoding="utf-8",
    )

    return {
        "case_dir": str(case_path),
        "round1": round1_eval,
        "round2": round2_eval,
        "cot_updates_path": str(cot_update_path),
        "html_report_path": str(html_path),
        "thresholds": {
            "precision": 0.80,
            "recall": 0.60,
            "high_risk_false_fill_max": 2,
        },
    }


def _load_gt_structured_profile(case_path: Path) -> Dict[str, Any]:
    exact = case_path / "profile_structured.json"
    if exact.exists():
        return json.loads(exact.read_text(encoding="utf-8"))
    matches = sorted(case_path.glob("*_profile_structured.json"))
    if not matches:
        return {}
    return json.loads(matches[0].read_text(encoding="utf-8"))


def _evaluate_structured_profile(
    *,
    predicted: Dict[str, Any],
    gt: Dict[str, Any],
    state: Any,
) -> Dict[str, Any]:
    fields = _build_eval_fields()
    rows: List[FieldEvalRow] = []
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "fp_fn": 0, "upstream_blocked": 0}
    high_risk_false_fill = 0

    for field_key in fields:
        blocked_reason = _upstream_block_reason(field_key, state)
        if blocked_reason:
            counts["upstream_blocked"] += 1
            rows.append(
                FieldEvalRow(
                    field_key=field_key,
                    status="upstream_blocked",
                    predicted_value=None,
                    gt_value=None,
                    blocked_reason=blocked_reason,
                )
            )
            continue

        predicted_value = _get_nested_value(predicted, field_key)
        gt_value = _get_nested_value(gt, field_key)
        status = _classify_field_result(predicted_value, gt_value)
        counts[status] += 1
        if status in {"fp", "fp_fn"} and FIELD_SPECS[field_key].risk_level == "P0":
            high_risk_false_fill += 1
        rows.append(
            FieldEvalRow(
                field_key=field_key,
                status=status,
                predicted_value=predicted_value,
                gt_value=gt_value,
            )
        )

    tp = counts["tp"]
    fp = counts["fp"] + counts["fp_fn"]
    fn = counts["fn"] + counts["fp_fn"]
    precision = round(tp / (tp + fp), 4) if (tp + fp) else 0.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) else 0.0

    return {
        "summary": {
            **counts,
            "precision": precision,
            "recall": recall,
            "high_risk_false_fill": high_risk_false_fill,
            "total_eval_fields": len(fields),
            "effective_eval_fields": len(fields) - counts["upstream_blocked"],
        },
        "fields": [row.to_dict() for row in rows],
    }


def _build_eval_fields() -> List[str]:
    selected = {key for key, spec in FIELD_SPECS.items() if spec.risk_level == "P0"}
    selected.update(EXTRA_EVAL_FIELDS)
    return sorted(selected)


def _upstream_block_reason(field_key: str, state: Any) -> str:
    spec = FIELD_SPECS[field_key]
    if "relationship" in spec.allowed_sources and not (state.relationships or []):
        return "upstream_blocked:missing_relationships"
    if "group" in spec.allowed_sources and not (state.groups or []):
        return "upstream_blocked:missing_groups"
    return ""


def _classify_field_result(predicted_value: Any, gt_value: Any) -> str:
    predicted_has = _has_value(predicted_value)
    gt_has = _has_value(gt_value)
    if not predicted_has and not gt_has:
        return "tn"
    if predicted_has and not gt_has:
        return "fp"
    if not predicted_has and gt_has:
        return "fn"
    return "tp" if _normalize_value(predicted_value) == _normalize_value(gt_value) else "fp_fn"


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) > 0
    return True


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, list):
        normalized = []
        for item in value:
            if isinstance(item, dict):
                if "name" in item:
                    normalized.append(str(item["name"]).strip().lower())
                else:
                    normalized.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
            else:
                normalized.append(str(item).strip().lower())
        return sorted(normalized)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _get_nested_value(payload: Dict[str, Any], field_key: str) -> Any:
    current: Any = payload
    for segment in field_key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(segment)
        if current is None:
            return None
    if isinstance(current, dict) and "value" in current:
        return current.get("value")
    return current


def _propose_cot_updates(round1_eval: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    fields = round1_eval.get("fields", [])
    target_fields = {
        "long_term_facts.material.brand_preference",
        "long_term_facts.geography.location_anchors",
        "short_term_facts.recent_interests",
    }
    updates: Dict[str, Dict[str, Any]] = {}
    for row in fields:
        field_key = row.get("field_key")
        status = row.get("status")
        if field_key not in target_fields or status not in {"fp", "fp_fn", "fn"}:
            continue
        spec = FIELD_SPECS[field_key]
        weak_evidence_caution = list(spec.weak_evidence_caution)
        cot_steps = list(spec.cot_steps)
        if status in {"fp", "fp_fn"}:
            weak_evidence_caution.append("若证据集中在单事件簇或单时间窗口，优先输出 null")
            cot_steps.append("反思阶段必须确认至少两个独立事件窗口，否则不输出非 null")
        else:
            cot_steps.append("若存在跨事件重复且主体归属明确，可放宽单字段保守阈值")
        updates[field_key] = {
            "cot_steps": list(dict.fromkeys(cot_steps)),
            "weak_evidence_caution": list(dict.fromkeys(weak_evidence_caution)),
        }
    return updates


def _build_html_report(
    *,
    case_name: str,
    round1_eval: Dict[str, Any],
    round2_eval: Dict[str, Any],
    cot_updates: Dict[str, Dict[str, Any]],
    generated_at: str,
) -> str:
    round1_summary = round1_eval.get("summary", {})
    round2_summary = round2_eval.get("summary", {})
    rows_html = []
    round2_map = {row["field_key"]: row for row in round2_eval.get("fields", [])}
    for row in round1_eval.get("fields", []):
        field_key = row["field_key"]
        round2_row = round2_map.get(field_key, {})
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(field_key)}</td>"
            f"<td>{html.escape(str(row.get('status', '')))}</td>"
            f"<td>{html.escape(str(round2_row.get('status', '')))}</td>"
            f"<td>{html.escape(str(row.get('predicted_value')))}</td>"
            f"<td>{html.escape(str(row.get('gt_value')))}</td>"
            f"<td>{html.escape(str(row.get('blocked_reason', '')))}</td>"
            "</tr>"
        )

    updates_html = html.escape(json.dumps(cot_updates, ensure_ascii=False, indent=2))

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>LP3 Offline Eval - {html.escape(case_name)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2937; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(2, minmax(220px, 1fr)); gap: 12px; margin-bottom: 20px; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; background: #f8fafc; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    pre {{ background: #0f172a; color: #e2e8f0; padding: 12px; border-radius: 8px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>LP3 离线双轮评测</h1>
  <div>Case: <b>{html.escape(case_name)}</b> | Generated: {html.escape(generated_at)}</div>
  <h2>Round Summary</h2>
  <div class="metrics">
    <div class="card">
      <b>Round1</b><br/>
      Precision: {round1_summary.get("precision", 0)}<br/>
      Recall: {round1_summary.get("recall", 0)}<br/>
      High-risk false fill: {round1_summary.get("high_risk_false_fill", 0)}
    </div>
    <div class="card">
      <b>Round2</b><br/>
      Precision: {round2_summary.get("precision", 0)}<br/>
      Recall: {round2_summary.get("recall", 0)}<br/>
      High-risk false fill: {round2_summary.get("high_risk_false_fill", 0)}
    </div>
  </div>
  <h2>Field-Level Diff</h2>
  <table>
    <thead>
      <tr>
        <th>field_key</th>
        <th>round1_status</th>
        <th>round2_status</th>
        <th>predicted_round1</th>
        <th>gt</th>
        <th>blocked_reason</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
  <h2>COT Auto-Updates</h2>
  <pre>{updates_html}</pre>
</body>
</html>
"""
