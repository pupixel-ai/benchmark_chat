#!/usr/bin/env python3
"""
记忆工程 v2.0 - 主入口
Pipeline: Load → HEIC → 去重 → 人脸识别 → 主角推断+画框 → 压缩 → VLM → LLM → 保存
"""
import os
import sys
import json
import argparse
import shutil
from datetime import datetime
from typing import List, Dict

# 预解析 --user-name 参数，在 config import 之前设置环境变量
def _pre_parse_user_name():
    for i, arg in enumerate(sys.argv):
        if arg == '--user-name' and i + 1 < len(sys.argv):
            os.environ['MEMORY_USER_NAME'] = sys.argv[i + 1]
            return

_pre_parse_user_name()

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 加载环境变量
from dotenv import load_dotenv
try:
    load_dotenv()
except:
    pass

from config import *
from services.image_processor import ImageProcessor
from services.face_recognition import FaceRecognition
from services.vlm_analyzer import VLMAnalyzer
from services.llm_processor import LLMProcessor
from utils import save_json, format_timestamp
from utils.output_artifacts import (
    build_artifacts_manifest,
    build_final_output_payload,
    build_internal_artifact,
    build_profile_debug_artifact,
    build_relationships_artifact,
    save_json_artifact,
)
from services.memory_pipeline.downstream_audit import (
    apply_downstream_profile_backflow,
    apply_downstream_protagonist_backflow,
    apply_downstream_relationship_backflow,
    inspect_profile_agent_runtime_health,
    run_downstream_profile_agent_audit,
)
from services.memory_pipeline.feedback_cases import persist_downstream_feedback_cases
from services.memory_pipeline.orchestrator import (
    build_memory_state,
    rerun_pipeline_from_primary_backflow,
    rerun_pipeline_from_relationship_backflow,
    run_memory_pipeline,
)


def show_progress(current: int, total: int, message: str = ""):
    """显示进度条"""
    if not SHOW_PROGRESS:
        return

    percent = (current / total) * 100
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)

    print(f"\r处理中... [{bar}] {current}/{total} ({percent:.0f}%) {message}", end='', flush=True)


def print_simple_summary(events: List, relationships: List, artifacts: Dict[str, str | None], profile_path: str = None):
    """打印简洁版摘要"""
    print("\n\n" + "=" * 50)
    print("记忆提取完成")
    print("=" * 50)

    print(f"\n事件：{len(events)}个")
    for event in events:
        print(f"  ├─ {event.date} {event.time_range}: {event.title} (置信度: {event.confidence:.0%})")

    print(f"\n人物关系：{len(relationships)}个")
    for rel in relationships:
        print(f"  ├─ {rel.person_id}: {rel.relationship_type} (intimacy: {rel.intimacy_score:.2f}, {rel.status})")

    print("\n用户画像：")
    if profile_path:
        print(f"  画像报告已生成: {profile_path}")
    else:
        print(f"  画像生成失败")

    print(f"\n结果已保存到: {OUTPUT_PATH}")
    print(f"详细报告已保存到: {DETAILED_OUTPUT_PATH}")
    print(f"VLM缓存已保存到: {VLM_CACHE_PATH}")
    print(f"关系调试输出已保存到: {artifacts.get('relationships_path')}")
    print(f"画像调试输出已保存到: {artifacts.get('profile_debug_path')}")
    print(f"下游审计输出已保存到: {artifacts.get('downstream_audit_report_path')}")
    print(f"下游反馈 case 已保存到: {artifacts.get('downstream_feedback_cases_path')}")
    print(f"下游通知状态已保存到: {artifacts.get('downstream_notification_status_path')}")


def save_detailed_report(events: List, relationships: List, face_db: dict):
    """保存详细版报告（Markdown格式）"""
    report = []
    report.append("# 记忆工程 v2.0 - 详细报告\n")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    report.append("## 事件详情\n")
    for i, event in enumerate(events, 1):
        report.append(f"### {i}. {event.title}\n")
        report.append(f"- **时间**: {event.date} {event.time_range}（{event.duration}）\n")
        report.append(f"- **类型**: {event.type}\n")
        report.append(f"- **地点**: {event.location}\n")
        report.append(f"- **参与者**: {', '.join(event.participants)}\n")
        report.append(f"- **照片数**: {event.photo_count}张\n")
        report.append(f"- **描述**: {event.description}\n")
        if event.narrative:
            report.append(f"- **客观叙事**: {event.narrative}\n")
        if event.social_interaction:
            core_ids = event.social_interaction.get('core_person_ids', [])
            if core_ids:
                report.append(f"- **核心同伴**: {', '.join(core_ids)}\n")
        if event.lifestyle_tags:
            report.append(f"- **生活标签**: {', '.join(event.lifestyle_tags)}\n")
        if event.social_slices:
            report.append(f"- **社交切片**:\n")
            for s in event.social_slices:
                report.append(f"  - {s.get('person_id', '未知')}: {s.get('interaction', '')} | {s.get('relationship', '')} ({s.get('confidence', 0):.0%})\n")
        if event.persona_evidence:
            evidence = event.persona_evidence
            report.append(f"- **人格线索**:\n")
            if evidence.get('behavioral'):
                report.append(f"  - 行为: {', '.join(evidence['behavioral'])}\n")
            if evidence.get('aesthetic'):
                report.append(f"  - 审美: {', '.join(evidence['aesthetic'])}\n")
        report.append(f"- **置信度**: {event.confidence:.0%}\n")
        report.append(f"- **推理依据**: {event.reason}\n\n")

    report.append("## 人物关系详情\n")
    for i, rel in enumerate(relationships, 1):
        report.append(f"### {i}. {rel.person_id} - {rel.relationship_type} (intimacy: {rel.intimacy_score:.2f}, {rel.status})\n")
        report.append(f"- **月均频率**: {rel.evidence.get('monthly_frequency', 0)}/月\n")
        report.append(f"- **置信度**: {rel.confidence:.0%}\n")
        report.append(f"- **证据**: 共同出现{rel.evidence.get('photo_count', 0)}次，时间跨度{rel.evidence.get('time_span', '未知')}\n")
        report.append(f"- **场景**: {', '.join(rel.evidence.get('scenes', []))}\n")
        co_persons = rel.evidence.get('co_appearing_persons', [])
        if co_persons:
            co_strs = [f"{c['person_id']}({c['co_ratio']:.0%})" for c in co_persons[:5]]
            report.append(f"- **第三方共现**: {', '.join(co_strs)}\n")
        anomalies = rel.evidence.get('anomalies', [])
        if anomalies:
            anomaly_strs = [f"{a['type']}@{a['date']}" for a in anomalies[:5]]
            report.append(f"- **异常**: {'; '.join(anomaly_strs)}\n")
        report.append(f"- **推理**: {rel.reasoning}\n\n")

    with open(DETAILED_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(report)


def save_final_output(events: List, relationships: List, face_db: dict, artifacts: Dict[str, str | None]):
    """保存最终输出（JSON格式）"""
    output = build_final_output_payload(
        events=events,
        relationships=relationships,
        face_db=face_db,
        artifacts=artifacts,
        models={
            "vlm": VLM_MODEL,
            "llm": LLM_MODEL,
            "face": f"InsightFace/{FACE_MODEL_NAME}",
        },
    )
    save_json(output, OUTPUT_PATH)


def save_profile_report(profile_markdown: str):
    """保存用户画像报告"""
    with open(PROFILE_REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(profile_markdown)
    return PROFILE_REPORT_PATH


def save_structured_profile(structured_profile: dict):
    """保存结构化画像标签"""
    save_json(structured_profile, PROFILE_STRUCTURED_PATH)
    return PROFILE_STRUCTURED_PATH


def save_relationships_output(relationships: List, primary_person_id: str | None):
    """保存独立 relationships 调试输出"""
    payload = build_relationships_artifact(
        relationships=relationships,
        primary_person_id=primary_person_id,
    )
    return save_json_artifact(payload, RELATIONSHIPS_OUTPUT_PATH)


def save_profile_debug(profile_result: dict | None, primary_person_id: str | None, total_events: int, total_relationships: int):
    """保存画像调试输出（debug + consistency）。"""
    payload = build_profile_debug_artifact(
        profile_result=profile_result,
        primary_person_id=primary_person_id,
        total_events=total_events,
        total_relationships=total_relationships,
    )
    return save_json_artifact(payload, PROFILE_DEBUG_PATH)


def save_internal_artifact(artifact_name: str, payload, path: str, **metadata):
    """保存多 agent 内部 artifact。"""
    artifact = build_internal_artifact(
        artifact_name=artifact_name,
        payload=payload,
        **metadata,
    )
    return save_json_artifact(artifact, path)


def run_memory_pipeline_entry(
    llm: LLMProcessor,
    photos: list,
    face_db: dict,
    vlm_results: list,
    primary_person_id: str | None,
):
    """运行当前主链路记忆生成链路。"""
    print("  - 运行主链路记忆生成...")
    memory_state = build_memory_state(
        photos=photos,
        face_db=face_db,
        vlm_results=vlm_results,
    )
    profile_result = run_memory_pipeline(
        state=memory_state,
        llm_processor=llm,
        fallback_primary_person_id=primary_person_id,
    )
    audit_album_id = f"{USER_NAME}_{RUN_TIMESTAMP}"
    downstream_audit_report = _run_downstream_audit_with_fallback(
        album_id=audit_album_id,
        primary_decision=profile_result.get("internal_artifacts", {}).get("primary_decision"),
        relationships=profile_result.get("relationships", []),
        structured_profile=profile_result.get("structured"),
        profile_fact_decisions=profile_result.get("internal_artifacts", {}).get("profile_fact_decisions", []),
    )
    audit_rounds = [_audit_round_snapshot("initial", downstream_audit_report)]

    updated_primary_decision, protagonist_changed = apply_downstream_protagonist_backflow(
        profile_result.get("internal_artifacts", {}).get("primary_decision"),
        downstream_audit_report,
    )
    if protagonist_changed:
        memory_state.primary_decision = updated_primary_decision
        if updated_primary_decision.get("mode") == "photographer_mode":
            memory_state.relationships = []
            memory_state.relationship_dossiers = []
            memory_state.groups = []
            profile_result = rerun_pipeline_from_relationship_backflow(
                state=memory_state,
                llm_processor=llm,
            )
            rerun_stage = "after_protagonist_photographer_rerun"
        else:
            profile_result = rerun_pipeline_from_primary_backflow(
                state=memory_state,
                llm_processor=llm,
            )
            rerun_stage = "after_protagonist_rerun"
        downstream_audit_report = _run_downstream_audit_with_fallback(
            album_id=audit_album_id,
            primary_decision=profile_result.get("internal_artifacts", {}).get("primary_decision"),
            relationships=profile_result.get("relationships", []),
            structured_profile=profile_result.get("structured"),
            profile_fact_decisions=profile_result.get("internal_artifacts", {}).get("profile_fact_decisions", []),
        )
        audit_rounds.append(_audit_round_snapshot(rerun_stage, downstream_audit_report))

    updated_relationships, updated_dossiers, relationship_changed = apply_downstream_relationship_backflow(
        profile_result.get("relationships", []),
        memory_state.relationship_dossiers,
        downstream_audit_report,
    )
    if relationship_changed:
        memory_state.relationships = updated_relationships
        memory_state.relationship_dossiers = updated_dossiers
        profile_result = rerun_pipeline_from_relationship_backflow(
            state=memory_state,
            llm_processor=llm,
        )
        downstream_audit_report = _run_downstream_audit_with_fallback(
            album_id=audit_album_id,
            primary_decision=profile_result.get("internal_artifacts", {}).get("primary_decision"),
            relationships=profile_result.get("relationships", []),
            structured_profile=profile_result.get("structured"),
            profile_fact_decisions=profile_result.get("internal_artifacts", {}).get("profile_fact_decisions", []),
        )
        audit_rounds.append(_audit_round_snapshot("after_relationship_rerun", downstream_audit_report))

    updated_structured_profile, updated_profile_fact_decisions = apply_downstream_profile_backflow(
        profile_result.get("structured"),
        downstream_audit_report,
        field_decisions=profile_result.get("internal_artifacts", {}).get("profile_fact_decisions", []),
    )
    profile_result["structured"] = updated_structured_profile
    profile_result.setdefault("internal_artifacts", {})["profile_fact_decisions"] = updated_profile_fact_decisions
    downstream_audit_report["feedback_loop"] = {
        "protagonist_rerun_applied": protagonist_changed,
        "relationship_rerun_applied": relationship_changed,
        "audit_rounds": audit_rounds,
    }
    final_primary_person_id = (
        (profile_result.get("internal_artifacts", {}).get("primary_decision") or {}).get("primary_person_id")
    )
    feedback_case_error = ""
    try:
        feedback_case_result = persist_downstream_feedback_cases(
            downstream_audit_report=downstream_audit_report,
            user_name=USER_NAME,
            run_timestamp=RUN_TIMESTAMP,
            album_id=audit_album_id,
            queue_path=DOWNSTREAM_BADCASE_QUEUE_PATH,
            run_output_path=DOWNSTREAM_FEEDBACK_CASES_OUTPUT_PATH,
            profile_agent_root=PROFILE_AGENT_ROOT,
        )
    except Exception as exc:
        feedback_case_error = str(exc)
        feedback_case_result = {
            "run_output_path": None,
            "queue_path": DOWNSTREAM_BADCASE_QUEUE_PATH,
            "written_count": 0,
            "pending_count": 0,
            "mirrored_count": 0,
            "case_ids": [],
            "error": feedback_case_error,
        }
    downstream_feedback_cases_path = feedback_case_result.get("run_output_path")
    notification_status = {
        "phase": "downstream_audit_pipeline",
        "state": "not_triggered" if not feedback_case_error else "feedback_case_write_failed",
        "message": "使用 scripts/downstream_badcase_bridge.py propose/apply 触发飞书审批链。",
        "feedback_case_error": feedback_case_error,
        "notification_failures": [],
    }
    downstream_audit_report.setdefault("metadata", {})["feedback_cases_written"] = feedback_case_result.get("written_count", 0)
    downstream_audit_report["metadata"]["notification_status"] = notification_status
    events = profile_result.get("events", [])
    relationships = profile_result.get("relationships", [])
    relationship_dossiers_path = save_internal_artifact(
        artifact_name="relationship_dossiers",
        payload=profile_result.get("internal_artifacts", {}).get("relationship_dossiers", []),
        path=RELATIONSHIP_DOSSIERS_PATH,
        primary_person_id=final_primary_person_id,
        total_dossiers=len(profile_result.get("internal_artifacts", {}).get("relationship_dossiers", [])),
    )
    group_artifacts_path = save_internal_artifact(
        artifact_name="group_artifacts",
        payload=profile_result.get("internal_artifacts", {}).get("group_artifacts", []),
        path=GROUP_ARTIFACTS_PATH,
        primary_person_id=final_primary_person_id,
        total_groups=len(profile_result.get("internal_artifacts", {}).get("group_artifacts", [])),
    )
    profile_fact_decisions_path = save_internal_artifact(
        artifact_name="profile_fact_decisions",
        payload=profile_result.get("internal_artifacts", {}).get("profile_fact_decisions", []),
        path=PROFILE_FACT_DECISIONS_PATH,
        primary_person_id=final_primary_person_id,
        total_fields=len(profile_result.get("internal_artifacts", {}).get("profile_fact_decisions", [])),
    )
    downstream_notification_status_path = save_internal_artifact(
        artifact_name="downstream_notification_status",
        payload=notification_status,
        path=DOWNSTREAM_NOTIFICATION_STATUS_PATH,
        primary_person_id=final_primary_person_id,
        queue_path=DOWNSTREAM_BADCASE_QUEUE_PATH,
        feedback_cases_written=feedback_case_result.get("written_count", 0),
    )
    downstream_audit_report_path = save_internal_artifact(
        artifact_name="downstream_audit_report",
        payload=downstream_audit_report,
        path=DOWNSTREAM_AUDIT_REPORT_PATH,
        primary_person_id=final_primary_person_id,
        total_audited_tags=downstream_audit_report.get("summary", {}).get("total_audited_tags", 0),
        rejected_count=downstream_audit_report.get("summary", {}).get("rejected_count", 0),
    )
    print(f"    提取了 {len(events)} 个事件")
    print(f"    推断了 {len(relationships)} 个关系")
    print(f"    生成了 {len(profile_result.get('internal_artifacts', {}).get('group_artifacts', []))} 个圈层 artifact")
    return {
        "events": events,
        "relationships": relationships,
        "profile_result": profile_result,
        "final_primary_person_id": final_primary_person_id,
        "relationship_dossiers_path": relationship_dossiers_path,
        "group_artifacts_path": group_artifacts_path,
        "profile_fact_decisions_path": profile_fact_decisions_path,
        "downstream_audit_report_path": downstream_audit_report_path,
        "downstream_feedback_cases_path": downstream_feedback_cases_path,
        "downstream_notification_status_path": downstream_notification_status_path,
    }


def _audit_round_snapshot(stage: str, report: Dict) -> Dict[str, Dict]:
    return {
        "stage": stage,
        "summary": dict(report.get("summary", {}) or {}),
        "backflow": dict(report.get("backflow", {}) or {}),
    }


def _run_downstream_audit_with_fallback(
    *,
    album_id: str,
    primary_decision: Dict | None,
    relationships: list | None,
    structured_profile: Dict | None,
    profile_fact_decisions: List[Dict] | None = None,
) -> Dict:
    runtime_health = inspect_profile_agent_runtime_health()
    if runtime_health.get("status") != "ok":
        return _build_skipped_downstream_audit_report(
            album_id=album_id,
            primary_decision=primary_decision or {},
            relationships=relationships or [],
            structured_profile=structured_profile or {},
            runtime_health=runtime_health,
            error=RuntimeError(
                f"downstream_runtime_unhealthy:{runtime_health.get('error_code', 'runtime_unhealthy')}"
            ),
        )
    try:
        return run_downstream_profile_agent_audit(
            album_id=album_id,
            primary_decision=primary_decision,
            relationships=relationships or [],
            structured_profile=structured_profile,
            profile_fact_decisions=profile_fact_decisions or [],
            runtime_health=runtime_health,
        )
    except Exception as exc:
        return _build_skipped_downstream_audit_report(
            album_id=album_id,
            primary_decision=primary_decision or {},
            relationships=relationships or [],
            structured_profile=structured_profile or {},
            runtime_health=runtime_health,
            error=exc,
        )


def _build_skipped_downstream_audit_report(
    *,
    album_id: str,
    primary_decision: Dict,
    relationships: list,
    structured_profile: Dict,
    runtime_health: Dict | None,
    error: Exception,
) -> Dict:
    error_type = type(error).__name__
    error_message = str(error)
    error_code = (
        str((runtime_health or {}).get("error_code") or "audit_runtime_failure")
        if runtime_health and runtime_health.get("status") != "ok"
        else "audit_runtime_failure"
    )
    protagonist_not_audited = [{"target_id": "primary_decision", "reason": "audit_runtime_failure"}] if primary_decision else []
    relationship_not_audited = []
    for relationship in relationships:
        if isinstance(relationship, dict):
            person_id = relationship.get("person_id")
        else:
            person_id = getattr(relationship, "person_id", None)
        relationship_not_audited.append(
            {
                "person_id": person_id,
                "reason": "audit_runtime_failure",
            }
        )
    total_not_audited = len(protagonist_not_audited) + len(relationship_not_audited)
    return {
        "metadata": {
            "downstream_engine": "profile_agent",
            "audit_mode": "selective_profile_domain_rules_facts_only",
            "audit_cycle_mode": "full_v1_critic_v2_judge",
            "profile_agent_root": PROFILE_AGENT_ROOT,
            "audit_status": "skipped_init_failure",
            "audit_error_code": error_code,
            "audit_error_type": error_type,
            "audit_error_message": error_message,
            "runtime_health": runtime_health or {},
            "feedback_cases_written": 0,
            "notification_status": {
                "phase": "downstream_audit_pipeline",
                "state": "skipped_init_failure",
                "notification_failures": [],
            },
        },
        "summary": {
            "total_audited_tags": 0,
            "challenged_count": 0,
            "accepted_count": 0,
            "downgraded_count": 0,
            "rejected_count": 0,
            "not_audited_count": total_not_audited,
        },
        "backflow": {
            "album_id": album_id,
            "storage_saved": False,
            "audit_failure": {
                "error_type": error_type,
                "error_message": error_message,
            },
            "protagonist": {
                "official_output_applied": False,
                "merged_output": {"agent_type": "protagonist", "tags": []},
                "actions": [],
            },
            "relationship": {
                "official_output_applied": False,
                "merged_output": {"agent_type": "relationship", "tags": []},
                "actions": [],
            },
            "profile": {
                "official_output_applied": False,
                "merged_output": {"agent_type": "profile", "tags": []},
                "field_actions": [],
            },
        },
        "protagonist": {
            "extractor_output": {},
            "critic_output": {"challenges": []},
            "judge_output": {"decisions": [], "hard_cases": []},
            "audit_flags": [],
            "not_audited": protagonist_not_audited,
        },
        "relationship": {
            "extractor_output": {},
            "critic_output": {"challenges": []},
            "judge_output": {"decisions": [], "hard_cases": []},
            "audit_flags": [],
            "not_audited": relationship_not_audited,
        },
        "profile": {
            "extractor_output": {},
            "critic_output": {"challenges": []},
            "judge_output": {"decisions": [], "hard_cases": []},
            "audit_flags": [],
            "not_audited": [],
        },
    }


def reset_cache():
    """重置缓存（清空 cache/ 和 output/）"""
    for path in (USER_CACHE_DIR, USER_OUTPUT_DIR):
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)


def main():
    """主流程 — 9步 pipeline"""
    parser = argparse.ArgumentParser(description='记忆工程 v2.0 - 从相册提取记忆和画像')
    parser.add_argument('--photos', type=str, required=True, help='照片目录路径')
    parser.add_argument('--user-name', type=str, required=True, help='用户名')
    parser.add_argument('--max-photos', type=int, default=MAX_PHOTOS, help='最多处理多少张照片')
    parser.add_argument('--use-cache', action='store_true', help='使用VLM缓存（跳过VLM分析）')
    parser.add_argument('--gps-only', action='store_true', help='只处理含GPS信息的照片')
    parser.add_argument('--task-version', default=DEFAULT_TASK_VERSION, help='人脸链路版本')
    parser.add_argument('--reset-cache', action='store_true', help='处理前清空缓存（v1.x 缓存不兼容）')
    args = parser.parse_args()

    print("=" * 50)
    print("记忆工程 v2.0")
    print(f"人脸引擎: InsightFace/{FACE_MODEL_NAME}")
    print(f"person_id 格式: Person_001/Person_002...")
    print("=" * 50)

    if args.reset_cache:
        print("\n[重置] 清空缓存...")
        reset_cache()

    # ─── [1/9] 加载照片 ───────────────────────────────────────────
    print("\n[1/9] 加载照片...")
    image_processor = ImageProcessor()
    photos, load_errors = image_processor.load_photos_with_errors(args.photos, args.max_photos)
    print(f"  加载 {len(photos)} 张，失败 {len(load_errors)} 张")

    if args.gps_only:
        before = len(photos)
        photos = [p for p in photos if p.location]
        print(f"  GPS过滤: {before} → {len(photos)} 张")

    # ─── [2/9] 转换 HEIC ──────────────────────────────────────────
    print("\n[2/9] 转换 HEIC / 方向归一化 JPEG...")
    photos = image_processor.convert_to_jpeg(photos)

    # ─── [3/9] 去重（在人脸识别前） ──────────────────────────────
    print("\n[3/9] 照片去重...")
    before_dedup = len(photos)
    photos = image_processor.dedupe_before_face_recognition(photos)
    dedupe_report = image_processor.last_dedupe_report
    save_json(dedupe_report, DEDUP_REPORT_PATH)
    print(f"  去重完成：{before_dedup} → {len(photos)} 张（去除 {before_dedup - len(photos)} 张）")

    # ─── [4/9] 人脸识别 ──────────────────────────────────────────
    print("\n[4/9] 人脸识别（InsightFace + FAISS）...")
    face_rec = FaceRecognition(task_version=args.task_version)
    face_errors = []

    for i, photo in enumerate(photos):
        show_progress(i + 1, len(photos), "人脸识别")
        try:
            face_rec.process_photo(photo)
        except Exception as e:
            face_errors.append({
                "image_id": photo.photo_id,
                "filename": photo.filename,
                "step": "face_recognition",
                "error": str(e),
            })

    face_db = face_rec.get_all_persons()
    print(f"\n  识别了 {len(face_db)} 个人物，失败 {len(face_errors)} 张")

    # ─── [5/9] 主角推断 + 画框 ──────────────────────────────────
    print("\n[5/9] 主角推断并绘制人脸框...")
    face_rec.reorder_protagonist(photos)
    primary_person_id = face_rec.get_primary_person_id()
    face_db = face_rec.get_all_persons()

    if primary_person_id:
        primary_info = face_db.get(primary_person_id)
        count = primary_info.photo_count if hasattr(primary_info, 'photo_count') else 0
        print(f"  主角：{primary_person_id}（出现 {count} 次）")
    else:
        print("  主角：未稳定识别")

    boxed_count = 0
    for photo in photos:
        photo.primary_person_id = primary_person_id
        if photo.faces:
            boxed_path = image_processor.draw_face_boxes(photo)
            if boxed_path:
                photo.boxed_path = boxed_path
                boxed_count += 1
    print(f"  绘制了 {boxed_count} 张带框图片")

    face_rec.save()

    # ─── [6/9] 压缩照片 ──────────────────────────────────────────
    print("\n[6/9] 压缩照片...")
    photos = image_processor.preprocess(photos)
    print(f"  压缩完成")

    # ─── [7/9] VLM 分析 ──────────────────────────────────────────
    vlm = VLMAnalyzer()

    if args.use_cache and vlm.load_cache():
        print("\n[7/9] 使用VLM缓存（跳过VLM分析）...")
    else:
        print("\n[7/9] VLM分析（并发处理）...")
        vlm.analyze_photos_concurrent(photos, face_db, protagonist=primary_person_id, max_workers=2)
        vlm.save_cache()
        print("  VLM分析完成，结果已缓存")

    # ─── [8/9] LLM 处理 ──────────────────────────────────────────
    print("\n[8/9] LLM处理（事件提取、关系推断、画像生成）...")
    llm = LLMProcessor(primary_person_id=primary_person_id)

    pipeline_result = run_memory_pipeline_entry(
        llm=llm,
        photos=photos,
        face_db=face_db,
        vlm_results=vlm.results,
        primary_person_id=primary_person_id,
    )
    events = pipeline_result["events"]
    relationships = pipeline_result["relationships"]
    profile_result = pipeline_result["profile_result"]
    relationship_dossiers_path = pipeline_result["relationship_dossiers_path"]
    group_artifacts_path = pipeline_result["group_artifacts_path"]
    profile_fact_decisions_path = pipeline_result["profile_fact_decisions_path"]
    downstream_audit_report_path = pipeline_result["downstream_audit_report_path"]
    downstream_feedback_cases_path = pipeline_result["downstream_feedback_cases_path"]
    downstream_notification_status_path = pipeline_result["downstream_notification_status_path"]
    final_primary_person_id = pipeline_result.get("final_primary_person_id")
    profile_path = None
    structured_path = None
    profile_debug_path = save_profile_debug(profile_result, final_primary_person_id, len(events), len(relationships))
    relationships_path = save_relationships_output(relationships, final_primary_person_id)
    if profile_result:
        if profile_result.get("structured"):
            structured_path = save_structured_profile(profile_result["structured"])
            print(f"    结构化画像已保存: {structured_path}")
        if profile_result.get("report"):
            profile_path = save_profile_report(profile_result["report"])
            print(f"    画像报告已保存: {profile_path}")
        if not structured_path and not profile_path:
            print("    画像生成失败")
    else:
        print("    画像生成失败")

    # ─── [9/9] 保存结果 ──────────────────────────────────────────
    print("\n[9/9] 保存结果...")
    artifacts = build_artifacts_manifest(
        dedupe_report_path=DEDUP_REPORT_PATH,
        face_state_path=FACE_STATE_PATH,
        face_output_path=FACE_OUTPUT_PATH,
        vlm_cache_path=VLM_CACHE_PATH,
        relationships_path=relationships_path,
        relationship_dossiers_path=relationship_dossiers_path,
        group_artifacts_path=group_artifacts_path,
        structured_profile_path=structured_path,
        profile_report_path=profile_path,
        profile_debug_path=profile_debug_path,
        profile_fact_decisions_path=profile_fact_decisions_path,
        downstream_audit_report_path=downstream_audit_report_path,
        downstream_feedback_cases_path=downstream_feedback_cases_path,
        downstream_notification_status_path=downstream_notification_status_path,
        detailed_report_path=DETAILED_OUTPUT_PATH,
    )
    save_final_output(events, relationships, face_db, artifacts)
    save_detailed_report(events, relationships, face_db)

    print_simple_summary(events, relationships, artifacts, profile_path)


if __name__ == "__main__":
    main()
