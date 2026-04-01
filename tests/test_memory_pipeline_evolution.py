from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class MemoryPipelineEvolutionTests(unittest.TestCase):
    def test_build_memory_run_trace_caps_completion_when_verification_fails(self) -> None:
        from services.memory_pipeline.evolution import build_memory_run_trace

        trace = build_memory_run_trace(
            run_type="mainline",
            user_name="vigar",
            stage_reports=[
                {"stage": "relationships", "status": "ok"},
                {"stage": "profile_lp3", "status": "ok"},
            ],
            downstream_audit_report={
                "metadata": {"audit_status": "skipped_init_failure"},
                "summary": {"rejected_count": 0, "not_audited_count": 2},
            },
            profile_llm_batch_debug=[
                {"used_offline_fallback": True, "fallback_reason": "parse_failure"},
                {"used_offline_fallback": False},
            ],
            test_issue_log={"summary": {"issue_count": 2, "high_risk_issue_count": 1}},
            artifacts={"structured_profile_path": "/tmp/structured_profile.json"},
            generated_at="2026-03-27T01:02:03",
        )

        self.assertFalse(trace["completion"]["verification_passed"])
        self.assertLessEqual(trace["scores"]["completion_score"], 0.59)
        self.assertLess(trace["scores"]["quality_score"], 0.7)
        self.assertEqual(trace["metrics"]["llm_fallback_batch_count"], 1)

    def test_persist_memory_run_trace_writes_output_and_user_scoped_ledger(self) -> None:
        from services.memory_pipeline.evolution import build_memory_run_trace, persist_memory_run_trace

        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            output_dir = project_root / "output" / "case_1"
            output_dir.mkdir(parents=True, exist_ok=True)

            payload = build_memory_run_trace(
                run_type="precomputed_bundle",
                user_name="vigar",
                stage_reports=[{"stage": "profile_lp3", "status": "ok"}],
                downstream_audit_report={"metadata": {"audit_status": "ok"}, "summary": {"rejected_count": 0}},
                profile_llm_batch_debug=[],
                test_issue_log={"summary": {"issue_count": 0, "high_risk_issue_count": 0}},
                artifacts={"downstream_audit_report_path": str(output_dir / "downstream_audit_report.json")},
                generated_at="2026-03-27T10:11:12",
            )
            result = persist_memory_run_trace(
                project_root=str(project_root),
                output_dir=str(output_dir),
                trace_payload=payload,
            )

            trace_json_path = Path(result["trace_json_path"])
            ledger_path = Path(result["trace_ledger_path"])
            self.assertTrue(trace_json_path.exists())
            self.assertTrue(ledger_path.exists())
            self.assertIn("memory/evolution/traces/vigar/2026-03-27.jsonl", str(ledger_path))

            lines = [line for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(lines), 1)
            stored = json.loads(lines[0])
            self.assertEqual(stored["user_name"], "vigar")
            self.assertEqual(stored["run_type"], "precomputed_bundle")

    def test_run_memory_nightly_evaluation_generates_insights_and_proposals(self) -> None:
        from services.memory_pipeline.evolution import (
            build_memory_run_trace,
            persist_memory_run_trace,
            run_memory_nightly_evaluation,
        )

        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            output_dir = project_root / "output" / "case_2"
            output_dir.mkdir(parents=True, exist_ok=True)

            low_quality = build_memory_run_trace(
                run_type="mainline",
                user_name="vigar",
                stage_reports=[{"stage": "profile_lp3", "status": "ok"}],
                downstream_audit_report={
                    "metadata": {"audit_status": "skipped_init_failure"},
                    "summary": {"rejected_count": 0, "not_audited_count": 3},
                },
                profile_llm_batch_debug=[{"used_offline_fallback": True, "fallback_reason": "exception"}],
                test_issue_log={"summary": {"issue_count": 1, "high_risk_issue_count": 1}},
                artifacts={},
                generated_at="2026-03-27T08:00:00",
            )
            persist_memory_run_trace(
                project_root=str(project_root),
                output_dir=str(output_dir),
                trace_payload=low_quality,
            )

            result = run_memory_nightly_evaluation(
                project_root=str(project_root),
                user_name="vigar",
                date_str="2026-03-27",
            )

            self.assertTrue(Path(result["report_path"]).exists())
            self.assertTrue(Path(result["insights_path"]).exists())
            self.assertTrue(Path(result["proposals_path"]).exists())

            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            insight_types = [item.get("type") for item in report.get("insights", [])]
            self.assertIn("downstream_audit_skipped", insight_types)
            self.assertGreaterEqual(len(report.get("proposals", [])), 1)

    def test_apply_memory_evolution_proposal_updates_rule_assets(self) -> None:
        from services.memory_pipeline.evolution import apply_memory_evolution_proposal
        from services.memory_pipeline.rule_asset_loader import ensure_rule_asset_files

        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            ensure_rule_asset_files(project_root=str(project_root))
            proposal = {
                "proposal_id": "prop_20260327_001",
                "user_name": "vigar",
                "proposal_type": "field_spec_patch",
                "patch_preview": {
                    "field_spec_overrides": {
                        "long_term_facts.social_identity.education": {
                            "null_preferred_when": [
                                "证据仅来自单事件窗口时输出 null",
                            ]
                        }
                    }
                },
            }

            result = apply_memory_evolution_proposal(
                project_root=str(project_root),
                proposal=proposal,
                actor="manual_test",
            )

            self.assertEqual(result["status"], "applied")
            rule_path = Path(result["asset_paths"]["field_specs_overrides_path"])
            saved = json.loads(rule_path.read_text(encoding="utf-8"))
            self.assertIn("long_term_facts.social_identity.education", saved)

            action_log = Path(result["proposal_actions_path"])
            self.assertTrue(action_log.exists())
            rows = [json.loads(line) for line in action_log.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(rows[-1]["proposal_id"], "prop_20260327_001")
            self.assertEqual(rows[-1]["actor"], "manual_test")

    def test_run_memory_nightly_evaluation_uses_gt_to_focus_top_three_fields(self) -> None:
        from services.memory_pipeline.evolution import (
            build_memory_run_trace,
            persist_memory_run_trace,
            run_memory_nightly_evaluation,
        )

        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            output_dir = project_root / "output" / "case_3"
            output_dir.mkdir(parents=True, exist_ok=True)
            trace = build_memory_run_trace(
                run_type="precomputed_bundle",
                user_name="alice",
                stage_reports=[{"stage": "profile_lp3", "status": "ok"}],
                downstream_audit_report={"metadata": {"audit_status": "ok"}, "summary": {"rejected_count": 0}},
                profile_llm_batch_debug=[],
                test_issue_log={"summary": {"issue_count": 0, "high_risk_issue_count": 0}},
                artifacts={"structured_profile_path": str(output_dir / "structured_profile.json")},
                generated_at="2026-03-27T12:00:00",
            )
            persist_memory_run_trace(
                project_root=str(project_root),
                output_dir=str(output_dir),
                trace_payload=trace,
            )

            reflection_dir = project_root / "memory" / "reflection"
            reflection_dir.mkdir(parents=True, exist_ok=True)
            gt_comparisons = [
                {
                    "field_key": "short_term_expression.motivation_shift",
                    "comparison_result": {
                        "grade": "mismatch",
                        "score": 0.0,
                        "severity": "high",
                        "output_value": "由兴趣导向向高端体验消费偏移",
                        "gt_value": "同步性",
                    },
                    "gt_payload": {"notes": "关键证据: EVT_0001、photo_001"},
                },
                {
                    "field_key": "long_term_facts.social_identity.education",
                    "comparison_result": {
                        "grade": "partial_match",
                        "score": 0.3,
                        "severity": "medium",
                        "output_value": "student",
                        "gt_value": "在读学生",
                    },
                    "gt_payload": {"notes": "关键证据: EVT_0002"},
                },
                {
                    "field_key": "long_term_facts.material.brand_preference",
                    "comparison_result": {
                        "grade": "mismatch",
                        "score": 0.0,
                        "severity": "high",
                        "output_value": "luxury",
                        "gt_value": "喜茶",
                    },
                    "gt_payload": {"notes": "关键证据: photo_010"},
                },
            ]
            (reflection_dir / "gt_comparisons_alice.jsonl").write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in gt_comparisons) + "\n",
                encoding="utf-8",
            )
            gt_rows = [
                {
                    "field_key": "short_term_expression.motivation_shift",
                    "gt_value": "同步性",
                    "original_confidence": 0.2,
                    "notes": "关键证据: EVT_0001",
                },
                {
                    "field_key": "long_term_facts.social_identity.education",
                    "gt_value": "在读学生",
                    "original_confidence": 0.4,
                    "notes": "关键证据: EVT_0002",
                },
                {
                    "field_key": "long_term_facts.material.brand_preference",
                    "gt_value": "喜茶",
                    "original_confidence": 0.3,
                    "notes": "关键证据: photo_010",
                },
                {
                    "field_key": "long_term_facts.identity.role",
                    "gt_value": "student",
                    "original_confidence": 0.1,
                    "notes": "观察字段",
                },
            ]
            (reflection_dir / "profile_field_gt_alice.jsonl").write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in gt_rows) + "\n",
                encoding="utf-8",
            )

            result = run_memory_nightly_evaluation(
                project_root=str(project_root),
                user_name="alice",
                date_str="2026-03-27",
                top_k_fields=3,
            )

            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(len(report.get("focus_fields", [])), 3)
            self.assertEqual(len(report.get("field_cycles", [])), 3)
            top_fields = [item.get("field_key") for item in report["focus_fields"]]
            self.assertIn("short_term_expression.motivation_shift", top_fields)
            self.assertIn("long_term_facts.material.brand_preference", top_fields)
            self.assertTrue(any(item.get("field_key") for item in report.get("proposals", [])))
            self.assertTrue(Path(result["focus_fields_path"]).exists())
            self.assertTrue(Path(result["field_cycles_path"]).exists())

    def test_run_memory_nightly_user_set_evaluation_supports_multiple_users(self) -> None:
        from services.memory_pipeline.evolution import (
            build_memory_run_trace,
            persist_memory_run_trace,
            run_memory_nightly_user_set_evaluation,
        )

        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            for user in ("u1", "u2"):
                output_dir = project_root / "output" / f"case_{user}"
                output_dir.mkdir(parents=True, exist_ok=True)
                trace = build_memory_run_trace(
                    run_type="precomputed_bundle",
                    user_name=user,
                    stage_reports=[{"stage": "profile_lp3", "status": "ok"}],
                    downstream_audit_report={"metadata": {"audit_status": "ok"}, "summary": {"rejected_count": 0}},
                    profile_llm_batch_debug=[],
                    test_issue_log={"summary": {"issue_count": 0, "high_risk_issue_count": 0}},
                    artifacts={"structured_profile_path": str(output_dir / "structured_profile.json")},
                    generated_at="2026-03-27T12:00:00",
                )
                persist_memory_run_trace(
                    project_root=str(project_root),
                    output_dir=str(output_dir),
                    trace_payload=trace,
                )

            result = run_memory_nightly_user_set_evaluation(
                project_root=str(project_root),
                user_names=["u1", "u2"],
                date_str="2026-03-27",
                top_k_fields=3,
            )
            self.assertEqual(result["total_users"], 2)
            self.assertEqual(len(result["users"]), 2)
            self.assertTrue(Path(result["report_path"]).exists())

    def test_run_memory_nightly_evaluation_skips_throttled_field_cycle(self) -> None:
        from services.memory_pipeline.evolution import (
            build_memory_run_trace,
            persist_memory_run_trace,
            run_memory_nightly_evaluation,
        )

        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            output_dir = project_root / "output" / "case_throttle"
            output_dir.mkdir(parents=True, exist_ok=True)
            trace = build_memory_run_trace(
                run_type="precomputed_bundle",
                user_name="bob",
                stage_reports=[{"stage": "profile_lp3", "status": "ok"}],
                downstream_audit_report={"metadata": {"audit_status": "ok"}, "summary": {"rejected_count": 0}},
                profile_llm_batch_debug=[],
                test_issue_log={"summary": {"issue_count": 0, "high_risk_issue_count": 0}},
                artifacts={"structured_profile_path": str(output_dir / "structured_profile.json")},
                generated_at="2026-03-27T12:00:00",
            )
            persist_memory_run_trace(
                project_root=str(project_root),
                output_dir=str(output_dir),
                trace_payload=trace,
            )

            reflection_dir = project_root / "memory" / "reflection"
            reflection_dir.mkdir(parents=True, exist_ok=True)
            field_key = "short_term_expression.motivation_shift"
            gt_comparisons = [
                {
                    "field_key": field_key,
                    "comparison_result": {
                        "grade": "mismatch",
                        "score": 0.0,
                        "severity": "high",
                        "output_value": "由兴趣导向向高端体验消费偏移",
                        "gt_value": "同步性",
                    },
                    "gt_payload": {"notes": "关键证据: EVT_0001"},
                }
            ]
            (reflection_dir / "gt_comparisons_bob.jsonl").write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in gt_comparisons) + "\n",
                encoding="utf-8",
            )
            gt_rows = [
                {
                    "field_key": field_key,
                    "gt_value": "同步性",
                    "original_confidence": 0.2,
                    "notes": "关键证据: EVT_0001",
                }
            ]
            (reflection_dir / "profile_field_gt_bob.jsonl").write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in gt_rows) + "\n",
                encoding="utf-8",
            )

            state_path = project_root / "memory" / "evolution" / "field_loop_state" / "bob.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps(
                    {
                        "fields": {
                            field_key: {
                                "cycle_count": 2,
                                "no_new_signal_streak": 2,
                                "cooldown_remaining": 2,
                                "seen_signal_keys": ["gt_token:同步性|evidence_ref:EVT_0001"],
                            }
                        }
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            result = run_memory_nightly_evaluation(
                project_root=str(project_root),
                user_name="bob",
                date_str="2026-03-27",
                top_k_fields=1,
            )

            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(len(report.get("field_cycles", [])), 1)
            cycle = report["field_cycles"][0]
            self.assertEqual(cycle["field_key"], field_key)
            self.assertEqual(cycle["cycle_status"], "throttled")
            self.assertFalse(cycle["new_signal_found"])
            self.assertFalse(
                any(item.get("field_key") == field_key for item in report.get("proposals", []))
            )

            persisted_state = json.loads(state_path.read_text(encoding="utf-8"))
            persisted_field = persisted_state["fields"][field_key]
            self.assertEqual(persisted_field["cooldown_remaining"], 1)
            self.assertEqual(persisted_field["no_new_signal_streak"], 2)

    def test_run_memory_nightly_evaluation_prioritizes_active_fields_over_throttled(self) -> None:
        from services.memory_pipeline.evolution import run_memory_nightly_evaluation

        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            reflection_dir = project_root / "memory" / "reflection"
            reflection_dir.mkdir(parents=True, exist_ok=True)
            throttled_field = "short_term_expression.motivation_shift"
            active_field = "long_term_facts.social_identity.education"
            gt_comparisons = [
                {
                    "field_key": throttled_field,
                    "comparison_result": {
                        "grade": "mismatch",
                        "score": 0.0,
                        "severity": "high",
                        "output_value": "偏消费动机",
                        "gt_value": "同步性",
                    },
                    "gt_payload": {"notes": "关键证据: EVT_001"},
                },
                {
                    "field_key": active_field,
                    "comparison_result": {
                        "grade": "partial_match",
                        "score": 0.4,
                        "severity": "medium",
                        "output_value": "student",
                        "gt_value": "在读学生",
                    },
                    "gt_payload": {"notes": "关键证据: EVT_002"},
                },
            ]
            (reflection_dir / "gt_comparisons_alex.jsonl").write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in gt_comparisons) + "\n",
                encoding="utf-8",
            )
            gt_rows = [
                {"field_key": throttled_field, "gt_value": "同步性", "original_confidence": 0.2},
                {"field_key": active_field, "gt_value": "在读学生", "original_confidence": 0.4},
            ]
            (reflection_dir / "profile_field_gt_alex.jsonl").write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in gt_rows) + "\n",
                encoding="utf-8",
            )

            state_path = project_root / "memory" / "evolution" / "field_loop_state" / "alex.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps(
                    {
                        "fields": {
                            throttled_field: {
                                "cycle_count": 3,
                                "no_new_signal_streak": 3,
                                "cooldown_remaining": 2,
                            }
                        }
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            result = run_memory_nightly_evaluation(
                project_root=str(project_root),
                user_name="alex",
                date_str="2026-03-27",
                top_k_fields=1,
            )
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(len(report.get("focus_fields", [])), 1)
            self.assertEqual(report["focus_fields"][0]["field_key"], active_field)

    def test_run_memory_nightly_evaluation_prioritizes_recent_gain_fields(self) -> None:
        from services.memory_pipeline.evolution import run_memory_nightly_evaluation

        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            reflection_dir = project_root / "memory" / "reflection"
            reflection_dir.mkdir(parents=True, exist_ok=True)
            stale_field = "short_term_expression.motivation_shift"
            gain_field = "long_term_facts.social_identity.education"
            gt_comparisons = [
                {
                    "field_key": stale_field,
                    "comparison_result": {
                        "grade": "mismatch",
                        "score": 0.0,
                        "severity": "high",
                        "output_value": "偏消费动机",
                        "gt_value": "同步性",
                    },
                    "gt_payload": {"notes": "关键证据: EVT_001"},
                },
                {
                    "field_key": gain_field,
                    "comparison_result": {
                        "grade": "partial_match",
                        "score": 0.4,
                        "severity": "medium",
                        "output_value": "student",
                        "gt_value": "在读学生",
                    },
                    "gt_payload": {"notes": "关键证据: EVT_002"},
                },
            ]
            (reflection_dir / "gt_comparisons_eva.jsonl").write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in gt_comparisons) + "\n",
                encoding="utf-8",
            )
            gt_rows = [
                {"field_key": stale_field, "gt_value": "同步性", "original_confidence": 0.2},
                {"field_key": gain_field, "gt_value": "在读学生", "original_confidence": 0.4},
            ]
            (reflection_dir / "profile_field_gt_eva.jsonl").write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in gt_rows) + "\n",
                encoding="utf-8",
            )

            state_path = project_root / "memory" / "evolution" / "field_loop_state" / "eva.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps(
                    {
                        "fields": {
                            stale_field: {
                                "cycle_count": 6,
                                "no_new_signal_streak": 5,
                                "cooldown_remaining": 0,
                                "last_status": "throttle_armed",
                            },
                            gain_field: {
                                "cycle_count": 2,
                                "no_new_signal_streak": 0,
                                "cooldown_remaining": 0,
                                "last_status": "new_insight_found",
                            },
                        }
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            result = run_memory_nightly_evaluation(
                project_root=str(project_root),
                user_name="eva",
                date_str="2026-03-27",
                top_k_fields=1,
            )
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(report["focus_fields"][0]["field_key"], gain_field)


if __name__ == "__main__":
    unittest.main()
