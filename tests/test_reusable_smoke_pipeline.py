from __future__ import annotations

import json
import os
import runpy
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


def _write_reusable_case(base: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)

    (base / f"{base.name}_face_recognition_output.json").write_text(
        json.dumps(
            {
                "primary_person_id": "Person_004",
                "persons": [
                    {
                        "person_id": "Person_004",
                        "photo_count": 6,
                        "first_seen": "2026-03-01T10:00:00",
                        "last_seen": "2026-03-02T10:00:00",
                        "avg_score": 0.92,
                        "avg_quality": 0.85,
                        "label": "main",
                    },
                    {
                        "person_id": "Person_023",
                        "photo_count": 4,
                        "first_seen": "2026-03-01T10:00:00",
                        "last_seen": "2026-03-02T10:00:00",
                        "avg_score": 0.88,
                        "avg_quality": 0.81,
                        "label": "partner",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    (base / f"{base.name}_vlm_cache.json").write_text(
        json.dumps(
            {
                "metadata": {"total_photos": 2, "model": "gemini-2.0-flash"},
                "photos": [
                    {
                        "photo_id": "photo_001",
                        "filename": "photo_001.jpg",
                        "timestamp": "2026-03-01T10:00:00",
                        "location": {"name": "沈阳"},
                        "vlm_analysis": {
                            "summary": "2026年3月1日，Person_004【主角】与 Person_023 在公园自拍合影。",
                            "people": [
                                {
                                    "person_id": "Person_004",
                                    "appearance": "male",
                                    "clothing": "hoodie",
                                    "activity": "selfie",
                                    "interaction": "自拍",
                                    "contact_type": "selfie_together",
                                    "expression": "happy",
                                },
                                {
                                    "person_id": "Person_023",
                                    "appearance": "female",
                                    "clothing": "dress",
                                    "activity": "selfie",
                                    "interaction": "自拍",
                                    "contact_type": "selfie_together",
                                    "expression": "happy",
                                },
                            ],
                            "relations": [
                                {"subject": "Person_004", "relation": "自拍合影", "object": "Person_023"}
                            ],
                            "scene": {"location_detected": "公园", "location_type": "户外"},
                            "event": {"activity": "自拍", "social_context": "两人同行", "mood": "开心", "story_hints": []},
                            "details": ["情侣自拍"],
                        },
                    },
                    {
                        "photo_id": "photo_002",
                        "filename": "photo_002.jpg",
                        "timestamp": "2026-03-02T21:00:00",
                        "location": {"name": "沈阳"},
                        "vlm_analysis": {
                            "summary": "2026年3月2日，Person_004【主角】在卧室与 Person_023 亲密合影。",
                            "people": [
                                {
                                    "person_id": "Person_004",
                                    "appearance": "male",
                                    "clothing": "tshirt",
                                    "activity": "kiss",
                                    "interaction": "亲密合影",
                                    "contact_type": "kiss",
                                    "expression": "happy",
                                },
                                {
                                    "person_id": "Person_023",
                                    "appearance": "female",
                                    "clothing": "pajamas",
                                    "activity": "kiss",
                                    "interaction": "亲密合影",
                                    "contact_type": "kiss",
                                    "expression": "happy",
                                },
                            ],
                            "relations": [
                                {"subject": "Person_004", "relation": "亲吻", "object": "Person_023"}
                            ],
                            "scene": {"location_detected": "私人卧室", "location_type": "室内"},
                            "event": {"activity": "亲密互动", "social_context": "两人独处", "mood": "甜蜜", "story_hints": []},
                            "details": ["卧室亲密互动"],
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    (base / f"{base.name}_events.json").write_text(
        json.dumps(
            {
                "metadata": {"generated_at": "2026-03-25T00:00:00", "version": "legacy"},
                "events": [
                    {
                        "event_id": "event_001",
                        "date": "2026-03-01",
                        "time_range": "10:00 - 11:00",
                        "duration": "1h",
                        "title": "公园约会",
                        "type": "休闲",
                        "participants": ["Person_004", "Person_023"],
                        "location": "沈阳公园",
                        "description": "一起自拍散步。",
                        "photo_count": 2,
                        "confidence": 0.91,
                        "reason": "legacy event",
                        "narrative": "两人在公园约会。",
                        "social_interaction": {},
                        "lifestyle_tags": ["#约会"],
                        "social_slices": [],
                        "persona_evidence": {"behavioral": ["约会"]},
                    }
                ],
                "relationships": [
                    {
                        "person_id": "Person_023",
                        "relationship_type": "friend",
                        "intimacy_score": 0.35,
                        "status": "stable",
                        "confidence": 0.7,
                        "reason": "legacy relationship reference only",
                        "shared_events": [{"event_id": "event_001", "date": "2026-03-01", "narrative": "legacy"}],
                        "evidence": {"photo_count": 2},
                    }
                ],
                "face_db": {
                    "Person_004": {"photo_count": 999},
                    "Person_023": {"photo_count": 888},
                },
            }
        ),
        encoding="utf-8",
    )

    (base / f"{base.name}_profile_structured.json").write_text(
        json.dumps(
            {
                "long_term_facts": {
                    "social_identity": {
                        "education": {"value": "higher_education", "confidence": 0.7}
                    }
                },
                "short_term_facts": {},
                "long_term_expression": {},
                "short_term_expression": {},
            }
        ),
        encoding="utf-8",
    )

    (base / f"{base.name}_face_recognition_state.json").write_text(
        json.dumps({"primary_person_id": "Person_004", "persons": {"Person_004": {"photo_count": 6}}}),
        encoding="utf-8",
    )


class ReusableSmokeLoaderTests(unittest.TestCase):
    def test_loader_only_maps_face_vlm_and_events_into_runtime_state(self) -> None:
        from services.memory_pipeline.reusable_smoke_loader import load_reusable_smoke_case

        with TemporaryDirectory() as temp_dir:
            case_dir = Path(temp_dir) / "06_case"
            _write_reusable_case(case_dir)

            result = load_reusable_smoke_case(case_dir)

        self.assertEqual(result.fallback_primary_person_id, "Person_004")
        self.assertIsNone(result.state.primary_decision)
        self.assertEqual(result.state.relationships, [])
        self.assertEqual(len(result.state.events), 1)
        self.assertEqual(result.state.events[0].participants, ["Person_004", "Person_023"])
        self.assertEqual(result.reference_relationships[0]["person_id"], "Person_023")
        self.assertEqual(result.reference_profile_path, case_dir / "06_case_profile_structured.json")
        self.assertTrue(
            any(
                item["kind"] == "profile_structured" and item["reason"] == "reference_only_not_fed"
                for item in result.mapping_debug["ignored_reference_inputs"]
            )
        )
        self.assertTrue(
            any(
                item["key"] == "relationships" and item["reason"] == "reference_only_not_fed"
                for item in result.mapping_debug["ignored_embedded_payloads"]
            )
        )

    def test_loader_fails_fast_when_required_whitelist_input_is_missing(self) -> None:
        from services.memory_pipeline.reusable_smoke_loader import load_reusable_smoke_case

        with TemporaryDirectory() as temp_dir:
            case_dir = Path(temp_dir) / "06_case"
            _write_reusable_case(case_dir)
            (case_dir / "06_case_face_recognition_output.json").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "face_recognition_output"):
                load_reusable_smoke_case(case_dir)

    def test_loader_accepts_legacy_events_with_empty_participants(self) -> None:
        from services.memory_pipeline.reusable_smoke_loader import load_reusable_smoke_case

        with TemporaryDirectory() as temp_dir:
            case_dir = Path(temp_dir) / "06_case"
            _write_reusable_case(case_dir)

            events_path = case_dir / "06_case_events.json"
            payload = json.loads(events_path.read_text(encoding="utf-8"))
            payload["events"][0]["participants"] = []
            events_path.write_text(json.dumps(payload), encoding="utf-8")

            result = load_reusable_smoke_case(case_dir)

        self.assertEqual(result.state.events[0].participants, [])
        self.assertEqual(result.state.events[0].event_id, "event_001")


class ReusableSmokeRunnerTests(unittest.TestCase):
    def test_runner_writes_isolated_outputs_and_reference_only_comparison(self) -> None:
        from services.memory_pipeline.reusable_smoke_runner import run_reusable_smoke_pipeline

        class StubLLMProcessor:
            def __init__(self) -> None:
                self.primary_person_id = "Person_004"

            def _collect_relationship_evidence(self, person_id, vlm_results, events=None):
                default = {
                    "photo_count": 0,
                    "time_span_days": 0,
                    "recent_gap_days": 0,
                    "scenes": [],
                    "private_scene_ratio": 0.0,
                    "dominant_scene_ratio": 0.0,
                    "interaction_behavior": [],
                    "with_user_only": True,
                    "contact_types": [],
                    "rela_events": [],
                    "monthly_frequency": 0.0,
                    "trend_detail": {},
                    "co_appearing_persons": [],
                    "anomalies": [],
                }
                if person_id != "Person_023":
                    return default
                payload = dict(default)
                payload.update(
                    {
                        "photo_count": 4,
                        "time_span_days": 30,
                        "recent_gap_days": 1,
                        "scenes": ["公园", "私人卧室"],
                        "private_scene_ratio": 0.5,
                        "dominant_scene_ratio": 0.5,
                        "interaction_behavior": ["亲密互动"],
                        "with_user_only": True,
                        "contact_types": ["kiss", "selfie_together"],
                        "rela_events": [
                            {
                                "event_id": "event_001",
                                "date": "2026-03-01",
                                "title": "公园约会",
                                "location": "沈阳公园",
                                "description": "一起自拍散步。",
                                "participants": ["Person_004", "Person_023"],
                                "narrative_synthesis": "一起自拍散步。",
                                "social_dynamics": [],
                            }
                        ],
                        "monthly_frequency": 3.0,
                        "trend_detail": {"direction": "up"},
                    }
                )
                return payload

        with TemporaryDirectory() as temp_dir:
            case_dir = Path(temp_dir) / "06_case"
            _write_reusable_case(case_dir)

            with patch(
                "services.memory_pipeline.reusable_smoke_runner.inspect_profile_agent_runtime_health",
                return_value={"status": "ok"},
            ), patch(
                "services.memory_pipeline.reusable_smoke_runner.run_downstream_profile_agent_audit",
                side_effect=RuntimeError("profile_agent unavailable"),
            ) as audit_mock:
                result = run_reusable_smoke_pipeline(
                    case_dir=case_dir,
                    run_id="smoke1",
                    llm_processor=StubLLMProcessor(),
                )

            output_dir = case_dir / "temp_smoke_run_smoke1"
            self.assertEqual(Path(result["output_dir"]), output_dir)
            self.assertEqual(result["final_primary_person_id"], "Person_004")
            self.assertEqual(audit_mock.call_args.kwargs["album_id"], "reusable_smoke_06_case_smoke1")

            for filename in (
                "normalized_state_snapshot.json",
                "mapping_debug.json",
                "relationships.json",
                "structured_profile.json",
                "relationship_dossiers.json",
                "group_artifacts.json",
                "profile_fact_decisions.json",
                "downstream_audit_report.json",
                "comparison_summary.json",
                "comparison_diff.json",
                "memory_pipeline_run_trace.json",
            ):
                self.assertTrue((output_dir / filename).exists(), filename)

            comparison_summary = json.loads((output_dir / "comparison_summary.json").read_text(encoding="utf-8"))
            comparison_diff = json.loads((output_dir / "comparison_diff.json").read_text(encoding="utf-8"))
            mapping_debug = json.loads((output_dir / "mapping_debug.json").read_text(encoding="utf-8"))

            self.assertTrue(comparison_summary["reference_only_not_fed"])
            self.assertEqual(comparison_summary["old_relationship_count"], 1)
            self.assertGreaterEqual(comparison_summary["new_relationship_count"], 1)
            self.assertTrue(comparison_diff["metadata"]["reference_only_not_fed"])
            self.assertTrue(
                any(
                    item["kind"] == "profile_structured" and item["reason"] == "reference_only_not_fed"
                    for item in mapping_debug["ignored_reference_inputs"]
                )
            )
            self.assertEqual(
                result["run_trace_path"],
                str(output_dir / "memory_pipeline_run_trace.json"),
            )
            self.assertIn("memory/evolution/traces", result["run_trace_ledger_path"])

    def test_cli_forwards_args_to_isolated_runner(self) -> None:
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_reusable_smoke_test.py"

        with patch(
            "services.memory_pipeline.reusable_smoke_runner.run_reusable_smoke_pipeline",
            return_value={"output_dir": "/tmp/out", "final_primary_person_id": "Person_004"},
        ) as runner_mock:
            argv_backup = sys.argv
            try:
                sys.argv = [
                    str(script_path),
                    "--case-dir",
                    "/tmp/case",
                    "--run-id",
                    "smoke1",
                ]
                with self.assertRaises(SystemExit) as exit_ctx:
                    runpy.run_path(str(script_path), run_name="__main__")
                self.assertEqual(exit_ctx.exception.code, 0)
            finally:
                sys.argv = argv_backup

        self.assertEqual(runner_mock.call_args.kwargs["case_dir"], "/tmp/case")
        self.assertEqual(runner_mock.call_args.kwargs["run_id"], "smoke1")


class ReusableSmokeOpenRouterResolverTests(unittest.TestCase):
    def test_resolver_builds_openrouter_processor_from_key_file(self) -> None:
        from services.memory_pipeline.reusable_smoke_llm import resolve_reusable_smoke_llm_processor

        with TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            (repo_root / ".env").write_text("", encoding="utf-8")
            (repo_root / "open router key.md").write_text(
                "# keys\n\n## first\n\n`sk-or-v1-testkey1111`\n\n## second\n\n`sk-or-v1-testkey2222`\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                processor = resolve_reusable_smoke_llm_processor(
                    primary_person_id="Person_004",
                    repo_root=repo_root,
                )

        self.assertIsNotNone(processor)
        self.assertEqual(processor.primary_person_id, "Person_004")
        self.assertEqual(processor.api_key, "sk-or-v1-testkey1111")
        self.assertEqual(processor.base_url, "https://openrouter.ai/api/v1")
        self.assertTrue(hasattr(processor, "_collect_relationship_evidence"))

    def test_resolver_prefers_openrouter_env_key_over_key_file(self) -> None:
        from services.memory_pipeline.reusable_smoke_llm import resolve_reusable_smoke_llm_processor

        with TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            (repo_root / ".env").write_text("", encoding="utf-8")
            (repo_root / "open router key.md").write_text(
                "`sk-or-v1-testkey1111`\n",
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "OPENROUTER_API_KEY": "sk-or-v1-envpreferred9999",
                    "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
                    "PROFILE_LLM_MODEL": "google/gemini-3.1-flash-lite-preview",
                },
                clear=True,
            ):
                processor = resolve_reusable_smoke_llm_processor(
                    primary_person_id="Person_004",
                    repo_root=repo_root,
                )

        self.assertIsNotNone(processor)
        self.assertEqual(processor.api_key, "sk-or-v1-envpreferred9999")


if __name__ == "__main__":
    unittest.main()
