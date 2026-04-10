from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from models import Photo
from services.llm_processor import LLMProcessor
from services.vlm_analyzer import VLMAnalyzer
from services.v0325.lp3_core.profile_fields import _resolve_profile_llm_processor
from services.v0325.pipeline import V0325PipelineFamily


class StubV0325LLM:
    provider = "openrouter"
    model = "google/gemini-3.1-pro-preview"
    relationship_model = "anthropic.claude-opus-4-6-v1"

    def __init__(self) -> None:
        self.json_calls: list[dict] = []

    def _lp1_batch_payload(self, batch_id: str) -> dict:
        return {
            "events": [
                {
                    "event_id": "TEMP_EVT_001",
                    "supporting_photo_ids": ["photo_001", "photo_002", "photo_003"],
                    "meta_info": {
                        "title": "Breakfast Together",
                        "location_context": "Home",
                        "photo_count": 3,
                    },
                    "objective_fact": {
                        "scene_description": "Breakfast together at home.",
                        "participants": ["Person_001", "Person_002"],
                    },
                    "participant_person_ids": ["Person_001", "Person_002"],
                    "depicted_person_ids": ["Person_001", "Person_002"],
                    "narrative_synthesis": "Shared breakfast at home.",
                    "social_dynamics": [
                        {
                            "target_id": "Person_002",
                            "interaction_type": "meal",
                            "social_clue": "same table",
                            "relation_hypothesis": "friend",
                            "confidence": 0.81,
                        }
                    ],
                    "persona_evidence": {
                        "behavioral": ["hosts breakfast"],
                        "aesthetic": [],
                        "socioeconomic": [],
                    },
                    "tags": ["meal", "home"],
                    "confidence": 0.84,
                    "reason": f"batch={batch_id}",
                    "started_at": "2026-03-23T09:00:00",
                    "ended_at": "2026-03-23T09:45:00",
                }
            ]
        }

    def _call_json_prompt(self, prompt: str, **kwargs):
        self.json_calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
        if "你是相册主人识别专家" in prompt:
            return {
                "mode": "person_id",
                "primary_person_id": "Person_001",
                "confidence": 0.93,
                "reasoning": "自拍、事件频次和证件锚点都指向 Person_001。",
            }
        if "你是关系分析专家" in prompt:
            return {
                "relationship_type": "close_friend",
                "stability": "long_term",
                "status": "stable",
                "confidence": 84,
                "strength_summary": "稳定共同用餐与居家共现",
                "reasoning": "多次一起吃饭并在居家场景共同出现。",
            }
        if "你是结构化画像的字段裁决 agent。" in prompt:
            fields = {}
            if "long_term_facts.identity.name" in prompt:
                fields["long_term_facts.identity.name"] = {
                    "value": "Vigar",
                    "confidence": 0.76,
                    "reasoning": "identity anchor points to the primary subject",
                    "supporting_ref_ids": ["photo_001"],
                    "contradicting_ref_ids": ["photo_004"],
                    "null_reason": None,
                }
            if "long_term_facts.identity.role" in prompt:
                fields["long_term_facts.identity.role"] = {
                    "value": "student",
                    "confidence": 0.74,
                    "reasoning": "school-like cues appear in the evidence bundle",
                    "supporting_ref_ids": ["photo_002", "EVT_0001"],
                    "contradicting_ref_ids": [],
                    "null_reason": None,
                }
            if "long_term_facts.social_identity.education" in prompt:
                fields["long_term_facts.social_identity.education"] = {
                    "value": "college student",
                    "confidence": 0.72,
                    "reasoning": "campus cues and study context repeat",
                    "supporting_ref_ids": ["photo_002", "EVT_0001"],
                    "contradicting_ref_ids": [],
                    "null_reason": None,
                }
            if "long_term_facts.geography.location_anchors" in prompt:
                fields["long_term_facts.geography.location_anchors"] = {
                    "value": ["Home"],
                    "confidence": 0.78,
                    "reasoning": "recurring home location across selected observations",
                    "supporting_ref_ids": ["photo_001", "photo_002", "EVT_0001"],
                    "contradicting_ref_ids": [],
                    "null_reason": None,
                }
            if "long_term_facts.hobbies.frequent_activities" in prompt:
                fields["long_term_facts.hobbies.frequent_activities"] = {
                    "value": ["meal"],
                    "confidence": 0.68,
                    "reasoning": "meal activity repeats across selected evidence",
                    "supporting_ref_ids": ["photo_001", "photo_003", "EVT_0001"],
                    "contradicting_ref_ids": [],
                    "null_reason": None,
                }
            return {"fields": fields}
        return {
            "value": None,
            "confidence": 0.0,
            "reasoning": "",
            "supporting_ref_ids": [],
            "contradicting_ref_ids": [],
            "null_reason": None,
        }

    def _call_json_prompt_raw_text(self, prompt: str, **kwargs) -> str:
        self.json_calls.append({"prompt": prompt, "kwargs": dict(kwargs), "raw_text_mode": True})
        if "LP1 Batch Analysis Task" in prompt:
            return "Batch analysis memo with grounded event grouping for breakfast at home."
        if "Convert the following LP1 batch analysis into JSON." in prompt:
            return json.dumps(self._lp1_batch_payload("BATCH_0001"), ensure_ascii=False)
        return json.dumps(self._call_json_prompt(prompt, **kwargs), ensure_ascii=False)

    def _call_json_prompt_raw_response(self, prompt: str, **kwargs) -> dict:
        text = self._call_json_prompt_raw_text(prompt, **kwargs)
        finish_reason = "stop"
        if "Convert the following LP1 batch analysis into JSON." in prompt:
            finish_reason = "tool"
        return {
            "text": text,
            "provider_response_id": "resp_test_001",
            "provider_finish_reason": finish_reason,
            "provider_usage": {"total_tokens": 1234},
        }

    def _call_markdown_prompt(self, prompt: str) -> str:
        return "# 用户画像\n\n- 稳定居家早餐社交。"


class TruncatedConvertStubLLM(StubV0325LLM):
    def _call_json_prompt_raw_response(self, prompt: str, **kwargs) -> dict:
        self.json_calls.append({"prompt": prompt, "kwargs": dict(kwargs), "raw_response_mode": True})
        if "LP1 Batch Analysis Task" in prompt:
            return {
                "text": "Batch analysis memo with grounded events.",
                "provider_response_id": "resp_analysis_partial",
                "provider_finish_reason": "stop",
                "provider_usage": {"total_tokens": 800},
            }
        if "Convert the following LP1 batch analysis into JSON." in prompt:
            partial = {
                "events": [
                    {
                        "event_id": "TEMP_EVT_001",
                        "supporting_photo_ids": ["photo_001", "photo_002"],
                        "meta_info": {
                            "title": "Breakfast Together",
                            "location_context": "Home",
                            "photo_count": 2,
                        },
                        "objective_fact": {
                            "scene_description": "Breakfast together at home.",
                            "participants": ["Person_001", "Person_002"],
                        },
                        "participant_person_ids": ["Person_001", "Person_002"],
                        "depicted_person_ids": ["Person_001", "Person_002"],
                        "narrative_synthesis": "Shared breakfast at home.",
                        "social_dynamics": [],
                        "persona_evidence": {"behavioral": ["hosts breakfast"], "aesthetic": [], "socioeconomic": []},
                        "tags": ["meal", "home"],
                        "confidence": 0.84,
                        "reason": "grounded",
                    },
                    {
                        "event_id": "TEMP_EVT_002",
                        "supporting_photo_ids": ["photo_003"],
                        "meta_info": {"title": "broken tail"},
                    },
                ]
            }
            partial_text = json.dumps(partial, ensure_ascii=False)[:-2]
            return {
                "text": partial_text,
                "provider_response_id": "resp_convert_partial",
                "provider_finish_reason": "length",
                "provider_usage": {"total_tokens": 1600},
            }
        return super()._call_json_prompt_raw_response(prompt, **kwargs)


class V0325PipelineTests(unittest.TestCase):
    def _public_url(self, path: Path) -> str:
        return f"/assets/{path.relative_to(path.parent.parent) if path.parent.parent.exists() else path.name}"

    def _build_inputs(self) -> tuple[list[Photo], list[dict], dict]:
        photos: list[Photo] = []
        vlm_results: list[dict] = []
        face_images: list[dict] = []
        base_time = datetime(2026, 3, 23, 9, 0)
        for index in range(1, 5):
            photo_id = f"photo_{index:03d}"
            timestamp = base_time + timedelta(minutes=5 * (index - 1))
            photo = Photo(
                photo_id=photo_id,
                filename=f"{photo_id}.jpg",
                path=f"/tmp/{photo_id}.jpg",
                timestamp=timestamp,
                location={"name": "Home"},
                source_hash=f"hash-{photo_id}",
            )
            faces = [{"person_id": "Person_001"}, {"person_id": "Person_002"}] if index <= 3 else [{"person_id": "Person_001"}]
            photo.faces = faces
            photos.append(photo)
            vlm_results.append(
                {
                    "photo_id": photo_id,
                    "filename": photo.filename,
                    "timestamp": timestamp.isoformat(),
                    "location": {"name": "Home"},
                    "face_person_ids": [face["person_id"] for face in faces],
                    "vlm_analysis": {
                        "summary": (
                            "【主角】(Person_001) 自拍 in mirror at home with student id visible"
                            if index == 4
                            else f"summary {photo_id}"
                        ),
                        "scene": {"location_detected": "Home", "environment_description": "home scene"},
                        "event": {"activity": "meal", "social_context": "friend"},
                        "people": [
                            {"person_id": "Person_001"},
                            {"person_id": "Person_002", "contact_type": "standing_near"},
                        ]
                        if index <= 3
                        else [{"person_id": "Person_001"}],
                        "relations": [],
                        "details": ["cup", "table"],
                        "ocr_hits": ["student club"] if index == 2 else [],
                        "brands": ["Local Cafe"] if index == 3 else [],
                        "place_candidates": ["Home"],
                    },
                }
            )
            face_images.append(
                {
                    "image_id": photo_id,
                    "photo_id": photo_id,
                    "filename": photo.filename,
                    "source_hash": photo.source_hash,
                    "timestamp": timestamp.isoformat(),
                    "location": {"name": "Home"},
                    "faces": [
                        {
                            "face_id": f"{photo_id}-face-{face_index}",
                            "person_id": face["person_id"],
                            "bbox_xywh": {"x": 10 * face_index, "y": 10, "w": 40, "h": 40},
                            "quality_score": 0.9,
                            "match_decision": "strong_match",
                        }
                        for face_index, face in enumerate(faces, start=1)
                    ],
                }
            )
        face_output = {
            "primary_person_id": "Person_001",
            "persons": [
                {
                    "person_id": "Person_001",
                    "label": "Primary",
                    "photo_count": 4,
                    "first_seen": base_time.isoformat(),
                    "last_seen": (base_time + timedelta(minutes=15)).isoformat(),
                    "avg_score": 0.96,
                    "avg_quality": 0.92,
                },
                {
                    "person_id": "Person_002",
                    "label": "Close Friend",
                    "photo_count": 3,
                    "first_seen": base_time.isoformat(),
                    "last_seen": (base_time + timedelta(minutes=10)).isoformat(),
                    "avg_score": 0.91,
                    "avg_quality": 0.88,
                },
            ],
            "images": face_images,
        }
        return photos, vlm_results, face_output

    def test_v0325_pipeline_generates_snapshot_and_raw_artifacts(self) -> None:
        photos, vlm_results, face_output = self._build_inputs()
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_dir = Path(tmp_dir)
            cache_dir = task_dir / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "face_recognition_output.json").write_text(json.dumps(face_output, ensure_ascii=False), encoding="utf-8")
            (cache_dir / "face_recognition_state.json").write_text(json.dumps({"primary_person_id": "Person_001"}, ensure_ascii=False), encoding="utf-8")
            (cache_dir / "vlm_cache.json").write_text(
                json.dumps({"photos": vlm_results}, ensure_ascii=False),
                encoding="utf-8",
            )
            (cache_dir / "dedupe_report.json").write_text(
                json.dumps({"retained_images": 4, "duplicate_backrefs": {}}, ensure_ascii=False),
                encoding="utf-8",
            )

            family = V0325PipelineFamily(
                task_id="task_v0325",
                task_dir=task_dir,
                user_id="user_1",
                asset_store=None,
                llm_processor=StubV0325LLM(),
                public_url_builder=lambda path: f"/assets/{Path(path).relative_to(task_dir).as_posix()}",
            )
            result = family.run(
                photos=photos,
                face_output=face_output,
                primary_person_id="Person_001",
                vlm_results=vlm_results,
                cached_photo_ids={"photo_004"},
                dedupe_report={"retained_images": 4},
            )

            self.assertEqual(result["pipeline_family"], "v0325")
            self.assertTrue(result["summary"]["no_drop_guarantee"])
            self.assertGreaterEqual(result["summary"]["raw_attachment_count"], 5)
            self.assertEqual(result["summary"]["primary_generation_mode"], "rejudged")
            self.assertEqual(len(result["lp2_relationships"]), 1)
            relationship = result["lp2_relationships"][0]
            self.assertEqual(relationship["person_id"], "Person_002")
            self.assertIn("relationship_type", relationship)
            self.assertIn("intimacy_score", relationship)
            self.assertIn("status", relationship)
            self.assertIn("confidence", relationship)
            self.assertIn("reasoning", relationship)
            self.assertIn("evidence", relationship)
            self.assertIn("photo_ids", relationship["evidence"])
            self.assertIn("event_ids", relationship["evidence"])
            self.assertEqual(result["lp1_events"][0]["participants"], ["Person_001", "Person_002"])
            self.assertEqual(result["lp1_events"][0]["evidence_photos"], ["photo_001", "photo_002", "photo_003"])
            self.assertIn("events", result["lp3_profile"])
            self.assertIn("relationships", result["lp3_profile"])
            self.assertIn("report", result["lp3_profile"])
            self.assertIn("debug", result["lp3_profile"])
            self.assertTrue(result["lp3_profile"]["report_markdown"])
            self.assertNotIn("field_decisions", result["lp3_profile"])
            self.assertEqual(
                set(result["lp3_profile"]["structured"]["long_term_facts"]["identity"]["name"]["evidence"].keys()),
                {
                    "photo_ids",
                    "event_ids",
                    "person_ids",
                    "group_ids",
                    "feature_names",
                    "supporting_ref_count",
                    "contradicting_ref_count",
                    "constraint_notes",
                    "summary",
                },
            )
            name_evidence = result["lp3_profile"]["structured"]["long_term_facts"]["identity"]["name"]["evidence"]
            self.assertEqual(name_evidence["photo_ids"], ["photo_001"])
            self.assertEqual(name_evidence["contradicting_ref_count"], 1)
            self.assertEqual(name_evidence["supporting_ref_count"], 1)
            self.assertEqual(
                result["lp3_profile"]["internal_artifacts"]["downstream_audit_report_path"],
                "v0325/downstream_audit_report.json",
            )
            self.assertEqual(
                result["lp3_profile"]["internal_artifacts"]["profile_fact_decisions_path"],
                "v0325/profile_fact_decisions.json",
            )
            self.assertEqual(
                result["lp3_profile"]["internal_artifacts"]["structured_profile_path"],
                "v0325/structured_profile.json",
            )
            self.assertTrue((task_dir / "v0325" / "raw_upstream_manifest.json").exists())
            self.assertTrue((task_dir / "v0325" / "raw_upstream_index.json").exists())
            self.assertTrue((task_dir / "v0325" / "structured_profile.json").exists())
            self.assertTrue((task_dir / "v0325" / "profile_fact_decisions.json").exists())
            self.assertTrue((task_dir / "v0325" / "downstream_audit_report.json").exists())
            self.assertTrue((task_dir / "v0325" / "memory_snapshot.json").exists())
            self.assertTrue((task_dir / "v0325" / "lp1_events_raw.json").exists())

            raw_manifest = json.loads((task_dir / "v0325" / "raw_upstream_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(raw_manifest["summary"]["no_drop_guarantee"])
            self.assertTrue(any(item["attachment_key"] == "raw_face_output" for item in raw_manifest["attachments"]))
            self.assertTrue(any(item["attachment_key"] == "raw_lp1_events" for item in raw_manifest["attachments"]))

            request_rows = (task_dir / "v0325" / "lp1_batch_requests.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertTrue(request_rows)
            first_request = json.loads(request_rows[0])
            self.assertIn("output_window_photo_ids", first_request)
            self.assertNotIn("new_region_photo_ids", first_request)

            attempt_rows = (task_dir / "v0325" / "lp1_batch_outputs.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertTrue(attempt_rows)
            first_attempt = json.loads(attempt_rows[0])
            self.assertEqual(first_attempt["analysis_finish_reason"], "stop")
            self.assertEqual(first_attempt["convert_finish_reason"], "tool")
            self.assertEqual(first_attempt["analysis_response_id"], "resp_test_001")
            self.assertEqual(first_attempt["convert_response_id"], "resp_test_001")
            self.assertEqual(first_attempt["contract_version"], "v0325.lp1.output_window.v1")

            lp2_artifact = json.loads((task_dir / "v0325" / "lp2_relationships.json").read_text(encoding="utf-8"))
            self.assertEqual(lp2_artifact["metadata"]["primary_person_id"], "Person_001")
            self.assertEqual(lp2_artifact["metadata"]["total_relationships"], 1)
            self.assertEqual(lp2_artifact["relationships"][0]["person_id"], "Person_002")
            self.assertNotIn("supporting_photo_ids", lp2_artifact["relationships"][0])

            structured_artifact = json.loads((task_dir / "v0325" / "structured_profile.json").read_text(encoding="utf-8"))
            self.assertIn("metadata", structured_artifact)
            self.assertIn("structured_profile", structured_artifact)

            decisions_artifact = json.loads((task_dir / "v0325" / "profile_fact_decisions.json").read_text(encoding="utf-8"))
            self.assertIn("metadata", decisions_artifact)
            self.assertIn("profile_fact_decisions", decisions_artifact)
            self.assertIn("reasoning", decisions_artifact["profile_fact_decisions"][0]["draft"])

    def test_v0325_pipeline_can_resume_from_precomputed_vp1_and_lp1(self) -> None:
        photos, vlm_results, face_output = self._build_inputs()
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_dir = Path(tmp_dir)
            cache_dir = task_dir / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "face_recognition_output.json").write_text(json.dumps(face_output, ensure_ascii=False), encoding="utf-8")
            (cache_dir / "face_recognition_state.json").write_text(
                json.dumps({"primary_person_id": "Person_001"}, ensure_ascii=False),
                encoding="utf-8",
            )
            (cache_dir / "vlm_cache.json").write_text(
                json.dumps({"photos": vlm_results}, ensure_ascii=False),
                encoding="utf-8",
            )
            (cache_dir / "dedupe_report.json").write_text(
                json.dumps({"retained_images": 4, "duplicate_backrefs": {}}, ensure_ascii=False),
                encoding="utf-8",
            )

            bootstrap_family = V0325PipelineFamily(
                task_id="task_bootstrap",
                task_dir=task_dir,
                user_id="user_1",
                asset_store=None,
                llm_processor=StubV0325LLM(),
                public_url_builder=lambda path: f"/assets/{Path(path).relative_to(task_dir).as_posix()}",
            )
            observations = bootstrap_family._build_vp1_observations(photos, vlm_results)
            lp1_events = StubV0325LLM()._lp1_batch_payload("BATCH_0001")["events"]

            family = V0325PipelineFamily(
                task_id="task_precomputed",
                task_dir=task_dir,
                user_id="user_1",
                asset_store=None,
                llm_processor=StubV0325LLM(),
                public_url_builder=lambda path: f"/assets/{Path(path).relative_to(task_dir).as_posix()}",
            )
            result = family.run_from_precomputed(
                observations=observations,
                face_output=face_output,
                primary_person_id="Person_001",
                cached_photo_ids=["photo_004"],
                dedupe_report={"retained_images": 4},
                lp1_events=lp1_events,
            )

            self.assertEqual(result["pipeline_family"], "v0325")
            self.assertEqual(len(result["lp2_relationships"]), 1)
            self.assertTrue(result["lp3_profile"]["report_markdown"])
            lp1_events_on_disk = json.loads((task_dir / "v0325" / "lp1_events_compact.json").read_text(encoding="utf-8"))
            self.assertEqual(len(lp1_events_on_disk), 1)
            self.assertEqual(lp1_events_on_disk[0]["event_id"], "TEMP_EVT_001")
            self.assertEqual(lp1_events_on_disk[0]["participants"], ["Person_001", "Person_002"])
            self.assertTrue((task_dir / "v0325" / "lp1_events_raw.json").exists())
            parse_failures = json.loads((task_dir / "v0325" / "lp1_parse_failures.json").read_text(encoding="utf-8"))
            self.assertEqual(parse_failures, [])
            self.assertFalse((task_dir / "v0325" / "lp1_batch_requests.jsonl").exists())
            self.assertFalse((task_dir / "v0325" / "lp1_batch_outputs.jsonl").exists())

    @patch("services.llm_processor.OPENROUTER_API_KEY", "test-openrouter-key")
    @patch("services.vlm_analyzer.OPENROUTER_API_KEY", "test-openrouter-key")
    @patch("services.llm_processor.BEDROCK_RELATIONSHIP_LLM_MODEL", "anthropic.claude-opus-4-6-v1")
    @patch("services.v0325.lp3_core.profile_fields.PROFILE_LLM_PROVIDER", "bedrock")
    @patch("services.v0325.lp3_core.profile_fields.PROFILE_LLM_MODEL", "anthropic.claude-opus-4-6-v1")
    def test_v0325_models_split_lp1_vs_lp2_lp3(self) -> None:
        llm = LLMProcessor(task_version="v0325")
        vlm = VLMAnalyzer(task_version="v0325")
        profile_llm = _resolve_profile_llm_processor({})

        self.assertEqual(llm.provider, "openrouter")
        self.assertEqual(llm.model, "google/gemini-3.1-pro-preview")
        self.assertEqual(llm.relationship_provider, "bedrock")
        self.assertEqual(llm.relationship_model, "anthropic.claude-opus-4-6-v1")
        self.assertEqual(vlm.provider, "openrouter")
        self.assertEqual(vlm.model, "google/gemini-3.1-pro-preview")
        self.assertIsNotNone(profile_llm)
        self.assertEqual(profile_llm.provider, "bedrock")
        self.assertEqual(profile_llm.model, "anthropic.claude-opus-4-6-v1")

    def test_v0325_lp1_prompt_uses_output_window_and_hard_contract(self) -> None:
        photos, vlm_results, face_output = self._build_inputs()
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_dir = Path(tmp_dir)
            family = V0325PipelineFamily(
                task_id="task_prompt",
                task_dir=task_dir,
                user_id="user_1",
                asset_store=None,
                llm_processor=StubV0325LLM(),
                public_url_builder=lambda path: f"/assets/{Path(path).relative_to(task_dir).as_posix()}",
            )
            observations = family._build_vp1_observations(photos, vlm_results)
            family.observation_index = {item["photo_id"]: item for item in observations}
            family.photo_order_index = {item["photo_id"]: int(item["sequence_index"]) for item in observations}
            batch = family._build_batches(observations)[0]
            prompt, request_record = family._build_lp1_batch_prompt(batch=batch, carryover_cards=[])
            convert_prompt = family._build_lp1_convert_prompt(batch=batch, analysis_text="memo")

            self.assertIn("OUTPUT_WINDOW_PHOTOS", prompt)
            self.assertIn("HARD_OUTPUT_CONTRACT", prompt)
            self.assertNotIn("NEW_REGION_PHOTOS", prompt)
            self.assertIn("OUTPUT_WINDOW_PHOTO_IDS", convert_prompt)
            self.assertIn("Do not stop mid-object, mid-array, or mid-string.", convert_prompt)
            self.assertIn("output_window_photo_ids", request_record)
            self.assertNotIn("new_region_photo_ids", request_record)

    def test_v0325_lp1_partial_salvage_writes_debug_artifacts_and_fails_batch(self) -> None:
        photos, vlm_results, _ = self._build_inputs()
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_dir = Path(tmp_dir)
            family = V0325PipelineFamily(
                task_id="task_salvage",
                task_dir=task_dir,
                user_id="user_1",
                asset_store=None,
                llm_processor=TruncatedConvertStubLLM(),
                public_url_builder=lambda path: f"/assets/{Path(path).relative_to(task_dir).as_posix()}",
            )
            observations = family._build_vp1_observations(photos, vlm_results)
            family.observation_index = {item["photo_id"]: item for item in observations}
            family.photo_order_index = {item["photo_id"]: int(item["sequence_index"]) for item in observations}

            with self.assertRaises(RuntimeError):
                family._run_lp1_batches(observations)

            salvage_report = json.loads((task_dir / "v0325" / "lp1_salvage_report.json").read_text(encoding="utf-8"))
            salvaged_events = (task_dir / "v0325" / "lp1_salvaged_events.jsonl").read_text(encoding="utf-8").strip().splitlines()
            parse_failures = json.loads((task_dir / "v0325" / "lp1_parse_failures.json").read_text(encoding="utf-8"))
            attempt_rows = (task_dir / "v0325" / "lp1_batch_outputs.jsonl").read_text(encoding="utf-8").strip().splitlines()

            self.assertTrue(salvage_report["summary"]["salvage_detected"])
            self.assertEqual(salvage_report["summary"]["salvaged_event_count"], 2)
            self.assertEqual(len(salvaged_events), 2)
            self.assertEqual(parse_failures[0]["salvage_status"], "detected")
            self.assertEqual(parse_failures[0]["salvaged_event_count"], 2)
            self.assertFalse((task_dir / "v0325" / "lp1_events_compact.json").exists())
            last_attempt = json.loads(attempt_rows[-1])
            self.assertEqual(last_attempt["salvage_status"], "detected")
            self.assertEqual(last_attempt["convert_finish_reason"], "length")
