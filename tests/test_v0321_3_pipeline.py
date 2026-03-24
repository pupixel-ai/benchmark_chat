from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from datetime import datetime, timedelta
from pathlib import Path

from models import Photo
from services.v0321_3.pipeline import V03213PipelineFamily
from services.v0321_3.retrieval_shadow import (
    build_memory_evidence_v2,
    build_memory_units_v2,
    build_profile_truth_v1,
)


class NullAssetStore:
    enabled = False
    bucket = None

    def upload_file(self, *args, **kwargs):
        return None

    def asset_url(self, *args, **kwargs):
        return None

    def sync_task_directory(self, *args, **kwargs):
        return None


class V03213PipelineFamilyTests(unittest.TestCase):
    def _build_family(self, task_dir: Path) -> V03213PipelineFamily:
        return V03213PipelineFamily(
            task_id="task_v03213_test",
            task_dir=task_dir,
            user_id="user_v03213_test",
            asset_store=NullAssetStore(),
            llm_processor=object(),
        )

    def test_unique_dedupes_object_wrapped_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            family = self._build_family(Path(tmpdir))

            values = family._unique(
                [
                    {"name": "Home"},
                    {"name": "Home"},
                    {"label": "Cafe"},
                    {"value": "Cafe"},
                    {"text": "Bookstore"},
                ]
            )

            self.assertEqual(values, ["Home", "Cafe", "Bookstore"])

    def test_normalize_llm_event_draft_coerces_object_wrapped_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            family = self._build_family(Path(tmpdir))
            fallback = {
                "title": "meal @ Home",
                "participant_person_ids": ["Person_001", "Person_002"],
                "depicted_person_ids": ["Person_003"],
                "place_refs": ["Home"],
                "confidence": 0.68,
                "atomic_evidence": [],
            }
            payload = {
                "title": "Breakfast",
                "summary": "Breakfast at home",
                "participant_person_ids": [{"person_id": "Person_001"}, {"id": "Person_999"}],
                "depicted_person_ids": [{"person_id": "Person_003"}],
                "place_refs": [{"name": "Home"}, {"name": "Elsewhere"}],
                "confidence": 0.9,
                "atomic_evidence": [
                    {
                        "evidence_type": "ocr",
                        "value_or_text": "Cafe receipt",
                        "photo_ids": [{"photo_id": "photo_001"}],
                        "confidence": 0.7,
                        "provenance": "ocr",
                    }
                ],
            }
            observations_by_photo = {
                "photo_001": {
                    "original_photo_ids": ["orig_001"],
                }
            }
            window = {"photo_ids": ["photo_001"]}

            normalized = family._normalize_llm_event_draft(
                payload=payload,
                fallback=fallback,
                observations_by_photo=observations_by_photo,
                window=window,
            )

            self.assertIsNotNone(normalized)
            assert normalized is not None
            self.assertEqual(normalized["participant_person_ids"], ["Person_001"])
            self.assertEqual(normalized["depicted_person_ids"], ["Person_003"])
            self.assertEqual(normalized["place_refs"], ["Home"])
            self.assertEqual(normalized["atomic_evidence"][0]["original_photo_ids"], ["orig_001"])

    def test_match_event_draft_handles_object_wrapped_overlap_signals(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            family = self._build_family(Path(tmpdir))
            draft = {
                "participant_person_ids": ["Person_001"],
                "place_refs": ["Home"],
                "started_at": "2026-03-15T09:40:00",
                "ended_at": "2026-03-15T10:00:00",
            }
            candidate = {
                "participant_person_ids": [{"person_id": "Person_001"}],
                "place_refs": [{"name": "Home"}],
                "started_at": "2026-03-15T09:00:00",
                "ended_at": "2026-03-15T09:30:00",
            }

            decision, matched = family._match_event_draft(draft, [candidate])

            self.assertEqual(decision, "merge")
            self.assertEqual(matched, candidate)

    def test_build_bursts_merges_short_same_place_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            family = self._build_family(Path(tmpdir))
            base = datetime(2026, 3, 23, 10, 0, 0)
            assets = [
                {
                    "photo_id": "photo_1",
                    "timestamp": base.isoformat(),
                    "place_key": "Cafe_A",
                },
                {
                    "photo_id": "photo_2",
                    "timestamp": (base + timedelta(minutes=4)).isoformat(),
                    "place_key": "Cafe_A",
                },
                {
                    "photo_id": "photo_3",
                    "timestamp": (base + timedelta(minutes=8)).isoformat(),
                    "place_key": "Cafe_A",
                },
            ]
            observations_by_photo = {
                "photo_1": {
                    "place_candidates": ["Cafe_A"],
                    "activity_hint": "meal",
                    "scene_hint": "cafe interior",
                },
                "photo_2": {
                    "place_candidates": ["Cafe_A"],
                    "activity_hint": "meal",
                    "scene_hint": "cafe interior",
                },
                "photo_3": {
                    "place_candidates": ["Cafe_A"],
                    "activity_hint": "meal",
                    "scene_hint": "cafe interior",
                },
            }
            appearances = [
                {"photo_id": "photo_1", "person_id": "Person_001", "appearance_mode": "live_presence"},
                {"photo_id": "photo_2", "person_id": "Person_001", "appearance_mode": "live_presence"},
            ]

            bursts = family._build_bursts(
                assets,
                observations_by_photo=observations_by_photo,
                appearances=appearances,
            )

            self.assertEqual(len(bursts), 1)
            self.assertEqual(bursts[0]["photo_ids"], ["photo_1", "photo_2", "photo_3"])

    def test_boundaries_keep_strong_continuity_as_same_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            family = self._build_family(Path(tmpdir))
            base = datetime(2026, 3, 23, 10, 0, 0)
            assets = [
                {
                    "photo_id": "photo_1",
                    "timestamp": base.isoformat(),
                    "place_key": "Cafe_A",
                },
                {
                    "photo_id": "photo_2",
                    "timestamp": (base + timedelta(minutes=24)).isoformat(),
                    "place_key": "Cafe_A",
                },
            ]
            observations_by_photo = {
                "photo_1": {
                    "place_candidates": ["Cafe_A"],
                    "activity_hint": "meal",
                    "scene_hint": "cafe interior",
                },
                "photo_2": {
                    "place_candidates": ["Cafe_A"],
                    "activity_hint": "meal",
                    "scene_hint": "cafe interior",
                },
            }
            appearances = [
                {"photo_id": "photo_1", "person_id": "Person_001", "appearance_mode": "live_presence"},
                {"photo_id": "photo_2", "person_id": "Person_001", "appearance_mode": "live_presence"},
            ]

            bursts = family._build_bursts(
                assets,
                observations_by_photo=observations_by_photo,
                appearances=appearances,
            )
            boundaries = family._score_boundaries(bursts)
            windows = family._build_event_windows(bursts, boundaries)

            self.assertEqual(len(bursts), 2)
            self.assertEqual(len(boundaries), 1)
            self.assertEqual(boundaries[0]["decision"], "continue")
            self.assertEqual(len(windows), 1)
            self.assertFalse(family._window_requires_event_finalize(windows[0]))

    def test_window_requires_finalize_when_multi_burst_signals_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            family = self._build_family(Path(tmpdir))
            window = {
                "burst_ids": ["burst_1", "burst_2", "burst_3"],
                "boundary_reason": "continue",
                "place_candidates": ["Cafe_A", "Office_B"],
                "activity_hints": ["meal", "meeting"],
                "scene_hints": ["cafe interior", "meeting room"],
            }

            self.assertTrue(family._window_requires_event_finalize(window))

    def test_build_photo_observation_packet_consumes_new_vlm_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            family = self._build_family(Path(tmpdir))
            photo = Photo(
                photo_id="photo_001",
                filename="img.jpg",
                path="/tmp/img.jpg",
                timestamp=datetime(2026, 3, 23, 18, 30, 0),
                location={"name": "校园咖啡店"},
                source_hash="src_001",
            )
            item = {
                "photo_id": "photo_001",
                "vlm_analysis": {
                    "summary": "傍晚【主角】在校园咖啡店与 Person_002 自拍合影。",
                    "people": [
                        {
                            "person_id": "Person_002",
                            "appearance": "青年女性，长发",
                            "clothing": "灰色连帽卫衣",
                            "activity": "举着饮品自拍",
                            "interaction": "与【主角】肩并肩自拍",
                            "contact_type": "selfie_together",
                            "expression": "微笑",
                        }
                    ],
                    "relations": [
                        {"subject": "Person_002", "relation": "holding", "object": "奶茶杯"},
                        {"subject": "奶茶杯", "relation": "placed_on", "object": "木质桌子"},
                    ],
                    "scene": {
                        "environment_details": ["木质桌子", "落地窗", "暖色灯光"],
                        "location_detected": "校园咖啡店",
                        "location_type": "室内",
                    },
                    "event": {
                        "activity": "喝咖啡",
                        "social_context": "和朋友",
                        "mood": "轻松",
                        "story_hints": ["可能是课间短暂休息"],
                    },
                    "details": ["学生证 学号 20261234", "WPS AI"],
                },
            }

            packet = family._build_photo_observation_packet(item, photo=photo)

            self.assertIn("校园咖啡店", packet["scene_hint"])
            self.assertIn("selfie_together", " ".join(packet["interaction_clues"]))
            self.assertIn("奶茶杯", " ".join(packet["object_clues"]))
            self.assertIn("学生证 学号 20261234", packet["detail_clues"])
            self.assertEqual(packet["location_type"], "室内")

            evidence = family._event_evidence_for_window(
                {"photo_ids": ["photo_001"]},
                {"photo_001": packet},
            )
            evidence_types = [item["evidence_type"] for item in evidence]
            self.assertIn("person_interaction", evidence_types)
            self.assertIn("object_last_seen", evidence_types)

    def test_profile_input_pack_extracts_identity_and_lifestyle_signals(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            family = self._build_family(Path(tmpdir))
            event_revisions = [
                {
                    "event_revision_id": "evt_student",
                    "title": "学生证与PLC教学实验台记录",
                    "event_summary": "学生证、学号、PLC 教学实验台接线练习与课程记录",
                    "started_at": "2026-03-20T10:00:00",
                    "ended_at": "2026-03-20T10:20:00",
                    "participant_person_ids": ["Person_002"],
                    "depicted_person_ids": [],
                    "place_refs": ["{'name': '安徽芜湖镜湖区', 'confidence': 0.75}"],
                    "original_photo_ids": ["orig_evt_student"],
                    "atomic_evidence": [
                        {"evidence_type": "ocr", "value_or_text": "姓名 陈美伊", "original_photo_ids": ["orig_evt_student"]},
                        {"evidence_type": "ocr", "value_or_text": "学号 20261234", "original_photo_ids": ["orig_evt_student"]},
                    ],
                },
                {
                    "event_revision_id": "evt_life",
                    "title": "茶饮店用餐与取餐记录",
                    "event_summary": "茶饮取餐、用餐和轻社交场景",
                    "started_at": "2026-03-20T18:00:00",
                    "ended_at": "2026-03-20T18:20:00",
                    "participant_person_ids": ["Person_002"],
                    "depicted_person_ids": [],
                    "place_refs": ["{'name': '茶饮店', 'confidence': 0.6}"],
                    "original_photo_ids": ["orig_evt_life"],
                    "atomic_evidence": [
                        {"evidence_type": "object_last_seen", "value_or_text": "茶饮杯", "original_photo_ids": ["orig_evt_life"]},
                    ],
                },
                {
                    "event_revision_id": "evt_self",
                    "title": "室内自拍记录",
                    "event_summary": "室内自拍与自我展示",
                    "started_at": "2026-03-20T20:00:00",
                    "ended_at": "2026-03-20T20:05:00",
                    "participant_person_ids": ["Person_002"],
                    "depicted_person_ids": [],
                    "place_refs": ["{'name': '室内', 'confidence': 0.5}"],
                    "original_photo_ids": ["orig_evt_self"],
                    "atomic_evidence": [],
                },
            ]
            reference_signals = [
                {
                    "signal_id": "sig_shop",
                    "profile_bucket": "consumption_intent",
                    "evidence_text": "近期关注商品、购买或支付相关内容",
                    "source_kind": "screenshot",
                }
            ]

            partial_pack = family._build_profile_input_pack_partial(
                primary_person_id="Person_002",
                event_revisions=event_revisions,
                atomic_evidence=[],
                reference_signals=reference_signals,
                scope="cumulative",
            )

            age_hints = partial_pack["identity_signals"]["age_range_hints"]
            career_hints = partial_pack["identity_signals"]["career_direction_hints"]
            lifestyle_hints = partial_pack["lifestyle_consumption_signals"]

            self.assertTrue(age_hints)
            self.assertTrue(any("学生" in item["label"] for item in age_hints))
            self.assertTrue(career_hints)
            self.assertTrue(any("自动化/电气控制方向" == item["label"] for item in career_hints))
            self.assertTrue(lifestyle_hints["diet_hints"])
            self.assertTrue(lifestyle_hints["self_presentation_hints"])
            self.assertTrue(lifestyle_hints["consumption_hints"])
            self.assertTrue(partial_pack["event_persona_clues"]["self_presentation_clues"])
            self.assertTrue(partial_pack["event_persona_clues"]["routine_lifestyle_clues"])

    def test_project_relationships_emits_semantic_relation_and_axes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            family = self._build_family(Path(tmpdir))
            event_revisions = [
                {
                    "event_root_id": "root_001",
                    "event_revision_id": "evt_001",
                    "revision": 1,
                    "title": "双人自拍与晚餐",
                    "event_summary": "两人自拍后共进晚餐",
                    "started_at": "2026-03-10T19:00:00",
                    "ended_at": "2026-03-10T20:00:00",
                    "participant_person_ids": ["Person_001", "Person_002"],
                    "depicted_person_ids": [],
                    "place_refs": ["餐饮店"],
                    "original_photo_ids": ["orig_001"],
                    "atomic_evidence": [],
                },
                {
                    "event_root_id": "root_002",
                    "event_revision_id": "evt_002",
                    "revision": 1,
                    "title": "双人出行候车合影",
                    "event_summary": "两人在车站候车并合影",
                    "started_at": "2026-03-17T09:00:00",
                    "ended_at": "2026-03-17T09:30:00",
                    "participant_person_ids": ["Person_001", "Person_002"],
                    "depicted_person_ids": [],
                    "place_refs": ["车站"],
                    "original_photo_ids": ["orig_002"],
                    "atomic_evidence": [],
                },
            ]
            with mock.patch.object(
                family,
                "_infer_relationship_with_llm",
                return_value={
                    "semantic_relation": "高概率伴侣",
                    "semantic_confidence": 0.86,
                    "semantic_reason_summary": "两人跨时间反复以双人自拍、餐饮、出行场景出现，排他性与亲密度都很高。",
                    "relation_axes": {
                        "intimacy": 0.88,
                        "exclusivity": 0.82,
                        "familyness": 0.16,
                        "schoolness": 0.08,
                        "workness": 0.05,
                        "caregiving": 0.11,
                        "continuity": 0.71,
                        "groupness": 0.12,
                        "conflict": 0.03,
                    },
                    "uncertainty_note": "",
                },
            ):
                revisions, _, changed = family._project_relationships(
                    primary_person_id="Person_001",
                    event_revisions=event_revisions,
                    changed_event_revisions=event_revisions,
                    atomic_evidence=[],
                )

            self.assertEqual(len(revisions), 1)
            relationship = revisions[0]
            self.assertEqual(relationship["target_person_id"], "Person_002")
            self.assertEqual(relationship["semantic_relation"], "伴侣/情侣关系")
            self.assertGreater(relationship["relation_axes"]["intimacy"], 0.8)
            self.assertEqual(relationship["relationship_type"], "semantic_relation")
            self.assertEqual(changed[0]["semantic_relation"], "伴侣/情侣关系")

            partial_pack = family._build_profile_input_pack_partial(
                primary_person_id="Person_001",
                event_revisions=event_revisions,
                atomic_evidence=[],
                reference_signals=[],
                scope="cumulative",
            )
            final_pack = family._build_profile_input_pack(
                profile_input_pack_partial=partial_pack,
                relationship_revisions=revisions,
            )
            self.assertEqual(final_pack["social_patterns"]["top_relationships"][0]["semantic_relation"], "伴侣/情侣关系")
            self.assertTrue(final_pack["social_clues"])

    def test_build_memory_units_v2_uses_event_revision_as_first_layer_unit(self) -> None:
        event_revisions = [
            {
                "event_root_id": "event_root_001",
                "event_revision_id": "event_rev_001",
                "revision": 2,
                "title": "Dinner with friends @ Bund",
                "event_summary": "Evening dinner and drinks with two friends near the Bund.",
                "started_at": "2026-03-20T19:00:00",
                "ended_at": "2026-03-20T21:00:00",
                "place_refs": ["Shanghai Bund", "Rooftop Restaurant"],
                "participant_person_ids": ["Person_001", "Person_002"],
                "depicted_person_ids": [],
                "original_photo_ids": ["orig_001", "orig_002"],
                "confidence": 0.88,
                "status": "active",
                "sealed_state": "sealed",
                "atomic_evidence": [
                    {"evidence_id": "ev_ocr", "value_or_text": "MOET", "evidence_type": "brand"},
                    {"evidence_id": "ev_place", "value_or_text": "Shanghai Bund", "evidence_type": "place_candidate"},
                ],
            }
        ]

        records = build_memory_units_v2(
            user_id="user_v03213_test",
            pipeline_family="v0321_3",
            event_revisions=event_revisions,
        )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["unit_id"], "event_rev_001")
        self.assertEqual(record["source_type"], "event_revision")
        self.assertEqual(record["event_root_id"], "event_root_001")
        self.assertEqual(record["supporting_evidence_count"], 2)
        self.assertIn("Dinner with friends", record["retrieval_text"])
        self.assertIn("Shanghai Bund", record["retrieval_text"])

    def test_build_memory_evidence_v2_uses_atomic_evidence_as_second_layer(self) -> None:
        event_revisions = [
            {
                "event_root_id": "event_root_001",
                "event_revision_id": "event_rev_001",
                "title": "Dinner with friends @ Bund",
                "event_summary": "Evening dinner and drinks with two friends near the Bund.",
                "place_refs": ["Shanghai Bund"],
                "participant_person_ids": ["Person_001", "Person_002"],
            }
        ]
        atomic_evidence = [
            {
                "evidence_id": "ev_001",
                "root_event_revision_id": "event_rev_001",
                "evidence_type": "brand",
                "value_or_text": "MOET",
                "provenance": "brand",
                "original_photo_ids": ["orig_001"],
                "confidence": 0.76,
            }
        ]

        records = build_memory_evidence_v2(
            user_id="user_v03213_test",
            pipeline_family="v0321_3",
            atomic_evidence=atomic_evidence,
            event_revisions=event_revisions,
        )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["evidence_id"], "ev_001")
        self.assertEqual(record["source_type"], "atomic_evidence")
        self.assertEqual(record["parent_unit_id"], "event_rev_001")
        self.assertEqual(record["event_root_id"], "event_root_001")
        self.assertEqual(record["original_photo_ids"], ["orig_001"])
        self.assertIn("MOET", record["retrieval_text"])
        self.assertIn("Dinner with friends", record["retrieval_text"])

    def test_build_profile_truth_v1_splits_strong_and_weak_layers(self) -> None:
        profile_revision = {
            "profile_revision_id": "profile_rev_001",
            "scope": "cumulative",
            "primary_person_id": "Person_001",
            "generation_mode": "profile_input_pack_llm",
            "original_photo_ids": ["orig_evt_001", "orig_sig_001"],
        }
        profile_input_pack = {
            "profile_input_pack_id": "profile_pack_001",
            "time_range": {"start": "2026-03-01T10:00:00", "end": "2026-03-22T20:00:00"},
            "baseline_rhythm": {"dominant_activity_window": "evening"},
            "place_patterns": {"top_place_refs": [{"place_ref": "Shanghai Bund", "count": 2}]},
            "activity_patterns": {"top_activities": [{"activity_type": "用餐/轻社交", "count": 3}]},
            "identity_signals": {
                "role_hints": [
                    {
                        "label": "学生或培训阶段个体",
                        "confidence": 0.66,
                        "evidence_level": "event_grounded",
                        "supporting_event_ids": ["evt_001"],
                        "supporting_signal_ids": [],
                    },
                    {
                        "label": "近期有较强的 AI 工具接触或试用倾向",
                        "confidence": 0.42,
                        "evidence_level": "weak_reference",
                        "supporting_event_ids": [],
                        "supporting_signal_ids": ["sig_001"],
                    },
                ]
            },
            "lifestyle_consumption_signals": {
                "diet_hints": [
                    {
                        "label": "餐饮与咖啡场景重复出现",
                        "confidence": 0.62,
                        "evidence_level": "event_grounded",
                        "supporting_event_ids": ["evt_001"],
                        "supporting_signal_ids": [],
                    }
                ]
            },
            "event_grounded_signals": {
                "interest_signals": [
                    {"label": "用餐/轻社交", "count": 3, "supporting_event_ids": ["evt_001"]}
                ]
            },
            "reference_media_weak_signals": {
                "aesthetic_hints": [
                    {
                        "label": "近期关注极简穿搭",
                        "count": 2,
                        "supporting_signal_ids": ["sig_001"],
                        "source_kinds": ["saved_web_image"],
                    }
                ]
            },
            "social_patterns": {
                "top_relationships": [
                    {
                        "relationship_revision_id": "rel_001",
                        "target_person_id": "Person_002",
                        "relationship_type": "semantic_relation",
                        "semantic_relation": "高概率伴侣",
                        "semantic_confidence": 0.82,
                        "semantic_reason_summary": "跨时间重复的双人亲密关系",
                        "relation_axes": {"intimacy": 0.82, "exclusivity": 0.76},
                    }
                ],
                "relationship_summary": {"high_intimacy_count": 1},
                "social_style_hints": {"one_on_one_bias": 0.7},
            },
            "event_persona_clues": {
                "self_presentation_clues": [
                    {
                        "label": "持续通过自拍和人物主体照片进行自我记录与表达",
                        "confidence": 0.72,
                        "evidence_level": "event_grounded",
                        "supporting_event_ids": ["evt_001"],
                        "supporting_signal_ids": [],
                    }
                ]
            },
            "social_clues": [
                {
                    "relationship_revision_id": "rel_001",
                    "target_person_id": "Person_002",
                    "semantic_relation": "高概率伴侣",
                    "semantic_confidence": 0.82,
                    "relation_axes": {"intimacy": 0.82, "exclusivity": 0.76},
                    "why_important": "跨时间重复的双人亲密关系",
                    "profile_implication": "这段关系直接影响情感状态与亲密关系判断。",
                    "supporting_event_ids": ["evt_001"],
                }
            ],
            "change_points": [{"type": "relationship_change", "summary": "新关系开始"}],
            "key_event_refs": [{"event_revision_id": "evt_001", "title": "Dinner"}],
            "key_relationship_refs": [{"relationship_revision_id": "rel_001"}],
            "evidence_guardrails": {"forbidden_direct_inference_from_reference_media": ["真实到访"]},
        }
        relationship_revisions = [
            {
                "relationship_revision_id": "rel_001",
                "target_person_id": "Person_002",
            }
        ]

        truth = build_profile_truth_v1(
            user_id="user_v03213_test",
            pipeline_family="v0321_3",
            profile_revision=profile_revision,
            profile_input_pack=profile_input_pack,
            relationship_revisions=relationship_revisions,
            profile_markdown="# Profile\n\nhello",
        )

        self.assertEqual(truth["schema_version"], "profile_truth.v1")
        self.assertEqual(truth["profile_truth_id"], "profile_rev_001:truth")
        self.assertTrue(truth["truth_layers"]["strong_identity"]["role_hints"])
        self.assertTrue(truth["truth_layers"]["weak_identity"]["role_hints"])
        self.assertTrue(truth["truth_layers"]["strong_lifestyle"]["diet_hints"])
        self.assertTrue(truth["truth_layers"]["event_persona_clues"]["self_presentation_clues"])
        self.assertTrue(truth["truth_layers"]["weak_reference_signals"]["aesthetic_hints"])
        self.assertTrue(truth["truth_layers"]["social_clues"])
        self.assertEqual(
            truth["truth_layers"]["relationship_truth"]["relationship_revision_ids"],
            ["rel_001"],
        )


if __name__ == "__main__":
    unittest.main()
