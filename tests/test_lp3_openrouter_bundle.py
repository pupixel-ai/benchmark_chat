from __future__ import annotations

import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from services.memory_pipeline.profile_fields import build_profile_context, generate_structured_profile
from services.memory_pipeline.profile_tools import (
    analyze_evidence_stats,
    check_subject_ownership,
    extract_metadata_evidence,
    fetch_field_evidence,
)
from services.memory_pipeline.types import MemoryState


class OpenRouterProfileLLMProcessorTests(unittest.TestCase):
    def test_json_mode_uses_chat_completions_and_response_format(self) -> None:
        from services.memory_pipeline.profile_llm import OpenRouterProfileLLMProcessor

        class FakeResponse:
            status_code = 200

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "{\"value\": \"ok\", \"confidence\": 0.91}"
                            }
                        }
                    ]
                }

            @property
            def text(self) -> str:
                return json.dumps(self.json(), ensure_ascii=False)

        with patch("services.memory_pipeline.profile_llm.requests.post", return_value=FakeResponse()) as post:
            processor = OpenRouterProfileLLMProcessor(
                api_key="sk-test",
                base_url="https://openrouter.ai/api/v1",
                primary_person_id="Person_001",
            )
            result = processor._call_llm_via_official_api(
                "Return JSON",
                response_mime_type="application/json",
            )

        self.assertEqual(result["value"], "ok")
        self.assertEqual(post.call_args.args[0], "https://openrouter.ai/api/v1/chat/completions")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "google/gemini-3.1-flash-lite-preview")
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_call_debug_tracks_http_status_and_response_preview(self) -> None:
        from services.memory_pipeline.profile_llm import OpenRouterProfileLLMProcessor

        class FakeResponse:
            status_code = 200

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "{\"value\": \"ok\", \"confidence\": 0.91}"
                            }
                        }
                    ]
                }

            @property
            def text(self) -> str:
                return json.dumps(self.json(), ensure_ascii=False)

        with patch("services.memory_pipeline.profile_llm.requests.post", return_value=FakeResponse()):
            processor = OpenRouterProfileLLMProcessor(
                api_key="sk-test",
                base_url="https://openrouter.ai/api/v1",
                primary_person_id="Person_001",
            )
            processor._call_llm_via_official_api(
                "Return JSON",
                response_mime_type="application/json",
            )

        debug_info = processor._consume_last_call_debug()
        self.assertTrue(debug_info["api_call_attempted"])
        self.assertEqual(debug_info["http_status_code"], 200)
        self.assertEqual(debug_info["model"], "google/gemini-3.1-flash-lite-preview")
        self.assertTrue(debug_info["raw_response_preview"].startswith('{"choices"'))
        self.assertFalse(debug_info["raw_response_truncated"])


class BundleLoaderTests(unittest.TestCase):
    def test_load_precomputed_memory_state_supports_face_vlm_lp1_bundle_layout(self) -> None:
        from services.memory_pipeline.precomputed_loader import load_precomputed_memory_state

        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            (base / "face").mkdir()
            (base / "vlm").mkdir()
            (base / "lp1").mkdir()
            (base / "face" / "face_recognition_output.json").write_text(
                json.dumps(
                    {
                        "primary_person_id": None,
                        "persons": [
                            {
                                "person_id": "Person_001",
                                "photo_count": 2,
                                "first_seen": "2026-03-01T10:00:00",
                                "last_seen": "2026-03-02T10:00:00",
                                "avg_score": 0.88,
                                "avg_quality": 0.77,
                                "label": "Person",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (base / "vlm" / "vp1_observations.json").write_text(
                json.dumps(
                    [
                        {
                            "photo_id": "photo_001",
                            "timestamp": "2026-03-01T10:00:00",
                            "face_person_ids": [],
                            "media_kind": "screenshot",
                            "is_reference_like": True,
                            "sequence_index": 1,
                            "vlm_analysis": {
                                "summary": "【主角】正在浏览校园社媒截图。",
                                "people": [],
                                "relations": [],
                                "scene": {"location_detected": "校园", "location_type": "室内"},
                                "event": {"activity": "浏览", "social_context": "独自", "mood": "平静", "story_hints": []},
                                "details": [],
                                "ocr_hits": ["Instagram"],
                                "brands": ["Nike"],
                                "place_candidates": ["清华大学"],
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (base / "lp1" / "lp1_events_compact.json").write_text(
                json.dumps(
                    [
                        {
                            "event_id": "EVT_0001",
                            "title": "校园截图浏览",
                            "started_at": "2026-03-01T10:00:00",
                            "ended_at": "2026-03-01T10:05:00",
                            "participant_person_ids": ["主角"],
                            "depicted_person_ids": [],
                            "place_refs": ["校园"],
                            "supporting_photo_ids": ["photo_001"],
                            "anchor_photo_id": "photo_001",
                            "batch_id": "BATCH_0001",
                            "source_temp_event_id": "TMP_1",
                            "meta_info": {"title": "校园截图浏览", "location_context": "校园", "photo_count": 1},
                            "objective_fact": {"scene_description": "主角查看校园相关截图。", "participants": ["主角"]},
                            "narrative_synthesis": "主角浏览校园社媒截图。",
                            "social_dynamics": [],
                            "persona_evidence": {"behavioral": [], "aesthetic": [], "socioeconomic": []},
                            "tags": ["#校园"],
                            "confidence": 0.81,
                            "reason": "截图内容明确",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            state = load_precomputed_memory_state(base)

        self.assertEqual(state.primary_decision["primary_person_id"], "主角")
        self.assertEqual(len(state.vlm_results), 1)
        self.assertEqual(len(state.events), 1)
        self.assertEqual(state.events[0].participants, ["主角"])
        self.assertEqual(
            state.events[0].meta_info["trace"]["supporting_photo_ids"],
            ["photo_001"],
        )

    def test_load_precomputed_memory_state_supports_flat_bundle_file_names(self) -> None:
        from services.memory_pipeline.precomputed_loader import load_precomputed_memory_state

        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            (base / "face_recognition_output.json").write_text(
                json.dumps(
                    {
                        "primary_person_id": "Person_004",
                        "persons": [
                            {
                                "person_id": "Person_004",
                                "photo_count": 6,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (base / "vp1_observations.json").write_text(
                json.dumps(
                    [
                        {
                            "photo_id": "photo_001",
                            "timestamp": "2026-03-01T10:00:00",
                            "face_person_ids": ["Person_004"],
                            "vlm_analysis": {
                                "summary": "【主角】在户外活动。",
                                "people": [{"person_id": "主角", "contact_type": "no_contact"}],
                                "relations": [],
                                "scene": {"location_detected": "公园"},
                                "event": {"activity": "散步"},
                                "details": [],
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (base / "lp1_events_compact.json").write_text(
                json.dumps(
                    [
                        {
                            "event_id": "EVT_0001",
                            "title": "公园散步",
                            "started_at": "2026-03-01T10:00:00",
                            "ended_at": "2026-03-01T10:05:00",
                            "participant_person_ids": ["【主角】"],
                            "depicted_person_ids": ["Person_004"],
                            "place_refs": ["公园"],
                            "supporting_photo_ids": ["photo_001"],
                            "anchor_photo_id": "photo_001",
                            "batch_id": "BATCH_0001",
                            "source_temp_event_id": "TMP_1",
                            "meta_info": {"title": "公园散步", "location_context": "公园", "photo_count": 1},
                            "objective_fact": {"scene_description": "主角在公园散步。", "participants": ["【主角】"]},
                            "narrative_synthesis": "主角在公园散步。",
                            "social_dynamics": [],
                            "persona_evidence": {"behavioral": []},
                            "tags": ["#散步"],
                            "confidence": 0.81,
                            "reason": "照片内容明确",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            state = load_precomputed_memory_state(base)

        self.assertEqual(state.primary_decision["primary_person_id"], "Person_004")
        self.assertEqual(len(state.vlm_results), 1)
        self.assertEqual(state.vlm_results[0]["vlm_analysis"]["people"][0]["person_id"], "Person_004")
        self.assertEqual(len(state.events), 1)
        self.assertEqual(state.events[0].participants, ["Person_004"])


class LP3BundleEvidenceMappingTests(unittest.TestCase):
    def test_build_profile_context_preserves_bundle_specific_vlm_fields_and_subject_role(self) -> None:
        state = MemoryState(
            photos=[],
            face_db={},
            vlm_results=[
                {
                    "photo_id": "photo_001",
                    "timestamp": "2026-03-01T10:00:00",
                    "face_person_ids": [],
                    "media_kind": "screenshot",
                    "is_reference_like": True,
                    "vlm_analysis": {
                        "summary": "【主角】在校园里浏览 Instagram 内容。",
                        "people": [],
                        "relations": [],
                        "scene": {"location_detected": "校园"},
                        "event": {"activity": "浏览"},
                        "details": [],
                        "ocr_hits": ["Instagram"],
                        "brands": ["Nike"],
                        "place_candidates": ["清华大学"],
                    },
                }
            ],
        )
        state.primary_decision = {"primary_person_id": "主角"}

        context = build_profile_context(state)
        observation = context["vlm_observations"][0]

        self.assertEqual(observation["ocr_hits"], ["Instagram"])
        self.assertEqual(observation["brands"], ["Nike"])
        self.assertEqual(observation["place_candidates"], ["清华大学"])
        self.assertEqual(observation["media_kind"], "screenshot")
        self.assertTrue(observation["is_reference_like"])
        self.assertEqual(observation["subject_role"], "protagonist_view")

    def test_structured_brand_place_and_ocr_fields_feed_lp3_tools(self) -> None:
        context = {
            "primary_person_id": "主角",
            "events": [],
            "relationships": [],
            "groups": [],
            "feature_refs": [],
            "resolved_facts": {},
            "social_media_available": False,
            "vlm_observations": [
                {
                    "photo_id": "photo_001",
                    "timestamp": "2026-03-01T10:00:00",
                    "summary": "【主角】浏览校园相关内容。",
                    "location": "校园",
                    "activity": "浏览",
                    "people": [],
                    "details": [],
                    "ocr_hits": ["Instagram"],
                    "brands": ["Nike"],
                    "place_candidates": ["清华大学"],
                    "subject_role": "protagonist_view",
                }
            ],
        }

        brand_bundle = fetch_field_evidence("long_term_facts.material.brand_preference", context)
        brand_ownership = check_subject_ownership("long_term_facts.material.brand_preference", brand_bundle)
        brand_stats = analyze_evidence_stats(
            "long_term_facts.material.brand_preference",
            brand_bundle,
            ownership_bundle=brand_ownership,
        )
        location_bundle = fetch_field_evidence("long_term_facts.geography.location_anchors", context)
        metadata_bundle = extract_metadata_evidence(context)

        self.assertEqual(brand_stats["brand_summary"]["top_brands"][0]["brand_name"], "Nike")
        self.assertEqual(location_bundle["compact"]["top_candidates"][0]["city_name"], "北京")
        self.assertTrue(metadata_bundle["has_social_media_evidence"])
        self.assertNotIn("raw_text_snippets", metadata_bundle)
        self.assertEqual(metadata_bundle["metadata_ids"], ["photo_001"])

    def test_language_culture_query_profile_adds_search_index_and_feature_refs(self) -> None:
        context = {
            "primary_person_id": "主角",
            "events": [
                type(
                    "Evt",
                    (),
                    {
                        "event_id": "EVT_LANG",
                        "date": "2026-03-01",
                        "type": "课堂",
                        "participants": ["主角"],
                        "location": "深圳校园",
                        "description": "主角在课堂上进行英语和粤语交流。",
                        "photo_count": 1,
                        "title": "粤语与英语课堂",
                        "narrative_synthesis": "双语交流课堂。",
                        "tags": ["课堂", "语言"],
                        "persona_evidence": {},
                    },
                )(),
                type(
                    "Evt",
                    (),
                    {
                        "event_id": "EVT_WORK",
                        "date": "2026-03-02",
                        "type": "工作",
                        "participants": ["主角"],
                        "location": "办公室",
                        "description": "主角整理工资表。",
                        "photo_count": 1,
                        "title": "工资整理",
                        "narrative_synthesis": "办公室文档记录。",
                        "tags": ["工资"],
                        "persona_evidence": {},
                    },
                )(),
            ],
            "relationships": [],
            "groups": [],
            "feature_refs": [
                {"feature_name": "event_count", "value": 2},
                {"feature_name": "relationship_count", "value": 0},
            ],
            "resolved_facts": {},
            "social_media_available": False,
            "vlm_observations": [
                {
                    "photo_id": "photo_lang",
                    "timestamp": "2026-03-01T09:00:00",
                    "summary": "【主角】拍下英文与粤语双语招牌。",
                    "location": "深圳校园",
                    "activity": "记录",
                    "people": [],
                    "details": ["双语文化活动海报"],
                    "ocr_hits": ["English", "粤语"],
                    "brands": [],
                    "place_candidates": ["深圳"],
                    "subject_role": "protagonist_view",
                }
            ],
        }

        evidence_bundle = fetch_field_evidence("long_term_facts.social_identity.language_culture", context)

        self.assertGreater(len(evidence_bundle["allowed_refs"]["feature_refs"]), 0)
        event_ref = evidence_bundle["allowed_refs"]["events"][0]
        feature_ref = evidence_bundle["allowed_refs"]["feature_refs"][0]
        self.assertIn("search_text", event_ref)
        self.assertIn("normalized_places", event_ref)
        self.assertIn("normalized_topics", event_ref)
        self.assertIn("time_keys", event_ref)
        self.assertIn("subject_binding", event_ref)
        self.assertIn("search_text", feature_ref)
        self.assertEqual(
            [ref["event_id"] for ref in evidence_bundle["supporting_refs"]["events"]],
            ["EVT_LANG"],
        )

    def test_activity_interest_query_profile_prefers_interest_clues(self) -> None:
        context = {
            "primary_person_id": "主角",
            "events": [
                type(
                    "Evt",
                    (),
                    {
                        "event_id": "EVT_INTEREST",
                        "date": "2026-03-01",
                        "type": "运动",
                        "participants": ["主角"],
                        "location": "体育馆",
                        "description": "主角参加篮球训练。",
                        "photo_count": 1,
                        "title": "篮球训练",
                        "narrative_synthesis": "主角持续参加篮球活动。",
                        "tags": ["篮球", "运动"],
                        "persona_evidence": {},
                    },
                )(),
                type(
                    "Evt",
                    (),
                    {
                        "event_id": "EVT_NOISE",
                        "date": "2026-03-02",
                        "type": "工作",
                        "participants": ["主角"],
                        "location": "办公室",
                        "description": "主角整理工资表。",
                        "photo_count": 1,
                        "title": "工资记录",
                        "narrative_synthesis": "办公室文档记录。",
                        "tags": ["工资"],
                        "persona_evidence": {},
                    },
                )(),
            ],
            "relationships": [],
            "groups": [],
            "feature_refs": [{"feature_name": "event_count", "value": 2}],
            "resolved_facts": {},
            "social_media_available": False,
            "vlm_observations": [
                {
                    "photo_id": "photo_interest",
                    "timestamp": "2026-03-01T10:00:00",
                    "summary": "【主角】在体育馆打篮球。",
                    "location": "体育馆",
                    "activity": "basketball training",
                    "people": [],
                    "details": ["basketball practice"],
                    "ocr_hits": [],
                    "brands": [],
                    "place_candidates": [],
                    "subject_role": "protagonist_view",
                },
                {
                    "photo_id": "photo_noise",
                    "timestamp": "2026-03-02T10:00:00",
                    "summary": "【主角】在办公室整理文档。",
                    "location": "办公室",
                    "activity": "work",
                    "people": [],
                    "details": ["salary spreadsheet"],
                    "ocr_hits": [],
                    "brands": [],
                    "place_candidates": [],
                    "subject_role": "protagonist_view",
                },
            ],
        }

        evidence_bundle = fetch_field_evidence("long_term_facts.hobbies.interests", context)

        self.assertLessEqual(len(evidence_bundle["supporting_refs"]["events"]), 4)
        self.assertLessEqual(len(evidence_bundle["supporting_refs"]["vlm_observations"]), 5)
        self.assertEqual(
            [ref["event_id"] for ref in evidence_bundle["supporting_refs"]["events"]],
            ["EVT_INTEREST"],
        )
        self.assertEqual(
            [ref["photo_id"] for ref in evidence_bundle["supporting_refs"]["vlm_observations"]],
            ["photo_interest"],
        )

    def test_education_filter_accepts_protagonist_view_when_primary_is_alias(self) -> None:
        context = {
            "primary_person_id": "主角",
            "events": [],
            "relationships": [],
            "groups": [],
            "feature_refs": [],
            "resolved_facts": {},
            "social_media_available": False,
            "vlm_observations": [
                {
                    "photo_id": "photo_002",
                    "timestamp": "2026-03-01T10:00:00",
                    "summary": "【主角】在 campus classroom 浏览课程截图。",
                    "location": "校园教室",
                    "activity": "浏览",
                    "people": [],
                    "details": [],
                    "ocr_hits": [],
                    "brands": [],
                    "place_candidates": [],
                    "subject_role": "protagonist_view",
                }
            ],
        }

        evidence_bundle = fetch_field_evidence("long_term_facts.social_identity.education", context)

        self.assertEqual(len(evidence_bundle["supporting_refs"]["vlm_observations"]), 1)

    def test_name_field_prunes_generic_vlm_refs_without_explicit_identity_clues(self) -> None:
        context = {
            "primary_person_id": "主角",
            "events": [],
            "relationships": [],
            "groups": [],
            "feature_refs": [],
            "resolved_facts": {},
            "social_media_available": False,
            "vlm_observations": [
                {
                    "photo_id": "photo_010",
                    "timestamp": "2026-03-01T10:00:00",
                    "summary": "【主角】在公园散步。",
                    "location": "公园",
                    "activity": "散步",
                    "people": [],
                    "details": ["天气晴朗"],
                    "ocr_hits": [],
                    "brands": [],
                    "place_candidates": [],
                    "subject_role": "protagonist_view",
                },
                {
                    "photo_id": "photo_011",
                    "timestamp": "2026-03-01T10:10:00",
                    "summary": "【主角】在餐厅吃饭。",
                    "location": "餐厅",
                    "activity": "用餐",
                    "people": [],
                    "details": ["桌上有饮料"],
                    "ocr_hits": [],
                    "brands": [],
                    "place_candidates": [],
                    "subject_role": "protagonist_view",
                },
            ],
        }

        evidence_bundle = fetch_field_evidence("long_term_facts.identity.name", context)

        self.assertEqual(evidence_bundle["supporting_refs"]["vlm_observations"], [])

    def test_attitude_style_field_limits_supporting_refs_to_top_k(self) -> None:
        context = {
            "primary_person_id": "主角",
            "events": [
                type(
                    "Evt",
                    (),
                    {
                        "event_id": f"EVT_{idx:03d}",
                        "date": f"2026-03-{idx:02d}",
                        "type": "日常",
                        "participants": ["主角"],
                        "location": "城市街区",
                        "description": "主角展示穿搭与摄影构图。",
                        "photo_count": idx + 1,
                        "title": f"Look {idx}",
                        "narrative_synthesis": "持续记录穿搭和构图风格。",
                        "tags": ["#ootd", "#摄影"],
                        "persona_evidence": {},
                    },
                )()
                for idx in range(1, 7)
            ],
            "relationships": [],
            "groups": [],
            "feature_refs": [],
            "resolved_facts": {},
            "social_media_available": False,
            "vlm_observations": [
                {
                    "photo_id": f"photo_{idx:03d}",
                    "timestamp": f"2026-03-{idx:02d}T10:00:00",
                    "summary": "【主角】记录今日穿搭与镜面自拍。",
                    "location": "城市街区",
                    "activity": "拍照",
                    "people": [],
                    "details": ["黑色外套", "构图讲究", "风格统一"],
                    "ocr_hits": [],
                    "brands": [],
                    "place_candidates": [],
                    "subject_role": "protagonist_view",
                }
                for idx in range(1, 9)
            ],
        }

        evidence_bundle = fetch_field_evidence("long_term_expression.attitude_style", context)

        self.assertLessEqual(len(evidence_bundle["supporting_refs"]["events"]), 3)
        self.assertLessEqual(len(evidence_bundle["supporting_refs"]["vlm_observations"]), 5)

    def test_brand_ownership_returns_compact_candidate_signals_without_refs(self) -> None:
        context = {
            "primary_person_id": "主角",
            "events": [
                type(
                    "Evt",
                    (),
                    {
                        "event_id": "EVT_BRAND_001",
                        "date": "2026-03-01",
                        "type": "旅行",
                        "participants": ["主角"],
                        "location": "北京",
                        "description": "主角在旅行中持续拍照记录。",
                        "photo_count": 2,
                        "title": "北京旅行记录",
                        "narrative_synthesis": "主角连续使用同一部手机记录旅途。",
                        "tags": ["拍照"],
                        "persona_evidence": {
                            "behavioral": [],
                            "aesthetic": [],
                            "socioeconomic": ["使用华为旗舰手机"],
                        },
                    },
                )()
            ],
            "relationships": [],
            "groups": [],
            "feature_refs": [],
            "resolved_facts": {},
            "social_media_available": False,
            "vlm_observations": [
                {
                    "photo_id": "photo_101",
                    "timestamp": "2026-03-01T10:00:00",
                    "summary": "【主角】拍摄街景。",
                    "location": "北京",
                    "activity": "拍摄",
                    "people": [],
                    "details": ["左下角设备水印：HUAWEI Pura70 Pro+ | XMAGE"],
                    "ocr_hits": ["HUAWEI Pura 70"],
                    "brands": [],
                    "place_candidates": [],
                    "subject_role": "protagonist_view",
                },
                {
                    "photo_id": "photo_102",
                    "timestamp": "2026-03-02T10:00:00",
                    "summary": "【主角】继续记录旅途。",
                    "location": "北京",
                    "activity": "记录",
                    "people": [],
                    "details": ["照片左下角手机水印：Shot on HUAWEI Pura70 Pro+"],
                    "ocr_hits": ["Shot on HUAWEI"],
                    "brands": [],
                    "place_candidates": [],
                    "subject_role": "protagonist_view",
                },
                {
                    "photo_id": "photo_103",
                    "timestamp": "2026-03-03T10:00:00",
                    "summary": "【主角】在 MANNER COFFEE 店内休息。",
                    "location": "MANNER COFFEE",
                    "activity": "休息",
                    "people": [],
                    "details": ["桌上有咖啡杯"],
                    "ocr_hits": [],
                    "brands": ["MANNER COFFEE"],
                    "place_candidates": ["MANNER COFFEE"],
                    "subject_role": "protagonist_view",
                },
                {
                    "photo_id": "photo_104",
                    "timestamp": "2026-03-04T10:00:00",
                    "summary": "朋友穿着 CAMEL 外套合影。",
                    "location": "商场",
                    "activity": "合影",
                    "people": ["Person_002"],
                    "face_person_ids": ["Person_002"],
                    "details": ["CAMEL 外套"],
                    "ocr_hits": [],
                    "brands": ["CAMEL"],
                    "place_candidates": [],
                    "subject_role": "other_people_only",
                },
            ],
        }

        evidence_bundle = fetch_field_evidence("long_term_facts.material.brand_preference", context)
        ownership_bundle = check_subject_ownership("long_term_facts.material.brand_preference", evidence_bundle)
        candidate_signals = {
            item["candidate"]: item["signal"]
            for item in ownership_bundle["candidate_signals"]
        }

        self.assertEqual(candidate_signals["HUAWEI"], "owned_or_used")
        self.assertEqual(candidate_signals["MANNER COFFEE"], "venue_context")
        self.assertEqual(candidate_signals["CAMEL"], "other_person")
        self.assertEqual(ownership_bundle["ownership_signal"], "owned_or_used")
        self.assertNotIn("ownership_refs", ownership_bundle)

    def test_location_stats_compact_summary_uses_city_normalization(self) -> None:
        def event(event_id: str, date: str, location: str, description: str):
            return type(
                "Evt",
                (),
                {
                    "event_id": event_id,
                    "date": date,
                    "type": "日常",
                    "participants": ["主角"],
                    "location": location,
                    "description": description,
                    "photo_count": 2,
                    "title": location,
                    "narrative_synthesis": description,
                    "tags": [],
                    "persona_evidence": {},
                },
            )()

        context = {
            "primary_person_id": "主角",
            "events": [
                event("EVT_001", "2026-01-05", "北京故宫", "主角在北京故宫参观。"),
                event("EVT_002", "2026-01-20", "天安门城楼", "主角继续在北京活动。"),
                event("EVT_003", "2026-02-02", "深圳文华东方酒店", "主角在深圳住宿。"),
                event("EVT_004", "2026-02-18", "星河双子塔", "主角在深圳巡查。"),
                event("EVT_005", "2026-03-01", "陆丰市公安局", "主角在陆丰办理事务。"),
            ],
            "relationships": [],
            "groups": [],
            "feature_refs": [],
            "resolved_facts": {},
            "social_media_available": False,
            "vlm_observations": [
                {
                    "photo_id": "photo_201",
                    "timestamp": "2026-01-06T10:00:00",
                    "summary": "【主角】在北京国博附近。",
                    "location": "北京国博",
                    "activity": "参观",
                    "people": [],
                    "details": [],
                    "ocr_hits": [],
                    "brands": [],
                    "place_candidates": ["北京国家博物馆"],
                    "subject_role": "protagonist_view",
                },
                {
                    "photo_id": "photo_202",
                    "timestamp": "2026-02-10T10:00:00",
                    "summary": "【主角】在深圳深业上城。 ",
                    "location": "深圳深业上城",
                    "activity": "打卡",
                    "people": [],
                    "details": [],
                    "ocr_hits": [],
                    "brands": [],
                    "place_candidates": ["深业上城"],
                    "subject_role": "protagonist_view",
                },
            ],
        }

        evidence_bundle = fetch_field_evidence("long_term_facts.geography.location_anchors", context)
        stats_bundle = analyze_evidence_stats("long_term_facts.geography.location_anchors", evidence_bundle)
        top_cities = [
            item["city_name"]
            for item in stats_bundle["location_summary"]["top_city_candidates"][:3]
        ]

        self.assertEqual(set(top_cities), {"北京", "深圳", "陆丰"})

    def test_location_stats_do_not_special_case_yuhuan(self) -> None:
        def event(event_id: str, date: str, location: str, description: str):
            return type(
                "Evt",
                (),
                {
                    "event_id": event_id,
                    "date": date,
                    "type": "日常",
                    "participants": ["主角"],
                    "location": location,
                    "description": description,
                    "photo_count": 2,
                    "title": location,
                    "narrative_synthesis": description,
                    "tags": [],
                    "persona_evidence": {},
                },
            )()

        context = {
            "primary_person_id": "主角",
            "events": [
                event("EVT_101", "2026-03-01", "浙江台州玉环市", "主角在玉环办事。"),
            ],
            "relationships": [],
            "groups": [],
            "feature_refs": [],
            "resolved_facts": {},
            "social_media_available": False,
            "vlm_observations": [
                {
                    "photo_id": "photo_301",
                    "timestamp": "2026-03-01T10:00:00",
                    "summary": "【主角】在玉环街区记录行程。",
                    "location": "玉环街区",
                    "activity": "记录",
                    "people": [],
                    "details": [],
                    "ocr_hits": [],
                    "brands": [],
                    "place_candidates": ["玉环市"],
                    "subject_role": "protagonist_view",
                }
            ],
        }

        evidence_bundle = fetch_field_evidence("long_term_facts.geography.location_anchors", context)
        stats_bundle = analyze_evidence_stats("long_term_facts.geography.location_anchors", evidence_bundle)

        self.assertEqual(
            stats_bundle["location_summary"]["top_city_candidates"][0]["city_name"],
            "玉环",
        )
        self.assertTrue(stats_bundle["suggested_strong_evidence_met"])

    def test_time_stats_include_month_histogram_and_distinct_months(self) -> None:
        def event(event_id: str, date: str):
            return type(
                "Evt",
                (),
                {
                    "event_id": event_id,
                    "date": date,
                    "type": "日常",
                    "participants": ["主角"],
                    "location": "城市街区",
                    "description": "主角记录日常活动。",
                    "photo_count": 2,
                    "title": event_id,
                    "narrative_synthesis": "连续活动记录。",
                    "tags": [],
                    "persona_evidence": {},
                },
            )()

        context = {
            "primary_person_id": "主角",
            "events": [
                event("EVT_301", "2026-01-05"),
                event("EVT_302", "2026-02-14"),
                event("EVT_303", "2026-03-20"),
            ],
            "relationships": [],
            "groups": [],
            "feature_refs": [],
            "resolved_facts": {},
            "social_media_available": False,
            "vlm_observations": [],
        }

        evidence_bundle = fetch_field_evidence("long_term_facts.time.event_cycles", context)
        stats_bundle = analyze_evidence_stats("long_term_facts.time.event_cycles", evidence_bundle)

        self.assertEqual(stats_bundle["time_summary"]["distinct_months"], 3)
        self.assertEqual(
            stats_bundle["time_summary"]["month_histogram"],
            {"2026-01": 1, "2026-02": 1, "2026-03": 1},
        )
        self.assertGreater(stats_bundle["time_summary"]["span_days"], 0)

    def test_event_driven_fields_do_not_collect_vlm_refs(self) -> None:
        context = {
            "primary_person_id": "主角",
            "events": [
                type(
                    "Evt",
                    (),
                    {
                        "event_id": "EVT_WORK",
                        "date": "2026-03-01",
                        "type": "工作",
                        "participants": ["主角"],
                        "location": "深圳办公室",
                        "description": "主角在办公室完成打卡和值班工作。",
                        "photo_count": 2,
                        "title": "工作日常",
                        "narrative_synthesis": "稳定的工作与异地流动记录。",
                        "tags": ["工作", "打卡"],
                        "persona_evidence": {},
                    },
                )()
            ],
            "relationships": [],
            "groups": [],
            "feature_refs": [{"feature_name": "event_count", "value": 1}],
            "resolved_facts": {},
            "social_media_available": False,
            "vlm_observations": [
                {
                    "photo_id": "photo_noise",
                    "timestamp": "2026-03-01T10:00:00",
                    "summary": "【主角】在办公室。",
                    "location": "深圳办公室",
                    "activity": "工作",
                    "people": [],
                    "details": ["办公桌"],
                    "ocr_hits": [],
                    "brands": [],
                    "place_candidates": ["深圳"],
                    "subject_role": "protagonist_view",
                }
            ],
        }

        for field_key in (
            "long_term_facts.social_identity.career",
            "long_term_facts.geography.cross_border",
            "short_term_facts.current_displacement",
        ):
            evidence_bundle = fetch_field_evidence(field_key, context)
            self.assertEqual(evidence_bundle["allowed_refs"]["vlm_observations"], [], field_key)
            self.assertEqual(evidence_bundle["supporting_refs"]["vlm_observations"], [], field_key)


class LP3BatchObservabilityTests(unittest.TestCase):
    def test_generate_structured_profile_records_exception_fallback_debug(self) -> None:
        class ExplodingLLMProcessor:
            def _call_llm_via_official_api(self, prompt, response_mime_type=None):
                raise RuntimeError("openrouter timeout")

            def _consume_last_call_debug(self):
                return {
                    "api_call_attempted": True,
                    "http_status_code": None,
                    "raw_response_preview": "",
                    "raw_response_truncated": False,
                    "model": "google/gemini-3.1-flash-lite-preview",
                }

        state = MemoryState(
            photos=[],
            face_db={},
            vlm_results=[
                {
                    "photo_id": "photo_001",
                    "timestamp": "2026-03-01T10:00:00",
                    "face_person_ids": [],
                    "vlm_analysis": {
                        "summary": "【主角】正在处理一份带实名信息的表单。",
                        "people": [],
                        "relations": [],
                        "scene": {"location_detected": "车店"},
                        "event": {"activity": "签字"},
                        "details": [],
                        "ocr_hits": ["姓名：郑逸朗"],
                        "brands": [],
                        "place_candidates": ["深圳"],
                    },
                }
            ],
        )
        state.primary_decision = {"primary_person_id": "Person_011"}

        with patch(
            "services.memory_pipeline.profile_agent.DOMAIN_SPECS",
            [
                {
                    "domain_key": "test_domain",
                    "display_name": "Test Domain",
                    "fields": ["long_term_facts.identity.name"],
                }
            ],
        ):
            result = generate_structured_profile(state, llm_processor=ExplodingLLMProcessor())

        self.assertIn("llm_batch_debug", result)
        self.assertEqual(len(result["llm_batch_debug"]), 1)
        batch_debug = result["llm_batch_debug"][0]
        self.assertEqual(batch_debug["batch_name"], "Test Domain::batch_1")
        self.assertTrue(batch_debug["api_call_attempted"])
        self.assertEqual(batch_debug["fallback_reason"], "exception")
        self.assertEqual(batch_debug["exception_type"], "RuntimeError")
        self.assertFalse(batch_debug["raw_result_parseable"])
        self.assertEqual(batch_debug["recovered_field_count"], 0)

    def test_generate_structured_profile_records_parse_failure_debug(self) -> None:
        class UnparseableLLMProcessor:
            def _call_llm_via_official_api(self, prompt, response_mime_type=None):
                return "not-json-response"

            def _consume_last_call_debug(self):
                return {
                    "api_call_attempted": True,
                    "http_status_code": 200,
                    "raw_response_preview": "not-json-response",
                    "raw_response_truncated": False,
                    "model": "google/gemini-3.1-flash-lite-preview",
                }

        state = MemoryState(
            photos=[],
            face_db={},
            vlm_results=[
                {
                    "photo_id": "photo_001",
                    "timestamp": "2026-03-01T10:00:00",
                    "face_person_ids": [],
                    "vlm_analysis": {
                        "summary": "【主角】正在处理一份带实名信息的表单。",
                        "people": [],
                        "relations": [],
                        "scene": {"location_detected": "车店"},
                        "event": {"activity": "签字"},
                        "details": [],
                        "ocr_hits": ["姓名：郑逸朗"],
                        "brands": [],
                        "place_candidates": ["深圳"],
                    },
                }
            ],
        )
        state.primary_decision = {"primary_person_id": "Person_011"}

        with patch(
            "services.memory_pipeline.profile_agent.DOMAIN_SPECS",
            [
                {
                    "domain_key": "test_domain",
                    "display_name": "Test Domain",
                    "fields": ["long_term_facts.identity.name"],
                }
            ],
        ):
            result = generate_structured_profile(state, llm_processor=UnparseableLLMProcessor())

        batch_debug = result["llm_batch_debug"][0]
        self.assertTrue(batch_debug["api_call_attempted"])
        self.assertEqual(batch_debug["http_status_code"], 200)
        self.assertEqual(batch_debug["fallback_reason"], "parse_failure")
        self.assertFalse(batch_debug["raw_result_parseable"])
        self.assertEqual(batch_debug["raw_response_preview"], "not-json-response")


if __name__ == "__main__":
    unittest.main()
