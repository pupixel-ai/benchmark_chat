from __future__ import annotations

import unittest

from models import Event
from services.memory_pipeline.types import FieldSpec, ProfileState
from services.memory_pipeline.profile_fields import build_profile_context, generate_structured_profile
from services.memory_pipeline.types import MemoryState


class ProfileAgentOfflineFallbackTests(unittest.TestCase):
    def test_tool_registry_execute_returns_native_bundle_contract(self) -> None:
        from services.memory_pipeline.profile_tools import get_tool

        context = {
            "primary_person_id": "Person_001",
            "events": [
                Event(
                    event_id="EVT_001",
                    date="2026-03-01",
                    time_range="10:00 - 11:00",
                    duration="1小时",
                    title="校园活动",
                    type="日常",
                    participants=["Person_001"],
                    location="校园",
                    description="主角在校园参加活动。",
                    photo_count=2,
                    confidence=0.8,
                    reason="",
                    narrative="",
                    narrative_synthesis="校园活动重复出现",
                    tags=["campus"],
                )
            ],
            "relationships": [],
            "vlm_observations": [
                {
                    "photo_id": "PHOTO_001",
                    "timestamp": "2026-03-01T10:00:00",
                    "summary": "主角在校园活动",
                    "location": "校园",
                    "activity": "campus activity",
                    "people": [{"person_id": "Person_001"}],
                    "details": ["campus"],
                    "subject_role": "protagonist_present",
                }
            ],
            "group_artifacts": [],
            "feature_refs": [],
        }

        evidence_bundle = get_tool("fetch_field_evidence").execute(
            field_key="long_term_facts.social_identity.education",
            context=context,
            profile_state={},
        )
        self.assertIsInstance(evidence_bundle, dict)
        self.assertIn("compact", evidence_bundle)
        self.assertIn("allowed_refs", evidence_bundle)
        self.assertNotIn("candidates", evidence_bundle)

        ownership_bundle = get_tool("check_subject_ownership").execute(
            field_key="long_term_facts.material.brand_preference",
            evidence_bundle={
                "field_key": "long_term_facts.material.brand_preference",
                "allowed_refs": {"events": [], "relationships": [], "vlm_observations": [], "group_artifacts": [], "feature_refs": []},
                "supporting_refs": {"events": [], "relationships": [], "vlm_observations": [], "group_artifacts": [], "feature_refs": []},
                "compact": {},
                "top_candidates": [{"candidate": "HUAWEI"}],
                "material_candidate_bundle": {"candidates": [{"candidate": "HUAWEI"}]},
            },
        )
        self.assertIsInstance(ownership_bundle, dict)
        self.assertIn("ownership_signal", ownership_bundle)
        self.assertNotIn("candidates", ownership_bundle)

        metadata_bundle = get_tool("extract_metadata_evidence").execute(context=context)
        self.assertIsInstance(metadata_bundle, dict)
        self.assertIn("has_social_media_evidence", metadata_bundle)
        self.assertNotIn("candidates", metadata_bundle)

    def test_run_batch_recovers_string_json_response_instead_of_offline_fallback(self) -> None:
        from services.memory_pipeline.profile_agent import ProfileAgent

        spec = FieldSpec(
            field_key="long_term_facts.hobbies.interests",
            risk_level="P1",
            allowed_sources=["event"],
            strong_evidence=["同类活动跨事件重复"],
            cot_steps=["有任何兴趣线索就可输出"],
            owner_resolution_steps=[],
            time_reasoning_steps=[],
            counter_evidence_checks=[],
            weak_evidence=[],
            hard_blocks=[],
            owner_checks=[],
            time_layer_rule="flexible",
            null_preferred_when=[],
            reflection_questions=[],
            reflection_rounds=1,
            requires_social_media=False,
        )
        agent = ProfileAgent(field_specs={spec.field_key: spec})
        batch = [
            {
                "field_key": spec.field_key,
                "spec": spec,
                "tool_trace": {
                    "evidence_bundle": {
                        "field_key": spec.field_key,
                        "allowed_refs": {"events": [], "relationships": [], "vlm_observations": [], "group_artifacts": [], "feature_refs": []},
                        "supporting_refs": {"events": [], "relationships": [], "vlm_observations": [], "group_artifacts": [], "feature_refs": []},
                        "source_coverage": {"events": 0, "relationships": 0, "vlm_observations": 0, "group_artifacts": 0, "feature_refs": 0},
                        "compact": {"summary": "tarot summary", "source_coverage": {"events": 0}, "top_candidates": [], "evidence_ids": {}, "representative_events": [], "representative_photos": []},
                    },
                    "stats_bundle": {"support_count": 1},
                    "ownership_bundle": {"field_key": spec.field_key, "ownership_signal": "owned_or_used", "candidate_signals": []},
                    "counter_bundle": {"field_key": spec.field_key, "conflict_types": [], "conflict_strength": 0, "conflict_summary": "", "contradicting_ids": []},
                },
            }
        ]

        class StringJsonLLM:
            def _call_llm_via_official_api(self, prompt: str, response_mime_type: str = None):
                return """```json
{
  "fields": {
    "long_term_facts.hobbies.interests": {
      "value": ["tarot_reading"],
      "confidence": 0.73,
      "reasoning": "近期出现稳定兴趣线索。",
      "supporting_ref_ids": [],
      "contradicting_ref_ids": [],
      "null_reason": null
    }
  }
}
```"""

        result = agent._run_batch(
            domain_spec={"display_name": "Taste & Interests"},
            batch=batch,
            context={},
            profile_state=ProfileState(structured_profile={}),
            llm_processor=StringJsonLLM(),
        )

        self.assertEqual(result[spec.field_key]["value"], ["tarot_reading"])

    def test_run_batch_recovers_fields_list_response_instead_of_offline_fallback(self) -> None:
        from services.memory_pipeline.profile_agent import ProfileAgent

        spec = FieldSpec(
            field_key="long_term_facts.hobbies.interests",
            risk_level="P1",
            allowed_sources=["event"],
            strong_evidence=["同类活动跨事件重复"],
            cot_steps=["有任何兴趣线索就可输出"],
            owner_resolution_steps=[],
            time_reasoning_steps=[],
            counter_evidence_checks=[],
            weak_evidence=[],
            hard_blocks=[],
            owner_checks=[],
            time_layer_rule="flexible",
            null_preferred_when=[],
            reflection_questions=[],
            reflection_rounds=1,
            requires_social_media=False,
        )
        agent = ProfileAgent(field_specs={spec.field_key: spec})
        batch = [
            {
                "field_key": spec.field_key,
                "spec": spec,
                "tool_trace": {
                    "evidence_bundle": {
                        "field_key": spec.field_key,
                        "allowed_refs": {"events": [], "relationships": [], "vlm_observations": [], "group_artifacts": [], "feature_refs": []},
                        "supporting_refs": {"events": [], "relationships": [], "vlm_observations": [], "group_artifacts": [], "feature_refs": []},
                        "source_coverage": {"events": 0, "relationships": 0, "vlm_observations": 0, "group_artifacts": 0, "feature_refs": 0},
                        "compact": {"summary": "tarot summary", "source_coverage": {"events": 0}, "top_candidates": [], "evidence_ids": {}, "representative_events": [], "representative_photos": []},
                    },
                    "stats_bundle": {"support_count": 1},
                    "ownership_bundle": {"field_key": spec.field_key, "ownership_signal": "owned_or_used", "candidate_signals": []},
                    "counter_bundle": {"field_key": spec.field_key, "conflict_types": [], "conflict_strength": 0, "conflict_summary": "", "contradicting_ids": []},
                },
            }
        ]

        class ListFieldsLLM:
            def _call_llm_via_official_api(self, prompt: str, response_mime_type: str = None):
                return {
                    "fields": [
                        {
                            "field_key": "long_term_facts.hobbies.interests",
                            "value": ["tarot_reading"],
                            "confidence": 0.68,
                            "reasoning": "列表格式返回。",
                            "supporting_ref_ids": [],
                            "contradicting_ref_ids": [],
                            "null_reason": None,
                        }
                    ]
                }

        result = agent._run_batch(
            domain_spec={"display_name": "Taste & Interests"},
            batch=batch,
            context={},
            profile_state=ProfileState(structured_profile={}),
            llm_processor=ListFieldsLLM(),
        )

        self.assertEqual(result[spec.field_key]["value"], ["tarot_reading"])

    def test_profile_agent_normalizes_literal_field_key_response(self) -> None:
        from services.memory_pipeline.profile_agent import ProfileAgent

        agent = ProfileAgent(field_specs={})
        batch = [{"field_key": "long_term_facts.social_identity.education"}]
        payload = {
            "field_key": {
                "value": "college_student",
                "confidence": 0.8,
                "reasoning": "test",
                "supporting_ref_ids": [],
                "contradicting_ref_ids": [],
                "null_reason": None,
            }
        }

        normalized = agent._normalize_batch_result(batch, payload)
        self.assertIn("long_term_facts.social_identity.education", normalized)
        self.assertEqual(normalized["long_term_facts.social_identity.education"]["value"], "college_student")

    def test_profile_agent_prompt_does_not_inject_null_preferred_or_reflection_language(self) -> None:
        from services.memory_pipeline.profile_agent import ProfileAgent

        spec = FieldSpec(
            field_key="long_term_facts.hobbies.interests",
            risk_level="P1",
            allowed_sources=["event", "vlm", "feature"],
            strong_evidence=["同类活动跨事件重复"],
            cot_steps=["有任何兴趣线索就可输出，confidence反映投入强度"],
            owner_resolution_steps=["确认主体归属主角本人"],
            time_reasoning_steps=["允许单事件证据输出，但 confidence 较低"],
            counter_evidence_checks=["检查是否只是一次体验"],
            weak_evidence=[],
            hard_blocks=[],
            owner_checks=[],
            time_layer_rule="flexible",
            null_preferred_when=["证据不足时优先 null"],
            reflection_questions=["是不是一次体验"],
            reflection_rounds=2,
            requires_social_media=False,
        )
        agent = ProfileAgent(field_specs={spec.field_key: spec})
        profile_state = ProfileState(structured_profile={})
        batch = [
            {
                "field_key": spec.field_key,
                "spec": spec,
                "tool_trace": {
                    "evidence_bundle": {
                        "field_key": spec.field_key,
                        "allowed_refs": {
                            "events": [{"event_id": "EVT_001", "signal": "VERY_RAW_SIGNAL_SHOULD_NOT_APPEAR"}],
                            "relationships": [],
                            "vlm_observations": [],
                            "group_artifacts": [],
                            "feature_refs": [],
                        },
                        "supporting_refs": {
                            "events": [{"event_id": "EVT_001", "signal": "tarot reading"}],
                            "relationships": [],
                            "vlm_observations": [],
                            "group_artifacts": [],
                            "feature_refs": [],
                        },
                        "source_coverage": {"events": 1},
                        "compact": {
                            "summary": "tarot interest evidence",
                            "source_coverage": {"events": 1},
                            "top_candidates": [{"topic_name": "tarot_reading"}],
                            "evidence_ids": {"event_ids": ["EVT_001"], "photo_ids": [], "person_ids": [], "group_ids": [], "feature_names": []},
                            "representative_events": [{"event_id": "EVT_001", "summary": "tarot reading repeated"}],
                            "representative_photos": [],
                        },
                    },
                    "stats_bundle": {
                        "support_count": 1,
                        "recent_topic_summary": {"top_topics": [{"topic_name": "tarot_reading", "event_count": 1}]},
                    },
                    "ownership_bundle": {
                        "field_key": spec.field_key,
                        "ownership_signal": "owned_or_used",
                        "candidate_signals": [{"candidate": "tarot_reading", "signal": "owned_or_used"}],
                        "ownership_refs": [{"photo_id": "OWNERSHIP_RAW_REF"}],
                    },
                    "counter_bundle": {
                        "field_key": spec.field_key,
                        "conflict_types": [],
                        "conflict_strength": 0,
                        "conflict_summary": "",
                        "contradicting_ids": ["EVT_COUNTER_1"],
                        "contradicting_refs": [{"photo_id": "PHOTO_RAW_1", "signal": "RAW_COUNTER_SHOULD_NOT_APPEAR"}],
                    },
                },
            }
        ]

        prompt = agent._build_batch_prompt(
            domain_spec={"display_name": "Taste & Interests"},
            batch=batch,
            profile_state=profile_state,
        )

        self.assertNotIn("Null Preferred", prompt)
        self.assertNotIn("reflection_questions", prompt)
        self.assertNotIn("是不是一次体验", prompt)
        self.assertIn("只要存在任何可靠支持证据", prompt)
        self.assertNotIn("VERY_RAW_SIGNAL_SHOULD_NOT_APPEAR", prompt)
        self.assertNotIn("OWNERSHIP_RAW_REF", prompt)
        self.assertNotIn("RAW_COUNTER_SHOULD_NOT_APPEAR", prompt)
        self.assertIn("owned_or_used", prompt)

    def test_offline_brand_preference_respects_candidate_ownership_signals(self) -> None:
        from services.memory_pipeline.profile_agent import ProfileAgent

        spec = FieldSpec(
            field_key="long_term_facts.material.brand_preference",
            risk_level="P1",
            allowed_sources=["event", "vlm"],
            strong_evidence=[],
            cot_steps=[],
            owner_resolution_steps=[],
            time_reasoning_steps=[],
            counter_evidence_checks=[],
            weak_evidence=[],
            hard_blocks=[],
            owner_checks=[],
            time_layer_rule="flexible",
            null_preferred_when=[],
            reflection_questions=[],
            reflection_rounds=1,
            requires_social_media=False,
        )
        agent = ProfileAgent(field_specs={spec.field_key: spec})
        tool_trace = {
            "evidence_bundle": {
                "field_key": spec.field_key,
                "allowed_refs": {"events": [], "relationships": [], "vlm_observations": [], "group_artifacts": [], "feature_refs": []},
                "supporting_refs": {"events": [], "relationships": [], "vlm_observations": [], "group_artifacts": [], "feature_refs": []},
                "source_coverage": {"events": 0, "relationships": 0, "vlm_observations": 0, "group_artifacts": 0, "feature_refs": 0},
                "compact": {"summary": "brand summary", "source_coverage": {}, "top_candidates": [], "evidence_ids": {}, "representative_events": [], "representative_photos": []},
            },
            "stats_bundle": {
                "brand_summary": {
                    "top_brands": [
                        {"brand_name": "HUAWEI", "source_count": 2, "event_count": 2, "scene_count": 2},
                        {"brand_name": "CAMEL", "source_count": 3, "event_count": 2, "scene_count": 2},
                    ]
                }
            },
            "ownership_bundle": {
                "field_key": spec.field_key,
                "ownership_signal": "owned_or_used",
                "candidate_signals": [
                    {"candidate": "HUAWEI", "signal": "owned_or_used"},
                    {"candidate": "CAMEL", "signal": "other_person"},
                ],
            },
            "counter_bundle": {
                "field_key": spec.field_key,
                "conflict_types": [],
                "conflict_strength": 0,
                "conflict_summary": "",
                "contradicting_ids": [],
            },
        }

        result = agent._offline_field_output(spec.field_key, tool_trace)

        self.assertEqual(result["value"], ["HUAWEI"])

    def test_offline_income_model_uses_work_signal_summary(self) -> None:
        from services.memory_pipeline.profile_agent import ProfileAgent

        spec = FieldSpec(
            field_key="long_term_facts.material.income_model",
            risk_level="P1",
            allowed_sources=["event", "vlm"],
            strong_evidence=[],
            cot_steps=[],
            owner_resolution_steps=[],
            time_reasoning_steps=[],
            counter_evidence_checks=[],
            weak_evidence=[],
            hard_blocks=[],
            owner_checks=[],
            time_layer_rule="flexible",
            null_preferred_when=[],
            reflection_questions=[],
            reflection_rounds=1,
            requires_social_media=False,
        )
        agent = ProfileAgent(field_specs={spec.field_key: spec})
        tool_trace = {
            "evidence_bundle": {
                "field_key": spec.field_key,
                "allowed_refs": {"events": [], "relationships": [], "vlm_observations": [], "group_artifacts": [], "feature_refs": []},
                "supporting_refs": {"events": [], "relationships": [], "vlm_observations": [], "group_artifacts": [], "feature_refs": []},
                "source_coverage": {"events": 0, "relationships": 0, "vlm_observations": 0, "group_artifacts": 0, "feature_refs": 0},
                "compact": {"summary": "income summary", "source_coverage": {}, "top_candidates": [], "evidence_ids": {}, "representative_events": [], "representative_photos": []},
            },
            "stats_bundle": {
                "work_signal_summary": {
                    "payroll_sheet": 1,
                    "attendance_sheet": 1,
                    "punch_in_scene": 1,
                    "operation_scene": 2,
                    "coworker_cluster": 0,
                    "total_signal_count": 5,
                }
            },
            "ownership_bundle": {"field_key": spec.field_key, "ownership_signal": "owned_or_used", "candidate_signals": []},
            "counter_bundle": {"field_key": spec.field_key, "conflict_types": [], "conflict_strength": 0, "conflict_summary": "", "contradicting_ids": []},
        }

        result = agent._offline_field_output(spec.field_key, tool_trace)

        self.assertEqual(result["value"], "salary")

    def test_offline_location_anchors_does_not_special_case_yuhuan(self) -> None:
        from services.memory_pipeline.profile_agent import ProfileAgent

        spec = FieldSpec(
            field_key="long_term_facts.geography.location_anchors",
            risk_level="P1",
            allowed_sources=["event", "vlm"],
            strong_evidence=[],
            cot_steps=[],
            owner_resolution_steps=[],
            time_reasoning_steps=[],
            counter_evidence_checks=[],
            weak_evidence=[],
            hard_blocks=[],
            owner_checks=[],
            time_layer_rule="flexible",
            null_preferred_when=[],
            reflection_questions=[],
            reflection_rounds=1,
            requires_social_media=False,
        )
        agent = ProfileAgent(field_specs={spec.field_key: spec})
        tool_trace = {
            "evidence_bundle": {
                "field_key": spec.field_key,
                "allowed_refs": {"events": [], "relationships": [], "vlm_observations": [], "group_artifacts": [], "feature_refs": []},
                "supporting_refs": {"events": [], "relationships": [], "vlm_observations": [], "group_artifacts": [], "feature_refs": []},
                "source_coverage": {"events": 0, "relationships": 0, "vlm_observations": 0, "group_artifacts": 0, "feature_refs": 0},
                "compact": {"summary": "location summary", "source_coverage": {}, "top_candidates": [], "evidence_ids": {}, "representative_events": [], "representative_photos": []},
            },
            "stats_bundle": {
                "location_summary": {
                    "top_city_candidates": [
                        {
                            "city_name": "玉环",
                            "primary_role": "named_place",
                            "event_count": 1,
                            "photo_count": 2,
                            "city_hit_count": 2,
                            "window_count": 1,
                        }
                    ]
                }
            },
            "ownership_bundle": {"field_key": spec.field_key, "ownership_signal": "owned_or_used", "candidate_signals": []},
            "counter_bundle": {"field_key": spec.field_key, "conflict_types": [], "conflict_strength": 0, "conflict_summary": "", "contradicting_ids": []},
        }

        result = agent._offline_field_output(spec.field_key, tool_trace)

        self.assertEqual(result["value"], ["玉环"])

    def test_profile_agent_keeps_non_null_value_even_if_model_returns_null_reason(self) -> None:
        from services.memory_pipeline.profile_agent import ProfileAgent

        spec = FieldSpec(
            field_key="long_term_facts.hobbies.interests",
            risk_level="P1",
            allowed_sources=["event"],
            strong_evidence=["同类活动跨事件重复"],
            cot_steps=["有任何兴趣线索就可输出，confidence反映投入强度"],
            owner_resolution_steps=[],
            time_reasoning_steps=[],
            counter_evidence_checks=[],
            weak_evidence=[],
            hard_blocks=[],
            owner_checks=[],
            time_layer_rule="flexible",
            null_preferred_when=[],
            reflection_questions=[],
            reflection_rounds=1,
            requires_social_media=False,
        )
        agent = ProfileAgent(field_specs={spec.field_key: spec})
        tool_trace = {
            "evidence_bundle": {
                "field_key": spec.field_key,
                "ref_index": {
                    "EVT_001": {"event_id": "EVT_001", "signal": "tarot reading"},
                },
                "allowed_refs": {
                    "events": [{"event_id": "EVT_001", "signal": "tarot reading"}],
                    "relationships": [],
                    "vlm_observations": [],
                    "group_artifacts": [],
                    "feature_refs": [],
                },
                "supporting_refs": {
                    "events": [{"event_id": "EVT_001", "signal": "tarot reading"}],
                    "relationships": [],
                    "vlm_observations": [],
                    "group_artifacts": [],
                    "feature_refs": [],
                },
            }
        }
        draft = {
            "value": ["tarot_reading"],
            "confidence": 0.62,
            "evidence": {},
        }
        field_output = {
            "value": ["tarot_reading"],
            "confidence": 0.62,
            "reasoning": "近期重复出现 tarot reading 线索。",
            "supporting_ref_ids": ["EVT_001"],
            "contradicting_ref_ids": [],
            "null_reason": "expression_event_conflict",
        }

        final, null_reason = agent._build_final(spec.field_key, draft, tool_trace, field_output)

        self.assertEqual(final["value"], ["tarot_reading"])
        self.assertGreater(final["confidence"], 0.0)
        self.assertIn("expression_event_conflict", final["evidence"]["constraint_notes"])
        self.assertEqual(null_reason, "expression_event_conflict")

    def test_offline_batch_fallback_recalls_key_fields_without_llm(self) -> None:
        events = [
            Event(
                event_id="EVT_001",
                date="2026-03-01",
                time_range="14:00 - 16:00",
                duration="2小时",
                title="Nike 生活记录",
                type="日常",
                participants=["Person_001"],
                location="朝阳社区",
                description="主角在朝阳社区背着 Nike bag，穿着 Nike shoes。",
                photo_count=3,
                confidence=0.88,
                reason="",
                narrative="",
                narrative_synthesis="朝阳社区重复出现 Nike 使用线索",
                tags=["tarot_reading"],
            ),
            Event(
                event_id="EVT_002",
                date="2026-03-10",
                time_range="15:00 - 17:00",
                duration="2小时",
                title="Nike 日常穿搭",
                type="日常",
                participants=["Person_001"],
                location="朝阳社区",
                description="主角在朝阳社区穿着 Nike hoodie，继续记录 tarot reading。",
                photo_count=2,
                confidence=0.86,
                reason="",
                narrative="",
                narrative_synthesis="同地点同品牌重复出现",
                tags=["tarot_reading"],
            ),
            Event(
                event_id="EVT_003",
                date="2026-03-20",
                time_range="19:00 - 20:00",
                duration="1小时",
                title="近期兴趣追踪",
                type="兴趣",
                participants=["Person_001"],
                location="朝阳社区",
                description="继续 tarot reading 主题，形成最近窗口重复兴趣。",
                photo_count=1,
                confidence=0.82,
                reason="",
                narrative="",
                narrative_synthesis="近期主题持续",
                tags=["tarot_reading"],
            ),
        ]

        vlm_results = [
            {
                "photo_id": "PHOTO_001",
                "timestamp": "2026-03-01T14:20:00",
                "vlm_analysis": {
                    "summary": "主角在朝阳社区展示 Nike bag。",
                    "people": [{"person_id": "Person_001"}],
                    "scene": {"location_detected": "朝阳社区"},
                    "event": {"activity": "tarot reading"},
                    "details": ["Nike shoes", "tarot reading notes"],
                },
            },
            {
                "photo_id": "PHOTO_002",
                "timestamp": "2026-03-10T15:30:00",
                "vlm_analysis": {
                    "summary": "主角继续在朝阳社区使用 Nike hoodie。",
                    "people": [{"person_id": "Person_001"}],
                    "scene": {"location_detected": "朝阳社区"},
                    "event": {"activity": "tarot reading"},
                    "details": ["Nike hoodie", "tarot reading cards"],
                },
            },
        ]

        state = MemoryState(
            photos=[],
            face_db={},
            vlm_results=vlm_results,
            primary_decision={
                "mode": "person_id",
                "primary_person_id": "Person_001",
                "confidence": 0.95,
                "evidence": {},
                "reasoning": "test",
            },
            events=events,
            relationships=[],
            groups=[],
        )
        state.profile_context = build_profile_context(state)

        result = generate_structured_profile(state, llm_processor=None)
        profile = result["structured"]

        self.assertIsNotNone(profile["long_term_facts"]["material"]["brand_preference"]["value"])
        self.assertIsNotNone(profile["long_term_facts"]["geography"]["location_anchors"]["value"])
        self.assertIsNotNone(profile["short_term_facts"]["recent_interests"]["value"])

    def test_structured_profile_only_keeps_traceable_evidence_while_field_decisions_keep_full_refs(self) -> None:
        events = [
            Event(
                event_id="EVT_001",
                date="2026-03-01",
                time_range="14:00 - 16:00",
                duration="2小时",
                title="Nike 生活记录",
                type="日常",
                participants=["Person_001"],
                location="朝阳社区",
                description="主角在朝阳社区背着 Nike bag，穿着 Nike shoes。",
                photo_count=3,
                confidence=0.88,
                reason="",
                narrative="",
                narrative_synthesis="朝阳社区重复出现 Nike 使用线索",
                tags=["tarot_reading"],
            ),
            Event(
                event_id="EVT_002",
                date="2026-03-10",
                time_range="15:00 - 17:00",
                duration="2小时",
                title="Nike 日常穿搭",
                type="日常",
                participants=["Person_001"],
                location="朝阳社区",
                description="主角在朝阳社区穿着 Nike hoodie，继续记录 tarot reading。",
                photo_count=2,
                confidence=0.86,
                reason="",
                narrative="",
                narrative_synthesis="同地点同品牌重复出现",
                tags=["tarot_reading"],
            ),
        ]

        vlm_results = [
            {
                "photo_id": "PHOTO_001",
                "timestamp": "2026-03-01T14:20:00",
                "vlm_analysis": {
                    "summary": "主角在朝阳社区展示 Nike bag。",
                    "people": [{"person_id": "Person_001"}],
                    "scene": {"location_detected": "朝阳社区"},
                    "event": {"activity": "tarot reading"},
                    "details": ["Nike shoes", "tarot reading notes"],
                },
            }
        ]

        state = MemoryState(
            photos=[],
            face_db={},
            vlm_results=vlm_results,
            primary_decision={
                "mode": "person_id",
                "primary_person_id": "Person_001",
                "confidence": 0.95,
                "evidence": {},
                "reasoning": "test",
            },
            events=events,
            relationships=[],
            groups=[],
        )
        state.profile_context = build_profile_context(state)

        result = generate_structured_profile(state, llm_processor=None)
        brand_tag = result["structured"]["long_term_facts"]["material"]["brand_preference"]
        brand_decision = next(
            item for item in result["field_decisions"]
            if item["field_key"] == "long_term_facts.material.brand_preference"
        )

        self.assertIn("photo_ids", brand_tag["evidence"])
        self.assertIn("event_ids", brand_tag["evidence"])
        self.assertIn("summary", brand_tag["evidence"])
        self.assertNotIn("supporting_refs", brand_tag["evidence"])
        self.assertNotIn("events", brand_tag["evidence"])
        self.assertNotIn("vlm_observations", brand_tag["evidence"])

        self.assertIn("supporting_refs", brand_decision["final"]["evidence"])
        self.assertIn("events", brand_decision["final"]["evidence"])


if __name__ == "__main__":
    unittest.main()
