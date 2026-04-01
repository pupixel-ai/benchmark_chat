from __future__ import annotations

import json
import os
import sys
import typing
import unittest
from dataclasses import dataclass, field
from unittest.mock import patch

from models import Relationship
from services.memory_pipeline.types import RelationshipDossier


class DownstreamProfileAgentIntegrationTests(unittest.TestCase):
    def _clear_profile_agent_modules(self) -> None:
        for module_name in list(sys.modules):
            if module_name == "profile_agent" or module_name.startswith("profile_agent."):
                sys.modules.pop(module_name, None)

    def test_downstream_runtime_loader_supports_profile_agent_two_root(self) -> None:
        import services.memory_pipeline.downstream_audit as audit

        storage_cls, critic_cls, judge_cls = audit._load_profile_agent_runtime()

        self.assertEqual(storage_cls.__module__, "profile_agent.storage")
        self.assertEqual(critic_cls.__module__, "profile_agent.agents.critic")
        self.assertEqual(judge_cls.__module__, "profile_agent.agents.judge")

    def test_prepare_profile_agent_runtime_env_clears_legacy_gemini_vars(self) -> None:
        import services.memory_pipeline.downstream_audit as audit

        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "sk-or-test",
                "GEMINI_API_KEY": "sk-ant-legacy",
                "GOOGLE_GEMINI_BASE_URL": "https://legacy.example/gemini",
                "GEMINI_API_BASE_URL": "https://legacy.example/direct",
            },
            clear=True,
        ):
            audit._prepare_profile_agent_runtime_env()

            self.assertEqual(os.environ["OPENROUTER_API_KEY"], "sk-or-test")
            self.assertEqual(os.environ["OPENROUTER_BASE_URL"], "https://openrouter.ai/api/v1")
            self.assertNotIn("GEMINI_API_KEY", os.environ)
            self.assertNotIn("GOOGLE_GEMINI_BASE_URL", os.environ)
            self.assertNotIn("GEMINI_API_BASE_URL", os.environ)

    def test_loaded_profile_agent_api_uses_openrouter_chat_completions(self) -> None:
        import services.memory_pipeline.downstream_audit as audit

        class FakeHTTPResponse:
            def __init__(self, payload: dict[str, typing.Any]):
                self._payload = payload

            def read(self) -> bytes:
                return json.dumps(self._payload, ensure_ascii=False).encode("utf-8")

            def __enter__(self) -> "FakeHTTPResponse":
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

        captured: dict[str, typing.Any] = {}

        def fake_urlopen(req, timeout=180):
            captured["url"] = req.full_url
            captured["headers"] = {key.lower(): value for key, value in req.header_items()}
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeHTTPResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "{\"status\":\"ok\"}",
                            }
                        }
                    ]
                }
            )

        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "sk-or-test",
                "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
                "GEMINI_API_KEY": "sk-ant-legacy",
                "GOOGLE_GEMINI_BASE_URL": "https://legacy.example/gemini",
            },
            clear=True,
        ):
            audit._prepare_profile_agent_runtime_env()
            self._clear_profile_agent_modules()
            audit._load_profile_agent_runtime()

            import profile_agent.api as api

            with patch.object(api.urllib.request, "urlopen", side_effect=fake_urlopen):
                result = api.call_gemini("hello world")

        self.assertEqual(result, "{\"status\":\"ok\"}")
        self.assertEqual(captured["url"], "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(captured["headers"]["authorization"], "Bearer sk-or-test")
        self.assertEqual(captured["payload"]["model"], "google/gemini-3.1-flash-lite-preview")
        self.assertEqual(captured["payload"]["messages"][-1]["content"], "hello world")

    def test_loaded_profile_agent_api_switches_to_free_model_after_primary_forbidden(self) -> None:
        import services.memory_pipeline.downstream_audit as audit

        class FakeHTTPResponse:
            def __init__(self, payload: dict[str, typing.Any]):
                self._payload = payload

            def read(self) -> bytes:
                return json.dumps(self._payload, ensure_ascii=False).encode("utf-8")

            def __enter__(self) -> "FakeHTTPResponse":
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

        attempts: list[tuple[str, int]] = []

        def fake_urlopen(req, timeout=180):
            payload = json.loads(req.data.decode("utf-8"))
            model = str(payload.get("model") or "")
            attempts.append((model, timeout))
            if model == "google/gemini-3.1-flash-lite-preview":
                raise RuntimeError("HTTP Error 403: Forbidden")
            return FakeHTTPResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "{\"status\":\"fallback_ok\"}",
                            }
                        }
                    ]
                }
            )

        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "sk-or-test",
                "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
                "PROFILE_AGENT_MODEL_CANDIDATES": "google/gemini-3.1-flash-lite-preview,google/gemini-2.0-flash-exp:free",
                "PROFILE_AGENT_REQUEST_MAX_RETRIES": "1",
                "PROFILE_AGENT_REQUEST_TIMEOUT_SECONDS": "33",
            },
            clear=True,
        ):
            audit._prepare_profile_agent_runtime_env()
            self._clear_profile_agent_modules()
            audit._load_profile_agent_runtime()

            import profile_agent.api as api

            with patch.object(api.urllib.request, "urlopen", side_effect=fake_urlopen):
                result = api.call_gemini("switch to free")

        self.assertEqual(result, "{\"status\":\"fallback_ok\"}")
        self.assertEqual(attempts[0][0], "google/gemini-3.1-flash-lite-preview")
        self.assertEqual(attempts[1][0], "google/gemini-2.0-flash-exp:free")
        self.assertEqual(attempts[0][1], 33)
        self.assertEqual(attempts[1][1], 33)

    def test_profile_adapter_maps_only_selected_facts_dimensions(self) -> None:
        from services.memory_pipeline.profile_agent_adapter import build_profile_agent_extractor_outputs

        structured_profile = {
            "long_term_facts": {
                "social_identity": {
                    "education": {
                        "value": "college_student",
                        "confidence": 0.84,
                        "evidence": {
                            "event_ids": ["EVT_001"],
                            "photo_ids": ["PHOTO_001"],
                            "person_ids": ["Person_002"],
                            "feature_names": ["campus_signal"],
                        },
                        "reasoning": "跨事件校园学习场景支持 education。",
                    },
                    "career": {
                        "value": "student_or_early_career",
                        "confidence": 0.72,
                        "evidence": {
                            "event_ids": ["EVT_001A"],
                            "photo_ids": ["PHOTO_001A"],
                            "person_ids": ["Person_002"],
                            "feature_names": ["career_signal"],
                        },
                        "reasoning": "当前主要还是学生主线，但已出现早期职业化线索。",
                    },
                    "career_phase": {
                        "value": "student",
                        "confidence": 0.76,
                        "evidence": {
                            "event_ids": ["EVT_002"],
                            "photo_ids": ["PHOTO_002"],
                            "person_ids": ["Person_002"],
                            "feature_names": ["career_phase_signal"],
                        },
                        "reasoning": "当前仍以校园主线为主，更接近 student。",
                    }
                    },
                "material": {
                    "asset_level": {
                        "value": "middle_class_signal",
                        "confidence": 0.68,
                        "evidence": {
                            "event_ids": ["EVT_003"],
                            "photo_ids": ["PHOTO_003"],
                            "person_ids": ["Person_002"],
                            "feature_names": ["asset_signal"],
                        },
                        "reasoning": "多次日常使用资产线索支持该判断。",
                    },
                },
                "geography": {
                    "location_anchors": {
                        "value": ["campus", "dorm"],
                        "confidence": 0.81,
                        "evidence": {
                            "event_ids": ["EVT_004"],
                            "photo_ids": ["PHOTO_004"],
                            "person_ids": ["Person_002"],
                            "feature_names": ["location_anchor_signal"],
                        },
                        "reasoning": "跨事件出现校园和宿舍生活场景。",
                    },
                    "mobility_pattern": {
                        "value": "campus_city_commute",
                        "confidence": 0.63,
                        "evidence": {
                            "event_ids": ["EVT_005"],
                            "photo_ids": ["PHOTO_005"],
                            "person_ids": ["Person_002"],
                            "feature_names": ["mobility_signal"],
                        },
                        "reasoning": "近期主要表现为校区与城市通勤。",
                    },
                },
                "relationships": {
                    "parenting": {
                        "value": "not_parenting",
                        "confidence": 0.74,
                        "evidence": {
                            "event_ids": ["EVT_006"],
                            "photo_ids": ["PHOTO_006"],
                            "person_ids": ["Person_002"],
                            "feature_names": ["parenting_signal"],
                        },
                        "reasoning": "未见稳定育儿日常，当前更接近非育儿状态。",
                    },
                    "pets": {
                        "value": "cat_owner_signal",
                        "confidence": 0.77,
                        "evidence": {
                            "event_ids": ["EVT_007"],
                            "photo_ids": ["PHOTO_007"],
                            "person_ids": ["Person_002"],
                            "feature_names": ["pet_signal"],
                        },
                        "reasoning": "多次家庭场景中出现同一宠物与照料行为。",
                    },
                },
                "hobbies": {
                    "frequent_activities": {
                        "value": ["badminton", "study_group"],
                        "confidence": 0.73,
                        "evidence": {
                            "event_ids": ["EVT_008"],
                            "photo_ids": ["PHOTO_008"],
                            "person_ids": ["Person_002"],
                            "feature_names": ["activity_signal"],
                        },
                        "reasoning": "这些活动跨事件重复出现。",
                    },
                },
            },
            "long_term_expression": {
                "attitude_style": {
                    "value": "playful",
                    "confidence": 0.66,
                    "evidence": {
                        "photo_ids": ["PHOTO_010"],
                        "person_ids": ["Person_002"],
                        "feature_names": ["attitude_style_signal"],
                    },
                    "reasoning": "自拍和互动中呈现轻松 playful 风格。",
                },
            },
        }

        outputs = build_profile_agent_extractor_outputs(
            primary_decision=None,
            relationships=[],
            structured_profile=structured_profile,
        )

        profile_tags = outputs["profile"]["tags"]
        by_dimension = {tag["dimension"]: tag for tag in profile_tags}

        self.assertIn("社会身份>教育背景", by_dimension)
        self.assertIn("社会身份>职业/专业", by_dimension)
        self.assertIn("社会身份>职业阶段", by_dimension)
        self.assertIn("物质>资产", by_dimension)
        self.assertIn("地理>地点锚点", by_dimension)
        self.assertIn("出行>mobility_pattern", by_dimension)
        self.assertIn("家庭状况>parenting", by_dimension)
        self.assertIn("物质>宠物归属", by_dimension)
        self.assertIn("长期爱好>活动", by_dimension)
        self.assertEqual(by_dimension["社会身份>教育背景"]["value"], "college_student")
        self.assertIn("reasoning", by_dimension["社会身份>职业阶段"])
        self.assertEqual(
            by_dimension["地理>地点锚点"]["evidence"][0]["photo_id"],
            "PHOTO_004",
        )
        self.assertEqual(
            by_dimension["地理>地点锚点"]["evidence"][0]["person_id"],
            "Person_002",
        )
        self.assertEqual(
            by_dimension["地理>地点锚点"]["evidence"][0]["feature_names"],
            ["location_anchor_signal"],
        )
        self.assertNotIn("长期表达>态度风格", by_dimension)

    def test_downstream_report_marks_unmapped_profile_fields_as_not_audited(self) -> None:
        from services.memory_pipeline.downstream_audit import build_downstream_audit_report

        structured_profile = {
            "long_term_facts": {
                "social_identity": {
                    "education": {
                        "value": "college_student",
                        "confidence": 0.84,
                        "evidence": {
                            "event_ids": ["EVT_001"],
                            "photo_ids": ["PHOTO_001"],
                            "person_ids": ["Person_002"],
                            "feature_names": ["campus_signal"],
                        },
                        "reasoning": "跨事件校园学习场景支持 education。",
                    }
                }
            },
            "long_term_expression": {
                "attitude_style": {
                    "value": "playful",
                    "confidence": 0.66,
                    "evidence": {
                        "photo_ids": ["PHOTO_010"],
                        "person_ids": ["Person_002"],
                        "feature_names": ["attitude_style_signal"],
                    },
                    "reasoning": "自拍和互动中呈现轻松 playful 风格。",
                }
            },
        }

        adapter_outputs = {
            "protagonist": {"tags": [], "reasoning_trace": ""},
            "relationship": {"tags": [], "reasoning_trace": ""},
            "profile": {
                "tags": [
                    {
                        "dimension": "社会身份>教育背景",
                        "value": "college_student",
                        "confidence": 84,
                        "stability": "long_term",
                        "evidence": [{"event_id": "EVT_001"}],
                        "extraction_gap": "",
                        "reasoning": "跨事件校园学习场景支持 education。",
                    }
                ],
                "reasoning_trace": "",
            },
        }
        audit_outputs = {
            "protagonist": {"critic_output": {"challenges": []}, "judge_output": {"decisions": [], "hard_cases": []}},
            "relationship": {"critic_output": {"challenges": []}, "judge_output": {"decisions": [], "hard_cases": []}},
            "profile": {"critic_output": {"challenges": []}, "judge_output": {"decisions": [], "hard_cases": []}},
        }

        report = build_downstream_audit_report(
            album_id="album_not_audited",
            primary_decision={},
            relationships=[],
            structured_profile=structured_profile,
            adapter_outputs=adapter_outputs,
            audit_outputs=audit_outputs,
            merged_outputs={
                "protagonist": {"agent_type": "protagonist", "tags": []},
                "relationship": {"agent_type": "relationship", "tags": []},
                "profile": {
                    "agent_type": "profile",
                    "tags": list(adapter_outputs["profile"]["tags"]),
                },
            },
            storage_saved=False,
        )

        not_audited = report["profile"]["not_audited"]
        self.assertEqual(report["metadata"]["audit_mode"], "selective_profile_domain_rules_facts_only")
        self.assertTrue(any(item["field_key"] == "long_term_expression.attitude_style" for item in not_audited))

    def test_loaded_profile_agent_schema_accepts_multi_anchor_evidence(self) -> None:
        import services.memory_pipeline.downstream_audit as audit

        audit._load_profile_agent_runtime()

        from profile_agent.schemas import Evidence, HardCase

        evidence = Evidence(
            event_id="EVT_001",
            photo_id="PHOTO_001",
            person_id="Person_002",
            feature_names=["mood_signal"],
            description="最近事件中出现 excited 线索。",
        )
        hints = typing.get_type_hints(HardCase)

        self.assertEqual(evidence.photo_id, "PHOTO_001")
        self.assertEqual(evidence.person_id, "Person_002")
        self.assertEqual(evidence.feature_names, ["mood_signal"])
        self.assertEqual(str(hints["evidence_used"]), "list[dict]")

    def test_build_downstream_audit_report_includes_profile_backflow_actions(self) -> None:
        from services.memory_pipeline.downstream_audit import build_downstream_audit_report

        structured_profile = {
            "long_term_facts": {
                "social_identity": {
                    "education": {
                        "value": "college_student",
                        "confidence": 0.84,
                        "evidence": {"event_ids": ["EVT_001"]},
                        "reasoning": "跨事件校园学习场景支持 education。",
                    },
                    "career_phase": {
                        "value": "student",
                        "confidence": 0.72,
                        "evidence": {"event_ids": ["EVT_002"]},
                        "reasoning": "当前仍以校园主线为主。",
                    },
                }
            }
        }
        adapter_outputs = {
            "protagonist": {"tags": [], "reasoning_trace": ""},
            "relationship": {"tags": [], "reasoning_trace": ""},
            "profile": {
                "tags": [
                    {
                        "dimension": "社会身份>教育背景",
                        "value": "college_student",
                        "confidence": 84,
                        "stability": "long_term",
                        "evidence": [{"event_id": "EVT_001"}],
                        "extraction_gap": "",
                        "reasoning": "跨事件校园学习场景支持 education。",
                    },
                    {
                        "dimension": "社会身份>职业阶段",
                        "value": "student",
                        "confidence": 72,
                        "stability": "long_term",
                        "evidence": [{"event_id": "EVT_002"}],
                        "extraction_gap": "",
                        "reasoning": "当前仍以校园主线为主。",
                    },
                ],
                "reasoning_trace": "",
            },
        }
        audit_outputs = {
            "protagonist": {"critic_output": {"challenges": []}, "judge_output": {"decisions": [], "hard_cases": []}},
            "relationship": {"critic_output": {"challenges": []}, "judge_output": {"decisions": [], "hard_cases": []}},
            "profile": {
                "critic_output": {"challenges": []},
                "judge_output": {
                    "decisions": [
                        {
                            "dimension": "社会身份>教育背景",
                            "value": None,
                            "verdict": "nullify",
                            "final_confidence": 0,
                            "reason": "证据不满足审计规则",
                        },
                        {
                            "dimension": "社会身份>职业阶段",
                            "value": "student",
                            "verdict": "downgrade",
                            "final_confidence": 58,
                            "reason": "更适合作为短期状态保留",
                        },
                    ],
                    "hard_cases": [],
                },
            },
        }
        merged_outputs = {
            "protagonist": {"agent_type": "protagonist", "tags": []},
            "relationship": {"agent_type": "relationship", "tags": []},
            "profile": {
                "agent_type": "profile",
                "tags": [
                    {
                        "dimension": "社会身份>教育背景",
                        "value": None,
                        "confidence": 84,
                        "stability": "long_term",
                        "evidence": [{"event_id": "EVT_001"}],
                        "reasoning": "跨事件校园学习场景支持 education。",
                    },
                    {
                        "dimension": "社会身份>职业阶段",
                        "value": "student",
                        "confidence": 72,
                        "stability": "short_term",
                        "evidence": [{"event_id": "EVT_002"}],
                        "reasoning": "当前仍以校园主线为主。",
                    },
                ],
            },
        }

        report = build_downstream_audit_report(
            album_id="album_demo",
            primary_decision={},
            relationships=[],
            structured_profile=structured_profile,
            adapter_outputs=adapter_outputs,
            audit_outputs=audit_outputs,
            merged_outputs=merged_outputs,
            storage_saved=True,
        )

        backflow = report["backflow"]["profile"]["field_actions"]
        by_field = {item["field_key"]: item for item in backflow}
        self.assertEqual(report["backflow"]["album_id"], "album_demo")
        self.assertTrue(report["backflow"]["storage_saved"])
        self.assertEqual(by_field["long_term_facts.social_identity.education"]["verdict"], "nullify")
        self.assertEqual(by_field["long_term_facts.social_identity.education"]["applied_change"], "nullify_value")
        self.assertEqual(by_field["long_term_facts.social_identity.career_phase"]["verdict"], "downgrade")
        self.assertEqual(by_field["long_term_facts.social_identity.career_phase"]["applied_change"], "annotate_short_term_downgrade")

    def test_run_profile_agent_judges_saves_merged_outputs_to_downstream_storage(self) -> None:
        import services.memory_pipeline.downstream_audit as audit

        @dataclass
        class FakeChallenge:
            target_tag: str
            challenge_type: str
            description: str
            counterfactual: str
            evidence_request: str
            evidence_gate: str = ""
            severity: str = "hard"

        @dataclass
        class FakeCriticOutput:
            challenges: list[FakeChallenge]

            def has_challenges(self) -> bool:
                return bool(self.challenges)

        @dataclass
        class FakeDecision:
            dimension: str
            value: str | None
            verdict: str
            final_confidence: int
            reason: str

        @dataclass
        class FakeJudgeOutput:
            decisions: list[FakeDecision]
            hard_cases: list[dict] = field(default_factory=list)

        class FakeStorage:
            last_instance = None

            def __init__(self):
                self.saved = None
                FakeStorage.last_instance = self

            def save_profile(self, album_id, protagonist, relationships, profile):
                self.saved = {
                    "album_id": album_id,
                    "protagonist": protagonist,
                    "relationships": relationships,
                    "profile": profile,
                }

        class FakeCriticAgent:
            def __init__(self, storage):
                self.storage = storage

            def run(self, extractor_output):
                if extractor_output["agent_type"] == "profile":
                    return FakeCriticOutput(
                        challenges=[
                            FakeChallenge(
                                target_tag="社会身份>教育背景:college_student",
                                challenge_type="overclaim",
                                description="证据不足",
                                counterfactual="也可能只是短期场景",
                                evidence_request="补充跨事件证据",
                            )
                        ]
                    )
                return FakeCriticOutput(challenges=[])

        class FakeJudgeAgent:
            def __init__(self, storage):
                self.storage = storage

            def run(self, extractor_output, critic_output, album_id=""):
                return FakeJudgeOutput(
                    decisions=[
                        FakeDecision(
                            dimension="社会身份>教育背景",
                            value=None,
                            verdict="nullify",
                            final_confidence=0,
                            reason="证据不满足审计规则",
                        )
                    ]
                )

            def run_no_challenges(self, extractor_output):
                return FakeJudgeOutput(decisions=[])

        adapter_outputs = {
            "protagonist": {"agent_type": "protagonist", "tags": [], "reasoning_trace": ""},
            "relationship": {"agent_type": "relationship", "tags": [], "reasoning_trace": ""},
            "profile": {
                "agent_type": "profile",
                "tags": [
                    {
                        "dimension": "社会身份>教育背景",
                        "value": "college_student",
                        "confidence": 84,
                        "stability": "long_term",
                        "evidence": [{"event_id": "EVT_001"}],
                        "extraction_gap": "",
                        "reasoning": "跨事件校园学习场景支持 education。",
                    }
                ],
                "reasoning_trace": "",
            },
        }

        with patch.object(
            audit,
            "_load_profile_agent_runtime",
            return_value=(FakeStorage, FakeCriticAgent, FakeJudgeAgent),
        ):
            audit_outputs, merged_outputs, storage_saved = audit._run_profile_agent_judges(
                adapter_outputs,
                album_id="album_merge",
            )

        self.assertTrue(storage_saved)
        self.assertEqual(
            FakeStorage.last_instance.saved["album_id"],
            "album_merge",
        )
        self.assertIsNone(merged_outputs["profile"]["tags"][0]["value"])
        self.assertEqual(
            audit_outputs["profile"]["judge_output"]["decisions"][0]["verdict"],
            "nullify",
        )

    def test_apply_downstream_protagonist_backflow_demotes_to_photographer_mode(self) -> None:
        from services.memory_pipeline.downstream_audit import apply_downstream_protagonist_backflow

        primary_decision = {
            "mode": "person_id",
            "primary_person_id": "Person_001",
            "confidence": 0.91,
            "evidence": {"photo_ids": ["PHOTO_001"]},
            "reasoning": "自拍与身份锚点都领先。",
        }
        report = {
            "backflow": {
                "protagonist": {
                    "actions": [
                        {
                            "mapped_dimension": "主角>身份确认",
                            "verdict": "nullify",
                            "judge_reason": "证据不足以确认主角身份",
                        }
                    ]
                }
            }
        }

        updated, changed = apply_downstream_protagonist_backflow(primary_decision, report)

        self.assertTrue(changed)
        self.assertEqual(updated["mode"], "photographer_mode")
        self.assertIsNone(updated["primary_person_id"])
        self.assertEqual(updated["confidence"], 0.0)
        self.assertIn("下游 Judge", updated["reasoning"])
        self.assertIn("downstream_judge:nullify", updated["evidence"]["constraint_notes"][0])

    def test_apply_downstream_relationship_backflow_updates_official_relationships_and_dossiers(self) -> None:
        from services.memory_pipeline.downstream_audit import apply_downstream_relationship_backflow

        relationships = [
            Relationship(
                person_id="Person_002",
                relationship_type="bestie",
                intimacy_score=0.82,
                status="stable",
                confidence=0.86,
                reasoning="多场景稳定互动。",
                shared_events=[{"event_id": "EVT_001", "date": "2026-03-01", "narrative": "一起吃饭"}],
                evidence={"constraint_notes": []},
            ),
            Relationship(
                person_id="Person_003",
                relationship_type="romantic",
                intimacy_score=0.91,
                status="growing",
                confidence=0.88,
                reasoning="亲密互动明显。",
                shared_events=[{"event_id": "EVT_002", "date": "2026-03-02", "narrative": "单独约会"}],
                evidence={},
            ),
        ]
        dossiers = [
            RelationshipDossier(
                person_id="Person_002",
                person_kind="real_person",
                memory_value="candidate",
                photo_count=6,
                time_span_days=40,
                recent_gap_days=5,
                monthly_frequency=3.5,
                scene_profile={"scenes": ["campus", "cafe"], "private_scene_ratio": 0.2, "dominant_scene_ratio": 0.5, "with_user_only": True},
                interaction_signals=["selfie_together"],
                shared_events=[{"event_id": "EVT_001"}],
                trend_detail={},
                co_appearing_persons=[],
                anomalies=[],
                evidence_refs=[],
                retention_decision="keep",
                retention_reason="relationship_retained",
                group_eligible=True,
                group_block_reason=None,
                group_weight=0.8,
                relationship_result={"relationship_type": "bestie", "status": "stable", "confidence": 0.86},
            ),
            RelationshipDossier(
                person_id="Person_003",
                person_kind="real_person",
                memory_value="candidate",
                photo_count=5,
                time_span_days=30,
                recent_gap_days=2,
                monthly_frequency=4.0,
                scene_profile={"scenes": ["home", "restaurant"], "private_scene_ratio": 0.8, "dominant_scene_ratio": 0.5, "with_user_only": True},
                interaction_signals=["hug"],
                shared_events=[{"event_id": "EVT_002"}],
                trend_detail={},
                co_appearing_persons=[],
                anomalies=[],
                evidence_refs=[],
                retention_decision="keep",
                retention_reason="relationship_retained",
                group_eligible=False,
                group_block_reason="relationship_type=romantic",
                group_weight=0.0,
                relationship_result={"relationship_type": "romantic", "status": "growing", "confidence": 0.88},
            ),
        ]
        report = {
            "backflow": {
                "relationship": {
                    "actions": [
                        {
                            "person_id": "Person_002",
                            "verdict": "downgrade",
                            "judge_reason": "更接近 close_friend 而非 bestie",
                        },
                        {
                            "person_id": "Person_003",
                            "verdict": "nullify",
                            "judge_reason": "romantic 证据不足",
                        },
                    ]
                }
            }
        }

        updated_relationships, updated_dossiers, changed = apply_downstream_relationship_backflow(
            relationships,
            dossiers,
            report,
        )

        self.assertTrue(changed)
        by_person = {rel.person_id: rel for rel in updated_relationships}
        self.assertEqual(by_person["Person_002"].relationship_type, "close_friend")
        self.assertIn("downstream Judge", by_person["Person_002"].reasoning)
        self.assertIn("downstream_judge:downgrade", by_person["Person_002"].evidence["constraint_notes"][0])
        self.assertNotIn("Person_003", by_person)

        dossier_by_person = {d.person_id: d for d in updated_dossiers}
        self.assertEqual(dossier_by_person["Person_003"].retention_decision, "drop")
        self.assertFalse(dossier_by_person["Person_003"].group_eligible)

    def test_apply_downstream_profile_backflow_updates_structured_profile_and_field_decisions(self) -> None:
        from services.memory_pipeline.downstream_audit import apply_downstream_profile_backflow

        structured_profile = {
            "long_term_facts": {
                "social_identity": {
                    "education": {
                        "value": "college_student",
                        "confidence": 0.84,
                        "evidence": {"event_ids": ["EVT_001"]},
                        "reasoning": "跨事件校园学习场景支持 education。",
                    },
                    "career_phase": {
                        "value": "student",
                        "confidence": 0.72,
                        "evidence": {"event_ids": ["EVT_002"]},
                        "reasoning": "当前仍以校园主线为主。",
                    },
                }
            }
        }
        field_decisions = [
            {
                "field_key": "long_term_facts.social_identity.education",
                "draft": {"value": "college_student"},
                "final": {
                    "value": "college_student",
                    "confidence": 0.84,
                    "evidence": {"event_ids": ["EVT_001"]},
                    "reasoning": "跨事件校园学习场景支持 education。",
                },
                "tool_trace": {},
                "null_reason": None,
            },
            {
                "field_key": "long_term_facts.social_identity.career_phase",
                "draft": {"value": "student"},
                "final": {
                    "value": "student",
                    "confidence": 0.72,
                    "evidence": {"event_ids": ["EVT_002"]},
                    "reasoning": "当前仍以校园主线为主。",
                },
                "tool_trace": {},
                "null_reason": None,
            },
        ]
        report = {
            "backflow": {
                "profile": {
                    "field_actions": [
                        {
                            "field_key": "long_term_facts.social_identity.education",
                            "verdict": "nullify",
                            "judge_reason": "证据不满足审计规则",
                            "applied_change": "nullify_value",
                        },
                        {
                            "field_key": "long_term_facts.social_identity.career_phase",
                            "verdict": "downgrade",
                            "judge_reason": "更适合作为短期状态保留",
                            "applied_change": "annotate_short_term_downgrade",
                        },
                    ]
                }
            }
        }

        updated_profile, updated_decisions = apply_downstream_profile_backflow(
            structured_profile,
            report,
            field_decisions=field_decisions,
        )

        education = updated_profile["long_term_facts"]["social_identity"]["education"]
        career_phase = updated_profile["long_term_facts"]["social_identity"]["career_phase"]
        education_decision = next(item for item in updated_decisions if item["field_key"] == "long_term_facts.social_identity.education")
        career_phase_decision = next(item for item in updated_decisions if item["field_key"] == "long_term_facts.social_identity.career_phase")

        self.assertIsNone(education["value"])
        self.assertIsNone(education_decision["final"]["value"])
        self.assertEqual(education_decision["final_before_backflow"]["value"], "college_student")
        self.assertEqual(education_decision["backflow"]["verdict"], "nullify")
        self.assertEqual(career_phase["value"], "student")
        self.assertIn("downstream Judge 建议降为 short_term", career_phase_decision["final"]["reasoning"])
        self.assertEqual(career_phase_decision["draft"]["value"], "student")
        self.assertEqual(career_phase_decision["backflow"]["verdict"], "downgrade")


if __name__ == "__main__":
    unittest.main()
