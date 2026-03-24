from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
import json
from pathlib import Path

from config import (
    RELATIONSHIP_MAX_RETRIES,
    V0323_LP1_MAX_ATTEMPTS,
    V0323_LP1_MAX_OUTPUT_TOKENS,
    V0323_LP2_TIMEOUT_SCHEDULE_SECONDS,
)
from backend.memory_full_retrieval import build_task_memory_core_payload
from memory_module.query import MemoryQueryService
from models import Photo
from services.v0323.pipeline import V0323PipelineFamily


class StubV0323LLM:
    provider = "openrouter"
    model = "google/gemini-3.1-pro-preview"
    relationship_model = "google/gemini-3.1-pro-preview"

    def __init__(self) -> None:
        self.json_calls: list[dict] = []

    def _lp1_batch_payload(self, batch_id: str) -> dict:
        if batch_id == "BATCH_0001":
            return {
                "events": [
                    {
                        "event_id": "TEMP_EVT_001",
                        "supporting_photo_ids": [f"photo_{index:03d}" for index in range(1, 201)],
                        "meta_info": {
                            "title": "Breakfast",
                            "location_context": "Home",
                            "photo_count": 200,
                        },
                        "objective_fact": {
                            "scene_description": "Breakfast with a friend at home.",
                            "participants": ["Person_001", "Person_002"],
                        },
                        "narrative_synthesis": "Breakfast with friend.",
                        "social_dynamics": [
                            {
                                "target_id": "Person_002",
                                "interaction_type": "meal",
                                "social_clue": "same table",
                                "relation_hypothesis": "friend",
                                "confidence": 0.7,
                            }
                        ],
                        "persona_evidence": {
                            "behavioral": ["hosts breakfast"],
                            "aesthetic": [],
                            "socioeconomic": [],
                        },
                        "tags": ["meal"],
                        "confidence": 0.8,
                        "reason": "same time and place",
                    }
                ]
            }
        return {
            "events": [
                {
                    "event_id": "TEMP_EVT_002",
                    "supporting_photo_ids": [f"photo_{index:03d}" for index in range(193, 202)],
                    "meta_info": {
                        "title": "Extended Breakfast",
                        "location_context": "Home",
                        "photo_count": 9,
                    },
                    "objective_fact": {
                        "scene_description": "The breakfast gathering continues into a longer conversation.",
                        "participants": ["Person_001", "Person_002"],
                    },
                    "narrative_synthesis": "Moved into a longer conversation.",
                    "social_dynamics": [],
                    "persona_evidence": {
                        "behavioral": ["lingers socially"],
                        "aesthetic": [],
                        "socioeconomic": [],
                    },
                    "tags": ["conversation"],
                    "confidence": 0.82,
                    "reason": "cross-batch continuation",
                }
            ]
        }

    def _call_json_prompt(self, prompt: str, **kwargs):
        self.json_calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
        if "Relationship Types" in prompt:
            return {
                "relationship_type": "friend",
                "status": "stable",
                "confidence": 0.72,
                "reason": "shared meal and steady co-presence",
            }
        return {
            "primary_person_id": "Person_001",
            "event_grounded": {
                "behavioral_traits": ["hosts breakfast"],
                "aesthetic_traits": [],
                "socioeconomic_traits": [],
                "activity_patterns": ["meal"],
                "place_patterns": ["Home"],
            },
            "relationship_grounded": {
                "top_relationships": ["Person_002:friend"],
                "social_style": ["small-circle hangouts"],
            },
            "weak_reference": [],
            "summary": "Grounded in breakfast and friend co-presence.",
        }

    def _call_json_prompt_raw_text(self, prompt: str, **kwargs) -> str:
        self.json_calls.append({"prompt": prompt, "kwargs": dict(kwargs), "raw_text_mode": True})
        if "LP1 Batch Analysis Task" in prompt:
            batch_id = "BATCH_0001" if "BATCH_0001" in prompt else "BATCH_0002"
            payload = self._lp1_batch_payload(batch_id)
            return f"Batch analysis memo\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
        if "Convert the following LP1 batch analysis into JSON." in prompt:
            batch_id = "BATCH_0001" if "BATCH_0001" in prompt else "BATCH_0002"
            return json.dumps(self._lp1_batch_payload(batch_id), ensure_ascii=False)
        payload = self._call_json_prompt(prompt, **kwargs)
        if isinstance(payload, str):
            return payload
        return json.dumps(payload, ensure_ascii=False)

    def _call_markdown_prompt(self, prompt: str) -> str:
        return "# 用户画像\n\n- 有稳定的小圈子早餐社交。"


class StubFailingLP2LLM(StubV0323LLM):
    def __init__(self) -> None:
        super().__init__()
        self.relationship_calls = 0

    def _lp1_batch_payload(self, batch_id: str) -> dict:
        return {
            "events": [
                {
                    "event_id": "TEMP_EVT_001",
                    "supporting_photo_ids": ["photo_001", "photo_002", "photo_003"],
                    "meta_info": {
                        "title": "Breakfast",
                        "location_context": "Home",
                        "photo_count": 3,
                    },
                    "objective_fact": {
                        "scene_description": "Breakfast with two companions.",
                        "participants": ["Person_001", "Person_002", "Person_003"],
                    },
                    "narrative_synthesis": "Breakfast with two companions.",
                    "social_dynamics": [],
                    "persona_evidence": {
                        "behavioral": ["hosts breakfast"],
                        "aesthetic": [],
                        "socioeconomic": [],
                    },
                    "tags": ["meal"],
                    "confidence": 0.8,
                    "reason": "same time and place",
                }
            ]
        }

    def _call_json_prompt(self, prompt: str, **kwargs):
        self.json_calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
        if "Relationship Types" in prompt:
            self.relationship_calls += 1
            if self.relationship_calls == 1:
                return {
                    "relationship_type": "friend",
                    "status": "stable",
                    "confidence": 0.72,
                    "reason": "shared meal and steady co-presence",
                }
            raise RuntimeError("synthetic LP2 timeout")
        return {
            "primary_person_id": "Person_001",
            "event_grounded": {
                "behavioral_traits": ["hosts breakfast"],
                "aesthetic_traits": [],
                "socioeconomic_traits": [],
                "activity_patterns": ["meal"],
                "place_patterns": ["Home"],
            },
            "relationship_grounded": {
                "top_relationships": ["Person_002:friend"],
                "social_style": ["small-circle hangouts"],
            },
            "weak_reference": [],
            "summary": "Grounded in breakfast and friend co-presence.",
        }


class StubRetryingLP2LLM(StubV0323LLM):
    def __init__(self) -> None:
        super().__init__()
        self.relationship_attempts: dict[str, int] = {}

    def _lp1_batch_payload(self, batch_id: str) -> dict:
        return {
            "events": [
                {
                    "event_id": "TEMP_EVT_001",
                    "supporting_photo_ids": ["photo_001", "photo_002", "photo_003"],
                    "meta_info": {
                        "title": "Breakfast",
                        "location_context": "Home",
                        "photo_count": 3,
                    },
                    "objective_fact": {
                        "scene_description": "Breakfast with two companions.",
                        "participants": ["Person_001", "Person_002", "Person_003"],
                    },
                    "narrative_synthesis": "Breakfast with two companions.",
                    "social_dynamics": [],
                    "persona_evidence": {
                        "behavioral": ["hosts breakfast"],
                        "aesthetic": [],
                        "socioeconomic": [],
                    },
                    "tags": ["meal"],
                    "confidence": 0.8,
                    "reason": "same time and place",
                }
            ]
        }

    def _call_json_prompt(self, prompt: str, **kwargs):
        self.json_calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
        if "Relationship Types" in prompt:
            marker = "figure out how "
            person_id = "Person_002"
            if marker in prompt and " fits into " in prompt:
                person_id = prompt.split(marker, 1)[1].split(" fits into ", 1)[0].strip() or "Person_002"
            attempt = int(self.relationship_attempts.get(person_id, 0)) + 1
            self.relationship_attempts[person_id] = attempt
            if person_id == "Person_003" and attempt == 1:
                raise RuntimeError("synthetic LP2 timeout")
            return {
                "relationship_type": "friend",
                "status": "stable",
                "confidence": 0.72,
                "reason": f"shared meal and steady co-presence for {person_id}",
            }
        return {
            "primary_person_id": "Person_001",
            "event_grounded": {
                "behavioral_traits": ["hosts breakfast"],
                "aesthetic_traits": [],
                "socioeconomic_traits": [],
                "activity_patterns": ["meal"],
                "place_patterns": ["Home"],
            },
            "relationship_grounded": {
                "top_relationships": ["Person_002:friend"],
                "social_style": ["small-circle hangouts"],
            },
            "weak_reference": [],
            "summary": "Grounded in breakfast and friend co-presence.",
        }


class StubRetryingLP1RawLLM(StubV0323LLM):
    def __init__(self) -> None:
        super().__init__()
        self.lp1_attempts = 0

    def _call_json_prompt_raw_text(self, prompt: str, **kwargs) -> str:
        self.json_calls.append({"prompt": prompt, "kwargs": dict(kwargs), "raw_text_mode": True})
        if "LP1 Batch Analysis Task" in prompt:
            self.lp1_attempts += 1
            if self.lp1_attempts == 1:
                return "Batch analysis memo without valid JSON event block."
            payload = {
                "events": [
                    {
                        "event_id": "TEMP_EVT_001",
                        "supporting_photo_ids": ["photo_001", "photo_002"],
                        "meta_info": {
                            "title": "Breakfast",
                            "location_context": "Home",
                            "photo_count": 2,
                        },
                        "objective_fact": {
                            "scene_description": "Breakfast with friend.",
                            "participants": ["Person_001", "Person_002"],
                        },
                        "narrative_synthesis": "Breakfast with friend.",
                        "social_dynamics": [],
                        "persona_evidence": {
                            "behavioral": ["hosts breakfast"],
                            "aesthetic": [],
                            "socioeconomic": [],
                        },
                        "tags": ["meal"],
                        "confidence": 0.8,
                        "reason": "same time and place",
                    }
                ]
            }
            return f"Retry analysis memo\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
        if "Convert the following LP1 batch analysis into JSON." in prompt:
            if self.lp1_attempts == 1:
                return '{"events":[{"event_id":"TEMP_EVT_001""supporting_photo_ids":["photo_001"]}]}'
            payload = {
                "events": [
                    {
                        "event_id": "TEMP_EVT_001",
                        "supporting_photo_ids": ["photo_001", "photo_002"],
                        "meta_info": {
                            "title": "Breakfast",
                            "location_context": "Home",
                            "photo_count": 2,
                        },
                        "objective_fact": {
                            "scene_description": "Breakfast with friend.",
                            "participants": ["Person_001", "Person_002"],
                        },
                        "narrative_synthesis": "Breakfast with friend.",
                        "social_dynamics": [],
                        "persona_evidence": {
                            "behavioral": ["hosts breakfast"],
                            "aesthetic": [],
                            "socioeconomic": [],
                        },
                        "tags": ["meal"],
                        "confidence": 0.8,
                        "reason": "same time and place",
                    }
                ]
            }
            return json.dumps(payload, ensure_ascii=False)
        return super()._call_json_prompt_raw_text(prompt, **kwargs)


class StubConvertLP1LLM(StubV0323LLM):
    def _call_json_prompt_raw_text(self, prompt: str, **kwargs) -> str:
        self.json_calls.append({"prompt": prompt, "kwargs": dict(kwargs), "raw_text_mode": True})
        if "LP1 Batch Analysis Task" in prompt:
            return "Event A: photos photo_001 and photo_002 show breakfast at home with Person_001 and Person_002."
        if "Convert the following LP1 batch analysis into JSON." in prompt:
            payload = {
                "events": [
                    {
                        "event_id": "TEMP_EVT_001",
                        "supporting_photo_ids": ["photo_001", "photo_002"],
                        "meta_info": {
                            "title": "Breakfast",
                            "location_context": "Home",
                            "photo_count": 2,
                        },
                        "objective_fact": {
                            "scene_description": "Breakfast at home.",
                            "participants": ["Person_001", "Person_002"],
                        },
                        "narrative_synthesis": "Breakfast with friend.",
                        "social_dynamics": [],
                        "persona_evidence": {
                            "behavioral": ["hosts breakfast"],
                            "aesthetic": [],
                            "socioeconomic": [],
                        },
                        "tags": ["meal"],
                        "confidence": 0.8,
                        "reason": "same time and place",
                    }
                ]
            }
            return json.dumps(payload, ensure_ascii=False)
        return super()._call_json_prompt_raw_text(prompt, **kwargs)


class V0323PipelineTests(unittest.TestCase):
    def _public_url(self, path: Path) -> str:
        return f"/assets/{Path(path).name}"

    def _build_inputs(self) -> tuple[list[Photo], list[dict]]:
        photos: list[Photo] = []
        vlm_results: list[dict] = []
        for index in range(1, 202):
            photo_id = f"photo_{index:03d}"
            timestamp = datetime(2026, 3, 23, 9, 0) if index < 201 else datetime(2026, 3, 23, 10, 0)
            photo = Photo(
                photo_id=photo_id,
                filename=f"{photo_id}.jpg",
                path=f"/tmp/{photo_id}.jpg",
                timestamp=timestamp,
                location={"name": "Home"},
            )
            photo.faces = [{"person_id": "Person_001"}, {"person_id": "Person_002"}]
            photos.append(photo)
            vlm_results.append(
                {
                    "photo_id": photo_id,
                    "filename": f"{photo_id}.jpg",
                    "timestamp": timestamp.isoformat(),
                    "location": {"name": "Home"},
                    "face_person_ids": ["Person_001", "Person_002"],
                    "vlm_analysis": {
                        "summary": f"summary {photo_id}",
                        "scene": {"location_detected": "Home", "environment_description": "home scene"},
                        "event": {"activity": "meal"},
                        "people": [
                            {"person_id": "Person_001"},
                            {"person_id": "Person_002", "contact_type": "standing_near"},
                        ],
                        "relations": [],
                        "details": [],
                    },
                }
            )
        return photos, vlm_results

    def test_v0323_pipeline_batches_201_photos_with_overlap_and_continuation(self) -> None:
        photos, vlm_results = self._build_inputs()
        with tempfile.TemporaryDirectory() as tmp_dir:
            llm = StubV0323LLM()
            family = V0323PipelineFamily(
                task_id="task_v0323",
                task_dir=Path(tmp_dir),
                user_id="user_1",
                asset_store=None,
                llm_processor=llm,
                public_url_builder=self._public_url,
            )
            result = family.run(
                photos=photos,
                face_output={"persons": [{"person_id": "Person_001"}, {"person_id": "Person_002"}]},
                primary_person_id="Person_001",
                vlm_results=vlm_results,
                cached_photo_ids=set(),
                dedupe_report={},
            )

            self.assertEqual(result["pipeline_family"], "v0323")
            self.assertEqual(result["summary"]["lp1_batch_count"], 2)
            self.assertEqual(
                result["lp1_batches"][1]["overlap_context_photo_ids"],
                [f"photo_{index:03d}" for index in range(193, 201)],
            )
            self.assertEqual(len(result["lp1_events"]), 1)
            self.assertEqual(result["lp1_events"][0]["event_id"], "EVT_0001")
            self.assertEqual(result["lp1_events"][0]["title"], "Extended Breakfast")
            self.assertIn("photo_201", result["lp1_events"][0]["supporting_photo_ids"])
            self.assertEqual(len(result["lp1_event_continuation_log"]), 1)
            self.assertEqual(result["lp2_relationships"][0]["person_id"], "Person_002")
            self.assertTrue((Path(tmp_dir) / "v0323" / "lp1_batch_requests.jsonl").exists())
            self.assertTrue((Path(tmp_dir) / "v0323" / "lp1_event_continuation_log.jsonl").exists())
            analysis_calls = [
                call for call in llm.json_calls
                if "LP1 Batch Analysis Task" in call["prompt"]
            ]
            convert_calls = [
                call for call in llm.json_calls
                if "Convert the following LP1 batch analysis into JSON." in call["prompt"]
            ]
            self.assertTrue(analysis_calls)
            self.assertEqual(analysis_calls[0]["kwargs"]["max_tokens"], V0323_LP1_MAX_OUTPUT_TOKENS)
            self.assertFalse(convert_calls)

    def test_v0323_builds_reference_like_observations_from_source_type(self) -> None:
        photos = [
            Photo(
                photo_id="photo_001",
                filename="Screenshot 2026-03-25 at 18.12.11.png",
                path="/tmp/photo_001.png",
                timestamp=datetime(2026, 3, 25, 18, 12, 11),
                location={},
            ),
            Photo(
                photo_id="photo_002",
                filename="cake_reference.jpg",
                path="/tmp/photo_002.jpg",
                timestamp=datetime(2026, 3, 25, 18, 15, 0),
                location={},
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            family = V0323PipelineFamily(
                task_id="task_v0323_source_type",
                task_dir=Path(tmp_dir),
                user_id="user_1",
                asset_store=None,
                llm_processor=StubV0323LLM(),
                public_url_builder=self._public_url,
            )
            observations = family._build_vp1_observations(
                photos,
                [
                    {
                        "photo_id": "photo_001",
                        "filename": photos[0].filename,
                        "timestamp": photos[0].timestamp.isoformat(),
                        "location": {},
                        "face_person_ids": [],
                        "source_type": "screenshot",
                        "vlm_analysis": {"summary": "聊天记录截图", "source_type": "screenshot"},
                    },
                    {
                        "photo_id": "photo_002",
                        "filename": photos[1].filename,
                        "timestamp": photos[1].timestamp.isoformat(),
                        "location": {},
                        "face_person_ids": [],
                        "source_type": "ai_generated_image",
                        "vlm_analysis": {"summary": "AI generated image of a cake", "source_type": "ai_generated_image"},
                    },
                ],
            )

        self.assertEqual(observations[0]["source_type"], "screenshot")
        self.assertEqual(observations[0]["media_kind"], "screenshot")
        self.assertTrue(observations[0]["is_reference_like"])
        self.assertEqual(observations[1]["source_type"], "ai_generated_image")
        self.assertEqual(observations[1]["media_kind"], "reference_media")
        self.assertTrue(observations[1]["is_reference_like"])

    def test_v0323_pipeline_lp1_full_retry_persists_raw_preview(self) -> None:
        photos: list[Photo] = []
        vlm_results: list[dict] = []
        for index in range(1, 3):
            photo_id = f"photo_{index:03d}"
            timestamp = datetime(2026, 3, 23, 9, index)
            photo = Photo(
                photo_id=photo_id,
                filename=f"{photo_id}.jpg",
                path=f"/tmp/{photo_id}.jpg",
                timestamp=timestamp,
                location={"name": "Home"},
            )
            photo.faces = [{"person_id": "Person_001"}, {"person_id": "Person_002"}]
            photos.append(photo)
            vlm_results.append(
                {
                    "photo_id": photo_id,
                    "filename": f"{photo_id}.jpg",
                    "timestamp": timestamp.isoformat(),
                    "location": {"name": "Home"},
                    "face_person_ids": ["Person_001", "Person_002"],
                    "vlm_analysis": {
                        "summary": f"summary {photo_id}",
                        "scene": {"location_detected": "Home", "environment_description": "home scene"},
                        "event": {"activity": "meal"},
                        "people": [
                            {"person_id": "Person_001"},
                            {"person_id": "Person_002", "contact_type": "standing_near"},
                        ],
                        "relations": [],
                        "details": [],
                    },
                }
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            llm = StubRetryingLP1RawLLM()
            family = V0323PipelineFamily(
                task_id="task_v0323_lp1_retry",
                task_dir=Path(tmp_dir),
                user_id="user_1",
                asset_store=None,
                llm_processor=llm,
                public_url_builder=self._public_url,
            )
            result = family.run(
                photos=photos,
                face_output={"persons": [{"person_id": "Person_001"}, {"person_id": "Person_002"}]},
                primary_person_id="Person_001",
                vlm_results=vlm_results,
                cached_photo_ids=set(),
                dedupe_report={},
            )

            self.assertEqual(result["summary"]["lp1_batch_count"], 1)
            self.assertEqual(result["lp1_batches"][0]["parse_status"], "retry_ok")
            self.assertEqual(len(result["lp1_events"]), 1)

            request_lines = [
                json.loads(line)
                for line in (Path(tmp_dir) / "v0323" / "lp1_batch_requests.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            output_lines = [
                json.loads(line)
                for line in (Path(tmp_dir) / "v0323" / "lp1_batch_outputs.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(request_lines), V0323_LP1_MAX_ATTEMPTS)
            self.assertEqual([line["prompt_kind"] for line in request_lines], ["primary", "retry"])
            self.assertEqual(output_lines[0]["parse_status"], "convert_failed")
            self.assertIn("Batch analysis memo without valid JSON", output_lines[0]["analysis_response_preview"])
            self.assertIn('"event_id":"TEMP_EVT_001"', output_lines[0]["convert_response_preview"])
            self.assertEqual(output_lines[1]["parse_status"], "analysis_ok")
            self.assertEqual(output_lines[1]["prompt_kind"], "retry")
            self.assertEqual(request_lines[0]["prompt_char_count"], request_lines[1]["prompt_char_count"])

    def test_v0323_pipeline_lp1_convert_step_recovers_analysis_memo(self) -> None:
        photos: list[Photo] = []
        vlm_results: list[dict] = []
        for index in range(1, 3):
            photo_id = f"photo_{index:03d}"
            timestamp = datetime(2026, 3, 23, 9, index)
            photo = Photo(
                photo_id=photo_id,
                filename=f"{photo_id}.jpg",
                path=f"/tmp/{photo_id}.jpg",
                timestamp=timestamp,
                location={"name": "Home"},
            )
            photo.faces = [{"person_id": "Person_001"}, {"person_id": "Person_002"}]
            photos.append(photo)
            vlm_results.append(
                {
                    "photo_id": photo_id,
                    "filename": f"{photo_id}.jpg",
                    "timestamp": timestamp.isoformat(),
                    "location": {"name": "Home"},
                    "face_person_ids": ["Person_001", "Person_002"],
                    "vlm_analysis": {
                        "summary": f"summary {photo_id}",
                        "scene": {"location_detected": "Home", "environment_description": "home scene"},
                        "event": {"activity": "meal"},
                        "people": [
                            {"person_id": "Person_001"},
                            {"person_id": "Person_002", "contact_type": "standing_near"},
                        ],
                        "relations": [],
                        "details": [],
                    },
                }
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            llm = StubConvertLP1LLM()
            family = V0323PipelineFamily(
                task_id="task_v0323_lp1_convert",
                task_dir=Path(tmp_dir),
                user_id="user_1",
                asset_store=None,
                llm_processor=llm,
                public_url_builder=self._public_url,
            )
            result = family.run(
                photos=photos,
                face_output={"persons": [{"person_id": "Person_001"}, {"person_id": "Person_002"}]},
                primary_person_id="Person_001",
                vlm_results=vlm_results,
                cached_photo_ids=set(),
                dedupe_report={},
            )

            self.assertEqual(result["summary"]["lp1_batch_count"], 1)
            self.assertEqual(result["lp1_batches"][0]["parse_status"], "ok")
            output_lines = [
                json.loads(line)
                for line in (Path(tmp_dir) / "v0323" / "lp1_batch_outputs.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(output_lines), 1)
            self.assertEqual(output_lines[0]["parse_status"], "convert_ok")
            self.assertIn("Event A: photos photo_001", output_lines[0]["analysis_response_preview"])
            self.assertEqual(output_lines[0]["event_count"], 1)

    def test_memory_query_service_answers_v0323_snapshot(self) -> None:
        query_service = MemoryQueryService(now=datetime(2026, 3, 23, 12, 0))
        memory_payload = {
            "memory": {
                "pipeline_family": "v0323",
                "vp1_observations": [
                    {
                        "photo_id": "photo_001",
                        "vlm_analysis": {
                            "summary": "home breakfast",
                            "scene": {"location_detected": "Home"},
                        },
                    }
                ],
                "lp1_events": [
                    {
                        "event_id": "EVT_0001",
                        "title": "Breakfast",
                        "narrative_synthesis": "Breakfast at home",
                        "started_at": "2026-03-20T09:00:00",
                        "ended_at": "2026-03-20T09:30:00",
                        "place_refs": ["Home"],
                        "participant_person_ids": ["Person_001", "Person_002"],
                        "depicted_person_ids": ["Person_001", "Person_002"],
                        "supporting_photo_ids": ["photo_001"],
                        "confidence": 0.8,
                        "persona_evidence": {"behavioral": ["hosts breakfast"]},
                        "tags": ["meal"],
                    }
                ],
                "lp2_relationships": [
                    {
                        "relationship_id": "REL_Person_002",
                        "person_id": "Person_002",
                        "relationship_type": "friend",
                        "status": "stable",
                        "confidence": 0.72,
                        "reason": "shared meal",
                        "supporting_event_ids": ["EVT_0001"],
                        "supporting_photo_ids": ["photo_001"],
                    }
                ],
                "lp3_profile": {
                    "structured": {
                        "summary": "喜欢在家吃早餐。",
                        "event_grounded": {"activity_patterns": ["在家吃早餐"]},
                    },
                    "report_markdown": "# 用户画像\n\n喜欢在家吃早餐。",
                },
            }
        }

        event_answer = query_service.answer(memory_payload, "我最近吃过什么")
        relationship_answer = query_service.answer(memory_payload, "帮我探索一下用户和朋友的关系")
        profile_answer = query_service.answer(memory_payload, "给我用户画像")

        self.assertEqual(event_answer["answer"]["answer_type"], "event_search")
        self.assertIn("Breakfast", event_answer["answer"]["summary"])
        self.assertEqual(relationship_answer["answer"]["answer_type"], "relationship_explore")
        self.assertIn("Person_002", relationship_answer["answer"]["summary"])
        self.assertEqual(profile_answer["answer"]["answer_type"], "profile_lookup")
        self.assertIn("喜欢在家吃早餐", profile_answer["answer"]["summary"])

    def test_memory_core_payload_reads_v0323_snapshot(self) -> None:
        task = {
            "user_id": "user_1",
            "task_dir": "/tmp/task_v0323",
            "uploads": [
                {
                    "image_id": "photo_001",
                    "source_hash": "photo_001",
                    "filename": "photo_001.jpg",
                    "path": "uploads/photo_001.jpg",
                    "url": "/assets/photo_001.jpg",
                }
            ],
            "result": {
                "face_recognition": {"images": []},
                "memory": {
                    "pipeline_family": "v0323",
                    "vp1_observations": [{"photo_id": "photo_001", "face_person_ids": ["Person_001", "Person_002"]}],
                    "lp1_events": [
                        {
                            "event_id": "EVT_0001",
                            "title": "Breakfast",
                            "narrative_synthesis": "Breakfast at home",
                            "participant_person_ids": ["Person_001", "Person_002"],
                            "depicted_person_ids": ["Person_001", "Person_002"],
                            "supporting_photo_ids": ["photo_001"],
                        }
                    ],
                    "lp2_relationships": [
                        {
                            "relationship_id": "REL_Person_002",
                            "person_id": "Person_002",
                            "supporting_photo_ids": ["photo_001"],
                        }
                    ],
                    "lp3_profile": {
                        "structured": {
                            "summary": "喜欢在家吃早餐。",
                            "event_grounded": {"activity_patterns": ["在家吃早餐"]},
                        },
                        "report_markdown": "# 用户画像\n\n喜欢在家吃早餐。",
                    },
                },
            },
        }

        payload = build_task_memory_core_payload(task)

        self.assertEqual(payload["events"][0]["llm_summary"], "Breakfast at home")
        self.assertEqual(payload["events"][0]["photo_ids"], ["photo_001"])
        self.assertEqual(payload["relationships"][0]["person_id"], "Person_002")
        self.assertEqual(payload["profile"]["summary"], "喜欢在家吃早餐。")
        self.assertEqual(payload["profile"]["report_markdown"], "# 用户画像\n\n喜欢在家吃早餐。")
        self.assertEqual(
            payload["profile"]["structured"]["event_grounded"]["activity_patterns"],
            ["在家吃早餐"],
        )

    def test_v0323_pipeline_persists_partial_lp2_results_and_skips_failed_candidate(self) -> None:
        photos: list[Photo] = []
        vlm_results: list[dict] = []
        for index in range(1, 4):
            photo_id = f"photo_{index:03d}"
            timestamp = datetime(2026, 3, 23, 9, 0)
            photo = Photo(
                photo_id=photo_id,
                filename=f"{photo_id}.jpg",
                path=f"/tmp/{photo_id}.jpg",
                timestamp=timestamp,
                location={"name": "Home"},
            )
            photo.faces = [{"person_id": "Person_001"}, {"person_id": "Person_002"}, {"person_id": "Person_003"}]
            photos.append(photo)
            vlm_results.append(
                {
                    "photo_id": photo_id,
                    "filename": f"{photo_id}.jpg",
                    "timestamp": timestamp.isoformat(),
                    "location": {"name": "Home"},
                    "face_person_ids": ["Person_001", "Person_002", "Person_003"],
                    "vlm_analysis": {
                        "summary": f"summary {photo_id}",
                        "scene": {"location_detected": "Home", "environment_description": "home scene"},
                        "event": {"activity": "meal"},
                        "people": [
                            {"person_id": "Person_001"},
                            {"person_id": "Person_002", "contact_type": "standing_near"},
                            {"person_id": "Person_003", "contact_type": "standing_near"},
                        ],
                        "relations": [],
                        "details": [],
                    },
                }
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            family = V0323PipelineFamily(
                task_id="task_v0323_failed_lp2",
                task_dir=Path(tmp_dir),
                user_id="user_1",
                asset_store=None,
                llm_processor=StubFailingLP2LLM(),
                public_url_builder=self._public_url,
            )
            result = family.run(
                photos=photos,
                face_output={"persons": [{"person_id": "Person_001"}, {"person_id": "Person_002"}, {"person_id": "Person_003"}]},
                primary_person_id="Person_001",
                vlm_results=vlm_results,
                cached_photo_ids=set(),
                dedupe_report={},
            )

            lp2_jsonl = Path(tmp_dir) / "v0323" / "lp2_relationships.jsonl"
            llm_failures = Path(tmp_dir) / "v0323" / "llm_failures.jsonl"
            self.assertTrue(lp2_jsonl.exists())
            self.assertTrue(llm_failures.exists())
            lp2_lines = [json.loads(line) for line in lp2_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
            failure_lines = [json.loads(line) for line in llm_failures.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(lp2_lines), 1)
            self.assertEqual(lp2_lines[0]["person_id"], "Person_002")
            self.assertEqual(len(failure_lines), RELATIONSHIP_MAX_RETRIES)
            self.assertEqual(failure_lines[-1]["step"], "lp2_relationship")
            self.assertEqual(failure_lines[-1]["person_id"], "Person_003")
            self.assertEqual(failure_lines[-1]["outcome"], "skipped_after_retries")
            self.assertEqual(
                [item["call_timeout_seconds"] for item in failure_lines],
                V0323_LP2_TIMEOUT_SCHEDULE_SECONDS[:RELATIONSHIP_MAX_RETRIES],
            )
            self.assertEqual(result["summary"]["relationship_count"], 1)
            self.assertEqual(result["summary"]["lp2_failed_candidate_count"], 1)
            self.assertEqual(result["summary"]["lp2_retry_count"], max(0, RELATIONSHIP_MAX_RETRIES - 1))

    def test_v0323_pipeline_retries_lp2_candidate_before_success(self) -> None:
        photos: list[Photo] = []
        vlm_results: list[dict] = []
        for index in range(1, 4):
            photo_id = f"photo_{index:03d}"
            timestamp = datetime(2026, 3, 23, 9, 0)
            photo = Photo(
                photo_id=photo_id,
                filename=f"{photo_id}.jpg",
                path=f"/tmp/{photo_id}.jpg",
                timestamp=timestamp,
                location={"name": "Home"},
            )
            photo.faces = [{"person_id": "Person_001"}, {"person_id": "Person_002"}, {"person_id": "Person_003"}]
            photos.append(photo)
            vlm_results.append(
                {
                    "photo_id": photo_id,
                    "filename": f"{photo_id}.jpg",
                    "timestamp": timestamp.isoformat(),
                    "location": {"name": "Home"},
                    "face_person_ids": ["Person_001", "Person_002", "Person_003"],
                    "vlm_analysis": {
                        "summary": f"summary {photo_id}",
                        "scene": {"location_detected": "Home", "environment_description": "home scene"},
                        "event": {"activity": "meal"},
                        "people": [
                            {"person_id": "Person_001"},
                            {"person_id": "Person_002", "contact_type": "standing_near"},
                            {"person_id": "Person_003", "contact_type": "standing_near"},
                        ],
                        "relations": [],
                        "details": [],
                    },
                }
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            family = V0323PipelineFamily(
                task_id="task_v0323_retry_lp2",
                task_dir=Path(tmp_dir),
                user_id="user_1",
                asset_store=None,
                llm_processor=StubRetryingLP2LLM(),
                public_url_builder=self._public_url,
            )
            result = family.run(
                photos=photos,
                face_output={"persons": [{"person_id": "Person_001"}, {"person_id": "Person_002"}, {"person_id": "Person_003"}]},
                primary_person_id="Person_001",
                vlm_results=vlm_results,
                cached_photo_ids=set(),
                dedupe_report={},
            )

            llm_failures = Path(tmp_dir) / "v0323" / "llm_failures.jsonl"
            self.assertTrue(llm_failures.exists())
            failure_lines = [json.loads(line) for line in llm_failures.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(failure_lines), 1)
            self.assertEqual(failure_lines[0]["person_id"], "Person_003")
            self.assertEqual(failure_lines[0]["outcome"], "retrying")
            self.assertEqual(failure_lines[0]["call_timeout_seconds"], V0323_LP2_TIMEOUT_SCHEDULE_SECONDS[0])
            relationship_person_ids = [item["person_id"] for item in result["lp2_relationships"]]
            self.assertEqual(relationship_person_ids, ["Person_002", "Person_003"])
            self.assertEqual(result["summary"]["relationship_count"], 2)
            self.assertEqual(result["summary"]["lp2_failed_candidate_count"], 0)
            self.assertEqual(result["summary"]["lp2_retry_count"], 1)
            relationship_timeouts = [
                item["kwargs"].get("timeout", (None, None))[1]
                for item in family.llm_processor.json_calls
                if "Relationship Types" in item["prompt"]
            ]
            self.assertEqual(relationship_timeouts, [60, 60, 120])
