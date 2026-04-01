from __future__ import annotations

import unittest

from services.v0325.lp3_core.relationships import build_relationship_dossiers
from services.v0325.lp3_core.types import MemoryState, PersonScreening


class StubRelationshipEvidenceLLM:
    def __init__(self, photo_counts: dict[str, int]) -> None:
        self.photo_counts = dict(photo_counts)

    def _collect_relationship_evidence(self, person_id: str, vlm_results, events):
        photo_count = int(self.photo_counts[person_id])
        return {
            "photo_count": photo_count,
            "time_span_days": max(photo_count, 1),
            "recent_gap_days": 0,
            "monthly_frequency": round(photo_count / 10.0, 2),
        }


class V0325RelationshipDossierTests(unittest.TestCase):
    def _build_state(
        self,
        *,
        photo_counts: dict[str, int],
        memory_values: dict[str, str],
    ) -> MemoryState:
        face_db = {
            person_id: {"person_id": person_id, "photo_count": photo_count}
            for person_id, photo_count in photo_counts.items()
        }
        screening = {
            person_id: PersonScreening(
                person_id=person_id,
                person_kind="real_person",
                memory_value=memory_values[person_id],
            )
            for person_id in photo_counts
        }
        return MemoryState(
            photos=[],
            face_db=face_db,
            vlm_results=[],
            screening=screening,
            primary_decision={"primary_person_id": "Primary_Only"},
            events=[],
        )

    def test_build_relationship_dossiers_keeps_core_then_top_photo_count(self) -> None:
        core_counts = {
            "Core_001": 5,
            "Core_002": 4,
            "Core_003": 3,
            "Core_004": 2,
            "Core_005": 1,
        }
        candidate_counts = {
            f"Candidate_{index:03d}": 200 - index
            for index in range(1, 71)
        }
        photo_counts = {**core_counts, **candidate_counts}
        memory_values = {person_id: "candidate" for person_id in photo_counts}
        for person_id in core_counts:
            memory_values[person_id] = "core"
        state = self._build_state(photo_counts=photo_counts, memory_values=memory_values)

        dossiers = build_relationship_dossiers(state, StubRelationshipEvidenceLLM(photo_counts))

        self.assertEqual(len(dossiers), 60)
        dossier_ids = [dossier.person_id for dossier in dossiers]
        self.assertEqual(set(core_counts), set(dossier_ids[: len(core_counts)]))
        self.assertTrue(all(person_id in dossier_ids for person_id in core_counts))

        expected_candidates = {
            person_id
            for person_id, _photo_count in sorted(
                candidate_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:55]
        }
        excluded_candidates = set(candidate_counts) - expected_candidates
        self.assertTrue(expected_candidates.issubset(set(dossier_ids)))
        self.assertTrue(excluded_candidates.isdisjoint(set(dossier_ids)))

    def test_build_relationship_dossiers_keeps_all_core_even_above_limit(self) -> None:
        core_counts = {
            f"Core_{index:03d}": 200 - index
            for index in range(1, 62)
        }
        candidate_counts = {
            f"Candidate_{index:03d}": 500 - index
            for index in range(1, 11)
        }
        photo_counts = {**core_counts, **candidate_counts}
        memory_values = {person_id: "candidate" for person_id in photo_counts}
        for person_id in core_counts:
            memory_values[person_id] = "core"
        state = self._build_state(photo_counts=photo_counts, memory_values=memory_values)

        dossiers = build_relationship_dossiers(state, StubRelationshipEvidenceLLM(photo_counts))

        dossier_ids = [dossier.person_id for dossier in dossiers]
        self.assertEqual(len(dossiers), len(core_counts))
        self.assertEqual(set(dossier_ids), set(core_counts))
        self.assertTrue(set(candidate_counts).isdisjoint(set(dossier_ids)))


if __name__ == "__main__":
    unittest.main()
