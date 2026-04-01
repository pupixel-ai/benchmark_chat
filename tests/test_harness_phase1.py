"""
Phase 1 harness verification tests.

Covers:
- A0  DataSufficiencyGate  (_run_data_sufficiency_gate)
- A1  resolution_signal     (history_recall + _apply_upstream_agent_reflection skip)
- B   CoverageProbe         (probe + enrich_case priority routing)
- A4  auto_generate_pseudo_gt
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_fact(**kwargs):
    from services.reflection import CaseFact

    defaults = dict(
        case_id="c001",
        user_name="vigar",
        album_id="album_001",
        entity_type="profile_field",
        entity_id="dim.field",
        dimension="dim.field",
        signal_source="mainline_profile",
        first_seen_stage="lp3",
        surfaced_stage="lp3",
        routing_result="strategy_candidate",
        business_priority="high",
        auto_confidence=0.70,
        accuracy_gap_status="open",
        resolution_route="strategy_fix",
        badcase_source="empty_output_candidate",
        decision_trace={},
        tool_usage_summary={},
        upstream_output={},
    )
    defaults.update(kwargs)
    return CaseFact(**defaults)


class DataSufficiencyGateTests(unittest.TestCase):
    """A0: cases tagged data_insufficient should become routing_result='difficult_case'."""

    def _run(self, facts):
        from services.reflection.tasks import _run_data_sufficiency_gate

        return _run_data_sufficiency_gate(facts)

    def test_modal_missing_null_reason_routes_to_difficult_case(self):
        fact = _make_fact(
            upstream_output={"value": None, "null_reason": "requires_social_media"},
        )
        result = self._run([fact])
        self.assertEqual(result[0].routing_result, "difficult_case")
        self.assertEqual(result[0].resolution_route, "difficult_case")

    def test_cleared_by_field_gate_routes_to_difficult_case(self):
        fact = _make_fact(
            upstream_output={"value": None, "null_reason": "cleared_by_field_gate"},
        )
        result = self._run([fact])
        self.assertEqual(result[0].routing_result, "difficult_case")

    def test_group_level_zero_refs_routes_to_difficult_case(self):
        """If this field AND all same-domain fields have zero refs → data gap."""
        def _make_zero_trace_fact(case_id, dim, domain="identity"):
            return _make_fact(
                case_id=case_id,
                dimension=dim,
                decision_trace={"tool_trace": {"evidence_bundle": {}}},
            )

        fact_a = _make_zero_trace_fact("a", "identity.age")
        fact_b = _make_zero_trace_fact("b", "identity.gender")

        with patch("services.reflection.tasks.FIELD_SPECS", {
            "identity.age": MagicMock(domain="identity"),
            "identity.gender": MagicMock(domain="identity"),
        }):
            result = self._run([fact_a, fact_b])

        for f in result:
            self.assertEqual(f.routing_result, "difficult_case", msg=f.case_id)

    def test_isolated_zero_refs_with_group_evidence_stays_open(self):
        """Only this field has zero refs; sibling has refs → tool issue, NOT data gap → stays open."""
        fact_zero = _make_fact(
            case_id="zero",
            dimension="identity.age",
            decision_trace={"tool_trace": {"evidence_bundle": {}}},
        )
        fact_hit = _make_fact(
            case_id="hit",
            dimension="identity.gender",
            accuracy_gap_status="open",
            decision_trace={"tool_trace": {"evidence_bundle": {"vlm": [{"ref": "r1"}]}}},
        )
        with patch("services.reflection.tasks.FIELD_SPECS", {
            "identity.age": MagicMock(domain="identity"),
            "identity.gender": MagicMock(domain="identity"),
        }):
            result = self._run([fact_zero, fact_hit])

        result_by_id = {f.case_id: f for f in result}
        self.assertEqual(result_by_id["zero"].routing_result, "strategy_candidate",
                         "solo zero-ref with sibling evidence should NOT be difficult_case")

    def test_non_mainline_facts_pass_through_unchanged(self):
        fact = _make_fact(signal_source="relationship", routing_result="other")
        result = self._run([fact])
        self.assertEqual(result[0].routing_result, "other")


class CoverageProbeTests(unittest.TestCase):
    """B: structural gap detection, zero LLM."""

    def _probe(self, **kwargs):
        from services.reflection.upstream_agent import CoverageProbe
        probe = CoverageProbe()
        defaults = dict(
            field_key="some_field",
            tool_trace={},
            allowed_sources=[],
            call_policies={},
            tool_rules={},
            group_tool_traces=None,
        )
        defaults.update(kwargs)
        return probe.probe(**defaults)

    def test_tool_rule_blocked_when_allowed_sources_empty(self):
        result = self._probe(allowed_sources=[])
        self.assertTrue(result["has_gap"])
        self.assertEqual(result["gap_type"], "tool_rule_blocked")

    def test_tool_rule_blocked_when_max_refs_zero(self):
        result = self._probe(
            field_key="f",
            allowed_sources=["vlm"],
            tool_rules={"f": {"max_refs_per_source": 0}},
            tool_trace={"evidence_bundle": {"vlm": []}},
        )
        self.assertTrue(result["has_gap"])
        self.assertEqual(result["gap_type"], "tool_rule_blocked")

    def test_source_unconfigured_when_source_in_allowed_but_no_call_policy(self):
        result = self._probe(
            field_key="f",
            allowed_sources=["vlm"],
            tool_trace={"evidence_bundle": {"vlm": []}},
            call_policies={},
        )
        self.assertTrue(result["has_gap"])
        self.assertEqual(result["gap_type"], "source_unconfigured")
        self.assertIn("vlm", result["affected_sources"])

    def test_tool_called_no_hit_when_source_in_bundle_but_zero(self):
        result = self._probe(
            field_key="f",
            allowed_sources=["vlm"],
            call_policies={"f": {"append_allowed_sources": ["vlm"]}},
            tool_trace={"evidence_bundle": {"vlm": []}},
        )
        self.assertTrue(result["has_gap"])
        self.assertEqual(result["gap_type"], "tool_called_no_hit")

    def test_index_path_suspect_when_only_this_field_has_zero(self):
        sibling_trace = {"evidence_bundle": {"vlm": [{"r": 1}]}}
        result = self._probe(
            field_key="f",
            allowed_sources=["vlm"],
            call_policies={"f": {"append_allowed_sources": ["vlm"]}},
            tool_trace={"evidence_bundle": {"vlm": []}},
            group_tool_traces=[sibling_trace],
        )
        self.assertTrue(result["has_gap"])
        self.assertIn(result["gap_type"], {"tool_called_no_hit", "index_path_suspect"})

    def test_no_gap_when_source_has_hits(self):
        result = self._probe(
            field_key="f",
            allowed_sources=["vlm"],
            tool_trace={"evidence_bundle": {"vlm": [{"r": 1}, {"r": 2}]}},
            call_policies={"f": {}},
        )
        self.assertFalse(result["has_gap"])
        self.assertEqual(result["gap_type"], "none")


class CoverageProbeTriageIntegrationTests(unittest.TestCase):
    """B: UpstreamTriageScorer.enrich_case() adopts coverage_gap root_cause over heuristic."""

    def test_coverage_gap_source_takes_priority_over_heuristic(self):
        from services.reflection.upstream_triage import UpstreamTriageScorer

        fact = _make_fact(
            tool_usage_summary={
                "coverage_gap": {
                    "has_gap": True,
                    "gap_type": "source_unconfigured",
                    "affected_sources": ["vlm"],
                    "detail": "vlm not in call_policy for this field",
                },
                "evidence_count": 0,
                "tool_called": True,
                "retrieval_hit_count": 0,
            },
        )
        scorer = UpstreamTriageScorer()
        enriched = scorer.enrich_case(fact, similar_patterns=[])
        self.assertEqual(enriched.root_cause_family, "coverage_gap_source")
        self.assertGreater(enriched.fix_surface_confidence, 0.8)

    def test_coverage_gap_tool_takes_priority(self):
        from services.reflection.upstream_triage import UpstreamTriageScorer

        fact = _make_fact(
            tool_usage_summary={
                "coverage_gap": {
                    "has_gap": True,
                    "gap_type": "tool_rule_blocked",
                    "affected_sources": [],
                    "detail": "max_refs == 0",
                },
            },
        )
        scorer = UpstreamTriageScorer()
        enriched = scorer.enrich_case(fact, similar_patterns=[])
        self.assertEqual(enriched.root_cause_family, "coverage_gap_tool")

    def test_no_gap_falls_through_to_heuristic(self):
        from services.reflection.upstream_triage import UpstreamTriageScorer

        fact = _make_fact(
            tool_usage_summary={
                "coverage_gap": {"has_gap": False, "gap_type": "none"},
                "evidence_count": 3,
                "tool_called": True,
                "retrieval_hit_count": 3,
            },
        )
        scorer = UpstreamTriageScorer()
        enriched = scorer.enrich_case(fact, similar_patterns=[])
        # Just verifies it doesn't crash and produces a root_cause from the normal 7
        from services.reflection.upstream_triage import ALLOWED_ROOT_CAUSE_FAMILIES
        self.assertIn(enriched.root_cause_family, ALLOWED_ROOT_CAUSE_FAMILIES)


class ResolutionSignalTests(unittest.TestCase):
    """A1: history_recall returns resolution_signal; already_fixed cases are skipped."""

    def _make_packet_with_history(self, *, outcome_status: str, dimension: str = "identity.age"):
        """Build a packet with pre-populated history so history_recall can produce a signal."""
        from services.reflection.upstream_agent import BadcasePacketAssembler
        fact = _make_fact(dimension=dimension, user_name="vigar")
        packet = {
            "case_fact": fact.to_dict(),
            "history_patterns": [
                {
                    "pattern_id": "pat1",
                    "dimension": dimension,
                    "root_cause_candidates": ["field_reasoning"],
                    "fix_surface_candidates": ["field_cot"],
                }
            ],
            "history_experiments": [
                {
                    "experiment_id": "exp1",
                    "pattern_id": "pat1",
                    "fix_surface": "field_cot",
                }
            ],
            "history_outcomes": [
                {
                    "outcome_id": "o1",
                    "experiment_id": "exp1",
                    "status": outcome_status,
                    "summary": "outcome",
                }
            ],
            "history_feedback": [],
        }
        return BadcasePacketAssembler(project_root="."), packet

    def test_history_recall_returns_already_fixed_when_success_outcome_exists(self):
        from services.reflection.upstream_agent import BadcasePacketAssembler

        _, packet = self._make_packet_with_history(outcome_status="success")
        assembler = BadcasePacketAssembler(project_root=".")
        recall = assembler.history_recall(packet)

        self.assertEqual(recall["resolution_signal"], "already_fixed",
                         f"Expected already_fixed, got: {recall.get('resolution_signal')}")

    def test_history_recall_returns_failed_before_when_only_failed_outcomes(self):
        _, packet = self._make_packet_with_history(outcome_status="failed")
        from services.reflection.upstream_agent import BadcasePacketAssembler
        assembler = BadcasePacketAssembler(project_root=".")
        recall = assembler.history_recall(packet)
        self.assertEqual(recall["resolution_signal"], "failed_before")

    def test_history_recall_returns_open_when_no_outcomes(self):
        from services.reflection.upstream_agent import BadcasePacketAssembler
        fact = _make_fact(dimension="identity.age")
        packet = {"case_fact": fact.to_dict(), "history_patterns": [], "history_experiments": [], "history_outcomes": [], "history_feedback": []}
        assembler = BadcasePacketAssembler(project_root=".")
        recall = assembler.history_recall(packet)
        self.assertEqual(recall["resolution_signal"], "open")

    def test_already_fixed_case_skipped_in_apply_upstream(self):
        from services.reflection.tasks import _apply_upstream_agent_reflection

        fact = _make_fact(dimension="identity.age", user_name="vigar")
        llm_called = []

        # Patch history_recall to return already_fixed
        with patch("services.reflection.tasks.BadcasePacketAssembler") as MockAssembler:
            mock_instance = MockAssembler.return_value
            mock_instance.assemble.return_value = {"case_fact": fact.to_dict()}
            mock_instance.history_recall.return_value = {"resolution_signal": "already_fixed"}

            with patch("services.reflection.upstream_agent.UpstreamReflectionAgent.reflect") as mock_reflect:
                mock_reflect.side_effect = lambda p: llm_called.append(1) or {"status": "ok"}
                _apply_upstream_agent_reflection(project_root=".", case_facts=[fact])

        self.assertEqual(len(llm_called), 0,
                         "LLM reflect() should NOT be called for already_fixed case")


class PseudoGTTests(unittest.TestCase):
    """A4: auto_generate_pseudo_gt writes only enum, high-confidence, Judge-accepted cases."""

    def test_writes_high_confidence_accepted_enum_field(self):
        from services.reflection.gt import auto_generate_pseudo_gt

        fact = _make_fact(
            auto_confidence=0.90,
            badcase_source="empty_output_candidate",
            upstream_output={"value": "college_student"},
            downstream_judge={"verdict": "accept"},
            causality_route="upstream_caused",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_path = str(Path(tmpdir) / "gt.jsonl")
            records = auto_generate_pseudo_gt(
                [fact], gt_path,
                enum_field_keys={"dim.field"},
                confidence_threshold=0.85,
            )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source"], "auto_high_confidence")
        self.assertEqual(records[0]["gt_value"], "college_student")

    def test_skips_low_confidence(self):
        from services.reflection.gt import auto_generate_pseudo_gt

        fact = _make_fact(
            auto_confidence=0.60,
            badcase_source="empty_output_candidate",
            upstream_output={"value": "college_student"},
            downstream_judge={"verdict": "accept"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_path = str(Path(tmpdir) / "gt.jsonl")
            records = auto_generate_pseudo_gt([fact], gt_path, enum_field_keys={"dim.field"})
        self.assertEqual(len(records), 0)

    def test_skips_non_enum_field(self):
        from services.reflection.gt import auto_generate_pseudo_gt

        fact = _make_fact(
            auto_confidence=0.92,
            badcase_source="empty_output_candidate",
            upstream_output={"value": "some free text"},
            downstream_judge={"verdict": "accept"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_path = str(Path(tmpdir) / "gt.jsonl")
            records = auto_generate_pseudo_gt(
                [fact], gt_path,
                enum_field_keys={"other.field"},  # dim.field not in enum whitelist
            )
        self.assertEqual(len(records), 0)

    def test_skips_judge_non_accept(self):
        from services.reflection.gt import auto_generate_pseudo_gt

        fact = _make_fact(
            auto_confidence=0.92,
            badcase_source="empty_output_candidate",
            upstream_output={"value": "college_student"},
            downstream_judge={"verdict": "reject"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_path = str(Path(tmpdir) / "gt.jsonl")
            records = auto_generate_pseudo_gt([fact], gt_path, enum_field_keys={"dim.field"})
        self.assertEqual(len(records), 0)

    def test_skips_mismatch_case(self):
        from services.reflection.gt import auto_generate_pseudo_gt

        fact = _make_fact(
            auto_confidence=0.92,
            badcase_source="gt_mismatch_candidate",
            upstream_output={"value": "college_student"},
            downstream_judge={"verdict": "accept"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_path = str(Path(tmpdir) / "gt.jsonl")
            records = auto_generate_pseudo_gt([fact], gt_path, enum_field_keys={"dim.field"})
        self.assertEqual(len(records), 0)

    def test_no_duplicate_writes(self):
        from services.reflection.gt import auto_generate_pseudo_gt

        fact = _make_fact(
            auto_confidence=0.92,
            badcase_source="empty_output_candidate",
            upstream_output={"value": "college_student"},
            downstream_judge={"verdict": "accept"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_path = str(Path(tmpdir) / "gt.jsonl")
            first = auto_generate_pseudo_gt([fact], gt_path, enum_field_keys={"dim.field"})
            second = auto_generate_pseudo_gt([fact], gt_path, enum_field_keys={"dim.field"})
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 0, "Second run should skip already-written GT")


if __name__ == "__main__":
    unittest.main()
