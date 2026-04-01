from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class ProfileRuleAssetTests(unittest.TestCase):
    def test_effective_field_specs_merge_repo_runtime_and_overlay_without_mutating_base_specs(self) -> None:
        from services.memory_pipeline.profile_fields import FIELD_SPECS, apply_runtime_field_spec_updates
        from services.memory_pipeline.rule_asset_loader import (
            clear_runtime_rule_overlays,
            get_effective_field_specs,
        )

        field_key = "long_term_facts.social_identity.education"
        original_spec = FIELD_SPECS[field_key].to_dict()

        with tempfile.TemporaryDirectory() as tmpdir:
            rule_asset_dir = Path(tmpdir) / "services" / "memory_pipeline" / "rule_assets"
            rule_asset_dir.mkdir(parents=True, exist_ok=True)
            (rule_asset_dir / "field_specs.overrides.json").write_text(
                json.dumps(
                    {
                        field_key: {
                            "cot_steps": ["repo_cot_step"],
                            "null_preferred_when": ["repo_null_clause"],
                        }
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (rule_asset_dir / "tool_rules.json").write_text("{}", encoding="utf-8")
            (rule_asset_dir / "call_policies.json").write_text("{}", encoding="utf-8")

            apply_runtime_field_spec_updates(
                {
                    field_key: {
                        "cot_steps": ["runtime_cot_step"],
                    }
                }
            )
            effective_specs = get_effective_field_specs(
                base_field_specs=FIELD_SPECS,
                project_root=tmpdir,
                overlay_bundle={
                    "field_spec_overrides": {
                        field_key: {
                            "counter_evidence_checks": ["overlay_counter_check"],
                        }
                    }
                },
            )

        self.assertEqual(FIELD_SPECS[field_key].to_dict(), original_spec)
        self.assertEqual(effective_specs[field_key].cot_steps, ["runtime_cot_step"])
        self.assertEqual(effective_specs[field_key].null_preferred_when, ["repo_null_clause"])
        self.assertEqual(effective_specs[field_key].counter_evidence_checks, ["overlay_counter_check"])

        clear_runtime_rule_overlays()
