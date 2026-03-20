from __future__ import annotations

import unittest

from backend.progress_utils import append_terminal_error, merge_stage_progress


class ProgressUtilsTests(unittest.TestCase):
    def test_merge_stage_progress_appends_logs_and_stages(self) -> None:
        progress = merge_stage_progress(
            None,
            "vlm",
            {"message": "进行视觉分析", "processed": 3, "photo_count": 10, "percent": 30},
        )

        self.assertEqual(progress["current_stage"], "vlm")
        self.assertIn("vlm", progress["stages"])
        self.assertEqual(len(progress["logs"]), 1)
        self.assertEqual(progress["logs"][0]["message"], "进行视觉分析")
        self.assertEqual(progress["logs"][0]["processed"], 3)
        self.assertEqual(progress["logs"][0]["total"], 10)

    def test_merge_stage_progress_dedupes_identical_consecutive_logs(self) -> None:
        progress = merge_stage_progress(
            None,
            "llm",
            {"message": "LLM 改写中", "substage": "slice_contract", "percent": 5},
        )
        progress = merge_stage_progress(
            progress,
            "llm",
            {"message": "LLM 改写中", "substage": "slice_contract", "percent": 5},
        )

        self.assertEqual(len(progress["logs"]), 1)

    def test_append_terminal_error_records_error_entry(self) -> None:
        progress = append_terminal_error(
            None,
            stage="failed",
            error="Response ended prematurely",
            substage="slice_contract",
        )

        self.assertEqual(progress["current_stage"], "failed")
        self.assertEqual(progress["logs"][0]["level"], "error")
        self.assertEqual(progress["logs"][0]["error"], "Response ended prematurely")


if __name__ == "__main__":
    unittest.main()
